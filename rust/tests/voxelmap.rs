// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! VIO-01's integration test: `VoxelMap::build` gives every lattice voxel
//! the cell that contains its centre - the lowest one on a tie (117.6) -
//! coarsens to the budget by (117.3)-(117.4), fills the gaps of 117.4's
//! pass 2, refuses by name as 117.12 says, and reports per 117.7. Every
//! number is evaluated in f64 in BOTH builds; the map itself is bit-identical
//! across the builds (117.2 item 1), which is what test 11 pins with three
//! FNV-1a literals.
//!
//! VIO-02's sampling tests follow: `linear` sampling is exact on a linear
//! field, never creates a new extremum, falls back on a mesh one cell
//! thick, and `cell` sampling conserves (117.9); the stencils' neighbour
//! lists match the test's own search, and the sample refusals and the
//! non-finite-field contract are pinned.
//!
//! Written from SPEC-LIT section 117 (117.2-117.5, 117.7).
//! Provenance: ORIGINAL.
//! No GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::error::Error;
use ofgpu::io::polymesh::{build_host_mesh, PolyMeshRaw};
use ofgpu::io::voxelmap::{
    coarsen_to_budget, lattice_dims, LatticeSample, LatticeSpec, LinearStencil, VoxelMap, NONE,
};
use ofgpu::mesh::refined::refined_core;
use ofgpu::mesh::{HostMesh, PatchInfo, PatchKind};
#[cfg(not(feature = "single"))]
use ofgpu::parcels::locate_cell;
use ofgpu::types::{DVec3, Vec3};
use ofgpu::{Label, Scalar};

// ==========================================================================
//  Fixtures
// ==========================================================================

/// A uniform axis of a block.
fn axis(lo: f64, hi: f64, n: usize) -> GradedAxis {
    GradedAxis {
        lo: lo as Scalar,
        hi: hi as Scalar,
        n,
        expansion: 1.0,
        two_sided: false,
    }
}

/// The unit cube, `n` cells of `1/n` per axis. Cell index `i + n (j + n k)`.
fn cube(n: usize) -> (PolyMeshRaw, HostMesh) {
    let spec = BlockSpec {
        x: axis(0.0, 1.0, n),
        y: axis(0.0, 1.0, n),
        z: axis(0.0, 1.0, n),
        ..Default::default()
    };
    let raw = blockgen::raw_mesh(&spec).expect("raw_mesh");
    let hm = build_host_mesh(&raw).expect("build_host_mesh");
    (raw, hm)
}

/// `[0,1] x [0,0.5] x [0,0.25]`, 8 x 4 x 2 cubes of 1/8.
fn slab() -> (PolyMeshRaw, HostMesh) {
    let spec = BlockSpec {
        x: axis(0.0, 1.0, 8),
        y: axis(0.0, 0.5, 4),
        z: axis(0.0, 0.25, 2),
        ..Default::default()
    };
    let raw = blockgen::raw_mesh(&spec).expect("raw_mesh");
    let hm = build_host_mesh(&raw).expect("build_host_mesh");
    (raw, hm)
}

/// The 2:1 refined box of `mesh/refined.rs`: 120 cells, 56 of edge 1/4 and
/// 64 of edge 1/8, with faces, edges and corners on the interface.
fn refined() -> (PolyMeshRaw, HostMesh) {
    let rb = refined_core([4, 4, 4], Vec3::new(0.25, 0.25, 0.25), 0.3, 1).expect("refined_core");
    let points: Vec<DVec3> = rb
        .points
        .iter()
        .map(|&p| DVec3::new(p.x as f64, p.y as f64, p.z as f64))
        .collect();
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

/// Carve the kept cells out of a raw mesh: kept cells renumbered in ascending
/// old order; first the internal faces whose two cells are both kept
/// (original order); then each original patch with its faces whose owner is
/// kept (original order, start/size recomputed, name/type/kind copied); then
/// ONE new patch `sphere` (type `wall`) holding the internal faces with
/// exactly one kept cell (original order), point list REVERSED when the kept
/// cell was the face's neighbour. Points are copied unchanged.
fn carve(raw: &PolyMeshRaw, keep: &[bool]) -> PolyMeshRaw {
    let n_if = raw.neighbour.len();
    let new_of_old: Vec<i64> = keep
        .iter()
        .scan(-1i64, |next, &k| {
            if k {
                *next += 1;
                Some(*next)
            } else {
                Some(-1)
            }
        })
        .collect();
    let mut faces: Vec<Vec<Label>> = Vec::new();
    let mut owner: Vec<Label> = Vec::new();
    let mut neighbour: Vec<Label> = Vec::new();
    for f in 0..n_if {
        let (o, n) = (raw.owner[f] as usize, raw.neighbour[f] as usize);
        if keep[o] && keep[n] {
            faces.push(raw.faces[f].clone());
            owner.push(new_of_old[o] as Label);
            neighbour.push(new_of_old[n] as Label);
        }
    }
    let mut patches: Vec<PatchInfo> = Vec::new();
    for p in &raw.patches {
        let start = faces.len() - neighbour.len();
        for bf in p.start..p.start + p.size {
            let f = n_if + bf;
            let o = raw.owner[f] as usize;
            if keep[o] {
                faces.push(raw.faces[f].clone());
                owner.push(new_of_old[o] as Label);
            }
        }
        patches.push(PatchInfo {
            name: p.name.clone(),
            type_name: p.type_name.clone(),
            kind: p.kind,
            start,
            size: faces.len() - neighbour.len() - start,
            nbr_patch: None,
        });
    }
    let start = faces.len() - neighbour.len();
    for f in 0..n_if {
        let (o, n) = (raw.owner[f] as usize, raw.neighbour[f] as usize);
        if keep[o] != keep[n] {
            let mut pts = raw.faces[f].clone();
            if !keep[o] {
                pts.reverse();
            }
            let kept = if keep[o] { o } else { n };
            faces.push(pts);
            owner.push(new_of_old[kept] as Label);
        }
    }
    patches.push(PatchInfo {
        name: "sphere".to_string(),
        type_name: "wall".to_string(),
        kind: PatchKind::Wall,
        start,
        size: faces.len() - neighbour.len() - start,
        nbr_patch: None,
    });
    PolyMeshRaw {
        points: raw.points.clone(),
        faces,
        owner,
        neighbour,
        patches,
    }
}

/// The test's own (117.6) oracle: cells ascending, the half-space test of
/// `parcels::locate_cell` evaluated in f64 over `geom64()`; the first cell
/// that contains `x`.
fn brute(hm: &HostMesh, x: DVec3) -> Option<usize> {
    let g = hm.geom64();
    for c in 0..hm.n_cells {
        let mut inside = true;
        for k in hm.cf_offset[c] as usize..hm.cf_offset[c + 1] as usize {
            let f = hm.cf_face[k] as usize;
            let sgn: f64 = if hm.cf_own[k] != 0 { 1.0 } else { -1.0 };
            if (x - g.cf[f]).dot(g.sf[f] * sgn) > 0.0 {
                inside = false;
                break;
            }
        }
        if inside {
            for k in hm.bcf_offset[c] as usize..hm.bcf_offset[c + 1] as usize {
                let bf = hm.bcf_face[k] as usize;
                if (x - g.b_cf[bf]).dot(g.b_sf[bf]) > 0.0 {
                    inside = false;
                    break;
                }
            }
        }
        if inside {
            return Some(c);
        }
    }
    None
}

/// FNV-1a 64 over each `u32`'s little-endian bytes.
fn fnv(cells: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &c in cells {
        for b in c.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// FNV-1a 64 over each f64's bit pattern, little-endian bytes - the
/// sampling hashes' companion to [`fnv`], same constants.
fn fnv_f64(values: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &s in values {
        for b in s.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// 117.4's test mesh: two cells over `[0,2] x [0,2] x [0,1]`. Cell 0 is the
/// L prism (the squares with low corner (0,0), (1,0), (0,1) extruded over
/// z), cell 1 the square with low corner (1,1). Points `p(i,j,k)` at index
/// `i + 3 j + 9 k`, coordinates `(i, j, k)`. Variant B keeps only cell 0,
/// its two ex-internal faces now its first two boundary faces.
fn l_mesh(variant_b: bool) -> (PolyMeshRaw, HostMesh) {
    let mut points = Vec::with_capacity(18);
    for k in 0..=1i32 {
        for j in 0..=2i32 {
            for i in 0..=2i32 {
                points.push(DVec3::new(f64::from(i), f64::from(j), f64::from(k)));
            }
        }
    }
    let f = |p: &[Label]| p.to_vec();
    let (faces, owner, neighbour): (Vec<Vec<Label>>, Vec<Label>, Vec<Label>) = if variant_b {
        (
            vec![
                f(&[4, 13, 14, 5]),
                f(&[4, 7, 16, 13]),
                f(&[0, 9, 15, 6]),
                f(&[0, 2, 11, 9]),
                f(&[2, 5, 14, 11]),
                f(&[6, 15, 16, 7]),
                f(&[0, 6, 7, 4, 5, 2]),
                f(&[9, 11, 14, 13, 16, 15]),
            ],
            vec![0; 8],
            vec![],
        )
    } else {
        (
            vec![
                f(&[4, 13, 14, 5]),
                f(&[4, 7, 16, 13]),
                f(&[0, 9, 15, 6]),
                f(&[0, 2, 11, 9]),
                f(&[2, 5, 14, 11]),
                f(&[6, 15, 16, 7]),
                f(&[0, 6, 7, 4, 5, 2]),
                f(&[9, 11, 14, 13, 16, 15]),
                f(&[5, 8, 17, 14]),
                f(&[7, 16, 17, 8]),
                f(&[4, 7, 8, 5]),
                f(&[13, 14, 17, 16]),
            ],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            vec![1, 1],
        )
    };
    let n_bf = faces.len() - neighbour.len();
    let patches = vec![PatchInfo {
        name: "walls".to_string(),
        type_name: "wall".to_string(),
        kind: PatchKind::Wall,
        start: 0,
        size: n_bf,
        nbr_patch: None,
    }];
    let raw = PolyMeshRaw {
        points,
        faces,
        owner,
        neighbour,
        patches,
    };
    let hm = build_host_mesh(&raw).expect("build_host_mesh on the L mesh");
    (raw, hm)
}

// ==========================================================================
//  Tests
// ==========================================================================

/// On a uniform Cartesian box with h equal to the cell size and the default
/// box, (117.2) puts every voxel centre on a cell centre - VIO-01's identity
/// gate (117.3).
#[test]
fn identity_cartesian() {
    let (raw, hm) = cube(16);
    let spec = LatticeSpec {
        voxel: Some(0.0625),
        ..Default::default()
    };
    let m = VoxelMap::build(&raw, &hm, &spec).expect("build");
    assert_eq!(m.n, [16, 16, 16]);
    assert_eq!(m.n_voxels(), 4096);
    for v in 0..4096usize {
        assert_eq!(m.cell[v], v as u32, "voxel {v}");
    }
    assert_eq!(m.report.n_fluid, 4096);
    assert_eq!(m.report.n_filled, 0);
    assert_eq!(m.report.fluid_fraction, 1.0);
    assert_eq!(m.report.coarsen, 1.0);
    assert_eq!(m.report.note, None);
    let ve = m.report.volume_error.expect("volume_error");
    assert!(ve.abs() <= 1e-12, "volume_error {ve}");
    assert_eq!(m.lo, DVec3::new(0.0, 0.0, 0.0));
    assert_eq!(m.origin(), DVec3::new(1.0 / 32.0, 1.0 / 32.0, 1.0 / 32.0));

    // The default voxel: (min_c V_c)^(1/3) = 1/16 here.
    let m = VoxelMap::build(&raw, &hm, &LatticeSpec::default()).expect("build");
    assert!(
        (m.voxel - 0.0625).abs() <= 1e-15,
        "default voxel {}",
        m.voxel
    );
    assert_eq!(m.n, [16, 16, 16]);
    for v in 0..4096usize {
        assert_eq!(m.cell[v], v as u32, "voxel {v}");
    }

    // The slab, voxel = cell size.
    let (raw, hm) = slab();
    let spec = LatticeSpec {
        voxel: Some(0.125),
        ..Default::default()
    };
    let m = VoxelMap::build(&raw, &hm, &spec).expect("build");
    assert_eq!(m.n, [8, 4, 2]);
    for v in 0..64usize {
        assert_eq!(m.cell[v], v as u32, "voxel {v}");
    }
}

/// (117.6) against a brute-force f64 oracle on the 2:1 mesh, voxel by voxel,
/// plus 117.7's coverage identity per cell: n_c h^3 = V_c.
#[test]
fn matches_locate_cell() {
    let (raw, hm) = refined();
    let spec = LatticeSpec {
        voxel: Some(1.0 / 24.0),
        ..Default::default()
    };
    let m = VoxelMap::build(&raw, &hm, &spec).expect("build");
    assert_eq!(m.n, [24, 24, 24]);
    assert_eq!(m.n_voxels(), 13824);
    assert_eq!(m.report.n_fluid, 13824);
    assert_eq!(m.report.n_filled, 0);
    let mut owned = vec![0usize; hm.n_cells];
    for k in 0..24 {
        for j in 0..24 {
            for i in 0..24 {
                let x = m.centre(i, j, k);
                let o = m.owner(i, j, k);
                assert_eq!(o, brute(&hm, x), "voxel ({i},{j},{k}) at ({:?})", x);
                #[cfg(not(feature = "single"))]
                assert_eq!(o, locate_cell(&hm, x), "locate_cell voxel ({i},{j},{k})");
                if let Some(c) = o {
                    owned[c] += 1;
                }
            }
        }
    }
    let g = hm.geom64();
    let h3 = m.voxel * m.voxel * m.voxel;
    for c in 0..hm.n_cells {
        let v = g.v[c];
        let err = ((owned[c] as f64) * h3 - v).abs();
        assert!(err <= 1e-12 * v, "cell {c}: {} voxels, V_c {v}", owned[c]);
    }
}

/// Centres on cell faces, edges and corners of the 2:1 mesh: every voxel
/// agrees with the brute-force oracle, i.e. ties go to the lowest cell
/// index (117.4, 117.6).
#[test]
fn ties_follow_the_brute_force_on_a_2to1_mesh() {
    let (raw, hm) = refined();
    let spec = LatticeSpec {
        bbox: Some([
            -1.0 / 32.0,
            -1.0 / 32.0,
            -1.0 / 32.0,
            1.0 + 1.0 / 32.0,
            1.0 + 1.0 / 32.0,
            1.0 + 1.0 / 32.0,
        ]),
        voxel: Some(0.0625),
        ..Default::default()
    };
    let m = VoxelMap::build(&raw, &hm, &spec).expect("build");
    assert_eq!(m.n, [17, 17, 17]);
    assert_eq!(m.n_voxels(), 4913);
    let mut n_owned = 0;
    for k in 0..17 {
        for j in 0..17 {
            for i in 0..17 {
                let x = m.centre(i, j, k);
                // Every centre is (i/16, j/16, k/16), on cell faces, edges
                // and corners - exact in f64.
                assert_eq!(x, DVec3::new(i as f64 / 16.0, j as f64 / 16.0, k as f64 / 16.0));
                assert_eq!(
                    m.owner(i, j, k),
                    brute(&hm, x),
                    "voxel ({i},{j},{k}) at ({:?})",
                    x
                );
                if m.owner(i, j, k).is_some() {
                    n_owned += 1;
                }
            }
        }
    }
    assert_eq!(n_owned, 4913);
    assert_eq!(m.report.n_fluid, 4913);
    assert_eq!(m.report.volume_error, None);
}

/// The 8 cells around an interior cell corner: the corner voxel goes to the
/// lowest of them (117.4's tie rule); every voxel off the mesh stays NONE.
#[test]
fn ties_go_to_the_lowest_cell() {
    let (raw, hm) = cube(16);
    let spec = LatticeSpec {
        bbox: Some([-0.5, -0.5, -0.5, 1.5, 1.5, 1.5]),
        voxel: Some(0.125),
        ..Default::default()
    };
    let m = VoxelMap::build(&raw, &hm, &spec).expect("build");
    assert_eq!(m.n, [16, 16, 16]);
    let mut n_owned = 0;
    for k in 0..16usize {
        for j in 0..16usize {
            for i in 0..16usize {
                let v = m.index(i, j, k);
                let on = |a: usize| (4..=11).contains(&a);
                if on(i) && on(j) && on(k) {
                    // The centre is the corner shared by 8 cells; the owner
                    // is the lowest, (a-1, b-1, c-1) with a = 2i-7 etc.
                    let (a, b, c) = (2 * i - 7, 2 * j - 7, 2 * k - 7);
                    let want = (a - 1) + 16 * ((b - 1) + 16 * (c - 1));
                    assert_eq!(m.cell[v], want as u32, "voxel ({i},{j},{k})");
                    n_owned += 1;
                } else {
                    assert_eq!(m.cell[v], NONE, "voxel ({i},{j},{k})");
                }
            }
        }
    }
    assert_eq!(n_owned, 512);
    assert_eq!(m.report.fluid_fraction, 0.125);
    assert_eq!(m.report.volume_error, None);
}

/// 117.7's sphere gate: a carved sphere of R = 0.3 in the unit cube, voxels
/// at half the cell size. fluid_fraction = 1 - V_s/V_b within A_s h / V_b.
#[test]
fn fluid_fraction_sphere() {
    let (raw, _hm) = cube(32);
    let keep: Vec<bool> = (0..32 * 32 * 32)
        .map(|idx| {
            let i = idx % 32;
            let j = idx / 32 % 32;
            let k = idx / 1024;
            let dx = (i as f64 + 0.5) / 32.0 - 0.5;
            let dy = (j as f64 + 0.5) / 32.0 - 0.5;
            let dz = (k as f64 + 0.5) / 32.0 - 0.5;
            dx * dx + dy * dy + dz * dz >= 0.09
        })
        .collect();
    assert_eq!(keep.iter().filter(|&&k| !k).count(), 3648);
    assert_eq!(keep.iter().filter(|&&k| k).count(), 29120);
    let raw2 = carve(&raw, &keep);
    let hm2 = build_host_mesh(&raw2).expect("build_host_mesh on the carved mesh");

    let spec = LatticeSpec {
        voxel: Some(1.0 / 64.0),
        ..Default::default()
    };
    let m = VoxelMap::build(&raw2, &hm2, &spec).expect("build");
    assert_eq!(m.n, [64, 64, 64]);
    let mut new_of_old = vec![NONE; 32 * 32 * 32];
    let mut next = 0usize;
    for (old, &k) in keep.iter().enumerate() {
        if k {
            new_of_old[old] = next as u32;
            next += 1;
        }
    }
    for k in 0..64usize {
        for j in 0..64usize {
            for i in 0..64usize {
                let old = i / 2 + 32 * (j / 2 + 32 * (k / 2));
                let v = m.index(i, j, k);
                assert_eq!(m.cell[v], new_of_old[old], "voxel ({i},{j},{k})");
            }
        }
    }
    assert_eq!(m.report.n_fluid, 232960);
    assert_eq!(m.report.n_filled, 0);
    assert_eq!(m.report.fluid_fraction, 0.888671875);
    let (r, v_b) = (0.3f64, 1.0f64);
    let v_s = 4.0 / 3.0 * std::f64::consts::PI * r * r * r;
    let a_s = 4.0 * std::f64::consts::PI * r * r;
    let gate = a_s * m.voxel / v_b;
    assert!(
        (m.report.fluid_fraction - (1.0 - v_s / v_b)).abs() <= gate,
        "fluid_fraction {} against gate {gate}",
        m.report.fluid_fraction
    );
    let ve = m.report.volume_error.expect("volume_error");
    assert!(ve.abs() <= 1e-12, "volume_error {ve}");
}

/// (117.3)-(117.4): the budget coarsens to the smallest fitting step; an
/// exact hold leaves coarsen 1.0 and no note.
#[test]
fn cap_coarsens() {
    let (raw, hm) = cube(16);

    // (a) 4096 asked, 1000 allowed: 10 x 10 x 10 at h' = 0.1, s = 1.6.
    let spec = LatticeSpec {
        voxel: Some(0.0625),
        max_voxels: 1000,
        ..Default::default()
    };
    let m = VoxelMap::build(&raw, &hm, &spec).expect("build");
    assert_eq!(m.n, [10, 10, 10]);
    assert_eq!(m.voxel, 0.1);
    assert_eq!(m.report.voxel_requested, 0.0625);
    assert_eq!(m.report.coarsen, 1.6);
    assert_eq!(m.report.n_voxels, 1000);
    assert_eq!(m.report.n_fluid, 1000);
    let note = m.report.note.clone().expect("note");
    for word in ["0.0625", "0.1", "1.6", "1000"] {
        assert!(note.contains(word), "note '{note}' lacks '{word}'");
    }

    // (b) the budget holding exactly, and the step just over it.
    let spec = LatticeSpec {
        voxel: Some(0.0625),
        max_voxels: 4096,
        ..Default::default()
    };
    let m = VoxelMap::build(&raw, &hm, &spec).expect("build");
    assert_eq!(m.report.coarsen, 1.0);
    assert_eq!(m.report.note, None);
    let spec = LatticeSpec {
        voxel: Some(0.0625),
        max_voxels: 4095,
        ..Default::default()
    };
    let m = VoxelMap::build(&raw, &hm, &spec).expect("build");
    assert_eq!(m.n, [15, 15, 15]);
    assert!((m.voxel - 1.0 / 15.0).abs() <= 1e-15, "voxel {}", m.voxel);

    // (c) the slab: 512 asked.
    let (raw, hm) = slab();
    let spec = LatticeSpec {
        voxel: Some(0.0625),
        max_voxels: 100,
        ..Default::default()
    };
    let m = VoxelMap::build(&raw, &hm, &spec).expect("build");
    assert_eq!(m.voxel, 0.125);
    assert_eq!(m.report.coarsen, 2.0);
    assert_eq!(m.n, [8, 4, 2]);
    for v in 0..64usize {
        assert_eq!(m.cell[v], v as u32, "voxel {v}");
    }
    let spec = LatticeSpec {
        voxel: Some(0.0625),
        max_voxels: 511,
        ..Default::default()
    };
    let m = VoxelMap::build(&raw, &hm, &spec).expect("build");
    assert_eq!(m.n, [15, 8, 4]);
    assert!((m.voxel - 1.0 / 15.0).abs() <= 1e-15, "voxel {}", m.voxel);
    assert!(
        (m.report.coarsen - 16.0 / 15.0).abs() <= 1e-15,
        "coarsen {}",
        m.report.coarsen
    );
}

/// 117.4's pass 2 on the non-convex L cell: pass 1's half-space intersection
/// of cell 0 is only the unit square below both internal-face planes, so
/// 128 voxels are filled by nearest centre. With cell 1 removed, the same
/// planes are boundary faces and every outside voxel stays solid.
#[test]
fn gap_fill_l_cell() {
    // Variant A: both cells.
    let (raw, hm) = l_mesh(false);
    let spec = LatticeSpec {
        voxel: Some(0.25),
        ..Default::default()
    };
    let m = VoxelMap::build(&raw, &hm, &spec).expect("build");
    assert_eq!(m.n, [8, 8, 4]);
    for k in 0..4usize {
        for j in 0..8usize {
            for i in 0..8usize {
                let want: u32 = if i >= 4 && j >= 4 { 1 } else { 0 };
                assert_eq!(m.cell[m.index(i, j, k)], want, "voxel ({i},{j},{k})");
            }
        }
    }
    assert_eq!(m.report.n_fluid, 256);
    assert_eq!(m.report.n_filled, 128);
    let ve = m.report.volume_error.expect("volume_error");
    assert!(ve.abs() <= 1e-12, "volume_error {ve}");

    // Variant B: cell 1 removed, its two faces now cell 0's boundary.
    let (raw, hm) = l_mesh(true);
    let m = VoxelMap::build(&raw, &hm, &spec).expect("build");
    for k in 0..4usize {
        for j in 0..8usize {
            for i in 0..8usize {
                let want: u32 = if i < 4 && j < 4 { 0 } else { NONE };
                assert_eq!(m.cell[m.index(i, j, k)], want, "voxel ({i},{j},{k})");
            }
        }
    }
    assert_eq!(m.report.n_fluid, 64);
    assert_eq!(m.report.n_filled, 0);
    let ve = m.report.volume_error.expect("volume_error");
    assert!((ve + 2.0 / 3.0).abs() <= 1e-12, "volume_error {ve}");
}

/// 117.12's refusals, each by name with its reason words; and the two
/// boundary cases that must NOT be refused (s exactly 16, and the default
/// voxel, which carries no limit).
#[test]
fn refusals_by_name() {
    let (raw, hm) = cube(16);
    let spec = |s: LatticeSpec| s;
    let expect_refused = |s: &LatticeSpec, key: &str, words: &[&str]| match VoxelMap::build(&raw, &hm, s) {
        Err(Error::Refused(msg)) => {
            assert!(msg.starts_with(key), "'{msg}' does not start with '{key}'");
            for w in words {
                assert!(msg.contains(w), "'{msg}' lacks '{w}'");
            }
        }
        other => panic!("expected Refused under '{key}', got {other:?}"),
    };
    let box_key = "output.visualisation.lattice.box";
    let voxel_key = "output.visualisation.lattice.voxel";
    let max_key = "output.visualisation.lattice.maxVoxels";
    let bg_key = "output.visualisation.lattice.background";

    expect_refused(
        &spec(LatticeSpec {
            bbox: Some([0.0, 0.0, 0.0, 0.0, 1.0, 1.0]),
            ..Default::default()
        }),
        box_key,
        &["the lattice would be empty"],
    );
    for v in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        expect_refused(
            &spec(LatticeSpec {
                voxel: Some(v),
                ..Default::default()
            }),
            voxel_key,
            &["there is no lattice"],
        );
    }
    expect_refused(
        &spec(LatticeSpec {
            bbox: Some([0.0, 0.0, f64::NAN, 1.0, 1.0, 1.0]),
            ..Default::default()
        }),
        box_key,
        &["there is no lattice"],
    );
    expect_refused(
        &spec(LatticeSpec {
            background: f64::NAN,
            ..Default::default()
        }),
        bg_key,
        &["there is no lattice"],
    );
    expect_refused(
        &spec(LatticeSpec {
            max_voxels: 0,
            ..Default::default()
        }),
        max_key,
        &["there is no lattice"],
    );
    expect_refused(
        &spec(LatticeSpec {
            voxel: Some(1.0 / 32.0),
            max_voxels: 1,
            ..Default::default()
        }),
        voxel_key,
        &["the budget cannot hold the lattice asked for"],
    );

    // NOT refused: s exactly 16 is allowed.
    let m = VoxelMap::build(
        &raw,
        &hm,
        &spec(LatticeSpec {
            voxel: Some(1.0 / 16.0),
            max_voxels: 1,
            ..Default::default()
        }),
    )
    .expect("s = 16 exactly must pass");
    assert_eq!(m.n, [1, 1, 1]);
    assert_eq!(m.report.coarsen, 16.0);
    // NOT refused: the default voxel coarsens without the limit.
    let m = VoxelMap::build(
        &raw,
        &hm,
        &spec(LatticeSpec {
            max_voxels: 1,
            ..Default::default()
        }),
    )
    .expect("the default voxel must coarsen without the limit");
    assert_eq!(m.n, [1, 1, 1]);
    assert!((m.report.coarsen - 16.0).abs() <= 1e-12);
}

/// (117.1).
#[test]
fn lattice_dims_follow_117_1() {
    assert_eq!(lattice_dims([1.0, 0.5, 0.25], 0.0625), [16, 8, 4]);
    assert_eq!(lattice_dims([1.0, 1.0, 1.0], 0.3), [4, 4, 4]);
    assert_eq!(lattice_dims([1.0, 1e-3, 1.0], 0.5), [2, 1, 2]);
    assert_eq!(lattice_dims([1.0, 1.0, 1.0], 0.1), [10, 10, 10]);
}

/// (117.3)-(117.4).
#[test]
fn coarsen_follows_117_3() {
    assert_eq!(coarsen_to_budget([1.0, 1.0, 1.0], 0.0625, 1000), (0.1, 1.6));
    assert_eq!(
        coarsen_to_budget([1.0, 0.5, 0.25], 0.0625, 100),
        (0.125, 2.0)
    );
    assert_eq!(
        coarsen_to_budget([1.0, 1.0, 1.0], 0.0625, 4096),
        (0.0625, 1.0)
    );
    assert_eq!(coarsen_to_budget([1.0, 1.0, 1.0], 0.0625, 1), (1.0, 16.0));
    assert_eq!(coarsen_to_budget([1.0, 1.0, 1.0], 0.03125, 1), (1.0, 32.0));
}

// ==========================================================================
//  Test 11's builders: the maps of tests 2, 3 and 5, hashed
// ==========================================================================

fn map_of_test2() -> VoxelMap {
    let (raw, hm) = refined();
    let spec = LatticeSpec {
        voxel: Some(1.0 / 24.0),
        ..Default::default()
    };
    VoxelMap::build(&raw, &hm, &spec).expect("build")
}

fn map_of_test3() -> VoxelMap {
    let (raw, hm) = refined();
    let spec = LatticeSpec {
        bbox: Some([
            -1.0 / 32.0,
            -1.0 / 32.0,
            -1.0 / 32.0,
            1.0 + 1.0 / 32.0,
            1.0 + 1.0 / 32.0,
            1.0 + 1.0 / 32.0,
        ]),
        voxel: Some(0.0625),
        ..Default::default()
    };
    VoxelMap::build(&raw, &hm, &spec).expect("build")
}

fn map_of_test5() -> VoxelMap {
    let (raw, _hm) = cube(32);
    let keep: Vec<bool> = (0..32 * 32 * 32)
        .map(|idx| {
            let i = idx % 32;
            let j = idx / 32 % 32;
            let k = idx / 1024;
            let dx = (i as f64 + 0.5) / 32.0 - 0.5;
            let dy = (j as f64 + 0.5) / 32.0 - 0.5;
            let dz = (k as f64 + 0.5) / 32.0 - 0.5;
            dx * dx + dy * dy + dz * dz >= 0.09
        })
        .collect();
    let raw2 = carve(&raw, &keep);
    let hm2 = build_host_mesh(&raw2).expect("build_host_mesh");
    let spec = LatticeSpec {
        voxel: Some(1.0 / 64.0),
        ..Default::default()
    };
    VoxelMap::build(&raw2, &hm2, &spec).expect("build")
}

/// SPEC-LIT 117.2 item 1: the map is bit-identical in both builds. The three
/// literals were printed from the f64 build; the `single` build must pass
/// with the same literals, no `cfg` anywhere.
#[test]
fn map_is_the_same_in_both_builds() {
    let m2 = map_of_test2();
    let m3 = map_of_test3();
    let m5 = map_of_test5();
    println!(
        "VoxelMap::build wall time: refined 1/24 {:.6} s, refined 1/16 wide {:.6} s, sphere 1/64 {:.6} s",
        m2.report.build_s, m3.report.build_s, m5.report.build_s
    );
    let h2 = fnv(&m2.cell);
    let h3 = fnv(&m3.cell);
    let h5 = fnv(&m5.cell);
    assert_eq!(h2, 0x70dd4aef220effa5, "test 2 map FNV, actual {h2:#018x}");
    assert_eq!(h3, 0xe3a4349cb6b98fb5, "test 3 map FNV, actual {h3:#018x}");
    assert_eq!(h5, 0x3a094c789e4b6f85, "test 5 map FNV, actual {h5:#018x}");
}

/// The `linear` sampling spec at `voxel` - the four samplings of the next
/// test differ only in mesh, box and field.
fn lin_spec(voxel: f64) -> LatticeSpec {
    LatticeSpec {
        voxel: Some(voxel),
        sample: LatticeSample::Linear,
        ..Default::default()
    }
}

/// 117.2 items 1-2 through `sample`: both fields here are the same numbers
/// in f32 and f64 (`r_c` carries 24 significant bits, the step field's
/// values one), the stencils and the sampling are f64 in BOTH builds, and
/// the maps are bit-identical (the test above) - so the four hashes are one
/// literal each, in either build, no `cfg` anywhere.
#[test]
fn sampling_is_the_same_in_both_builds() {
    // 1. refined(), voxel 1/24, the rough field.
    let (raw, hm) = refined();
    let phi: Vec<Scalar> = rough_field(hm.n_cells).iter().map(|&r| r as Scalar).collect();
    let m = VoxelMap::build(&raw, &hm, &lin_spec(1.0 / 24.0)).expect("build");
    let h1 = fnv_f64(&m.sample(&phi).expect("sample"));
    // 2. refined(), the box and voxel of map_of_test3, the rough field.
    let m3 = map_of_test3();
    let mut spec = lin_spec(m3.spec.voxel.expect("the test 3 voxel"));
    spec.bbox = m3.spec.bbox;
    let m = VoxelMap::build(&raw, &hm, &spec).expect("build");
    let h2 = fnv_f64(&m.sample(&phi).expect("sample"));
    // 3. the carved sphere, voxel 1/64, the rough field.
    let (raw, hm) = sphere_carved();
    let phi: Vec<Scalar> = rough_field(hm.n_cells).iter().map(|&r| r as Scalar).collect();
    let m = VoxelMap::build(&raw, &hm, &lin_spec(1.0 / 64.0)).expect("build");
    let h3 = fnv_f64(&m.sample(&phi).expect("sample"));
    // 4. cube(8), voxel 1/16, the step field of `no_new_extrema`.
    let (raw, hm) = cube(8);
    let m = VoxelMap::build(&raw, &hm, &lin_spec(1.0 / 16.0)).expect("build");
    let g = hm.geom64();
    let step: Vec<Scalar> = (0..hm.n_cells)
        .map(|c| if g.c[c].x > 0.5 { 1.0 } else { 0.0 })
        .collect();
    let h4 = fnv_f64(&m.sample(&step).expect("sample"));
    println!("sampling hash 1 refined() 1/24 rough: {h1:#018x}");
    println!("sampling hash 2 refined() test3 box: {h2:#018x}");
    println!("sampling hash 3 sphere 1/64 rough:   {h3:#018x}");
    println!("sampling hash 4 cube(8) 1/16 step:   {h4:#018x}");
    assert_eq!(h1, 0x635690dc32b97949, "sampling hash 1, actual {h1:#018x}");
    assert_eq!(h2, 0x32a66f5dc7a7cdcb, "sampling hash 2, actual {h2:#018x}");
    assert_eq!(h3, 0x9e8419bfe07c2c53, "sampling hash 3, actual {h3:#018x}");
    assert_eq!(h4, 0x528f800969e3f325, "sampling hash 4, actual {h4:#018x}");
}

// ==========================================================================
//  VIO-02: fixtures and oracles for the sampling tests (117.5)
// ==========================================================================

/// The linear field of the VIO-02 tests: `phi(x) = a . x + b` with
/// `a = (1, -2, 0.5)`, `b = 0.25`.
fn lin(x: DVec3) -> f64 {
    x.x - 2.0 * x.y + 0.5 * x.z + 0.25
}

/// The linear field at every cell centre, `phi(C_c)` with `C_c` from
/// `geom64()` - f64 in both builds.
fn lin_field(hm: &HostMesh) -> Vec<f64> {
    let g = hm.geom64();
    (0..hm.n_cells).map(|c| lin(g.c[c])).collect()
}

/// The second linear field of the VIO-02 tests: `phi2(x) = 0.3 x - 0.7 y
/// + 0.11 z + 0.2`, whose coefficients are NOT dyadic - `phi2` at a dyadic
/// centre is not representable in f32, so the widened Scalar field carries
/// f32 rounding under `single`.
fn lin2(x: DVec3) -> f64 {
    0.3 * x.x - 0.7 * x.y + 0.11 * x.z + 0.2
}

/// The second linear field at every cell centre, `phi2(C_c)` from
/// `geom64()` - f64 in both builds.
fn lin2_field(hm: &HostMesh) -> Vec<f64> {
    let g = hm.geom64();
    (0..hm.n_cells).map(|c| lin2(g.c[c])).collect()
}

/// The rough field `r_c = ((c * 2654435761) >> 8) / 2^24`, in `[0, 1)`.
fn rough_field(n_cells: usize) -> Vec<f64> {
    (0..n_cells)
        .map(|c| ((c as u32).wrapping_mul(2654435761) >> 8) as f64 / 16777216.0)
        .collect()
}

/// The test's own `N_f(c)`: the cells across the internal faces, a set - a
/// cell that shares two faces with `c` counts once - ascending, never `c`.
fn face_nbr_sets(raw: &PolyMeshRaw, hm: &HostMesh) -> Vec<Vec<u32>> {
    let mut sets: Vec<Vec<u32>> = vec![Vec::new(); hm.n_cells];
    for f in 0..hm.n_internal_faces {
        let (o, n) = (raw.owner[f] as usize, raw.neighbour[f] as usize);
        if o == n {
            continue;
        }
        if !sets[o].contains(&(n as u32)) {
            sets[o].push(n as u32);
        }
        if !sets[n].contains(&(o as u32)) {
            sets[n].push(o as u32);
        }
    }
    for s in sets.iter_mut() {
        s.sort_unstable();
    }
    sets
}

/// The test's own `N_p(c)`: the cells that share at least one point with
/// `c`, the points of a cell being the points of all its faces in
/// `raw.faces`; a set, ascending, never `c`.
fn point_nbr_sets(raw: &PolyMeshRaw, hm: &HostMesh) -> Vec<Vec<u32>> {
    let mut pts_of: Vec<Vec<u32>> = Vec::with_capacity(hm.n_cells);
    for c in 0..hm.n_cells {
        let mut pts: Vec<u32> = Vec::new();
        for k in hm.cf_offset[c] as usize..hm.cf_offset[c + 1] as usize {
            pts.extend(raw.faces[hm.cf_face[k] as usize].iter().map(|&p| p as u32));
        }
        for k in hm.bcf_offset[c] as usize..hm.bcf_offset[c + 1] as usize {
            let f = hm.n_internal_faces + hm.bcf_face[k] as usize;
            pts.extend(raw.faces[f].iter().map(|&p| p as u32));
        }
        pts.sort_unstable();
        pts.dedup();
        pts_of.push(pts);
    }
    let mut cells_of: Vec<Vec<u32>> = vec![Vec::new(); raw.points.len()];
    for (c, pts) in pts_of.iter().enumerate() {
        for &p in pts {
            cells_of[p as usize].push(c as u32);
        }
    }
    let mut sets: Vec<Vec<u32>> = Vec::with_capacity(hm.n_cells);
    let mut mark = vec![usize::MAX; hm.n_cells];
    for c in 0..hm.n_cells {
        let mut nbrs: Vec<u32> = Vec::new();
        for &p in &pts_of[c] {
            for &n in &cells_of[p as usize] {
                let n = n as usize;
                if n != c && mark[n] != c {
                    mark[n] = c;
                    nbrs.push(n as u32);
                }
            }
        }
        nbrs.sort_unstable();
        sets.push(nbrs);
    }
    sets
}

/// The points carried by `raw`'s boundary faces (`f >= n_internal_faces`).
fn boundary_points(raw: &PolyMeshRaw, hm: &HostMesh) -> Vec<bool> {
    let mut on = vec![false; raw.points.len()];
    for f in hm.n_internal_faces..raw.faces.len() {
        for &p in &raw.faces[f] {
            on[p as usize] = true;
        }
    }
    on
}

/// Whether cell `c` is interior: none of its points - the points of all its
/// faces in `raw.faces` - is a point of a boundary face.
fn is_interior_cell(raw: &PolyMeshRaw, hm: &HostMesh, on_boundary: &[bool], c: usize) -> bool {
    for k in hm.cf_offset[c] as usize..hm.cf_offset[c + 1] as usize {
        if raw.faces[hm.cf_face[k] as usize]
            .iter()
            .any(|&p| on_boundary[p as usize])
        {
            return false;
        }
    }
    for k in hm.bcf_offset[c] as usize..hm.bcf_offset[c + 1] as usize {
        let f = hm.n_internal_faces + hm.bcf_face[k] as usize;
        if raw.faces[f].iter().any(|&p| on_boundary[p as usize]) {
            return false;
        }
    }
    true
}

/// The carved sphere mesh behind `map_of_test5`, as a fixture.
fn sphere_carved() -> (PolyMeshRaw, HostMesh) {
    let (raw, _hm) = cube(32);
    let keep: Vec<bool> = (0..32 * 32 * 32)
        .map(|idx| {
            let i = idx % 32;
            let j = idx / 32 % 32;
            let k = idx / 1024;
            let dx = (i as f64 + 0.5) / 32.0 - 0.5;
            let dy = (j as f64 + 0.5) / 32.0 - 0.5;
            let dz = (k as f64 + 0.5) / 32.0 - 0.5;
            dx * dx + dy * dy + dz * dz >= 0.09
        })
        .collect();
    let raw2 = carve(&raw, &keep);
    let hm2 = build_host_mesh(&raw2).expect("build_host_mesh on the carved sphere");
    (raw2, hm2)
}

/// The interior voxels of a linear-sampled map, and the largest absolute
/// deviation of `s_v` from `phi(x_v)` over them, for the widened Scalar
/// sampling `s_sc` and the f64 sampling `s_f64`; `want_of` is the exact
/// field the deviations are measured against.
fn interior_maxima(
    map: &VoxelMap,
    interior: &[bool],
    s_sc: &[f64],
    s_f64: &[f64],
    want_of: fn(DVec3) -> f64,
) -> (usize, f64, f64) {
    let (mut n, mut a, mut b) = (0usize, 0.0f64, 0.0f64);
    for k in 0..map.n[2] {
        for j in 0..map.n[1] {
            for i in 0..map.n[0] {
                if let Some(c) = map.owner(i, j, k) {
                    if !interior[c] {
                        continue;
                    }
                    n += 1;
                    let want = want_of(map.centre(i, j, k));
                    let v = map.index(i, j, k);
                    a = a.max((s_sc[v] - want).abs());
                    b = b.max((s_f64[v] - want).abs());
                }
            }
        }
    }
    (n, a, b)
}

/// VIO-02's exactness gate (117.5): on a linear field the least-squares
/// gradient is exact, the clip is inactive on the interior voxels, and
/// `s_v = phi(x_v)` - to 1e-12 of the scale for the f64 field in BOTH
/// builds, and to 1e-5 for the solver's own `Scalar` values under `single`
/// (precision class R). Two fields per mesh: the dyadic `phi1`, exact in
/// f32 as well, and the non-dyadic `phi2`, whose f32-rounded coefficients
/// the `single` Scalar arm must see.
#[test]
fn linear_exact() {
    let cases = [
        ("A cube(8) h=1/16", cube(8), 1.0 / 16.0),
        ("B refined() h=1/24", refined(), 1.0 / 24.0),
    ];
    for (name, (raw, hm), voxel) in cases {
        let spec = LatticeSpec {
            voxel: Some(voxel),
            sample: LatticeSample::Linear,
            ..Default::default()
        };
        let map = VoxelMap::build(&raw, &hm, &spec).expect("build");
        assert_eq!(map.report.n_linear_fallback, 0, "{name}: no fallback");
        let on_boundary = boundary_points(&raw, &hm);
        let interior: Vec<bool> = (0..hm.n_cells)
            .map(|c| is_interior_cell(&raw, &hm, &on_boundary, c))
            .collect();
        // One field's bars (117.5): exact to 1e-12 of the scale through
        // `sample_f64` in BOTH builds; the widened Scalar field to 1e-12 in
        // the f64 build and to 1e-5 under `single` - and, for the second
        // field only, visibly above 1e-12 under `single`, so the f32
        // rounding of its coefficients is proven to reach the arm.
        let check = |label: &str, field: &[f64], want_of: fn(DVec3) -> f64| {
            let scale = field.iter().fold(0.0f64, |m, &p| m.max(p.abs()));
            let phi: Vec<Scalar> = field.iter().map(|&p| p as Scalar).collect();
            let s_sc = map.sample(&phi).expect("sample");
            let s_f64 = map.sample_f64(field).expect("sample_f64");
            let (n_interior, dev_sc, dev_64) =
                interior_maxima(&map, &interior, &s_sc, &s_f64, want_of);
            assert_eq!(n_interior, 1728, "{name}: interior voxels");
            let tol_sc = if cfg!(feature = "single") { 1e-5 } else { 1e-12 };
            let (max_sc, max_64) = (dev_sc / scale, dev_64 / scale);
            println!(
                "{name} {label}: interior voxels {n_interior}, |s_v - phi(x_v)|/scale: Scalar {max_sc:.3e} (tol {tol_sc:.0e}), f64 {max_64:.3e} (tol 1e-12)"
            );
            assert!(max_sc <= tol_sc, "{name} {label}: Scalar max {max_sc:e}");
            assert!(max_64 <= 1e-12, "{name} {label}: f64 max {max_64:e}");
            if cfg!(feature = "single") && label == "phi2" {
                assert!(
                    max_sc > 1e-12,
                    "{name} {label}: Scalar max {max_sc:e} not above 1e-12 - the single arm is not exercised"
                );
            }
        };
        let field1 = lin_field(&hm);
        check("phi1", &field1, lin);
        let field2 = lin2_field(&hm);
        check("phi2", &field2, lin2);
    }
}

/// The Barth & Jespersen clip (117.8): `linear` never creates a new
/// extremum - no `s_v` leaves the range of the cell values, every `s_v`
/// sits in `[m_c, M_c]` with the test's own point-neighbour bounds, and on
/// the step field the clip provably fires on exactly 512 voxels.
#[test]
fn no_new_extrema() {
    // (i) the step field on cube(8), cell index i + 8 (j + 8 k).
    let (raw, hm) = cube(8);
    let spec = LatticeSpec {
        voxel: Some(1.0 / 16.0),
        sample: LatticeSample::Linear,
        ..Default::default()
    };
    let map = VoxelMap::build(&raw, &hm, &spec).expect("build");
    let st = map.linear.as_ref().expect("linear stencil");
    let g = hm.geom64();
    let step: Vec<Scalar> = (0..hm.n_cells).map(|c| if g.c[c].x > 0.5 { 1.0 } else { 0.0 }).collect();
    let step64: Vec<f64> = step.iter().map(|&s| s as f64).collect();
    let sampled = map.sample(&step).expect("sample");
    for k in 0..8usize {
        for j in 0..8usize {
            for i in 0..8usize {
                let c = i + 8 * (j + 8 * k);
                let cx = g.c[c].x;
                for dk in 0..2usize {
                    for dj in 0..2usize {
                        for di in 0..2usize {
                            let (vi, vj, vk) = (2 * i + di, 2 * j + dj, 2 * k + dk);
                            let v = map.index(vi, vj, vk);
                            assert_eq!(map.cell[v], c as u32, "owner of ({vi},{vj},{vk})");
                            let x = map.centre(vi, vj, vk).x;
                            let s = sampled[v];
                            let side = if x < cx { -1i32 } else if x > cx { 1 } else { 0 };
                            assert_ne!(side, 0, "centre on the plane at ({vi},{vj},{vk})");
                            match (i, side) {
                                (3, -1) => assert_eq!(s, 0.0, "voxel ({vi},{vj},{vk})"),
                                (3, 1) => {
                                    assert!((s - 0.125).abs() <= 1e-14, "voxel ({vi},{vj},{vk}): {s}");
                                }
                                (4, -1) => {
                                    assert!((s - 0.875).abs() <= 1e-14, "voxel ({vi},{vj},{vk}): {s}");
                                }
                                (4, 1) => assert_eq!(s, 1.0, "voxel ({vi},{vj},{vk})"),
                                _ => assert_eq!(s, step64[c], "voxel ({vi},{vj},{vk})"),
                            }
                        }
                    }
                }
            }
        }
    }
    // The clip provably fires: 512 voxels leave [m_c, M_c] unclipped.
    let mut n_out = 0usize;
    for k in 0..16usize {
        for j in 0..16usize {
            for i in 0..16usize {
                let v = map.index(i, j, k);
                let c = map.cell[v] as usize;
                let grad = st.gradient(&step64, c);
                let (lo, hi) = st.bounds(&step64, c);
                let un = step64[c] + grad.dot(map.centre(i, j, k) - g.c[c]);
                if un < lo || un > hi {
                    n_out += 1;
                }
            }
        }
    }
    assert_eq!(n_out, 512, "clipped voxels on the step field");
    println!("(i) step field cube(8): unclipped outside [m_c, M_c]: {n_out} (expected 512)");
    // (ii) the rough field on refined(), (iii) on the carved sphere.
    let cases = [
        ("ii refined() rough", refined(), 1.0 / 24.0),
        ("iii carved sphere rough", sphere_carved(), 1.0 / 64.0),
    ];
    for (name, (raw, hm), voxel) in cases {
        let spec = LatticeSpec {
            voxel: Some(voxel),
            sample: LatticeSample::Linear,
            ..Default::default()
        };
        let map = VoxelMap::build(&raw, &hm, &spec).expect("build");
        let st = map.linear.as_ref().expect("linear stencil");
        let g = hm.geom64();
        let phi: Vec<Scalar> = rough_field(hm.n_cells).iter().map(|&r| r as Scalar).collect();
        let phi64: Vec<f64> = phi.iter().map(|&s| s as f64).collect();
        let sampled = map.sample(&phi).expect("sample");
        let pn = point_nbr_sets(&raw, &hm);
        let lo_all = phi64.iter().copied().fold(f64::INFINITY, f64::min);
        let hi_all = phi64.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut n_out = 0usize;
        for k in 0..map.n[2] {
            for j in 0..map.n[1] {
                for i in 0..map.n[0] {
                    let v = map.index(i, j, k);
                    let cv = map.cell[v];
                    if cv == NONE {
                        continue;
                    }
                    let c = cv as usize;
                    assert!(
                        lo_all <= sampled[v] && sampled[v] <= hi_all,
                        "{name}: new extremum at voxel {v}"
                    );
                    // The test's own [m_c, M_c], from `point_nbr_sets`.
                    let (mut m, mut mm) = (phi64[c], phi64[c]);
                    for &nn in &pn[c] {
                        m = m.min(phi64[nn as usize]);
                        mm = mm.max(phi64[nn as usize]);
                    }
                    assert!(
                        m <= sampled[v] && sampled[v] <= mm,
                        "{name}: voxel {v} outside [m_c, M_c]"
                    );
                    let grad = st.gradient(&phi64, c);
                    let un = phi64[c] + grad.dot(map.centre(i, j, k) - g.c[c]);
                    if un < m || un > mm {
                        n_out += 1;
                    }
                }
            }
        }
        assert!(n_out > 0, "{name}: the clip never fired");
        println!("{name}: unclipped outside [m_c, M_c]: {n_out}");
    }
}

/// `cell` sampling is the solver's own value, voxel for voxel (117.5), and
/// (117.9) holds as an identity of the map; the background value reaches
/// the NONE voxels under both samplers.
#[test]
fn cell_conserves() {
    let (raw, hm) = refined();
    let spec = LatticeSpec {
        voxel: Some(0.1),
        ..Default::default()
    };
    let map = VoxelMap::build(&raw, &hm, &spec).expect("build");
    assert!(map.linear.is_none(), "cell sampling builds no stencil");
    assert_eq!(map.report.n_linear_fallback, 0);
    let phi: Vec<Scalar> = rough_field(hm.n_cells).iter().map(|&r| r as Scalar).collect();
    let sampled = map.sample(&phi).expect("sample");
    let g = hm.geom64();
    let h3 = map.voxel * map.voxel * map.voxel;
    let mut owned = vec![0usize; hm.n_cells];
    for (v, &s) in sampled.iter().enumerate() {
        let cv = map.cell[v];
        if cv == NONE {
            continue;
        }
        assert_eq!(s, phi[cv as usize] as f64, "voxel {v}");
        owned[cv as usize] += 1;
    }
    // (117.9): h^3 sum_v s_v - sum_c phi_c V_c = sum_c phi_c (n_c h^3 - V_c).
    let (mut sum_s, mut sum_pv, mut sum_abs_pv, mut rhs, mut sum_dev) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for v in 0..sampled.len() {
        if map.cell[v] != NONE {
            sum_s += sampled[v];
        }
    }
    for c in 0..hm.n_cells {
        let (vc, pc) = (g.v[c] as f64, phi[c] as f64);
        sum_pv += pc * vc;
        sum_abs_pv += pc.abs() * vc;
        rhs += pc * (owned[c] as f64 * h3 - vc);
        sum_dev += (owned[c] as f64 * h3 - vc).abs();
    }
    let lhs = h3 * sum_s - sum_pv;
    println!(
        "cell_conserves: |lhs - rhs| = {:.3e} (bound {:.3e}), sum |n_c h^3 - V_c| = {:.6}",
        (lhs - rhs).abs(),
        1e-12 * sum_abs_pv,
        sum_dev
    );
    assert!((lhs - rhs).abs() <= 1e-12 * sum_abs_pv, "117.9 identity");
    let max_phi = (0..hm.n_cells).fold(0.0f64, |m, c| m.max((phi[c] as f64).abs()));
    assert!(lhs.abs() <= max_phi * sum_dev, "coverage bound");
    assert!(sum_dev > 0.0, "no cell-to-voxel volume deviation");
    // phi = 1: lhs / sum V_c is the report's coverage error (117.7).
    let ones: Vec<Scalar> = vec![1.0; hm.n_cells];
    let s1 = map.sample(&ones).expect("sample");
    let mut sum1 = 0.0f64;
    for v in 0..s1.len() {
        if map.cell[v] != NONE {
            sum1 += s1[v];
        }
    }
    let mut sum_v = 0.0f64;
    for c in 0..hm.n_cells {
        sum_v += g.v[c] as f64;
    }
    let lhs1 = h3 * sum1 - sum_v;
    let ve = map.report.volume_error.expect("volume_error for the default box");
    println!(
        "cell_conserves: phi=1 lhs/V = {lhs1:.3e} / {sum_v:.6} vs volume_error {ve:.3e}"
    );
    assert!((lhs1 / sum_v - ve).abs() <= 1e-12, "coverage error identity");
    // Background: the lattice of `ties_go_to_the_lowest_cell` - cube(16),
    // voxel 0.125, the box half a cell beyond the mesh on every side -
    // background -7.5, both samplers. 3584 of its 4096 voxels are NONE and
    // hold -7.5 exactly; the 512 owned voxels hold values of the rough
    // field, whose range is [0, 1), so none equals the background.
    let (raw16, hm16) = cube(16);
    let phi16: Vec<Scalar> = rough_field(hm16.n_cells)
        .iter()
        .map(|&r| r as Scalar)
        .collect();
    for sample in [LatticeSample::Cell, LatticeSample::Linear] {
        let spec = LatticeSpec {
            bbox: Some([-0.5, -0.5, -0.5, 1.5, 1.5, 1.5]),
            voxel: Some(0.125),
            background: -7.5,
            sample,
            ..Default::default()
        };
        let m = VoxelMap::build(&raw16, &hm16, &spec).expect("build");
        let s = m.sample(&phi16).expect("sample");
        let (mut n_none, mut n_owned) = (0usize, 0usize);
        for (v, &cv) in m.cell.iter().enumerate() {
            if cv == NONE {
                n_none += 1;
                assert_eq!(s[v], -7.5, "sample {sample:?}: background at voxel {v}");
            } else {
                n_owned += 1;
                assert!(
                    (0.0..1.0).contains(&s[v]),
                    "sample {sample:?}: owned voxel {v} holds {}, outside [0, 1)",
                    s[v]
                );
            }
        }
        assert_eq!(n_none, 3584, "sample {sample:?}: NONE voxels");
        assert_eq!(n_owned, 512, "sample {sample:?}: owned voxels");
        println!(
            "cell_conserves: sample {sample:?} - NONE voxels {n_none} (all exactly -7.5), owned {n_owned} (all in [0, 1))"
        );
    }
}

/// A mesh one cell thick has no 3-D stencil: every cell of (117.7) falls
/// back, `g_c = 0`, and `linear` samples bit for bit as `cell` - as does a
/// single-cell mesh, whose every voxel is exactly `phi_0`.
#[test]
fn linear_falls_back_on_a_one_cell_thick_mesh() {
    let spec32 = BlockSpec {
        x: axis(0.0, 1.0, 8),
        y: axis(0.0, 0.5, 4),
        z: axis(0.0, 0.125, 1),
        ..Default::default()
    };
    let raw = blockgen::raw_mesh(&spec32).expect("raw_mesh");
    let hm = build_host_mesh(&raw).expect("build_host_mesh");
    let lin_spec = LatticeSpec {
        voxel: Some(1.0 / 16.0),
        sample: LatticeSample::Linear,
        ..Default::default()
    };
    let map = VoxelMap::build(&raw, &hm, &lin_spec).expect("build");
    assert_eq!(map.report.n, [16, 8, 2]);
    assert_eq!(map.report.n_linear_fallback, 32, "every cell falls back");
    let st = map.linear.as_ref().expect("linear stencil");
    for c in 0..hm.n_cells {
        assert!(st.is_fallback(c), "cell {c} is not a fallback");
    }
    let field = lin_field(&hm);
    let phi: Vec<Scalar> = field.iter().map(|&p| p as Scalar).collect();
    let s_lin = map.sample(&phi).expect("sample");
    let cell_spec = LatticeSpec {
        voxel: Some(1.0 / 16.0),
        ..Default::default()
    };
    let map2 = VoxelMap::build(&raw, &hm, &cell_spec).expect("build");
    let s_cell = map2.sample(&phi).expect("sample");
    assert_eq!(s_lin.len(), s_cell.len());
    for (v, (a, b)) in s_lin.iter().zip(s_cell.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "voxel {v}: linear != cell");
    }
    // One cell on the unit cube, voxel 0.5: 8 voxels, all exactly phi_0.
    let spec1 = BlockSpec {
        x: axis(0.0, 1.0, 1),
        y: axis(0.0, 1.0, 1),
        z: axis(0.0, 1.0, 1),
        ..Default::default()
    };
    let raw1 = blockgen::raw_mesh(&spec1).expect("raw_mesh");
    let hm1 = build_host_mesh(&raw1).expect("build_host_mesh");
    let spec05 = LatticeSpec {
        voxel: Some(0.5),
        sample: LatticeSample::Linear,
        ..Default::default()
    };
    let map1 = VoxelMap::build(&raw1, &hm1, &spec05).expect("build");
    let st1 = map1.linear.as_ref().expect("linear stencil");
    assert_eq!(map1.report.n_linear_fallback, 1);
    assert!(st1.is_fallback(0));
    assert!(st1.face_neighbours(0).is_empty(), "no face neighbour");
    assert!(st1.point_neighbours(0).is_empty(), "no point neighbour");
    let f1 = lin_field(&hm1);
    let phi1: Vec<Scalar> = f1.iter().map(|&p| p as Scalar).collect();
    let s1 = map1.sample(&phi1).expect("sample");
    assert_eq!(s1.len(), 8);
    for (v, &s) in s1.iter().enumerate() {
        assert_eq!(s.to_bits(), (phi1[0] as f64).to_bits(), "voxel {v}");
    }
}

/// The stencil's neighbour lists equal the test's own sets, cell by cell
/// (117.5): `N_f(c)` from the internal faces, `N_p(c)` from the shared
/// points of `raw.faces` - ascending, deduped, never `c`.
#[test]
fn stencil_neighbours_match_brute_force() {
    let cases = [
        ("cube(8)", cube(8), 512usize),
        ("refined()", refined(), 120),
    ];
    for (name, (raw, hm), want) in cases {
        let st = LinearStencil::build(&raw, &hm).expect("stencil");
        assert_eq!(st.n_cells(), want, "{name}: n_cells");
        let f = face_nbr_sets(&raw, &hm);
        let p = point_nbr_sets(&raw, &hm);
        for c in 0..hm.n_cells {
            assert_eq!(st.face_neighbours(c), f[c].as_slice(), "{name}: N_f({c})");
            assert_eq!(st.point_neighbours(c), p[c].as_slice(), "{name}: N_p({c})");
        }
        if name == "cube(8)" {
            assert_eq!(st.face_neighbours(219).len(), 6, "cell 219 N_f");
            assert_eq!(st.point_neighbours(219).len(), 26, "cell 219 N_p");
            assert_eq!(st.face_neighbours(0).len(), 3, "cell 0 N_f");
            assert_eq!(st.point_neighbours(0).len(), 7, "cell 0 N_p");
        }
        if name == "refined()" {
            let nf: usize = (0..hm.n_cells).map(|c| st.face_neighbours(c).len()).sum();
            let np: usize = (0..hm.n_cells).map(|c| st.point_neighbours(c).len()).sum();
            println!("refined(): total N_f entries {nf}, total N_p entries {np}");
        }
    }
}

/// The two `sample` refusals, and the non-finite-field contract (117.5): a
/// NaN or infinite cell value reaches the output, never a panic, and never
/// moves a voxel outside the cell's own point neighbourhood.
#[test]
fn sample_refusals_and_non_finite_values() {
    let (raw, hm) = cube(8);
    let spec = LatticeSpec {
        voxel: Some(1.0 / 16.0),
        sample: LatticeSample::Linear,
        ..Default::default()
    };
    let map = VoxelMap::build(&raw, &hm, &spec).expect("build");
    let short: Vec<Scalar> = vec![0.0; hm.n_cells + 1];
    let err = short.len();
    match map.sample(&short) {
        Err(Error::Mesh(t)) => {
            assert!(t.starts_with("output.visualisation.lattice"), "'{t}'");
            assert!(t.contains("513") && t.contains("512"), "'{t}'");
        }
        other => panic!("expected Error::Mesh naming {err} and {}, got {other:?}", map.n_cells),
    }
    let short64: Vec<f64> = vec![0.0; 513];
    match map.sample_f64(&short64) {
        Err(Error::Mesh(t)) => {
            assert!(t.starts_with("output.visualisation.lattice"), "'{t}'");
            assert!(t.contains("513") && t.contains("512"), "'{t}'");
        }
        other => panic!("expected Error::Mesh, got {other:?}"),
    }
    // linear asked for, but the map carries no stencil.
    let mut no_stencil = map.clone();
    no_stencil.linear = None;
    let full: Vec<Scalar> = vec![0.0; hm.n_cells];
    match no_stencil.sample(&full) {
        Err(Error::Mesh(t)) => assert!(t.contains("linear"), "'{t}'"),
        other => panic!("expected Error::Mesh with 'linear', got {other:?}"),
    }
    match no_stencil.sample_f64(&vec![0.0f64; hm.n_cells]) {
        Err(Error::Mesh(t)) => assert!(t.contains("linear"), "'{t}'"),
        other => panic!("expected Error::Mesh with 'linear', got {other:?}"),
    }
    // Non-finite values must not panic and must stay local to cell 219's
    // point neighbourhood.
    let rough64 = rough_field(hm.n_cells);
    let clean = map.sample_f64(&rough64).expect("sample of the rough field");
    let pn = point_nbr_sets(&raw, &hm);
    let mut banned: Vec<usize> = vec![219];
    banned.extend(pn[219].iter().map(|&n| n as usize));
    for bad in [f64::NAN, f64::INFINITY] {
        let mut field = rough64.clone();
        field[219] = bad;
        let s = map
            .sample_f64(&field)
            .expect("sample must not panic on a non-finite value");
        assert_eq!(s.len(), map.n_voxels());
        let mut n_checked = 0usize;
        for k in 0..16usize {
            for j in 0..16usize {
                for i in 0..16usize {
                    let v = map.index(i, j, k);
                    let cv = map.cell[v];
                    if cv == NONE || banned.contains(&(cv as usize)) {
                        continue;
                    }
                    assert_eq!(
                        s[v].to_bits(),
                        clean[v].to_bits(),
                        "voxel {v} moved under a non-finite value"
                    );
                    n_checked += 1;
                }
            }
        }
        assert!(n_checked > 3000, "only {n_checked} voxels checked");
        println!("non-finite {bad}: {n_checked} voxels bit-identical to the clean field");
    }
}
