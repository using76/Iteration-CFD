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
//! Written from SPEC-LIT section 117 (117.2-117.4, 117.7).
//! Provenance: ORIGINAL.
//! No GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::error::Error;
use ofgpu::io::polymesh::{build_host_mesh, PolyMeshRaw};
use ofgpu::io::voxelmap::{coarsen_to_budget, lattice_dims, LatticeSpec, VoxelMap, NONE};
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
