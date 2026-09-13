// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

// Provenance: ORIGINAL - the tests for the region layout of the plan's §C
// (SPEC-LIT §97): dyadic identities and refusals the rules R1-R8 name.
// Nothing is compared against another CFD code.
// No GPL-licensed source was consulted.

use super::*;

use crate::blockgen::{raw_mesh, BlockSpec, GradedAxis};

/// The fixture: a 4x4x8 union block on `[0,1]^2 x [0,2]`, all six patches
/// plain. Every node coordinate is `i/4` - exact in binary - so everything
/// the geometry sweep sums is exact and the split's faces must match a
/// straight sub-block TO THE BIT.
fn union_block() -> BlockSpec {
    BlockSpec {
        x: GradedAxis { lo: 0.0, hi: 1.0, n: 4, expansion: 1.0, two_sided: false },
        y: GradedAxis { lo: 0.0, hi: 1.0, n: 4, expansion: 1.0, two_sided: false },
        z: GradedAxis { lo: 0.0, hi: 2.0, n: 8, expansion: 1.0, two_sided: false },
        patch_name: ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"].map(String::from),
        patch_type: ["patch"; 6].map(String::from),
        windows: Vec::new(),
        cyclic: Vec::new(),
    }
}

/// The reference the split's regions are held to: the 4x4x4 slab the zone
/// occupies, written straight from a block - `[0,1]^3` at `z_lo == 0`,
/// `[0,1]^2 x [1,2]` at `z_lo == 1`.
fn sub_block(z_lo: Scalar) -> BlockSpec {
    let mut b = union_block();
    b.z = GradedAxis { lo: z_lo, hi: z_lo + 1.0, n: 4, expansion: 1.0, two_sided: false };
    b
}

/// The fixture's zones from the cell centres: `lower` first, so a manifest
/// built from the split has regions[0] at the bottom.
fn zones_of(raw: &PolyMeshRaw) -> Vec<(String, Vec<Label>)> {
    let m = build_host_mesh(raw).expect("fixture host mesh");
    let (mut lower, mut upper) = (Vec::new(), Vec::new());
    for (c, z) in m.c.iter().enumerate() {
        if z.z < 1.0 {
            lower.push(c as Label);
        } else {
            upper.push(c as Label);
        }
    }
    vec![("lower".to_string(), lower), ("upper".to_string(), upper)]
}

fn scratch(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut d = std::env::temp_dir();
    d.push(format!("ofgpu-regions-{tag}-{}-{nanos}", std::process::id()));
    fs::create_dir_all(&d).unwrap();
    d
}

fn split_fixture() -> (Vec<(String, PolyMeshRaw)>, Vec<SplitInterface>, PolyMeshRaw) {
    let raw = raw_mesh(&union_block()).expect("fixture raw");
    let zones = zones_of(&raw);
    let (regions, ifaces) = split_by_zones(&raw, &zones).expect("split");
    (regions, ifaces, raw)
}

fn write_two_zone_layout(root: &Path) -> Vec<(String, PolyMeshRaw)> {
    let (regions, ifaces, _) = split_fixture();
    write_layout(
        &root.join("mesh"),
        &regions,
        &[RegionKind::Solid, RegionKind::Solid],
        &ifaces,
        None,
    )
    .expect("write_layout");
    regions
}

fn same_point(a: Vec3, b: Vec3) -> bool {
    a.x.to_bits() == b.x.to_bits()
        && a.y.to_bits() == b.y.to_bits()
        && a.z.to_bits() == b.z.to_bits()
}

fn face_coords(raw: &PolyMeshRaw, f: usize) -> Vec<Vec3> {
    raw.faces[f].iter().map(|&v| raw.points[v as usize]).collect()
}

fn patch_owners(raw: &PolyMeshRaw, patch: &str) -> Vec<Label> {
    let n_if = raw.neighbour.len();
    let p = raw.patches.iter().find(|p| p.name == patch).expect(patch);
    raw.owner[n_if + p.start..n_if + p.start + p.size].to_vec()
}

fn patch_face_coords(raw: &PolyMeshRaw, patch: &str) -> Vec<Vec<Vec3>> {
    let n_if = raw.neighbour.len();
    let p = raw.patches.iter().find(|p| p.name == patch).expect(patch);
    (0..p.size).map(|k| face_coords(raw, n_if + p.start + k)).collect()
}

/// Write one manifest body and return the refusal `read` raises on it.
fn refused<T: std::fmt::Debug>(read: impl Fn(&Path) -> Result<T>, body: &str) -> String {
    let d = scratch("refuse");
    let p = d.join("regions.json");
    fs::write(&p, body).unwrap();
    let err = read(&p).expect_err(body);
    fs::remove_dir_all(&d).ok();
    err.to_string()
}

/// A layout on disk with one mutation applied, and the refusal `load`
/// raises. This is where check_layout's R2/R3 refusals are exercised.
fn layout_with(mutate: impl FnOnce(&Path)) -> String {
    let d = scratch("layoutrefuse");
    write_two_zone_layout(&d);
    mutate(&d);
    let err = load(&d.join("mesh").join("regions.json")).expect_err("layout refusal");
    fs::remove_dir_all(&d).ok();
    err.to_string()
}

#[test]
fn a_manifest_that_breaks_a_rule_is_refused_by_name() {
    let fluid = r#"{"name":"f","kind":"fluid","polyMesh":"f/polyMesh"}"#;
    let solid = r#"{"name":"s","kind":"solid","polyMesh":"s/polyMesh"}"#;
    let two = format!(r#""version":1,"units":"m","regions":[{fluid},{solid}]"#);
    let cases: Vec<(String, &[&str])> = vec![
        (format!(r#"{{"version":2,"units":"m","regions":[{solid}]}}"#), &["2"]),
        (format!(r#"{{"version":1,"units":"ft","regions":[{solid}]}}"#), &["ft", "scale"]),
        (format!(r#"{{"version":1,"units":"m","regions":[{solid},{{"name":"s","kind":"solid","polyMesh":"x/polyMesh"}}]}}"#), &["twice", "'s'"]),
        (format!(r#"{{"version":1,"units":"m","regions":[{{"name":"a b","kind":"solid","polyMesh":"x/polyMesh"}}]}}"#), &["a b"]),
        (format!(r#"{{"version":1,"units":"m","regions":[{{"name":"s","kind":"gas","polyMesh":"x/polyMesh"}}]}}"#), &["gas"]),
        (format!(r#"{{"version":1,"units":"m","regions":[{{"name":"f","kind":"fluid","polyMesh":"x/polyMesh","material":"steel"}}]}}"#), &["steel"]),
        (format!(r#"{{"version":1,"units":"m","regions":[{fluid},{{"name":"g","kind":"fluid","polyMesh":"g/polyMesh"}}]}}"#), &["'f'", "'g'"]),
        (format!(r#"{{"version":1,"units":"m","regions":[{solid},{fluid}]}}"#), &["regions[1]", "'f'"]),
        (format!(r#"{{{two},"interfaces":[{{"regions":["f","f"],"patches":["f_to_f","f_to_f"]}}]}}"#), &["both sides"]),
        (format!(r#"{{{two},"interfaces":[{{"regions":["f","ghost"],"patches":["f_to_ghost","ghost_to_f"]}}]}}"#), &["ghost"]),
        (format!(r#"{{{two},"interfaces":[{{"regions":["f","s"],"patches":["a","b"]}}]}}"#), &["f_to_s", "s_to_f"]),
        (format!(r#"{{{two},"interfaces":[{{"regions":["f","s"],"patches":["f_to_s","s_to_f"],"tolerance":0.0}}]}}"#), &["tolerance"]),
    ];
    for (body, wants) in &cases {
        let e = refused(read_manifest, body);
        for w in *wants {
            assert!(e.contains(w), "want '{w}' in: {e}");
        }
    }
    // R6 is `load`'s check, because it is about the path against the
    // manifest's own directory.
    let e = refused(
        load,
        r#"{"version":1,"units":"m","regions":[{"name":"s","kind":"solid","polyMesh":"C:/elsewhere/polyMesh"}]}"#,
    );
    assert!(e.contains("R6") && e.contains("C:/elsewhere/polyMesh"), "{e}");
    let e = refused(
        load,
        r#"{"version":1,"units":"m","regions":[{"name":"s","kind":"solid","polyMesh":"../x/polyMesh"}]}"#,
    );
    assert!(e.contains("R6") && e.contains(".."), "{e}");

    // R1's added ordering check, on a hand-swapped two-face raw: face 1's
    // (owner, neighbour) sorts before face 0's.
    let raw = PolyMeshRaw {
        points: vec![Vec3::ZERO; 8],
        faces: vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
        owner: vec![1, 0],
        neighbour: vec![2, 1],
        patches: Vec::new(),
    };
    let e = check_r1(&raw, "test region").expect_err("ordering breaks").to_string();
    assert!(e.contains("R1") && e.contains("face 1"), "{e}");

    // check_layout, on a real layout: a renamed patch, a `type wall`
    // interface patch, and a `faces` that disagrees with the patches.
    let e = layout_with(|d| {
        let mj = d.join("mesh").join("regions.json");
        let ok = fs::read_to_string(&mj).unwrap();
        fs::write(&mj, ok.replace("\"lower_to_upper\"", "\"nonsense\"")).unwrap();
    });
    assert!(e.contains("nonsense") && e.contains("lower_to_upper"), "{e}");
    let e = layout_with(|d| {
        let bp = d.join("mesh").join("lower").join("polyMesh").join("boundary");
        let t = fs::read_to_string(&bp).unwrap();
        let wall = t.replace(
            "lower_to_upper\n    {\n        type            patch;",
            "lower_to_upper\n    {\n        type            wall;",
        );
        assert_ne!(t, wall, "boundary rewrite did not land");
        fs::write(&bp, wall).unwrap();
    });
    assert!(e.contains("wall") && e.contains("R3"), "{e}");
    let e = layout_with(|d| {
        let mj = d.join("mesh").join("regions.json");
        let ok = fs::read_to_string(&mj).unwrap();
        fs::write(&mj, ok.replace("\"faces\": 16", "\"faces\": 15")).unwrap();
    });
    assert!(e.contains("15") && e.contains("16"), "{e}");
}

/// The split layout written to disk loads back as the same layout,
/// and the pairing numbers R2 measures on it are zero to within the fan's
/// one-ulp centroid (exact areas and normals - the fixture is dyadic).
#[test]
fn the_layout_round_trips_through_disk_and_load_checks_r1_to_r6() {
    let d = scratch("roundtrip");
    let regions = write_two_zone_layout(&d);
    let layout = load(&d.join("mesh").join("regions.json")).expect("load");
    assert_eq!(layout.regions.len(), 2);
    for (i, (name, raw0)) in regions.iter().enumerate() {
        let lr = &layout.regions[i];
        assert_eq!(lr.name, *name);
        assert_eq!(lr.kind, RegionKind::Solid);
        assert_eq!(lr.material, None);
        assert_eq!(lr.raw.points.len(), raw0.points.len());
        for (a, b) in lr.raw.points.iter().zip(&raw0.points) {
            assert!(same_point(*a, *b), "point {i} did not round-trip bitwise");
        }
        assert_eq!(lr.raw.faces, raw0.faces);
        assert_eq!(lr.raw.owner, raw0.owner);
        assert_eq!(lr.raw.neighbour, raw0.neighbour);
        assert_eq!(lr.raw.patches.len(), raw0.patches.len());
        for (p, q) in lr.raw.patches.iter().zip(&raw0.patches) {
            assert_eq!(p.name, q.name);
            assert_eq!(p.type_name, q.type_name);
            assert_eq!(p.start, q.start);
            assert_eq!(p.size, q.size);
        }
    }
    assert_eq!(layout.interfaces.len(), 1);
    for pc in &layout.pairing {
        println!(
            "pairing {}: {} faces, worst centroid {:e}, area {:e}, normal {:e}",
            pc.name, pc.n_faces, pc.worst_centroid, pc.worst_area, pc.worst_normal
        );
        assert!(pc.worst_centroid <= 1e-14, "{pc:?}");
        assert!(pc.worst_area <= 1e-15, "{pc:?}");
        assert!(pc.worst_normal <= 1e-15, "{pc:?}");
    }
    fs::remove_dir_all(&d).ok();
}

/// The split's two regions ARE the sub-blocks - topology, coordinates
/// and built geometry bitwise, and one conformal pair between them.
#[test]
fn a_two_zone_block_splits_into_two_standalone_regions_with_one_conformal_pair() {
    let (regions, ifaces, _) = split_fixture();
    let (lo, up) = (&regions[0].1, &regions[1].1);
    let names: Vec<&str> = regions.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["lower", "upper"]);
    assert_eq!(ifaces.len(), 1);
    let si = &ifaces[0];
    assert_eq!((si.region_a, si.region_b), (0, 1));
    assert_eq!(si.patch_a, "lower_to_upper");
    assert_eq!(si.patch_b, "upper_to_lower");
    assert_eq!(si.n_faces, 16);
    let (lo_ref, up_ref) =
        (raw_mesh(&sub_block(0.0)).unwrap(), raw_mesh(&sub_block(1.0)).unwrap());
    for (name, raw) in [("lower", lo), ("upper", up)] {
        println!(
            "{name}: 64 cells, {} internal faces, {} boundary faces, 6 patches of 16",
            raw.neighbour.len(),
            raw.owner.len() - raw.neighbour.len()
        );
    }
    // (i) patch names, types, sizes.
    for (raw, want) in [
        (lo, &["xmin", "xmax", "ymin", "ymax", "zmin", "lower_to_upper"][..]),
        (up, &["xmin", "xmax", "ymin", "ymax", "zmax", "upper_to_lower"][..]),
    ] {
        let pn: Vec<&str> = raw.patches.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(pn, want);
        assert!(raw.patches.iter().all(|p| p.type_name == "patch" && p.size == 16));
    }
    // (ii) lower's owner/neighbour are the sub-block's, entry for entry.
    assert_eq!(lo.owner, lo_ref.owner);
    assert_eq!(lo.neighbour, lo_ref.neighbour);
    // (iii) upper: the internal owners and every neighbour match; the whole
    // boundary owner slice does NOT (its patch order differs - zmax comes
    // before upper_to_lower, against the sub-block's zmin then zmax), so the
    // boundary comparison is per patch, by name.
    let n_if = up.neighbour.len();
    assert_eq!(up.neighbour, up_ref.neighbour);
    assert_eq!(&up.owner[..n_if], &up_ref.owner[..n_if]);
    assert_ne!(up.owner, up_ref.owner, "patch order, not the mesh, differs");
    assert_eq!(patch_owners(up, "zmax"), patch_owners(&up_ref, "zmax"));
    assert_eq!(patch_owners(up, "upper_to_lower"), patch_owners(&up_ref, "zmin"));
    // (iv) every face of both regions, mapped to coordinates through its
    // own points, equals the corresponding sub-block face's coordinate
    // loop: internal faces by index, boundary faces by patch name and k
    // (lower_to_upper is sub-block zmax, upper_to_lower is sub-block zmin).
    for f in 0..144 {
        assert_eq!(face_coords(lo, f), face_coords(&lo_ref, f), "lower internal {f}");
        assert_eq!(face_coords(up, f), face_coords(&up_ref, f), "upper internal {f}");
    }
    for name in ["xmin", "xmax", "ymin", "ymax"] {
        assert_eq!(patch_face_coords(lo, name), patch_face_coords(&lo_ref, name));
        assert_eq!(patch_face_coords(up, name), patch_face_coords(&up_ref, name));
    }
    assert_eq!(patch_face_coords(lo, "zmin"), patch_face_coords(&lo_ref, "zmin"));
    assert_eq!(patch_face_coords(lo, "lower_to_upper"), patch_face_coords(&lo_ref, "zmax"));
    assert_eq!(patch_face_coords(up, "zmax"), patch_face_coords(&up_ref, "zmax"));
    assert_eq!(patch_face_coords(up, "upper_to_lower"), patch_face_coords(&up_ref, "zmin"));
    // (v) the built geometry is the sub-block's, bitwise: v, c, and the
    // interface faces' areas and centroids.
    let (lm, um) = (build_host_mesh(lo).unwrap(), build_host_mesh(up).unwrap());
    let (lref, uref) = (build_host_mesh(&lo_ref).unwrap(), build_host_mesh(&up_ref).unwrap());
    for (m, r, tag) in [(&lm, &lref, "lower"), (&um, &uref, "upper")] {
        for c in 0..m.n_cells {
            assert_eq!(m.v[c].to_bits(), r.v[c].to_bits(), "{tag} v cell {c}");
            assert!(same_point(m.c[c], r.c[c]), "{tag} c cell {c}");
        }
    }
    let ifl = lm.patches.iter().find(|p| p.name == "lower_to_upper").unwrap();
    let zl = lref.patches.iter().find(|p| p.name == "zmax").unwrap();
    let ifu = um.patches.iter().find(|p| p.name == "upper_to_lower").unwrap();
    let zu = uref.patches.iter().find(|p| p.name == "zmin").unwrap();
    for k in 0..16 {
        let (a, b) = (ifl.start + k, zl.start + k);
        assert!(same_point(lm.b_sf[a], lref.b_sf[b]), "lower interface b_sf {k}");
        assert!(same_point(lm.b_cf[a], lref.b_cf[b]), "lower interface b_cf {k}");
        let (a, b) = (ifu.start + k, zu.start + k);
        assert!(same_point(um.b_sf[a], uref.b_sf[b]), "upper interface b_sf {k}");
        assert!(same_point(um.b_cf[a], uref.b_cf[b]), "upper interface b_cf {k}");
    }
    // (vi) the two sides are the same faces wound oppositely: Sf bitwise
    // negated, centroids agreeing (the loops are first-vertex reverses, so
    // the fan order matches and the centroid is one ulp at worst).
    for k in 0..16 {
        let a = lm.b_sf[ifl.start + k];
        let b = um.b_sf[ifu.start + k];
        // Exact negation per component, checked as a sum so a component
        // that is zero on BOTH sides (+0.0, the fixture's x and y) does not
        // fail on the sign of zero: a.x + b.x is +0.0 iff a.x == -b.x
        // exactly, for every finite pair including the zeros.
        assert_eq!((a.x + b.x).to_bits(), 0.0f64.to_bits(), "Sf x {k}");
        assert_eq!((a.y + b.y).to_bits(), 0.0f64.to_bits(), "Sf y {k}");
        assert_eq!((a.z + b.z).to_bits(), 0.0f64.to_bits(), "Sf z {k}");
        assert!(
            (lm.b_cf[ifl.start + k] - um.b_cf[ifu.start + k]).mag() <= 1e-14,
            "interface centroid {k}"
        );
    }
}

/// The split's refusals, each named: a cell in no zone (the count and the first
/// cell), a cell in two zones (both names), a label out of range, an empty
/// zone, an unwritable zone name, and a cyclic patch in the parent.
#[test]
fn split_refuses_what_it_cannot_carry() {
    let raw = raw_mesh(&union_block()).unwrap();
    let e = split_by_zones(&raw, &[("lower".into(), (0..64).collect())]).unwrap_err();
    let e = e.to_string();
    assert!(e.contains("64 cells are in no zone") && e.contains("cell 64"), "{e}");
    let e = split_by_zones(&raw, &[("a".into(), vec![0, 1]), ("b".into(), vec![65, 1])])
        .unwrap_err()
        .to_string();
    assert!(e.contains("cell 1") && e.contains("'a'") && e.contains("'b'"), "{e}");
    let e = split_by_zones(&raw, &[("a".into(), vec![0, 500])]).unwrap_err().to_string();
    assert!(e.contains("500") && e.contains("128 cells"), "{e}");
    let e = split_by_zones(&raw, &[("a".into(), vec![0]), ("b".into(), vec![])])
        .unwrap_err()
        .to_string();
    assert!(e.contains("empty") && e.contains("'b'"), "{e}");
    let e = split_by_zones(&raw, &[("bad;name".into(), (0..128).collect())])
        .unwrap_err()
        .to_string();
    assert!(e.contains("bad;name"), "{e}");
    let mut cyc = raw.clone();
    cyc.patches.push(PatchInfo {
        name: "periodic".to_string(),
        type_name: "cyclic".to_string(),
        kind: PatchKind::Cyclic,
        start: 0,
        size: 1,
        nbr_patch: None,
    });
    let e = split_by_zones(&cyc, &zones_of(&raw)).unwrap_err().to_string();
    assert!(e.contains("periodic") && e.contains("31.1"), "{e}");
}
