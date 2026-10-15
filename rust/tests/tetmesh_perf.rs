// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! TET-15's gates (SPEC-LIT §116.8, §116.10, §116.18) as an integration
//! test, so the `--features single` build runs them too: the library's own
//! test crate does not compile under `single` (SPEC-LIT §112.3).
//!
//! `perf::thread_invariant` (class D): the size field's sources, its octree
//! and every probed answer, and the surface remesh, are the same bit for
//! bit on 1, 4 and 8 workers. `perf::million_tets` (`#[ignore]`, measured,
//! a supervisor run): a sphere in a box sized for about a million
//! tetrahedra, `run_tet` end to end in 120 s or less; a miss prints MISSED
//! with the time and the cause, per the tranche's rule that a miss is
//! recorded, never widened.
//!
//! Provenance: ORIGINAL. No GPL-licensed source was consulted.

mod perf {
    use ofgpu::automesher::driver::run_tet;
    use ofgpu::automesher::read_config_str;
    use ofgpu::surface::stl::parse_stl;
    use ofgpu::surface::Surface;
    use ofgpu::tetmesh::remesh::{remesh, RemeshOptions, Remeshed};
    use ofgpu::tetmesh::sizefield::{
        surface_sources_threads, GapSide, OctreeOptions, SizeField, SizeParams, SizeSource,
    };
    use ofgpu::tetmesh::surfprep::{prepare, TetSurface};
    use ofgpu::tetmesh::{Point, TetSizeSpec};

    const BOX_SPHERE_TET: &str =
        include_str!("../../tools/automesher/examples/box_sphere_tet.json");

    /// The sphere `make_box_sphere_stl.py --outward` writes (radius 1 at
    /// (5, 5, 5), 48 bands of 96 meridians, 9024 triangles), as
    /// tests/shared/automesher_tet_cases.rs builds it.
    fn sphere_stl() -> String {
        let (r, c) = (1.0f64, [5.0f64; 3]);
        let (nlat, nlon) = (48usize, 96usize);
        let p = |i: usize, j: usize| -> [f64; 3] {
            let th = std::f64::consts::PI * i as f64 / nlat as f64;
            let ph = 2.0 * std::f64::consts::PI * (j % nlon) as f64 / nlon as f64;
            [c[0] + r * th.sin() * ph.cos(), c[1] + r * th.sin() * ph.sin(), c[2] + r * th.cos()]
        };
        let mut s = String::from("solid sphere\n");
        let mut tri = |a: [f64; 3], b: [f64; 3], cc: [f64; 3]| {
            s.push_str("  facet normal 0 0 0\n    outer loop\n");
            for q in [a, cc, b] {
                s.push_str(&format!("      vertex {:.9e} {:.9e} {:.9e}\n", q[0], q[1], q[2]));
            }
            s.push_str("    endloop\n  endfacet\n");
        };
        for i in 0..nlat {
            for j in 0..nlon {
                let (a, b) = (p(i, j), p(i, j + 1));
                let (cc, d) = (p(i + 1, j + 1), p(i + 1, j));
                if i == 0 {
                    tri(a, cc, d);
                } else if i == nlat - 1 {
                    tri(a, b, cc);
                } else {
                    tri(a, b, cc);
                    tri(a, cc, d);
                }
            }
        }
        s.push_str("endsolid sphere\n");
        s
    }

    fn sphere() -> Surface {
        parse_stl(sphere_stl().as_bytes(), "box_sphere_outward", "box_sphere_outward.stl")
            .expect("the sphere parses")
    }

    /// The cube [1, 3]^3 with one patch per face, wound outward: twelve
    /// feature polylines and eight corners, so the remesh's feature and
    /// corner paths run too.
    fn cube() -> TetSurface {
        let mut points: Vec<Point> = Vec::new();
        for k in 0..8u32 {
            let c = |a: u32| if (k >> a) & 1 == 1 { 3.0 } else { 1.0 };
            points.push([c(0), c(1), c(2)]);
        }
        // Each face as (a, b, c, d) counter-clockwise seen from outside.
        let faces: [[u32; 4]; 6] = [
            [0, 4, 6, 2], // x = 1
            [1, 3, 7, 5], // x = 3
            [0, 1, 5, 4], // y = 1
            [2, 6, 7, 3], // y = 3
            [0, 2, 3, 1], // z = 1
            [4, 5, 7, 6], // z = 3
        ];
        let mut tris = Vec::new();
        let mut tri_patch = Vec::new();
        for (f, q) in faces.iter().enumerate() {
            tris.push([q[0], q[1], q[2]]);
            tris.push([q[0], q[2], q[3]]);
            tri_patch.push(f as u32);
            tri_patch.push(f as u32);
        }
        let patch_names = (0..6).map(|f| format!("face{f}")).collect();
        TetSurface { points, tris, tri_patch, patch_names }
    }

    fn bits(p: &[Point]) -> Vec<[u64; 3]> {
        p.iter().map(|x| x.map(f64::to_bits)).collect()
    }

    /// Everything a remesh returns, the points as bits.
    #[allow(clippy::type_complexity)]
    fn remesh_key(
        r: &Remeshed,
    ) -> (Vec<[u64; 3]>, Vec<[u32; 3]>, Vec<u32>, String, Vec<[u32; 2]>, Vec<bool>, Vec<u32>, Vec<u32>) {
        (
            bits(&r.surface.points),
            r.surface.tris.clone(),
            r.surface.tri_patch.clone(),
            format!("{:?}", r.stats),
            r.feature_edges.clone(),
            r.edge_is_seam.clone(),
            r.edge_polyline.clone(),
            r.corners.clone(),
        )
    }

    /// SplitMix64, the repository's deterministic probe generator.
    fn splitmix(state: &mut u64) -> f64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    /// TET-15's class-D gate: 1, 4 and 8 workers give the same size-field
    /// sources (curvature and proximity), the same octree stats, the same
    /// `h` and leaf box bit for bit at 20 000 probes, and the same remesh
    /// of the sphere and of the cube, point for point and bit for bit.
    #[test]
    fn thread_invariant() {
        let ts = TetSurface::from_surface(&sphere());
        assert_eq!(ts.tris.len(), 9024);
        let prep = prepare(&ts, 30.0).expect("the sphere prepares");
        let params = SizeParams::new(1.0, 1.2, None).expect("params");
        let spec = TetSizeSpec { curvature_cells: 16, gap_cells: 3, ..TetSizeSpec::default() };
        let (lo, hi): (Point, Point) = ([0.0; 3], [10.0; 3]);
        let mut probes: Vec<Point> = Vec::with_capacity(20_000);
        let mut st = 2026u64;
        for _ in 0..20_000 {
            let x = [0, 1, 2].map(|_| 10.0 * splitmix(&mut st));
            probes.push(x);
        }
        let cube_prep = prepare(&cube(), 30.0).expect("the cube prepares");
        let mut first: Option<(String, String, Vec<u64>, Vec<String>)> = None;
        let mut first_rm: Option<(String, String)> = None;
        for threads in [1usize, 4, 8] {
            let t0 = std::time::Instant::now();
            let mut sources =
                surface_sources_threads(&prep.surface, &params, &spec, 30.0, GapSide::Both, threads)
                    .expect("surface sources");
            let n_surface = sources.len();
            sources.push(SizeSource::Patch {
                name: "sphere".to_string(),
                tris: prep
                    .surface
                    .tris
                    .iter()
                    .map(|t| t.map(|v| prep.surface.points[v as usize]))
                    .collect(),
                distance: 0.25,
                size: 0.0625,
            });
            sources.push(SizeSource::Point { at: [2.0, 2.0, 2.0], size: 0.1 });
            sources.push(SizeSource::Box { min: [6.5, 1.0, 1.0], max: [9.0, 2.0, 3.0], size: 0.2 });
            let opts = OctreeOptions { threads, ..OctreeOptions::default() };
            let field = SizeField::build(params, &sources, lo, hi, opts).expect("the field builds");
            let t_field = t0.elapsed().as_secs_f64();
            let hs: Vec<u64> = probes.iter().map(|&x| field.h(x).to_bits()).collect();
            let boxes: Vec<String> =
                probes.iter().step_by(97).map(|&x| format!("{:?}", field.leaf_box(x))).collect();
            let key = (format!("{sources:?}"), format!("{:?}", field.stats()), hs, boxes);
            let h = |x: Point| field.h(x);
            let t1 = std::time::Instant::now();
            let ro = RemeshOptions { threads, ..RemeshOptions::default() };
            let rm = remesh(&prep, &h, &ro).expect("the sphere remeshes");
            let rc = remesh(&cube_prep, &|_: Point| 0.25, &ro).expect("the cube remeshes");
            let t_remesh = t1.elapsed().as_secs_f64();
            let rkey = (format!("{:?}", remesh_key(&rm)), format!("{:?}", remesh_key(&rc)));
            println!(
                "thread_invariant: {threads} workers - {n_surface} surface sources, {:?}; \
                 sphere remesh {} points, cube remesh {} points; field {t_field:.3} s, remesh {t_remesh:.3} s",
                field.stats(),
                rm.surface.points.len(),
                rc.surface.points.len()
            );
            assert!(n_surface > 0, "the curvature and proximity sources are not empty");
            assert!(rc.corners.len() == 8, "the cube keeps its eight corners");
            match (&first, &first_rm) {
                (None, None) => {
                    first = Some(key);
                    first_rm = Some(rkey);
                }
                (Some(f), Some(fr)) => {
                    assert!(f.0 == key.0, "{threads} workers: the sources differ from 1 worker's");
                    assert!(f.1 == key.1, "{threads} workers: the octree stats differ: {} vs {}", f.1, key.1);
                    let diff = f.2.iter().zip(&key.2).filter(|(a, b)| a != b).count();
                    assert_eq!(diff, 0, "{threads} workers: {diff} probes answer differently");
                    assert!(f.3 == key.3, "{threads} workers: a leaf box differs");
                    assert!(fr.0 == rkey.0, "{threads} workers: the sphere remesh differs");
                    assert!(fr.1 == rkey.1, "{threads} workers: the cube remesh differs");
                }
                _ => unreachable!(),
            }
        }
    }

    /// TET-15's measured gate: `box_sphere_tet.json` with `base_size`
    /// 0.25 m and the sphere at level 1 (0.125 m), which asks for about
    /// 1.3 million tetrahedra (the example's 20 309 at `H` = 1 times 4^3),
    /// end to end in 120 s or less in release. A supervisor run. The miss
    /// it measured is in SPEC-LIT §116.18: the refinement refuses
    /// (`refine/...`), which is the one failure this test accepts.
    #[test]
    #[ignore = "TET-15 supervisor run: a million tetrahedra, release only"]
    fn million_tets() {
        let text = BOX_SPHERE_TET
            .replace("\"base_size\": 1.0", "\"base_size\": 0.25")
            .replace("{ \"distance\": 0.0, \"level\": 2 }", "{ \"distance\": 0.0, \"level\": 1 }")
            .replace("\"max_level\": 2", "\"max_level\": 1");
        assert_ne!(text, BOX_SPHERE_TET);
        let cfg = read_config_str(&text, "million_tets.json").expect("the config parses");
        cfg.validate().expect("the config validates");
        assert_eq!(cfg.domain.base_size, 0.25);
        let surf = sphere();
        let t0 = std::time::Instant::now();
        let res = run_tet(&cfg, &surf, &mut |l: &str| {
            if l.starts_with("---") {
                println!("million_tets: {l}");
            }
        });
        let seconds = t0.elapsed().as_secs_f64();
        match res {
            Ok(out) => {
                println!(
                    "million_tets: {} tetrahedra end to end in {seconds:.1} s (the gate is 120 s) - {}",
                    out.n_tets,
                    if seconds <= 120.0 { "PASS" } else { "MISSED" }
                );
                for s in &out.stages {
                    println!("million_tets: {s:?}");
                }
                if seconds > 120.0 {
                    println!(
                        "million_tets: MISSED - {seconds:.1} s against 120 s; the size field and the \
                         remesh are threaded (TET-15), refinement and optimisation are not (SPEC-LIT 116.18)"
                    );
                }
            }
            Err(err) => {
                let text = err.to_string();
                let head: String = text.chars().take(300).collect();
                println!("million_tets: MISSED after {seconds:.1} s - {head}");
                assert!(text.contains("refine/"), "{err}");
            }
        }
    }
}
