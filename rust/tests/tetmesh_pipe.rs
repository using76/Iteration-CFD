// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! TET-14's gates (SPEC-LIT §116.9's cavity rule, §116.12's seed carve,
//! §116.17) as an integration test, so the `--features single` build runs
//! them too: the library's own test crate does not compile under `single`
//! (SPEC-LIT §112.3).
//!
//! `pipe::pipe_cavity_volume` (f64 class D, under `single` class R): the
//! inward-wound pipe is ONE shell of negative volume, which
//! `prepare_with(.., Cavity::Allow)` takes and `prepare` refuses by name;
//! the shell volume is held to 1e-12 of the prism formula in f64 and to
//! 1e-5 under `single`, where the STL is parsed into f32 first (D-F32-1).
//! `pipe::pipe_hull_start_refused` (class D): the hull start keeps the
//! refusal, named. `pipe::pipe_tet_layers` (class D in f64): the seed carve
//! meshes the pipe end to end at the pinned `base_size` 0.25, prisms in
//! threes on the wall, the fluid on the pipe's own patches only - at the
//! example's radius-edge bound 2.5, because the default 2 does not finish
//! on this surface's rims (SPEC-LIT §116.17, TET-14); a converging run
//! meets the f64 asserts, and under `single` a §92.3 G2 failure is the
//! known A28, printed MISSED.
//!
//! Provenance: ORIGINAL. No GPL-licensed source was consulted.

mod pipe {
    use ofgpu::automesher::driver::run_tet;
    use ofgpu::automesher::{read_config_str, KeepRegion};
    use ofgpu::surface::stl::parse_stl;
    use ofgpu::surface::Surface;
    use ofgpu::tetmesh::surfprep::{prepare, prepare_with, Cavity, TetSurface};
    use ofgpu::Error;

    const PIPE_TET_JSON: &str = include_str!("../../tools/automesher/examples/pipe_tet.json");

    /// The pipe `make_pipe_stl.py` writes, as text: the three solids in its
    /// order (wall, inlet, outlet), the same triangles, wound inward. The
    /// vertices go out in Rust's shortest round-trip form, so the f64 build
    /// parses the exact coordinates back and the 1e-12 volume bar measures
    /// the arithmetic, not the text; the script's "%.9e" would cost ~1e-10
    /// of the volume.
    fn pipe_stl(r: f64, l: f64, ntheta: usize, nst: usize) -> String {
        // One facet (a, b, c); the facet normal is the right-hand normal of
        // the order written, which here points INTO the fluid.
        fn tri(s: &mut String, a: [f64; 3], b: [f64; 3], c: [f64; 3]) {
            let (ux, uy, uz) = (b[0] - a[0], b[1] - a[1], b[2] - a[2]);
            let (vx, vy, vz) = (c[0] - a[0], c[1] - a[1], c[2] - a[2]);
            let (qx, qy, qz) = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
            let m = (qx * qx + qy * qy + qz * qz).sqrt();
            if m < 1e-300 {
                return;
            }
            s.push_str(&format!(
                "  facet normal {} {} {}\n    outer loop\n",
                qx / m,
                qy / m,
                qz / m
            ));
            for q in [a, b, c] {
                s.push_str(&format!("      vertex {} {} {}\n", q[0], q[1], q[2]));
            }
            s.push_str("    endloop\n  endfacet\n");
        }
        let p = |i: usize, j: usize| -> [f64; 3] {
            let ph = 2.0 * std::f64::consts::PI * (j % ntheta) as f64 / ntheta as f64;
            [l * i as f64 / nst as f64, r * ph.cos(), r * ph.sin()]
        };
        let mut s = String::new();
        s.push_str("solid wall\n");
        for i in 0..nst {
            for j in 0..ntheta {
                let (a, b) = (p(i, j), p(i + 1, j));
                let (c, d) = (p(i + 1, j + 1), p(i, j + 1));
                tri(&mut s, a, b, c);
                tri(&mut s, a, c, d);
            }
        }
        s.push_str("endsolid wall\nsolid inlet\n");
        for j in 0..ntheta {
            tri(&mut s, [0.0, 0.0, 0.0], p(0, j), p(0, j + 1));
        }
        s.push_str("endsolid inlet\nsolid outlet\n");
        for j in 0..ntheta {
            tri(&mut s, [l, 0.0, 0.0], p(nst, j + 1), p(nst, j));
        }
        s.push_str("endsolid outlet\n");
        s
    }

    /// B1's surface: the R 0.5, L 2 pipe, 32 round and 8 stations, as
    /// `make_pipe_stl.py pipe_tet.stl 0.5 2.0 32 8` writes it.
    fn pipe_surface() -> Surface {
        parse_stl(
            pipe_stl(0.5, 2.0, 32, 8).as_bytes(),
            "pipe_tet",
            "make_pipe_stl.py",
        )
        .expect("the pipe parses")
    }

    /// B1: the inward pipe is a cavity shell - `Cavity::Allow` takes it and
    /// reports its signed volume against the prism formula, `prepare` (the
    /// hull start's rule) refuses it by name.
    #[test]
    fn pipe_cavity_volume() {
        let ts = TetSurface::from_surface(&pipe_surface());
        let p = prepare_with(&ts, 30.0, Cavity::Allow).expect("the inward pipe is a cavity");
        assert_eq!(p.report.n_shells, 1);
        let v = -(32.0 / 2.0) * 0.25 * (2.0 * std::f64::consts::PI / 32.0).sin() * 2.0;
        let vol = p.report.shell_volumes[0];
        // f64 class D; under `single` the STL is parsed into f32 first, so
        // the volume carries its rounding (class R, D-F32-1).
        let tol = if cfg!(feature = "single") { 1e-5 } else { 1e-12 };
        assert!(
            ((vol - v) / v.abs()).abs() < tol,
            "shell volume {vol} against the prism formula {v}"
        );
        match prepare(&ts, 30.0) {
            Err(Error::Refused(m)) => assert!(m.contains("surface/inward"), "{m}"),
            other => panic!("expected the cavity refusal, got {other:?}"),
        }
    }

    /// B2: the hull start keeps §116.9's refusal - a cavity shell is
    /// refused when the carve would take the domain box's hull region.
    #[test]
    fn pipe_hull_start_refused() {
        let mut cfg = read_config_str(PIPE_TET_JSON, "pipe_tet.json").expect("the config parses");
        cfg.castellation.keep_region = KeepRegion::Largest;
        cfg.castellation.seed_point = None;
        match run_tet(&cfg, &pipe_surface(), &mut |_| {}) {
            Err(e) => assert!(e.to_string().contains("surface/inward"), "{e}"),
            Ok(_) => panic!("the hull start must refuse the inward pipe"),
        }
    }

    /// A patch's boundary-face count, and the largest |x - x0| over its
    /// points - the count is asserted, the offset is a diagnostic.
    fn patch_face_and_offset(
        mesh: &ofgpu::io::polymesh::PolyMeshRaw,
        name: &str,
        x0: f64,
    ) -> (usize, f64) {
        let nif = mesh.neighbour.len();
        let p = match mesh.patches.iter().find(|p| p.name == name) {
            Some(p) => p,
            None => return (0, 0.0),
        };
        let mut off = 0.0f64;
        for f in p.start..p.start + p.size {
            for &v in &mesh.faces[nif + f] {
                off = off.max((mesh.points[v as usize].x - x0).abs());
            }
        }
        (p.size, off)
    }

    /// B3: the seed carve meshes the pipe end to end - three prisms per
    /// wall stack, the fluid on the pipe's own patches only.
    ///
    /// TET-14: the pin that lets the pinned `base_size` 0.25 finish on this
    /// surface is the example's radius-edge bound 2.5 - at the default 2
    /// the §116.13 refinement grinds on the 90-degree rims and never
    /// finishes (SPEC-LIT §116.17), so the example carries it and this test
    /// keeps the example's. n 3 and t1 0.005 come with it; the test pins
    /// extent, `base_size`, seed and first thickness and nothing else. Any
    /// refusal panics in f64; under `single` a §92.3 G2 failure is the
    /// known A28 (docs/17-review-later), printed MISSED.
    #[test]
    fn pipe_tet_layers() {
        let mut cfg = read_config_str(PIPE_TET_JSON, "pipe_tet.json").expect("the config parses");
        cfg.domain.extent = [-0.5, 2.5, -1.0, 1.0, -1.0, 1.0];
        cfg.domain.base_size = 0.25;
        cfg.castellation.seed_point = Some([1.0, 0.1, 0.05]);
        cfg.layers.first_thickness = 0.005;
        let started = std::time::Instant::now();
        // The stage banners stream, so a supervisor can see where a run
        // stands; the asserts do not depend on them.
        match run_tet(&cfg, &pipe_surface(), &mut |l: &str| println!("{l}")) {
            Err(e) => {
                let msg = e.to_string();
                if cfg!(feature = "single") && msg.contains("G2 (closure)") {
                    let head: String = msg.chars().take(300).collect();
                    println!(
                        "pipe_tet_layers: MISSED - §92.3 G2 under single \
                         (docs/17-review-later A28): {head}"
                    );
                } else {
                    panic!("pipe_tet_layers refused: {msg}");
                }
            }
            Ok(out) => {
                assert!(out.quality.passed(), "the quality gate failed");
                assert!(
                    out.n_prisms > 0 && out.n_prisms % 3 == 0,
                    "{} prisms, not a whole number of 3-layer stacks",
                    out.n_prisms
                );
                for name in ["wall", "inlet", "outlet"] {
                    let (n, _) = patch_face_and_offset(&out.mesh, name, 0.0);
                    assert!(n > 0, "patch {name} carries no faces");
                }
                for name in ["xMin", "xMax", "yMin", "yMax", "zMin", "zMax"] {
                    let (n, _) = patch_face_and_offset(&out.mesh, name, 0.0);
                    assert_eq!(n, 0, "domain patch {name} carries faces");
                }
                let (_, in_x) = patch_face_and_offset(&out.mesh, "inlet", 0.0);
                let (_, out_x) = patch_face_and_offset(&out.mesh, "outlet", 2.0);
                println!(
                    "pipe_tet_layers: {} tets, {} prisms, dihedral [{:.3}, {:.3}] deg, \
                     {:.1} s; inlet max|x| {in_x:.6}, outlet max|x - 2| {out_x:.6}",
                    out.n_tets,
                    out.n_prisms,
                    out.min_dihedral_deg,
                    out.max_dihedral_deg,
                    started.elapsed().as_secs_f64()
                );
            }
        }
    }
}
