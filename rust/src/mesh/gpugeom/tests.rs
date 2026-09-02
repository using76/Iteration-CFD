// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
//! SPEC-LIT section 82.6. No GPL-licensed source was consulted.

use super::*;
use crate::blockgen::{BlockSpec, GradedAxis};
use crate::mesh::refined;

fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// A mesh, its points and its face point lists, under a name for the failure
/// message. The tuple is what both `geometry::compute` and the device sweep
/// take, so it is the shape a fixture has to be.
type Fixture = (String, HostMesh, Vec<Vec3>, Vec<Vec<Label>>);

/// The meshes the bitwise gate runs on, and why each one is here.
///
/// Every array the sweep writes has to be exercised by at least one of them,
/// or the gate is a claim about the arrays that happen to be non-zero on a
/// uniform box - which is most of them, but not `skew_corr`, not
/// `non_orth_corr`, not `b_weights` and not a `weight` away from one half.
fn fixtures() -> Vec<Fixture> {
    let mut out = Vec::new();
    let cube = Vec3::new(0.25, 0.2, 0.3);

    // (a) A uniform box. Orthogonal and unskewed, so `skew_corr` is exactly
    //     zero and `non_orth_corr` is the round-off `non_orth_split` leaves
    //     behind - which is the case a wrong port passes anyway, and is here
    //     as the floor.
    let r = refined::build([5, 4, 3], cube, &[0u32; 60]).expect("uniform box");
    out.push(("uniform 5x4x3".to_string(), r.mesh, r.points, r.faces));

    // (b) A 2:1 refined box with three levels. The interface faces are skewed
    //     - section 2.5 measures one at 0.1421 |d| - so this is the fixture
    //     that makes `skew_corr` a real array and drives `weights` away from
    //     one half. It is also the mesh an adapt produces.
    let at = |i: usize, j: usize, k: usize| i + 6 * (j + 6 * k);
    let mut lev = vec![0u32; 216];
    lev[at(3, 3, 3)] = 1;
    lev[at(1, 1, 1)] = 2;
    let r = refined::build([6, 6, 6], cube, &lev).expect("2:1 refined box");
    out.push(("2:1 refined 6x6x6, three levels".to_string(), r.mesh, r.points, r.faces));

    // (c) A graded block with two cyclic axes. Grading makes every internal
    //     face's weight and delta coefficient different from its neighbour's,
    //     and the cyclic pair is the only route into the branch of
    //     `meshBoundaryFaceMetrics` that writes `b_weights` and a boundary
    //     `non_orth_corr` at all.
    let mut b = BlockSpec {
        x: GradedAxis { lo: 0.0, hi: 1.0, n: 8, expansion: 4.0, two_sided: false },
        y: GradedAxis { lo: 0.0, hi: 0.6, n: 6, expansion: 3.0, two_sided: true },
        z: GradedAxis { lo: 0.0, hi: 0.4, n: 4, expansion: 1.0, two_sided: false },
        ..Default::default()
    };
    b.set_cyclic_axis(2).expect("z is cyclic");
    let raw = crate::blockgen::raw_mesh(&b).expect("graded cyclic block");
    let m = crate::io::polymesh::build_host_mesh(&raw).expect("graded cyclic host mesh");
    out.push(("graded 8x6x4, z cyclic".to_string(), m, raw.points, raw.faces));

    // (d) The 2:1 box again with every point displaced.
    //
    //     Fixtures (a)-(c) are all axis-aligned: `Sf` has two zero components,
    //     so `Sf . d` is one product plus two exact zeros and a compiler has
    //     nothing there to contract. Displacing the points makes every face
    //     non-planar and every area vector fully three-dimensional, and it is
    //     the only fixture here that exercises the median decomposition as
    //     anything but a rectangle: the four sub-triangles of a jittered quad
    //     have different areas, so `Cf` stops being the vertex average and
    //     `area` stops being exact.
    //
    //     This fixture was added because a host-side SIMULATION of a fused
    //     multiply-add on `Sf . (Cf - C_P)` found no difference on (a)-(c),
    //     and that looked like the bitwise gate proving nothing. Replacing the
    //     simulation with the real contracted build - see
    //     `the_contraction_this_unit_turns_off_is_real` - showed the
    //     simulation had simply probed the wrong expression: the contraction
    //     reaches the volume through the pyramid centroid on a uniform box
    //     too. The fixture stays, because a sweep written for arbitrary
    //     polyhedra should be gated on something that is not a rectangle.
    //
    //     The displacement is a fixed 32-bit LCG on the point index, so this
    //     mesh is the same on every run and on every machine, which a gate
    //     that reports ulp differences has to be.
    let r = refined::build([6, 6, 6], cube, &lev).expect("2:1 refined box");
    let jitter = |i: usize| -> Vec3 {
        let mut h = (i as u32).wrapping_mul(2_654_435_761);
        let mut next = || {
            h = h.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (h >> 8) as Scalar / (1u32 << 24) as Scalar - 0.5
        };
        Vec3::new(next(), next(), next())
    };
    let moved: Vec<Vec3> = r
        .points
        .iter()
        .enumerate()
        .map(|(i, &p)| p + jitter(i) * (0.08 * cube.x))
        .collect();
    out.push((
        "2:1 refined 6x6x6, points displaced".to_string(),
        r.mesh,
        moved,
        r.faces,
    ));

    // (e) A box whose every face carries a fifth vertex, on the midpoint of
    //     its first edge.
    //
    //     All four fixtures above are QUADRILATERAL, because every emitter in
    //     this crate makes quads - and the sweep is written for arbitrary
    //     polyhedra, which is a claim nothing here was testing. The subdivided
    //     face is geometrically the same polygon, so the mesh is still valid
    //     and still closes; what changes is the median decomposition of
    //     section 2.1, which now takes FIVE sub-triangles about the average of
    //     five points instead of four about four - and one of those triangles
    //     has the inserted vertex collinear with its neighbours, so its area
    //     is zero and it contributes `t_a = 0` to the centroid weighting.
    //     That is the branch a cut cell (section 24) reaches routinely and no
    //     other fixture here reaches at all.
    //
    //     The midpoint is appended per FACE rather than shared between the two
    //     faces of an edge, so the point set is non-conformal. Nothing in
    //     either sweep looks at point sharing - `validate` checks indices and
    //     the decomposition reads coordinates - and both sweeps are handed the
    //     identical mesh, which is what the comparison needs.
    let r = refined::build([4, 3, 2], Vec3::new(0.5, 0.25, 2.0), &[0u32; 24])
        .expect("box for the polygonal fixture");
    let mut pts = r.points.clone();
    let mut faces = Vec::with_capacity(r.faces.len());
    for fv in &r.faces {
        let a = pts[fv[0] as usize];
        let b = pts[fv[1] as usize];
        let mid = (a + b) * 0.5;
        let id = pts.len() as Label;
        pts.push(mid);
        let mut nf = vec![fv[0], id];
        nf.extend_from_slice(&fv[1..]);
        faces.push(nf);
    }
    out.push(("uniform 4x3x2, pentagonal faces".to_string(), r.mesh, pts, faces));

    out
}

/// Compare two `Scalar` arrays by their BITS, and say which entry first
/// differed and by how many units in the last place.
fn same_scalars(what: &str, a: &[Scalar], b: &[Scalar]) -> Option<String> {
    if a.len() != b.len() {
        return Some(format!("{what}: {} entries against {}", a.len(), b.len()));
    }
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if x.to_bits() != y.to_bits() {
            let ulp = (x.to_bits() as i64 - y.to_bits() as i64).abs();
            return Some(format!(
                "{what}[{i}]: host {x:.17e} ({:#x}), device {y:.17e} ({:#x}), {ulp} ulp",
                x.to_bits(),
                y.to_bits()
            ));
        }
    }
    None
}

fn same_vectors(what: &str, a: &[Vec3], b: &[Vec3]) -> Option<String> {
    if a.len() != b.len() {
        return Some(format!("{what}: {} entries against {}", a.len(), b.len()));
    }
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        for (c, (p, q)) in [(x.x, y.x), (x.y, y.y), (x.z, y.z)].iter().enumerate() {
            if p.to_bits() != q.to_bits() {
                let ulp = (p.to_bits() as i64 - q.to_bits() as i64).abs();
                return Some(format!(
                    "{what}[{i}].{}: host {p:.17e}, device {q:.17e}, {ulp} ulp",
                    ["x", "y", "z"][c]
                ));
            }
        }
    }
    None
}

/// Every geometric array of two meshes, compared bit for bit.
///
/// All sixteen of them, named, so that a failure says which quantity moved.
/// Nothing here is allowed to be "close": see the gate below.
fn differences(host: &HostMesh, dev: &HostMesh) -> Vec<String> {
    let mut bad: Vec<String> = Vec::new();
    for (w, a, b) in [
        ("v", &host.v, &dev.v),
        ("mag_sf", &host.mag_sf, &dev.mag_sf),
        ("weights", &host.weights, &dev.weights),
        ("delta_coeffs", &host.delta_coeffs, &dev.delta_coeffs),
        ("b_mag_sf", &host.b_mag_sf, &dev.b_mag_sf),
        ("b_delta_coeffs", &host.b_delta_coeffs, &dev.b_delta_coeffs),
        ("b_y", &host.b_y, &dev.b_y),
        ("b_weights", &host.b_weights, &dev.b_weights),
    ] {
        if let Some(m) = same_scalars(w, a, b) {
            bad.push(m);
        }
    }
    for (w, a, b) in [
        ("c", &host.c, &dev.c),
        ("sf", &host.sf, &dev.sf),
        ("cf", &host.cf, &dev.cf),
        ("non_orth_corr", &host.non_orth_corr, &dev.non_orth_corr),
        ("skew_corr", &host.skew_corr, &dev.skew_corr),
        ("b_sf", &host.b_sf, &dev.b_sf),
        ("b_cf", &host.b_cf, &dev.b_cf),
        ("b_non_orth_corr", &host.b_non_orth_corr, &dev.b_non_orth_corr),
    ] {
        if let Some(m) = same_vectors(w, a, b) {
            bad.push(m);
        }
    }
    bad
}

/// **THE GATE.** The device sweep and the host sweep write the same bits.
///
/// Not "agree to 1e-15". Moving where the geometry is computed must not move
/// what it is: `weights` feeds every interpolation, `delta_coeffs` every
/// Laplacian and `v` every source term, so a last-bit difference here is a
/// different answer everywhere, and a solver that changed its answer because
/// a mesh was rebuilt on the device would be a bug that no tolerance could
/// distinguish from a real one.
///
/// Both sweeps are run from the SAME starting mesh, so the comparison is
/// between two computations of one thing and not between two histories.
#[test]
fn the_device_sweep_is_bitwise_identical_to_the_host_sweep() {
    let Some(g) = gpu() else { return };
    let k = MeshGeomKernels::new(&g).expect("the meshgeom kernels must load");

    for (name, base, points, faces) in fixtures() {
        let mut host = base.clone();
        geometry::compute(&mut host, &points, &faces).expect("the host sweep");

        let mut dev = base.clone();
        gpu_compute_geometry(&g, &k, &mut dev, &points, &faces).expect("the device sweep");

        let bad = differences(&host, &dev);
        assert!(
            bad.is_empty(),
            "SPEC-LIT section 82.3 requires the device geometry sweep to be BITWISE \
             identical to the host sweep, and on '{name}' it is not:\n  {}",
            bad.join("\n  ")
        );
        assert_eq!(host.n_points, dev.n_points, "n_points on '{name}'");
        assert_eq!(host.b_nbr_face, dev.b_nbr_face, "b_nbr_face on '{name}'");
        assert_eq!(host.b_nbr_cell, dev.b_nbr_cell, "b_nbr_cell on '{name}'");
    }
}

/// The gate above is not vacuous: the SAME source compiled with nvcc's
/// default multiply-add contraction does NOT reproduce the host's bits.
///
/// This is a measurement, not an argument. `build.rs` compiles
/// `cuda/meshgeom.cu` twice - once with `-fmad=false` and once with
/// `-fmad=true` - and this test runs both against the host sweep. Without it,
/// `-fmad=false` could be deleted for the throughput and the gate might still
/// pass on fixtures where nothing happened to contract, which is precisely how
/// section 67.11's first draft came to assert an FMA had not happened.
///
/// **Measured on this machine, five fixtures:** the contracted module moves
/// 9, 14, 15, 14 and 8 of the sixteen geometry arrays, by 1 to 2 ulp. It moves
/// them on the UNIFORM box too, which is not where the argument for the flag
/// said to look: a box's area vectors are axis-aligned, so `Sf . d` has
/// nothing to fuse, but the pyramid centroid `0.75 Cf + 0.25 apex` does, and
/// that reaches `v` and `c` and through them everything else.
///
/// This test replaced a host-side SIMULATION of the contraction, which probed
/// `Sf . (Cf - C_P)` and found no difference on any box mesh - and so reported
/// that the flag was buying nothing. It was probing the wrong expression. That
/// is the argument for running the other compiler rather than reasoning about
/// it: a simulation can only find the contraction you thought of.
#[test]
fn the_contraction_this_unit_turns_off_is_real() {
    let Some(g) = gpu() else { return };
    let unfused = MeshGeomKernels::new(&g).expect("the meshgeom kernels must load");
    let fused = MeshGeomKernels::from_cubin(&g, crate::kernels::MESHGEOM_FMAD)
        .expect("the contracted twin must load");

    let mut separated = 0usize;
    for (name, base, points, faces) in fixtures() {
        let mut host = base.clone();
        geometry::compute(&mut host, &points, &faces).expect("the host sweep");

        let mut a = base.clone();
        gpu_compute_geometry(&g, &unfused, &mut a, &points, &faces).expect("unfused");
        let mut b = base.clone();
        gpu_compute_geometry(&g, &fused, &mut b, &points, &faces).expect("fused");

        let d_unfused = differences(&host, &a);
        let d_fused = differences(&host, &b);
        assert!(
            d_unfused.is_empty(),
            "the -fmad=false build must match the host on '{name}': {}",
            d_unfused.join("; ")
        );
        if d_fused.is_empty() {
            println!("  '{name}': the contraction changes nothing (axis-aligned)");
        } else {
            separated += 1;
            println!(
                "  '{name}': the contraction moves {} of 16 arrays; first is {}",
                d_fused.len(),
                d_fused[0]
            );
        }
    }

    assert!(
        separated > 0,
        "cuda/meshgeom.cu is compiled with -fmad=false to keep the bitwise gate, \
         but on every fixture here the CONTRACTED build of the same source gives \
         the host's bits too - so the flag is costing FMA throughput and buying \
         nothing that these meshes can see. Either add a fixture that separates \
         them (a non-axis-aligned mesh does) or drop the flag and say so"
    );
}

/// `-fmad=false` is what makes the gate above pass, so the flag is held by a
/// test rather than by a comment.
///
/// `build.rs` runs before the crate compiles and cannot be reached from it, so
/// the list is read off disk - the same shape as section 80's citation census
/// and section 81's module census. A future edit that drops the flag for speed
/// fails here, naming the section that would break.
#[test]
fn the_translation_unit_is_compiled_with_fmad_off() {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("build.rs");
    let text = std::fs::read_to_string(&p).expect("build.rs must be readable");

    let decl = text
        .lines()
        .find(|l| l.trim_start().starts_with("const FMAD_OFF_UNITS"))
        .unwrap_or_else(|| {
            panic!(
                "build.rs no longer declares FMAD_OFF_UNITS; SPEC-LIT section 82.4 \
                 requires cuda/meshgeom.cu to be compiled with -fmad=false, and \
                 nothing else in this crate asks for that flag"
            )
        });
    assert!(
        decl.contains("\"meshgeom.cu\""),
        "build.rs declares FMAD_OFF_UNITS as `{decl}`, which does not name \
         meshgeom.cu. SPEC-LIT section 82.4: nvcc contracts a*b + c into one \
         rounding and rustc does not, so without the flag the device sweep is \
         off the host's answer in the last bit on nearly every face and the \
         bitwise gate cannot hold"
    );
    assert!(
        text.contains("-fmad=false"),
        "FMAD_OFF_UNITS names meshgeom.cu but build.rs never passes -fmad=false"
    );
    // And the flag must be passed BY NAME, not to everything: contracting is
    // the better answer everywhere else in this crate, and sections 28.5 and
    // 67.11 measure kernels that rely on it.
    assert!(
        text.contains("FMAD_OFF_UNITS.contains("),
        "build.rs passes -fmad=false without consulting FMAD_OFF_UNITS, which \
         would apply it to every translation unit; sections 28.5 and 67.11 \
         measure kernels whose answers depend on the contraction"
    );
}

/// The device sweep gathers, so it needs the cell -> face CSR the host sweep
/// does not. Refused by name, with the alternative - section 82.7.
#[test]
fn a_mesh_without_a_cell_face_csr_is_refused_by_name() {
    let Some(g) = gpu() else { return };
    let k = MeshGeomKernels::new(&g).expect("the meshgeom kernels must load");

    let r = refined::build([3, 3, 3], Vec3::new(0.3, 0.3, 0.3), &[0u32; 27]).unwrap();
    let mut m = r.mesh.clone();
    m.cf_face.clear();

    let e = gpu_compute_geometry(&g, &k, &mut m, &r.points, &r.faces)
        .expect_err("a mesh with no CSR must be refused");
    let s = e.to_string();
    assert!(
        s.contains("cf_face") && s.contains("build_cell_face_maps"),
        "the refusal must name the array and the call that fixes it, and it says: {s}"
    );

    // The host sweep, which needs no addressing, still succeeds on the same
    // mesh - so the refusal is a statement about this module and not about the
    // mesh being broken.
    let mut m2 = r.mesh.clone();
    m2.cf_face.clear();
    geometry::compute(&mut m2, &r.points, &r.faces)
        .expect("the host sweep needs no cell -> face CSR");
}

/// The flattening is a CSR of the face list and nothing else.
#[test]
fn the_flattened_face_list_is_the_face_list() {
    let r = refined::build([4, 3, 2], Vec3::new(0.5, 0.25, 2.0), &[0u32; 24]).unwrap();
    let csr = flatten_faces(&r.faces);
    assert_eq!(csr.offset.len(), r.faces.len() + 1);
    assert_eq!(csr.offset[0], 0);
    for (f, fv) in r.faces.iter().enumerate() {
        let (a, b) = (csr.offset[f] as usize, csr.offset[f + 1] as usize);
        assert_eq!(b - a, fv.len(), "face {f} length");
        assert_eq!(&csr.point[a..b], &fv[..], "face {f} points");
    }
    assert_eq!(csr.point.len(), csr.offset[r.faces.len()] as usize);
    assert_eq!(flatten_faces(&[]).offset, vec![0 as Label]);
}
