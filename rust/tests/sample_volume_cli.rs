// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! VIO-06's `sample_volume_cli` test, in BOTH builds: `ofgpu-sample volume`
//! (SPEC-LIT 117.11) run as the real binary on a case written to disk.
//!
//! 1. `box_case_writes_the_in_run_bytes`: a uniform Cartesian box. The
//!    `.vdb`, `.nvdb` and `.usda` the command writes equal, byte for byte,
//!    the files `build_writers` writes from one `WriteCtx` built the way a
//!    driver builds it (`cart` from `cartesian_info`, no lattice), on the
//!    fields read back from the same time directory.
//! 2. `refined_mesh_writes_the_lattice_bytes`: the 2:1 mesh of
//!    `mesh/refined.rs` with a `-lattice` block. The files equal the writers'
//!    on `VoxelMap::build` of the same keys, and the printed fluid fraction
//!    and free-stream median are the map's.
//! 3. `volume_refuses_by_name`: each refusal of the command names its reason.
//!
//! Written from SPEC-LIT section 117 (117.4, 117.6, 117.11).
//! Provenance: ORIGINAL.
//! No GPL-licensed source was consulted.

use std::path::{Path, PathBuf};
use std::process::Command;

use ofgpu::io::fields::{read_scalar_field, read_vector_field};
use ofgpu::io::nvdb::Precision;
use ofgpu::io::polymesh::{build_host_mesh, read_poly_mesh, write_poly_mesh_raw, PolyMeshRaw};
use ofgpu::io::voxelmap::{LatticeActivation, LatticeSample, LatticeSpec, VoxelMap};
use ofgpu::io::{build_writers, cartesian_info, CartesianInfo, OutputField, OutputFormat, WriteCtx};
use ofgpu::mesh::refined::refined_core;
use ofgpu::mesh::HostMesh;
use ofgpu::types::{DVec3, Vec3};
use ofgpu::Scalar;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ofgpu_sample_volume_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// `refined_core`'s mesh as a raw polyMesh.
fn raw_of(n: [usize; 3], d: Vec3, levels: u32) -> PolyMeshRaw {
    let rb = refined_core(n, d, 0.3, levels).expect("refined_core");
    let points = rb.points.iter().map(|&p| DVec3::new(p.x as f64, p.y as f64, p.z as f64)).collect();
    let mut owner = rb.mesh.owner.clone();
    owner.extend(rb.mesh.b_face_cells.iter().copied());
    PolyMeshRaw {
        points,
        faces: rb.faces.clone(),
        owner,
        neighbour: rb.mesh.neighbour.clone(),
        patches: rb.mesh.patches.clone(),
    }
}

fn header(class: &str, object: &str) -> String {
    format!("FoamFile\n{{\n    version     2.0;\n    format      ascii;\n    class       {class};\n    object      {object};\n}}\n")
}

/// The case: its polyMesh, and the time directory `0.5` with `U`, `p` and a
/// `surfaceScalarField` `phi` (which the command skips), plus `0.25` with
/// `p` alone. Every value is exact in f32 and in six digits.
fn write_case(case: &Path, raw: &PolyMeshRaw) {
    write_poly_mesh_raw(&case.join("constant").join("polyMesh"), raw).expect("write polyMesh");
    let n = raw.owner.iter().copied().max().map_or(0, |m| m as usize + 1);
    let p: Vec<String> = (0..n).map(|c| format!("{}", ((37 * c) % 101) as f64 / 8.0 - 3.0)).collect();
    let u: Vec<String> = (0..n)
        .map(|c| {
            let t = ((c % 7) as f64 + 1.0) * 0.25;
            format!("({} {} 2)", 3.0 * t + ((c % 11) as f64) * 0.125, -4.0 * t)
        })
        .collect();
    let scalar = format!(
        "{}dimensions [0 2 -2 0 0 0 0];\ninternalField nonuniform List<scalar> {n}\n(\n{}\n)\n;\nboundaryField\n{{\n}}\n",
        header("volScalarField", "p"),
        p.join("\n")
    );
    let vector = format!(
        "{}dimensions [0 1 -1 0 0 0 0];\ninternalField nonuniform List<vector> {n}\n(\n{}\n)\n;\nboundaryField\n{{\n}}\n",
        header("volVectorField", "U"),
        u.join("\n")
    );
    let phi = format!("{}dimensions [0 3 -1 0 0 0 0];\ninternalField uniform 0;\n", header("surfaceScalarField", "phi"));
    for (time, files) in [("0.5", vec![("p", &scalar), ("U", &vector), ("phi", &phi)]), ("0.25", vec![("p", &scalar)])] {
        let dir = case.join(time);
        std::fs::create_dir_all(&dir).expect("time dir");
        for (name, text) in files {
            std::fs::write(dir.join(name), text).expect("field file");
        }
    }
}

fn run(args: &[&str]) -> (i32, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_ofgpu-sample")).args(args).output().expect("run ofgpu-sample");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// The mesh and the fields exactly as the command reads them back.
fn read_back(case: &Path) -> (PolyMeshRaw, HostMesh, Vec<Vec3>, Vec<Scalar>) {
    let raw = read_poly_mesh(case).expect("read polyMesh");
    let hm = build_host_mesh(&raw).expect("build_host_mesh");
    let u = read_vector_field(&case.join("0.5").join("U"), hm.n_cells).expect("U").internal;
    let p = read_scalar_field(&case.join("0.5").join("p"), hm.n_cells).expect("p").internal;
    (raw, hm, u, p)
}

/// What a driver's `OutputPipeline::write` hands its writers at one write.
fn write_reference(dir: &Path, hm: &HostMesh, cart: Option<&CartesianInfo>, vox: Option<&VoxelMap>, u: &[Vec3], p: &[Scalar]) {
    let formats = [OutputFormat::Vdb, OutputFormat::Nvdb, OutputFormat::Usda];
    let fields = [OutputField::vector("U", u), OutputField::scalar("p", p)];
    let mut writers = build_writers(dir, "ref", &formats, Precision::F32).expect("build_writers");
    let ctx = WriteCtx { time: 0.5, step: 0, name: "0.5", mesh: hm, cart, vox, fields: &fields, foam: &[] };
    for w in &mut writers {
        w.write_step(&ctx).expect("write_step");
    }
}

fn assert_same_files(got: &Path, want: &Path) {
    for rel in ["VDB/ref_000000.vdb", "VDB/ref_000000.nvdb", "ref.usda"] {
        let a = std::fs::read(got.join(rel)).unwrap_or_else(|e| panic!("{rel} from the command: {e}"));
        let b = std::fs::read(want.join(rel)).unwrap_or_else(|e| panic!("{rel} from the writers: {e}"));
        assert!(a == b, "{rel}: the command's {} bytes differ from the in-run writers' {} bytes", a.len(), b.len());
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

#[test]
fn box_case_writes_the_in_run_bytes() {
    let dir = scratch("box");
    let case = dir.join("case");
    write_case(&case, &raw_of([6, 4, 3], Vec3::new(0.25, 0.5, 0.125), 0));
    let out = dir.join("out");
    let (code, stdout, stderr) = run(&[
        "volume", case.to_str().unwrap(), "latest", "-vdb", "-nvdb", "-usda", "-freestream", "2",
        "-out", out.to_str().unwrap(), "-stem", "ref",
    ]);
    assert_eq!(code, 0, "stdout:\n{stdout}\nstderr:\n{stderr}");
    assert!(stdout.contains("at time 0.5: 72 cells; fields U, p (of U, p)"), "{stdout}");
    assert!(stdout.contains("skipped, not a volScalarField or volVectorField: phi"), "{stdout}");
    assert!(stdout.contains("uniform Cartesian box 6 x 4 x 3"), "{stdout}");
    assert!(stdout.contains("fluid fraction 1"), "{stdout}");

    let (_raw, hm, u, p) = read_back(&case);
    let cart = ofgpu::pressure::cartesian::detect(&hm).expect("the box is Cartesian");
    let info = cartesian_info(&hm, &cart);
    let want = dir.join("want");
    write_reference(&want, &hm, Some(&info), None, &u, &p);
    assert_same_files(&out, &want);

    // The last x plane by the cells' own centres: within half a cell of the
    // largest centre x, so a rounded centre cannot drop a cell.
    let x_max = hm.c[..hm.n_cells].iter().map(|c| c.x).fold(Scalar::NEG_INFINITY, Scalar::max);
    let plane: Vec<f64> =
        (0..hm.n_cells).filter(|&i| hm.c[i].x > x_max - 0.125).map(|i| f64::from(u[i].x)).collect();
    assert_eq!(plane.len(), 12);
    let m = median(plane);
    let line = format!("median U.x over the 12 cells of the xMax cell plane (i = 5) = {m}; |median - Ux|/|Ux| = {}", (m - 2.0).abs() / 2.0);
    assert!(stdout.contains(&line), "want '{line}' in:\n{stdout}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn refined_mesh_writes_the_lattice_bytes() {
    let dir = scratch("refined");
    let case = dir.join("case");
    write_case(&case, &raw_of([4, 4, 4], Vec3::new(0.25, 0.25, 0.25), 1));
    let out = dir.join("out");
    let lattice = r#"{"voxel": 0.0625, "box": [-0.25, -0.25, -0.25, 0.75, 1.25, 1.25], "sample": "linear", "activation": "fluid", "background": -7.5}"#;
    let (code, stdout, stderr) = run(&[
        "volume", case.to_str().unwrap(), "0.5", "-vdb", "-nvdb", "-usda", "-lattice", lattice,
        "-freestream", "2", "-out", out.to_str().unwrap(), "-stem", "ref",
    ]);
    assert_eq!(code, 0, "stdout:\n{stdout}\nstderr:\n{stderr}");

    let (raw, hm, u, p) = read_back(&case);
    assert!(ofgpu::pressure::cartesian::detect(&hm).is_err(), "the 2:1 mesh is not a Cartesian box");
    let spec = LatticeSpec {
        voxel: Some(0.0625),
        bbox: Some([-0.25, -0.25, -0.25, 0.75, 1.25, 1.25]),
        sample: LatticeSample::Linear,
        activation: LatticeActivation::Fluid,
        background: -7.5,
        ..Default::default()
    };
    let vm = VoxelMap::build(&raw, &hm, &spec).expect("VoxelMap::build");
    assert_eq!(vm.n, [16, 24, 24]);
    assert_eq!(vm.report.n_fluid, 3072, "12 x 16 x 16 voxel centres lie in the unit cube");
    assert_eq!(vm.report.fluid_fraction, 3072.0 / 9216.0);
    let line = format!("16 x 24 x 24 of h = 0.0625 (requested 0.0625, coarsen 1): 9216 voxels, 3072 fluid, 0 gap-filled; fluid fraction {}", vm.report.fluid_fraction);
    assert!(stdout.contains(&line), "want '{line}' in:\n{stdout}");

    let want = dir.join("want");
    write_reference(&want, &hm, None, Some(&vm), &u, &p);
    assert_same_files(&out, &want);

    let ux: Vec<Scalar> = u.iter().map(|w| w.x).collect();
    let s = vm.sample(&ux).expect("sample");
    let mut plane = Vec::new();
    for j in 0..24 {
        for k in 0..24 {
            if vm.owner(15, j, k).is_some() {
                plane.push(s[vm.index(15, j, k)]);
            }
        }
    }
    assert_eq!(plane.len(), 256);
    let m = median(plane);
    let line = format!("median U.x over the 256 fluid voxels of the xMax lattice plane (i = 15) = {m}; |median - Ux|/|Ux| = {}", (m - 2.0).abs() / 2.0);
    assert!(stdout.contains(&line), "want '{line}' in:\n{stdout}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn volume_refuses_by_name() {
    let dir = scratch("refuse");
    let case = dir.join("case");
    write_case(&case, &raw_of([6, 4, 3], Vec3::new(0.25, 0.5, 0.125), 0));
    let c = case.to_str().unwrap();
    let out = dir.join("out");
    let o = out.to_str().unwrap();
    let cases: [(&[&str], &str); 8] = [
        (&["volume", c, "0.5", "-out", o], "no format to write"),
        (&["volume", c, "0.5", "-usda", "-out", o], "-usda: a .usda scene points at .vdb or .nvdb files"),
        (&["volume", c, "0.5", "-vdb", "-isaac", "index", "-out", o], "-isaac: the Isaac Sim profile"),
        (&["volume", c, "0.75", "-vdb", "-out", o], "does not exist"),
        (&["volume", c, "0.5", "-vdb", "-lattice", r#"{"voxels": 1}"#, "-out", o], "not an output.visualisation.lattice block"),
        (&["volume", c, "0.5", "-vdb", "-precision", "fp8", "-out", o], "output.visualisation.precision"),
        (&["volume", c, "0.5", "-vdb", "-freestream", "0", "-out", o], "cannot be a free-stream U.x"),
        (&["volume", c, "0.25", "-vdb", "-freestream", "2", "-out", o], "has no volVectorField U"),
    ];
    for (args, phrase) in cases {
        let (code, stdout, stderr) = run(args);
        assert_eq!(code, 1, "{args:?} should be refused; stdout:\n{stdout}\nstderr:\n{stderr}");
        assert!(stderr.contains(phrase), "{args:?}: want '{phrase}' in:\n{stderr}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
