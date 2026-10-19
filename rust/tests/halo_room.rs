// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 139.1 (PAR-09): every cell buffer on the laminar SIMPLE + scalar
//! + energy path has room for the halo. A decomposed part is an ordinary
//! [`GpuMesh`] whose processor faces name neighbours PAST `n_cells`; after
//! this unit every cell buffer is `n_cells + n_halo` long, so those reads
//! land inside the buffer, while every kernel still launches over owned
//! cells only. Nothing exchanges the halo here (PAR-10a): it only has to
//! EXIST.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::decompose::{Decomposition, PartitionMethod};
use ofgpu::momentum::{BuoyancyCoeffs, MomentumControls};
use ofgpu::pressure::{PbicgstabBackend, PressureBackend, SystemProbe};
use ofgpu::simple::{Simple, SimpleControls};
use ofgpu::{
    BcKind, Gpu, GpuMesh, GpuScalarField, GpuVectorField, HostMesh, Label, Result, Scalar, Vec3,
};

/// A machine without a card makes every device test pass vacuously, which is
/// the convention the rest of the crate's tests follow.
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// The 8 x 8 x 1 cavity of SPEC-LIT 139.1's test: walls on x and y, empty z,
/// unit spacing, built with the public block generator.
fn cavity_8() -> Result<HostMesh> {
    let axis = |n: usize| GradedAxis {
        lo: 0.0,
        hi: 1.0,
        n,
        expansion: 1.0,
        two_sided: false,
    };
    let mut spec = BlockSpec::default();
    spec.x = axis(8);
    spec.y = axis(8);
    spec.z = axis(1);
    spec.patch_type = ["wall", "wall", "wall", "wall", "empty", "empty"].map(String::from);
    blockgen::build_mesh(&spec)
}

/// The two-part cut of the cavity, part 0 first.
fn two_parts() -> Result<(HostMesh, Decomposition)> {
    let hm = cavity_8()?;
    let d = Decomposition::build(&hm, 2, &PartitionMethod::Linear)?;
    Ok((hm, d))
}

/// The lid on `yMax`, no-slip on the other three walls, empty z. A processor
/// patch keeps the kind the field constructor seeded, `Cyclic`, so the
/// boundary evaluation reads its neighbour through `b_nbr_cell`, in the halo.
fn seat_lid(gpu: &Gpu, u: &mut GpuVectorField, hm: &HostMesh) -> Result<()> {
    let nbf = hm.n_boundary_faces;
    let mut kind: Vec<Label> = gpu.download(&u.bc_kind)?;
    let mut fr = vec![0.0 as Scalar; nbf];
    let mut rv = vec![Vec3::ZERO; nbf];
    let lid = Vec3::new(1.0, 0.0, 0.0);
    for patch in &hm.patches {
        let (k, v): (BcKind, Vec3) = match patch.name.as_str() {
            "yMax" => (BcKind::FixedValue, lid),
            "xMin" | "xMax" | "yMin" => (BcKind::FixedValue, Vec3::ZERO),
            "zMin" | "zMax" => (BcKind::Empty, Vec3::ZERO),
            _ => continue,
        };
        for f in patch.start..patch.start + patch.size {
            kind[f] = k as Label;
            fr[f] = 1.0;
            rv[f] = v;
        }
    }
    gpu.write(&mut u.bc_kind, &kind)?;
    gpu.write(&mut u.fr, &fr)?;
    gpu.write(&mut u.ref_value, &rv)
}

/// `nut = 0` with calculated boundary values - a laminar run.
fn laminar_nut(gpu: &Gpu, m: &GpuMesh, hm: &HostMesh) -> Result<GpuScalarField> {
    let mut nut = GpuScalarField::zeros(gpu, m, "nut")?;
    let kinds = vec![BcKind::Calculated as Label; hm.n_boundary_faces];
    gpu.write(&mut nut.bc_kind, &kinds)?;
    Ok(nut)
}

/// `T = 0` held at 0 on every wall, empty in z, the processor patch left
/// `Cyclic` as seeded - the passive scalar never moves, which is all this
/// test needs it for.
fn cold_t(gpu: &Gpu, m: &GpuMesh, hm: &HostMesh) -> Result<GpuScalarField> {
    let mut t = GpuScalarField::zeros(gpu, m, "T")?;
    let mut kind: Vec<Label> = gpu.download(&t.bc_kind)?;
    for patch in &hm.patches {
        let k = match patch.name.as_str() {
            "zMin" | "zMax" => BcKind::Empty,
            "xMin" | "xMax" | "yMin" | "yMax" => BcKind::FixedValue,
            _ => continue,
        };
        for f in patch.start..patch.start + patch.size {
            kind[f] = k as Label;
        }
    }
    gpu.write(&mut t.bc_kind, &kind)?;
    Ok(t)
}

fn cavity_controls(nu: Scalar) -> SimpleControls {
    SimpleControls {
        momentum: MomentumControls {
            nu,
            div_scheme: ofgpu::io::case::DivScheme::Upwind,
            sn_grad: ofgpu::fv::SnGradScheme::Uncorrected,
            ..MomentumControls::default()
        },
        ..SimpleControls::default()
    }
}

/// PAR-09: three SIMPLE iterations on each part of the 2-part cavity, no
/// halo exchange anywhere - so the only thing that keeps the kernels'
/// `psi[bNbrCell[f]]` reads inside a buffer is the room the buffers have.
#[test]
fn three_simple_iterations_on_a_part_stay_inside_their_buffers() -> Result<()> {
    let Some(gpu) = gpu() else {
        eprintln!("  [139.1] no GPU: three_simple_iterations skipped");
        return Ok(());
    };
    let (_, d) = two_parts()?;
    for part in &d.parts {
        assert!(part.n_halo > 0, "the 2-part cavity cut must have a halo");
        let hm = &part.mesh;
        let gm = GpuMesh::upload_with_halo(&gpu, hm, part.n_halo)?;
        let mut ctrl = cavity_controls(0.01);
        ctrl.momentum.u_relax = 0.7;
        ctrl.p_relax = 0.3;
        ctrl.momentum.steady = true;
        ctrl.momentum.ddt = ofgpu::timescheme::DdtScheme::SteadyState;
        let mut sim = Simple::new(&gpu, hm, &gm, ctrl, BuoyancyCoeffs {
            g: Vec3::ZERO,
            ..BuoyancyCoeffs::default()
        })?;
        seat_lid(&gpu, sim.u_mut(), hm)?;
        // The processor faces must stay coupled, or nothing would read the
        // halo through the boundary evaluation of U and p.
        for kinds in [gpu.download(&sim.u().bc_kind)?, gpu.download(&sim.p().bc_kind)?] {
            let cyclic = kinds.iter().filter(|&&k| k == BcKind::Cyclic as Label).count();
            assert!(cyclic > 0, "part {}: no processor face is seeded Cyclic", part.part);
        }
        sim.initialise(&gpu)?;
        let nut = laminar_nut(&gpu, &gm, hm)?;
        let t = cold_t(&gpu, &gm, hm)?;
        let mut backend = PbicgstabBackend::new(ctrl.p_solver);
        backend.setup(&gpu, hm, &gm, &SystemProbe::default())?;
        for _ in 0..3 {
            sim.correct(&gpu, &mut backend, &nut, &t)?;
        }
        let n_cells = part.mesh.n_cells;
        assert_eq!(sim.u().f.len(), part.n_local());
        assert_eq!(sim.p().f.len(), part.n_local());
        // Nothing exchanges the halo and no kernel launches over it, so the
        // tail every buffer carries stays exactly zero through the run.
        let u = gpu.download(&sim.u().f)?;
        let p = gpu.download(&sim.p().f)?;
        for (c, v) in u.iter().enumerate().skip(n_cells) {
            assert_eq!(
                *v,
                Vec3::ZERO,
                "part {} halo cell {c} of U moved: nothing may write the halo yet",
                part.part
            );
        }
        for (c, v) in p.iter().enumerate().skip(n_cells) {
            assert_eq!(
                *v,
                0.0 as Scalar,
                "part {} halo cell {c} of p moved: nothing may write the halo yet",
                part.part
            );
        }
        println!(
            "  [139.1] part {}: n_cells {} n_halo {} n_local {}",
            part.part,
            part.mesh.n_cells,
            part.n_halo,
            part.n_local()
        );
    }
    Ok(())
}

/// A whole mesh carries no halo at all: `n_local` IS `n_cells`, and the field
/// constructors size their buffers by it.
#[test]
fn a_whole_mesh_has_no_halo() -> Result<()> {
    let Some(gpu) = gpu() else {
        eprintln!("  [139.1] no GPU: a_whole_mesh_has_no_halo skipped");
        return Ok(());
    };
    let hm = cavity_8()?;
    let gm = GpuMesh::upload(&gpu, &hm)?;
    assert_eq!(gm.n_halo, 0);
    assert_eq!(gm.n_local(), hm.n_cells);
    let f = GpuScalarField::zeros(&gpu, &gm, "T")?;
    assert_eq!(f.f.len(), hm.n_cells);
    let again = GpuMesh::upload_with_halo(&gpu, &hm, 0)?;
    assert_eq!(again.n_cells, hm.n_cells);
    assert_eq!(again.n_local(), hm.n_cells);
    Ok(())
}

/// `upload_with_halo` refuses, by name, a part whose processor faces point
/// past the room it was given, and accepts exactly `part.n_halo`.
#[test]
fn upload_with_halo_refuses_a_halo_too_small() -> Result<()> {
    let Some(gpu) = gpu() else {
        eprintln!("  [139.1] no GPU: upload_with_halo_refuses skipped");
        return Ok(());
    };
    let (_, d) = two_parts()?;
    let part = &d.parts[0];
    assert!(part.n_halo > 0);
    let short = GpuMesh::upload_with_halo(&gpu, &part.mesh, part.n_halo - 1);
    let err = short.err().expect("a halo one short must be refused");
    let msg = format!("{err}");
    assert!(
        msg.contains("GpuMesh::upload_with_halo"),
        "the refusal must name its constructor: {msg}"
    );
    assert!(
        msg.contains(&(part.mesh.n_cells + part.n_halo - 1).to_string()),
        "the refusal must name n_cells + n_halo - 1: {msg}"
    );
    let fit = GpuMesh::upload_with_halo(&gpu, &part.mesh, part.n_halo)?;
    assert_eq!(fit.n_local(), part.n_local());
    Ok(())
}

// ==========================================================================
//  Test A: the audit of the nine files PAR-09 owns
// ==========================================================================

/// The files every cell buffer of the laminar SIMPLE + scalar + energy path
/// is allocated in. Audited up to the first `#[cfg(test)]`, comment lines
/// skipped.
const AUDITED: &[&str] = &[
    "field.rs",
    "fv.rs",
    "simple.rs",
    "momentum.rs",
    "energy.rs",
    "scalar_transport.rs",
    "ldu.rs",
    "solver.rs",
    "mesh.rs",
];

/// Every `.zeros(` argument that is neither a halo site (`n_local`) nor
/// face-sized (contains `faces`) must be listed here with the EXACT count of
/// its occurrences in that file's non-test part, and a reason. A count that
/// no longer matches the scan is itself a failure: update the list.
const ALLOWED: &[(&str, &str, usize, &str)] = &[
    ("simple.rs", "1", 2, "a device scalar: the pressure reduction slot, and the degenerate empty wall-face list"),
    ("momentum.rs", "one(nbf)", 3, "boundary-face sized: b_nu_lam, b_gdot, bw"),
    ("momentum.rs", "one(nif)", 1, "internal-face sized: the face scratch w"),
    ("energy.rs", "n_any", 1, "kernel scratch sized max(n_local, n_boundary_faces), not a per-cell field"),
    ("energy.rs", "1", 1, "a device scalar: dot_out"),
    ("energy.rs", "REDUCE_PARTIALS", 1, "reduction partials, not a cell field"),
    ("energy.rs", "one(nif)", 1, "internal-face sized: the face scratch w"),
    ("energy.rs", "one(nbf)", 1, "boundary-face sized: bw"),
    ("energy.rs", "nbf.max(1)", 1, "boundary-face sized: the conjugate solid condensation"),
    ("energy.rs", "nparts", 4, "reduction partials, not a cell field"),
    ("ldu.rs", "self.nnz", 1, "CSR value slots, sized by non-zeros, not a cell field"),
    ("solver.rs", "MAX_REDUCE_BLOCKS", 2, "reduction partials, not a cell field"),
    ("solver.rs", "1", 12, "a device scalar"),
    ("solver.rs", "3", 1, "three report slots, not a cell field"),
    ("solver.rs", "4", 1, "four doubles, one download (SPEC-LIT §118.4), not a cell field"),
];

/// The number of halo sites the scan must find, so that deleting an
/// `n_local` buffer fails this audit even when every remaining site is
/// still classified. 63 since lane G's merge 3b1170c sized
/// `EnergySources::q_div` (SPEC-LIT 137.4) `n_local` like `q` and `sp`.
const HALO_SITES: usize = 63;

/// Every `.zeros(` argument on this line - the site is the leading dot, so
/// `fn zeros(` and `GpuScalarField::zeros(` are not sites - balanced to the
/// closing `)` on the SAME line, whitespace removed.
fn zeros_args(line: &str) -> Vec<String> {
    let bytes = line.as_bytes();
    let mut args = Vec::new();
    let mut i = 0;
    while let Some(rel) = line[i..].find(".zeros") {
        let mut j = i + rel + ".zeros".len();
        if line[j..].starts_with("::<") {
            let Some(gt) = line[j..].find('>') else { break };
            j += gt + 1;
        }
        if !line[j..].starts_with('(') {
            i = j;
            continue;
        }
        let (mut depth, mut k) = (0usize, j);
        while k < bytes.len() {
            match bytes[k] {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                _ => {}
            }
            k += 1;
        }
        if depth != 0 {
            break;
        }
        args.push(line[j + 1..k].split_whitespace().collect());
        i = k + 1;
    }
    args
}

/// THE audit: every cell buffer on the laminar SIMPLE + scalar + energy path
/// spells `n_local`, every face buffer stays face-sized, and every other
/// allocation is on ALLOWED with the exact count the scan finds.
#[test]
fn every_simple_path_buffer_has_room_for_the_halo() {
    let mut halo = 0usize;
    let mut offenders: Vec<String> = Vec::new();
    let mut counts: Vec<(&str, String, usize)> = Vec::new();
    for name in AUDITED {
        let path = format!("{}/src/{name}", env!("CARGO_MANIFEST_DIR"));
        let text =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        assert!(!text.is_empty(), "{name} was read but is empty");
        for (no, line) in text.lines().enumerate() {
            if line.trim() == "#[cfg(test)]" {
                break;
            }
            if line.trim().starts_with("//") {
                continue;
            }
            for arg in zeros_args(line) {
                if arg.contains("n_local") {
                    halo += 1;
                } else if arg.contains("faces") {
                    // face-sized: face buffers never carry the halo
                } else if ALLOWED.iter().any(|(f, a, _, _)| *f == *name && *a == arg) {
                    match counts.iter_mut().find(|(f, a, _)| *f == *name && *a == arg) {
                        Some((_, _, c)) => *c += 1,
                        None => counts.push((name, arg, 1)),
                    }
                } else {
                    offenders.push(format!("{name}:{} {arg}", no + 1));
                }
            }
        }
    }
    let mut report: Vec<String> = offenders.clone();
    for (f, a, want, _) in ALLOWED {
        match counts.iter().find(|(cf, ca, _)| *cf == *f && *ca == *a) {
            Some((_, _, n)) if n == want => {}
            Some((_, _, n)) => report.push(format!(
                "{f}: {a:?} is scanned {n} times but ALLOWED says {want} - update the list"
            )),
            None => report.push(format!(
                "{f}: {a:?} is on ALLOWED for {want} but never scanned - update the list"
            )),
        }
    }
    assert!(
        report.is_empty(),
        "{} unclassified or stale .zeros() sites:\n{}",
        report.len(),
        report.join("\n")
    );
    assert_eq!(
        halo, HALO_SITES,
        "the halo-site count moved: every n_local buffer is one of them, so \
         one was added or deleted - update HALO_SITES with the audit"
    );
    println!(
        "  [139.1] audited {} files: {halo} halo sites, {} ALLOWED kinds, {} allowed sites",
        AUDITED.len(),
        ALLOWED.len(),
        counts.iter().map(|(_, _, c)| c).sum::<usize>()
    );
}
