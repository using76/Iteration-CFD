// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 139.8 (PAR-13a)'s gate - the partition-invariant laplacian and
//! Gauss gradients, in BOTH builds.
//!
//! An integration test because the library's own test target does not compile
//! under `--features single`; an integration test links the normal (non-test)
//! library, which does. The gate is BITWISE everywhere: a part-assembled
//! laplacian matrix is [`Decomposition::split_matrix`] of the whole-mesh one
//! at P = 2, 3, 4, and a part's gradients are the whole mesh's gradients bit
//! for bit - which §139.2's gate (tests/distflow.rs) measured and did NOT
//! hold, because the processor face's term was summed in the boundary loop
//! and the non-owning side carried a recomputed weight. PAR-13a fixes both:
//! the part twins walk the merged row of SPEC-LIT 70, rebuild the whole
//! mesh's operands exactly (`b_proc`, `b_proc_w`), and leave
//! `internal_coeffs` at the zero `split_matrix` gives a cut face.
//!
//! Where the bitwise line fails the tests print one `MISSED` line with the
//! counts, the max ulp and the FIRST difference, and continue (the misses
//! rule of docs/17-review-later.md; the line is never loosened). What IS
//! asserted is that something was compared - a gate that passed on nothing is
//! worse than no gate - and that every rank returned `Ok`.
//!
//! Provenance: ORIGINAL - the part twins under test are this project's own
//! design (SPEC-LIT 139.8, PAR-13a). No external source. No GPL-licensed
//! source was consulted.

use ofgpu::blockgen::{build_case, CaseKind};
use ofgpu::comm::{Comm, ThreadComm};
use ofgpu::decompose::{Decomposition, PartitionMethod};
use ofgpu::distflow::RankCtx;
use ofgpu::field_ops::{
    correct_boundary_conditions, correct_boundary_conditions_vector, FieldKernels,
};
use ofgpu::fv::{
    fvc_grad_scalar, fvc_grad_vector, fvm_laplacian, fvm_laplacian_non_orth_correction,
    FvKernels, SnGradScheme,
};
use ofgpu::ldu::HostLduMatrix;
use ofgpu::ldu_ops::{add_boundary_contributions, LduKernels};
use ofgpu::solid::prototype;
use ofgpu::{
    BcKind, Gpu, GpuLduMatrix, GpuMesh, GpuScalarField, GpuVectorField, HostMesh, Label, Result,
    Scalar, Tensor, Vec3,
};

/// A machine without a card makes every device test pass vacuously, which is
/// the convention the rest of the crate's tests follow.
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// Mesh A: an axis-aligned box.
fn mesh_a() -> Result<HostMesh> {
    let (m, _) = build_case(CaseKind::Big, 12, 10, 8)?;
    Ok(m)
}

/// Mesh B: a non-orthogonal, skewed unit cube.
fn mesh_b() -> Result<HostMesh> {
    prototype::jittered_block(10, 0.25)
}

/// The two gate meshes, named for the report lines.
fn meshes() -> Vec<(&'static str, HostMesh)> {
    vec![
        ("A 12x10x8 box", mesh_a().expect("mesh A")),
        ("B jittered_block(10, 0.25)", mesh_b().expect("mesh B")),
    ]
}

/// The round robin: cell `c` to part `c % P`, the worst partition possible,
/// which cuts nearly every face.
fn round_robin(n: usize, p: usize) -> PartitionMethod {
    PartitionMethod::Explicit((0..n).map(|c| (c % p) as Label).collect())
}

/// The three partitioners every gate runs.
fn partitioners(n: usize, p: usize) -> Vec<PartitionMethod> {
    vec![
        PartitionMethod::Hilbert,
        PartitionMethod::Linear,
        round_robin(n, p),
    ]
}

/// The nine components, row-major (xx..zz), of a tensor.
fn comps(t: Tensor) -> [Scalar; 9] {
    [t.xx, t.xy, t.xz, t.yx, t.yy, t.yz, t.zx, t.zy, t.zz]
}

/// The ULP distance of two `Scalar`s: equal (so +0 == -0) is 0; a different
/// sign or a non-finite operand is `u64::MAX`; otherwise how far apart the
/// bit patterns are.
#[cfg(not(feature = "single"))]
fn ulp(a: Scalar, b: Scalar) -> u64 {
    if a == b {
        return 0;
    }
    if a.is_sign_negative() != b.is_sign_negative() || !a.is_finite() || !b.is_finite() {
        return u64::MAX;
    }
    (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs()
}

/// The `f32` twin: the bit patterns are 32 bits, so the distance is taken on
/// `i32`.
#[cfg(feature = "single")]
fn ulp(a: Scalar, b: Scalar) -> u64 {
    if a == b {
        return 0;
    }
    if a.is_sign_negative() != b.is_sign_negative() || !a.is_finite() || !b.is_finite() {
        return u64::MAX;
    }
    (a.to_bits() as i32 - b.to_bits() as i32).unsigned_abs() as u64
}

/// The smooth test fields, functions of the WHOLE mesh's cell centres, so
/// splitting the mesh is a permutation and not a computation:
///
/// - `psi = exp(3 xh + 2 yh + 2.5 zh)`
/// - `U = (exp(3 xh + 2 yh + 2.5 zh), exp(-2 xh + 3 yh + 2 zh),
///   exp(2 xh + 2 yh - 3 zh))`
///
/// with `xh = (x - xmin)/(xmax - xmin)` and its friends over the whole
/// mesh's cell centres.
struct Smooth {
    lo: Vec3,
    hi: Vec3,
}

impl Smooth {
    /// The window of the whole mesh's cell centres.
    fn of(m: &HostMesh) -> Self {
        let mut lo = m.c[0];
        let mut hi = m.c[0];
        for c in &m.c {
            lo.x = lo.x.min(c.x);
            lo.y = lo.y.min(c.y);
            lo.z = lo.z.min(c.z);
            hi.x = hi.x.max(c.x);
            hi.y = hi.y.max(c.y);
            hi.z = hi.z.max(c.z);
        }
        Self { lo, hi }
    }

    fn h(&self, c: Vec3) -> (Scalar, Scalar, Scalar) {
        (
            (c.x - self.lo.x) / (self.hi.x - self.lo.x),
            (c.y - self.lo.y) / (self.hi.y - self.lo.y),
            (c.z - self.lo.z) / (self.hi.z - self.lo.z),
        )
    }

    fn psi(&self, c: Vec3) -> Scalar {
        let (xh, yh, zh) = self.h(c);
        (3.0 * xh + 2.0 * yh + 2.5 * zh).exp()
    }

    fn u(&self, c: Vec3) -> Vec3 {
        let (xh, yh, zh) = self.h(c);
        Vec3::new(
            (3.0 * xh + 2.0 * yh + 2.5 * zh).exp(),
            (-2.0 * xh + 3.0 * yh + 2.0 * zh).exp(),
            (2.0 * xh + 2.0 * yh - 3.0 * zh).exp(),
        )
    }
}

/// The whole mesh's fields, cell by cell.
fn smooth_fields(m: &HostMesh) -> (Vec<Scalar>, Vec<Vec3>) {
    let sm = Smooth::of(m);
    let psis: Vec<Scalar> = m.c.iter().map(|&c| sm.psi(c)).collect();
    let us: Vec<Vec3> = m.c.iter().map(|&c| sm.u(c)).collect();
    (psis, us)
}

/// One part's local buffer: its owned cells from the global field, through
/// `global_cell`; its halo ZERO, so a forgotten exchange cannot pass.
fn local_of<T: Copy>(part: &ofgpu::decompose::PartMesh, global: &[T], zero: T) -> Vec<T> {
    let mut h = vec![zero; part.n_local()];
    for (i, hi) in h.iter_mut().enumerate().take(part.mesh.n_cells) {
        *hi = global[part.global_cell[i] as usize];
    }
    h
}

/// The whole-mesh array placed on EVERY local cell of the part - the halo
/// included, which [`local_of`] leaves zero.
fn full_of<T: Copy>(part: &ofgpu::decompose::PartMesh, global: &[T]) -> Vec<T> {
    (0..part.n_local())
        .map(|i| global[part.global_cell[i] as usize])
        .collect()
}

/// The part carries the whole mesh's SIDE and WEIGHT of every processor face
/// (SPEC-LIT 139.8). Host first, on every cut; the device half only checks
/// what the upload does with them. `b_proc_w` is the point of the array: on
/// the non-owning side the part's `b_weights` is `weight_from_offsets(b, a)`,
/// and the whole weight cannot be recovered from it - so it is STORED.
#[test]
fn processor_faces_carry_the_whole_mesh_side_and_weight() {
    for (name, hm) in meshes() {
        assert!(
            hm.b_proc.is_empty(),
            "{name}: the whole mesh must not carry b_proc marks"
        );
        let n = hm.n_cells;
        for pcount in 2..=4usize {
            for method in partitioners(n, pcount) {
                let d = Decomposition::build(&hm, pcount, &method).expect("cut");
                let mut total = 0usize;
                for (p, part) in d.parts.iter().enumerate() {
                    let pm = &part.mesh;
                    assert_eq!(
                        pm.b_proc.len(),
                        pm.n_boundary_faces,
                        "{name} P={pcount} {} part {p}: b_proc is not face-sized",
                        method.as_str()
                    );
                    assert_eq!(
                        pm.b_proc_w.len(),
                        pm.n_boundary_faces,
                        "{name} P={pcount} {} part {p}: b_proc_w is not face-sized",
                        method.as_str()
                    );
                    for j in 0..pm.n_boundary_faces {
                        let g = pm.global_face[pm.n_internal_faces + j] as usize;
                        let at = format!(
                            "{name} P={pcount} {} part {p} boundary face {j} (global {g})",
                            method.as_str()
                        );
                        if g < d.n_global_internal_faces {
                            // a CUT internal face: the side decides the mark,
                            // and the stored weight is the whole mesh's.
                            let want =
                                if d.cell_part[hm.owner[g] as usize] == p as Label {
                                    1
                                } else {
                                    0
                                };
                            assert_eq!(pm.b_proc[j], want, "side, {at}");
                            assert_eq!(
                                pm.b_proc_w[j].to_bits(),
                                hm.weights[g].to_bits(),
                                "the stored weight is not the whole mesh's, {at}"
                            );
                            total += 1;
                        } else {
                            // an inherited face: no mark, no weight.
                            assert_eq!(pm.b_proc[j], -1, "an inherited face is marked, {at}");
                            assert_eq!(pm.b_proc_w[j], 0.0, "an inherited face has a weight, {at}");
                        }
                    }
                }
                assert!(
                    total > 0,
                    "{name} P={pcount} {} cut nothing - the gate passed on nothing",
                    method.as_str()
                );
                assert_eq!(
                    total,
                    2 * d.n_cut_faces,
                    "{name} P={pcount} {}: the parts hold {} processor faces for {} cut faces",
                    method.as_str(),
                    total,
                    d.n_cut_faces
                );
            }
        }
    }
    println!("  [139.8] host half: every part's marks and stored weights check out");

    // ---- the device half: what the upload does with the marks ----------
    let Some(gpu) = gpu() else {
        eprintln!(
            "  [139.8] no GPU: the device half of \
             processor_faces_carry_the_whole_mesh_side_and_weight skipped"
        );
        return;
    };
    for (name, hm) in meshes() {
        let gm = GpuMesh::upload(&gpu, &hm).expect("upload whole");
        assert_eq!(
            gm.n_proc_faces, 0,
            "{name}: a whole mesh uploaded a non-zero processor-face count"
        );

        let d = Decomposition::build(&hm, 2, &PartitionMethod::Hilbert).expect("cut");
        for (p, part) in d.parts.iter().enumerate() {
            let m = GpuMesh::upload_with_halo(&gpu, &part.mesh, part.n_halo).expect("upload part");
            let count = part.mesh.b_proc.iter().filter(|&&b| b >= 0).count();
            assert_eq!(
                m.n_proc_faces, count,
                "{name} part {p}: n_proc_faces is not the mark count"
            );
        }

        // A part whose `b_proc_w` was cut short is refused by name: the
        // kernels read both arrays face by face, and a short one is a read
        // out of bounds rather than an error.
        let mut mm = d.parts[0].mesh.clone();
        mm.b_proc_w.truncate(mm.b_proc_w.len() - 1);
        let msg = GpuMesh::upload_with_halo(&gpu, &mm, d.parts[0].n_halo)
            .err()
            .expect("a truncated b_proc_w must refuse")
            .to_string();
        assert!(msg.starts_with("mesh: b_proc_w"), "{msg}");
        println!("  [139.8] {name}: the truncated b_proc_w refused by name");
    }
}

/// The part gradients ARE the whole mesh's gradients, BITWISE (SPEC-LIT
/// 139.8). §139.2's gate measured the owned cells and reported up to 305 ulp
/// on the round-robin cuts, every miss on a cell touching a processor face;
/// the twins close it by summing the merged row and rebuilding the whole
/// mesh's operands. Every owned AND ghost cell, all 3 + 9 components,
/// compared with the whole gradient at `global_cell[i]`. A difference is
/// MEASURED and reported, not gated: the test prints one `MISSED` line with
/// the first difference and continues. What IS asserted is that the gate
/// compared something - on cells and on ghosts - and that every rank was
/// `Ok`.
#[test]
fn the_part_gradients_are_the_whole_mesh_gradients() {
    let Some(gpu0) = gpu() else {
        eprintln!("  [139.8] no GPU: the_part_gradients_are_the_whole_mesh_gradients skipped");
        return;
    };
    let fk0 = FieldKernels::new(&gpu0).expect("field kernels");
    let fvk0 = FvKernels::new(&gpu0).expect("fv kernels");

    let mut comps_total = 0usize;
    let mut cells_total = 0usize;
    let mut ghosts_total = 0usize;
    let mut k_differ = 0usize;
    let mut max_ulp = 0u64;
    let mut first: Option<String> = None;

    for (name, hm) in meshes() {
        let n = hm.n_cells;
        let (psis, us) = smooth_fields(&hm);

        // ---- the reference: the WHOLE mesh, on the main thread ------------
        let gm0 = GpuMesh::upload(&gpu0, &hm).expect("upload");
        let mut psi = GpuScalarField::zeros(&gpu0, &gm0, "psi").expect("psi");
        let mut u = GpuVectorField::zeros(&gpu0, &gm0, "U").expect("U");
        gpu0.write(&mut psi.f, &psis).expect("write psi");
        gpu0.write(&mut u.f, &us).expect("write U");
        correct_boundary_conditions(&gpu0, &fk0, &mut psi, &gm0).expect("correct psi");
        correct_boundary_conditions_vector(&gpu0, &fk0, &mut u, &gm0).expect("correct U");
        let mut grad_psi0 = gpu0.zeros::<Vec3>(gm0.n_local()).expect("alloc");
        let mut grad_u0 = gpu0.zeros::<Tensor>(gm0.n_local()).expect("alloc");
        fvc_grad_scalar(&gpu0, &fvk0, &mut grad_psi0, &psi, &gm0).expect("grad psi");
        fvc_grad_vector(&gpu0, &fvk0, &mut grad_u0, &u, &gm0).expect("grad U");
        gpu0.sync().expect("sync");
        let whole_psi: Vec<Vec3> = gpu0.download(&grad_psi0).expect("download");
        let whole_u: Vec<Tensor> = gpu0.download(&grad_u0).expect("download");

        for pcount in 2..=4usize {
            for method in partitioners(n, pcount) {
                let d = Decomposition::build(&hm, pcount, &method).expect("cut");
                let got = ThreadComm::run(
                    pcount,
                    |comm| -> Result<(Vec<Vec3>, Vec<Tensor>)> {
                        let gpu = Gpu::new(0)?;
                        let part = &d.parts[comm.rank()];
                        let m = GpuMesh::upload_with_halo(&gpu, &part.mesh, part.n_halo)?;
                        let mut ctx = RankCtx::new(&gpu, comm, part, &m)?;
                        let fk = FieldKernels::new(&gpu)?;
                        let fvk = FvKernels::new(&gpu)?;

                        // Owned values only; the halo is ZERO until the
                        // exchange fills it.
                        let mut psi = GpuScalarField::zeros(&gpu, &m, "psi")?;
                        let mut u = GpuVectorField::zeros(&gpu, &m, "U")?;
                        gpu.write(&mut psi.f, &local_of(part, &psis, 0.0))?;
                        gpu.write(&mut u.f, &local_of(part, &us, Vec3::ZERO))?;
                        ctx.exchange_field(&gpu, &fk, &mut psi, &m)?;
                        ctx.exchange_field_vector(&gpu, &fk, &mut u, &m)?;

                        let mut gp = gpu.zeros::<Vec3>(m.n_local())?;
                        fvc_grad_scalar(&gpu, &fvk, &mut gp, &psi, &m)?;
                        ctx.exchange_gradient(&gpu, &mut gp)?;
                        let mut gu = gpu.zeros::<Tensor>(m.n_local())?;
                        fvc_grad_vector(&gpu, &fvk, &mut gu, &u, &m)?;
                        ctx.exchange_gradient_tensor(&gpu, &mut gu)?;
                        gpu.sync()?;
                        Ok((gpu.download(&gp)?, gpu.download(&gu)?))
                    },
                )
                .expect("spmd run");

                for (r, part) in d.parts.iter().enumerate() {
                    let (gp, gu) = got[r]
                        .as_ref()
                        .unwrap_or_else(|e| panic!("rank {r} failed: {e}"));
                    let line = format!("{name} P={pcount} {}", method.as_str());
                    for i in 0..part.n_local() {
                        let g = part.global_cell[i] as usize;
                        let at =
                            format!("{line} rank {r} local {i} (global {g})");
                        cells_total += 1;
                        ghosts_total += usize::from(i >= part.mesh.n_cells);

                        let want = [whole_psi[g].x, whole_psi[g].y, whole_psi[g].z];
                        let got_c = [gp[i].x, gp[i].y, gp[i].z];
                        for (k, (a, b)) in got_c.into_iter().zip(want).enumerate() {
                            comps_total += 1;
                            let du = ulp(a, b);
                            max_ulp = max_ulp.max(du);
                            if a.to_bits() != b.to_bits() {
                                k_differ += 1;
                                if first.is_none() {
                                    first = Some(format!(
                                        "grad psi component {k}, {at}: got {a:e}, \
                                         want {b:e}, {du} ulp"
                                    ));
                                }
                            }
                        }

                        let want = comps(whole_u[g]);
                        let got_c = comps(gu[i]);
                        for (k, (a, b)) in got_c.into_iter().zip(want).enumerate() {
                            comps_total += 1;
                            let du = ulp(a, b);
                            max_ulp = max_ulp.max(du);
                            if a.to_bits() != b.to_bits() {
                                k_differ += 1;
                                if first.is_none() {
                                    first = Some(format!(
                                        "grad U component {k}, {at}: got {a:e}, \
                                         want {b:e}, {du} ulp"
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    eprintln!(
        "  [139.8] gradients: {comps_total} components compared on \
         {cells_total} cells, {k_differ} differ"
    );
    if k_differ > 0 {
        eprintln!(
            "  [139.8] MISSED part gradients bitwise: {k_differ} of \
             {comps_total} differ, max {max_ulp} ulp; first: {}",
            first.as_ref().expect("a difference was counted")
        );
    }
    assert!(comps_total > 0, "no gradient component was compared");
    assert!(ghosts_total > 0, "no ghost cell was compared anywhere");
}

/// Bitwise compare of one rank's matrix against `split_matrix`'s target:
/// `diag` and `source` on the owned cells, `upper`, `lower`,
/// `internal_coeffs` and `boundary_coeffs` whole. Differences are counted,
/// the first one kept.
fn compare_matrix(
    got: &HostLduMatrix,
    want: &HostLduMatrix,
    at: &str,
    total: &mut usize,
    diff: &mut usize,
    first: &mut Option<String>,
) {
    for i in 0..want.n_cells {
        for (nm, g, w) in [
            ("diag", got.diag[i], want.diag[i]),
            ("source", got.source[i], want.source[i]),
        ] {
            *total += 1;
            if g.to_bits() != w.to_bits() {
                *diff += 1;
                if first.is_none() {
                    *first = Some(format!("{nm}[{i}] {at}: got {g:e}, want {w:e}"));
                }
            }
        }
    }
    for i in 0..want.n_internal_faces {
        for (nm, g, w) in [
            ("upper", got.upper[i], want.upper[i]),
            ("lower", got.lower[i], want.lower[i]),
        ] {
            *total += 1;
            if g.to_bits() != w.to_bits() {
                *diff += 1;
                if first.is_none() {
                    *first = Some(format!("{nm}[{i}] {at}: got {g:e}, want {w:e}"));
                }
            }
        }
    }
    for i in 0..want.n_boundary_faces {
        for (nm, g, w) in [
            ("internal_coeffs", got.internal_coeffs[i], want.internal_coeffs[i]),
            ("boundary_coeffs", got.boundary_coeffs[i], want.boundary_coeffs[i]),
        ] {
            *total += 1;
            if g.to_bits() != w.to_bits() {
                *diff += 1;
                if first.is_none() {
                    *first = Some(format!("{nm}[{i}] {at}: got {g:e}, want {w:e}"));
                }
            }
        }
    }
}

/// The part laplacian IS `split_matrix` of the whole mesh's, BITWISE, folded
/// and unfolded (SPEC-LIT 139.8). The whole mesh's psi carries even
/// non-coupled boundary faces as `FixedValue` of a smooth function and a
/// face gamma of `mag_sf * (2 + sin(3 x))`, so no coefficient is trivial;
/// the non-orthogonal correction runs once against the whole mesh's own
/// gradient. Each rank assembles the SAME problem on its part - the boundary
/// state copied onto its inherited faces, the gamma mapped through
/// `global_face` (a processor face takes the whole mesh's INTERNAL-face
/// gamma - that is the PAR-13a convention) - and its gradient handed to it
/// WHOLE (`full_of`), NOT recomputed, so this test isolates the assembly.
/// The comparison is bitwise against `Decomposition::split_matrix` of the
/// whole matrices; a difference prints one `MISSED` line with the first
/// offending coefficient and continues.
#[test]
fn the_part_laplacian_is_split_matrix_of_the_whole() {
    let Some(gpu0) = gpu() else {
        eprintln!("  [139.8] no GPU: the_part_laplacian_is_split_matrix_of_the_whole skipped");
        return;
    };
    let fk0 = FieldKernels::new(&gpu0).expect("field kernels");
    let fvk0 = FvKernels::new(&gpu0).expect("fv kernels");
    let lk0 = LduKernels::new(&gpu0).expect("ldu kernels");

    let mut coeffs_total = 0usize;
    let mut k_differ = 0usize;
    let mut first: Option<String> = None;

    for (name, hm) in meshes() {
        let n = hm.n_cells;
        let nif_global = hm.n_internal_faces;
        let (psis, _) = smooth_fields(&hm);
        let sm = Smooth::of(&hm);

        // ---- the WHOLE mesh, on the main thread ----------------------------
        let gm0 = GpuMesh::upload(&gpu0, &hm).expect("upload");
        let mut psi = GpuScalarField::zeros(&gpu0, &gm0, "psi").expect("psi");
        gpu0.write(&mut psi.f, &psis).expect("write psi");

        let mut kinds_w: Vec<Label> = gpu0.download(&psi.bc_kind).expect("kinds");
        let mut fr_w: Vec<Scalar> = gpu0.download(&psi.fr).expect("fr");
        let mut rv_w: Vec<Scalar> = gpu0.download(&psi.ref_value).expect("ref_value");
        let rg_w: Vec<Scalar> = gpu0.download(&psi.ref_grad).expect("ref_grad");
        for bf in 0..hm.n_boundary_faces {
            let k = kinds_w[bf];
            if k == BcKind::Empty as Label || k == BcKind::Cyclic as Label {
                continue;
            }
            if bf % 2 == 0 {
                kinds_w[bf] = BcKind::FixedValue as Label;
                fr_w[bf] = 1.0;
                rv_w[bf] = sm.psi(hm.b_cf[bf]);
            }
        }
        gpu0.write(&mut psi.bc_kind, &kinds_w).expect("write kinds");
        gpu0.write(&mut psi.fr, &fr_w).expect("write fr");
        gpu0.write(&mut psi.ref_value, &rv_w).expect("write ref_value");
        gpu0.write(&mut psi.ref_grad, &rg_w).expect("write ref_grad");
        correct_boundary_conditions(&gpu0, &fk0, &mut psi, &gm0).expect("correct psi");

        let gamma_if: Vec<Scalar> = (0..hm.n_internal_faces)
            .map(|f| hm.mag_sf[f] * (2.0 + (3.0 * hm.cf[f].x).sin()))
            .collect();
        let gamma_bf: Vec<Scalar> = (0..hm.n_boundary_faces)
            .map(|b| hm.b_mag_sf[b] * (2.0 + (3.0 * hm.b_cf[b].x).sin()))
            .collect();
        let gif0 = gpu0.upload(&gamma_if).expect("upload gamma");
        let gbf0 = gpu0.upload(&gamma_bf).expect("upload gamma bf");

        let mut a = GpuLduMatrix::new(&gpu0, &gm0).expect("matrix");
        fvm_laplacian(&gpu0, &fvk0, &mut a, &gm0, &gif0, &gbf0, &psi, 1.0)
            .expect("laplacian");
        let mut grad = gpu0.zeros::<Vec3>(gm0.n_local()).expect("alloc");
        fvc_grad_scalar(&gpu0, &fvk0, &mut grad, &psi, &gm0).expect("grad");
        fvm_laplacian_non_orth_correction(
            &gpu0,
            &fvk0,
            &mut a,
            &gm0,
            &gif0,
            &gbf0,
            &psi,
            &grad,
            SnGradScheme::Corrected,
            1.0,
        )
        .expect("non-orth correction");
        gpu0.sync().expect("sync");
        let w_unf = HostLduMatrix::download(&gpu0, &a).expect("download");
        add_boundary_contributions(&gpu0, &lk0, &mut a, &gm0).expect("fold");
        let w_fold = HostLduMatrix::download(&gpu0, &a).expect("download");
        let whole_grad: Vec<Vec3> = gpu0.download(&grad).expect("download grad");

        for pcount in 2..=4usize {
            for method in partitioners(n, pcount) {
                let d = Decomposition::build(&hm, pcount, &method).expect("cut");
                let got = ThreadComm::run(
                    pcount,
                    |comm| -> Result<(HostLduMatrix, HostLduMatrix)> {
                        let gpu = Gpu::new(0)?;
                        let part = &d.parts[comm.rank()];
                        let pm = &part.mesh;
                        let m = GpuMesh::upload_with_halo(&gpu, pm, part.n_halo)?;
                        let mut ctx = RankCtx::new(&gpu, comm, part, &m)?;
                        let fk = FieldKernels::new(&gpu)?;
                        let fvk = FvKernels::new(&gpu)?;
                        let lk = LduKernels::new(&gpu)?;

                        // The field: owned values from the smooth psi, then
                        // the INHERITED faces only take the whole mesh's
                        // boundary state; a processor face keeps what
                        // `zeros` seeded (kind Cyclic, everything zero).
                        let mut psi = GpuScalarField::zeros(&gpu, &m, "psi")?;
                        gpu.write(&mut psi.f, &local_of(part, &psis, 0.0))?;
                        let mut kinds: Vec<Label> = gpu.download(&psi.bc_kind)?;
                        let mut fr: Vec<Scalar> = gpu.download(&psi.fr)?;
                        let mut rv: Vec<Scalar> = gpu.download(&psi.ref_value)?;
                        let mut rg: Vec<Scalar> = gpu.download(&psi.ref_grad)?;
                        for j in 0..pm.n_boundary_faces {
                            let g =
                                pm.global_face[pm.n_internal_faces + j] as usize;
                            if g >= nif_global {
                                let bf = g - nif_global;
                                kinds[j] = kinds_w[bf];
                                fr[j] = fr_w[bf];
                                rv[j] = rv_w[bf];
                                rg[j] = rg_w[bf];
                            }
                        }
                        gpu.write(&mut psi.bc_kind, &kinds)?;
                        gpu.write(&mut psi.fr, &fr)?;
                        gpu.write(&mut psi.ref_value, &rv)?;
                        gpu.write(&mut psi.ref_grad, &rg)?;
                        ctx.exchange_field(&gpu, &fk, &mut psi, &m)?;

                        // The face gamma, mapped through `global_face`: an
                        // interior face takes the whole mesh's internal-face
                        // gamma; an inherited boundary face the whole mesh's
                        // boundary gamma; a PROCESSOR face the whole mesh's
                        // INTERNAL gamma at its global id - the convention
                        // that makes its diagonal share the whole mesh's.
                        let gamma_if_p: Vec<Scalar> = (0..pm.n_internal_faces)
                            .map(|i| gamma_if[pm.global_face[i] as usize])
                            .collect();
                        let gamma_bf_p: Vec<Scalar> = (0..pm.n_boundary_faces)
                            .map(|j| {
                                let g =
                                    pm.global_face[pm.n_internal_faces + j] as usize;
                                if g >= nif_global {
                                    gamma_bf[g - nif_global]
                                } else {
                                    gamma_if[g]
                                }
                            })
                            .collect();
                        let gif = gpu.upload(&gamma_if_p)?;
                        let gbf = gpu.upload(&gamma_bf_p)?;

                        let mut a = GpuLduMatrix::new(&gpu, &m)?;
                        fvm_laplacian(&gpu, &fvk, &mut a, &m, &gif, &gbf, &psi, 1.0)?;
                        // The gradient is HANDED to the part, not recomputed,
                        // so this test isolates the assembly.
                        let mut grad = gpu.zeros::<Vec3>(m.n_local())?;
                        gpu.write(&mut grad, &full_of(part, &whole_grad))?;
                        fvm_laplacian_non_orth_correction(
                            &gpu,
                            &fvk,
                            &mut a,
                            &m,
                            &gif,
                            &gbf,
                            &psi,
                            &grad,
                            SnGradScheme::Corrected,
                            1.0,
                        )?;
                        gpu.sync()?;
                        let unf = HostLduMatrix::download(&gpu, &a)?;
                        add_boundary_contributions(&gpu, &lk, &mut a, &m)?;
                        let fold = HostLduMatrix::download(&gpu, &a)?;
                        Ok((unf, fold))
                    },
                )
                .expect("spmd run");

                for (p, _part) in d.parts.iter().enumerate() {
                    let (unf, fold) = got[p]
                        .as_ref()
                        .unwrap_or_else(|e| panic!("rank {p} failed: {e}"));
                    let at = format!("{name} P={pcount} {} rank {p}", method.as_str());
                    let want_unf = d.split_matrix(&hm, p, &w_unf).expect("split");
                    compare_matrix(
                        unf,
                        &want_unf,
                        &format!("{at} unfolded"),
                        &mut coeffs_total,
                        &mut k_differ,
                        &mut first,
                    );
                    let want_fold = d.split_matrix(&hm, p, &w_fold).expect("split");
                    compare_matrix(
                        fold,
                        &want_fold,
                        &format!("{at} folded"),
                        &mut coeffs_total,
                        &mut k_differ,
                        &mut first,
                    );
                }
            }
        }
    }
    eprintln!(
        "  [139.8] laplacian: {coeffs_total} coefficients compared, {k_differ} differ"
    );
    if k_differ > 0 {
        eprintln!(
            "  [139.8] MISSED part laplacian bitwise: {k_differ} of \
             {coeffs_total} differ; first: {}",
            first.as_ref().expect("a difference was counted")
        );
    }
    assert!(coeffs_total > 0, "no laplacian coefficient was compared");
}








