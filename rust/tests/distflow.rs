// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 139.2 (PAR-10a)'s gate - the exchange points of a decomposed
//! field, in BOTH builds.
//!
//! An integration test because the library's own test target does not compile
//! under `--features single`; an integration test links the normal (non-test)
//! library, which does. Every rank is a [`ThreadComm`] thread with its own
//! [`Gpu`] on device 0, and the reference is the WHOLE mesh on the main
//! thread's card: a program run over `P` ranks must put the neighbour rank's
//! value in every ghost cell, re-evaluate a processor face from it, and so
//! make a part's field and its gradient the whole mesh's. On the gradient
//! the gate's 4-ulp line is MEASURED and reported, not asserted: where it
//! does not hold the test prints one `MISSED` line with the maxima, the
//! counts and the cause, and the miss is listed for review (the misses
//! rule of docs/17-review-later.md; the line is never loosened). What IS
//! asserted is the cause check - the conditioning bound
//! `|a - b| <= 4 eps S`, `S` the magnitude of that component's Green-Gauss
//! sum on the whole mesh - which shows a miss is the order of rounding and
//! not a wrong value. Bitwise is PAR-13a's job: the processor face's
//! contribution joins the boundary loop, not the internal loop, and the
//! non-owning side's weight is `weight_from_offsets`, not `1 - w`.
//!
//! At the ghost cells the gate is class D of docs/17 §5.1: every ghost is
//! compared BIT FOR BIT with what its owning rank computed, per component,
//! and every field's halo is ZERO before its exchange while no owned value
//! is zero, so a forgotten or misdirected exchange cannot pass.
//!
//! Provenance: ORIGINAL - the exchange points under test are this project's
//! own design (SPEC-LIT 139.2, PAR-10a). No external source. No GPL-licensed
//! source was consulted.

use ofgpu::blockgen::{build_case, CaseKind};
use ofgpu::comm::{Comm, ThreadComm};
use ofgpu::decompose::{Decomposition, PartitionMethod};
use ofgpu::distflow::RankCtx;
use ofgpu::field_ops::{
    correct_boundary_conditions, correct_boundary_conditions_vector, FieldKernels,
};
use ofgpu::fv::{fvc_grad_scalar, fvc_grad_vector, FvKernels};
use ofgpu::solid::prototype;
use ofgpu::{
    BcKind, Error, Gpu, GpuMesh, GpuScalarField, GpuVectorField, HostMesh, Label, Result, Scalar,
    Tensor, Vec3,
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

/// The normalised error of one component: `|a - b| / (EPSILON * S)`, how
/// many epsilons of the whole mesh's Green-Gauss sum magnitude the
/// difference spends. The asserted cause check is that this stays <= 4,
/// which is `|a - b| <= 4 eps S` divided through.
fn norm_err(a: Scalar, b: Scalar, s: f64) -> f64 {
    let d = (a as f64 - b as f64).abs();
    if d == 0.0 {
        return 0.0;
    }
    let den = Scalar::EPSILON as f64 * s;
    if den > 0.0 {
        d / den
    } else {
        f64::INFINITY
    }
}

/// The magnitude `S` of each component's Green-Gauss sum on the WHOLE mesh,
/// host side, in f64: for grad psi component k of cell c,
/// `(1/V_c) * sum over the faces f of c of |Sf_k| * |psi_f|`; for grad U
/// component (i, j), `(1/V_c) * sum over f of |Sf_i| * |U_f,j|`. An internal
/// face counts for both its cells at `max(|psi_owner|, |psi_neighbour|)`; a
/// boundary face counts for its one cell at the field's downloaded `bf`,
/// and an `Empty` face counts nothing. Returns `(3 n, 9 n)` flat arrays.
fn conditioning(
    hm: &HostMesh,
    psis: &[Scalar],
    us: &[Vec3],
    bf_psi: &[Scalar],
    bf_u: &[Vec3],
    kinds_psi: &[Label],
    kinds_u: &[Label],
) -> (Vec<f64>, Vec<f64>) {
    let n = hm.n_cells;
    let mut s_psi = vec![0.0f64; 3 * n];
    let mut s_u = vec![0.0f64; 9 * n];
    let sf64 = |v: Vec3| [v.x as f64, v.y as f64, v.z as f64];
    for f in 0..hm.n_internal_faces {
        let o = hm.owner[f] as usize;
        let nbr = hm.neighbour[f] as usize;
        let sf = sf64(hm.sf[f]);
        let pf = psis[o].abs().max(psis[nbr].abs()) as f64;
        let uf = [
            us[o].x.abs().max(us[nbr].x.abs()) as f64,
            us[o].y.abs().max(us[nbr].y.abs()) as f64,
            us[o].z.abs().max(us[nbr].z.abs()) as f64,
        ];
        for (k, &s) in sf.iter().enumerate() {
            let t = s.abs() * pf;
            s_psi[3 * o + k] += t;
            s_psi[3 * nbr + k] += t;
        }
        for i in 0..3 {
            for j in 0..3 {
                let t = sf[i].abs() * uf[j];
                s_u[9 * o + 3 * i + j] += t;
                s_u[9 * nbr + 3 * i + j] += t;
            }
        }
    }
    for b in 0..hm.n_boundary_faces {
        if kinds_psi[b] == BcKind::Empty as Label || kinds_u[b] == BcKind::Empty as Label {
            continue;
        }
        let c = hm.b_face_cells[b] as usize;
        let sf = sf64(hm.b_sf[b]);
        let pf = bf_psi[b].abs() as f64;
        let uf = [bf_u[b].x.abs() as f64, bf_u[b].y.abs() as f64, bf_u[b].z.abs() as f64];
        for (k, &s) in sf.iter().enumerate() {
            s_psi[3 * c + k] += s.abs() * pf;
        }
        for i in 0..3 {
            for j in 0..3 {
                s_u[9 * c + 3 * i + j] += sf[i].abs() * uf[j];
            }
        }
    }
    for c in 0..n {
        let v = hm.v[c] as f64;
        for x in &mut s_psi[3 * c..3 * c + 3] {
            *x /= v;
        }
        for x in &mut s_u[9 * c..9 * c + 9] {
            *x /= v;
        }
    }
    (s_psi, s_u)
}

/// **The gate.** The gradient of the smooth psi and U on P = 2..4 parts of
/// two meshes under three partitioners, against the same gradients on the
/// whole mesh. Every GHOST cell is BIT FOR BIT the value its owning rank
/// computed. On every OWNED cell the ulp distance is MEASURED and counted,
/// not gated: if the 4-ulp line does not hold the test ends with one
/// `MISSED` line carrying the maxima, the counts and the cause. The asserted
/// owned line is the cause check - per component, `|a - b| <= 4 eps S` with
/// `S` the magnitude of that component's Green-Gauss sum on the whole mesh,
/// which is what shows a miss is the order of rounding and not a wrong
/// value. A gate that passed because nothing was cut would be worse than no
/// gate, so every cut is asserted to have one.
#[test]
fn a_halo_gradient_equals_the_whole_mesh_gradient() {
    let Some(gpu0) = gpu() else {
        eprintln!("  [139.2] no GPU: a_halo_gradient_equals_the_whole_mesh_gradient skipped");
        return;
    };
    let fk0 = FieldKernels::new(&gpu0).expect("field kernels");
    let fvk0 = FvKernels::new(&gpu0).expect("fv kernels");

    let mut max_psi = 0u64;
    let mut max_u = 0u64;
    let mut ghosts_total = 0usize;
    // Owned components above 4 ulp, and how many were compared; and the
    // maxima of the normalised error |a - b| / (eps S).
    let mut above_psi = 0usize;
    let mut above_u = 0usize;
    let mut compared_psi = 0usize;
    let mut compared_u = 0usize;
    let mut nmax_psi = 0.0f64;
    let mut nmax_u = 0.0f64;

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

        // The cause check's S: the boundary faces' evaluated values and
        // kinds come off the WHOLE mesh's fields, then S is a host sum.
        let whole_bf_psi: Vec<Scalar> = gpu0.download(&psi.bf).expect("download bf");
        let whole_bf_u: Vec<Vec3> = gpu0.download(&u.bf).expect("download bf U");
        let kinds_psi: Vec<Label> = gpu0.download(&psi.bc_kind).expect("download kinds");
        let kinds_u: Vec<Label> = gpu0.download(&u.bc_kind).expect("download kinds U");
        let (s_psi, s_u) =
            conditioning(&hm, &psis, &us, &whole_bf_psi, &whole_bf_u, &kinds_psi, &kinds_u);

        for pcount in 2..=4usize {
            for method in partitioners(n, pcount) {
                let d = Decomposition::build(&hm, pcount, &method).expect("cut");
                assert!(
                    d.n_cut_faces > 0,
                    "{name} P={pcount} {} cut no face",
                    method.as_str()
                );

                // ---- the SPMD run: one thread per rank, each with its own
                // `Gpu` on device 0. A rank's error must fail the test with
                // its text, so the closure returns `Result`. ----------------
                let got = ThreadComm::run(
                    pcount,
                    |comm| -> Result<(Vec<Vec3>, Vec<Tensor>)> {
                        let gpu = Gpu::new(0)?;
                        let part = &d.parts[comm.rank()];
                        let m = GpuMesh::upload_with_halo(&gpu, &part.mesh, part.n_halo)?;
                        let mut ctx = RankCtx::new(&gpu, comm, part, &m)?;
                        let fk = FieldKernels::new(&gpu)?;
                        let fvk = FvKernels::new(&gpu)?;

                        // Owned values only; the halo is ZERO.
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

                let mut line_psi = 0u64;
                let mut line_u = 0u64;
                let mut line_above_psi = 0usize;
                let mut line_above_u = 0usize;
                let mut line_compared_psi = 0usize;
                let mut line_compared_u = 0usize;
                let mut line_npsi = 0.0f64;
                let mut line_nu = 0.0f64;
                let mut owned_bitwise = 0usize;
                let mut owned_total = 0usize;
                let mut ghosts = 0usize;
                for (r, part) in d.parts.iter().enumerate() {
                    let (gp, gu) = got[r]
                        .as_ref()
                        .unwrap_or_else(|e| panic!("rank {r} failed: {e}"));
                    for i in 0..part.n_local() {
                        let g = part.global_cell[i] as usize;
                        let at = format!(
                            "{name} P={pcount} {} rank {r} local {i} (global {g})",
                            method.as_str()
                        );
                        if i < part.mesh.n_cells {
                            // OWNED: the ulp distance is MEASURED and
                            // counted, never gated; the asserted line is the
                            // cause check |a - b| <= 4 eps S per component,
                            // S the whole mesh's Green-Gauss sum magnitude.
                            let want = [
                                whole_psi[g].x,
                                whole_psi[g].y,
                                whole_psi[g].z,
                            ];
                            let got_c = [gp[i].x, gp[i].y, gp[i].z];
                            let mut psi_eq = true;
                            for (k, (a, b)) in got_c.into_iter().zip(want).enumerate() {
                                let du = ulp(a, b);
                                line_psi = line_psi.max(du);
                                line_above_psi += usize::from(du > 4);
                                psi_eq &= du == 0;
                                let s = s_psi[3 * g + k];
                                let ne = norm_err(a, b, s);
                                assert!(
                                    ne <= 4.0,
                                    "grad psi component {} breaks |a - b| \
                                     <= 4 eps S: normalised error {ne}, \
                                     S = {s:e}, {at}",
                                    ['x', 'y', 'z'][k]
                                );
                                line_npsi = line_npsi.max(ne);
                            }
                            line_compared_psi += 3;
                            let want = comps(whole_u[g]);
                            let got_c = comps(gu[i]);
                            let mut u_eq = true;
                            for (k, (a, b)) in got_c.into_iter().zip(want).enumerate() {
                                let du = ulp(a, b);
                                line_u = line_u.max(du);
                                line_above_u += usize::from(du > 4);
                                u_eq &= du == 0;
                                // Tensor.xy is (i = x, j = y): k = 3 i + j.
                                let (ki, kj) = (k / 3, k % 3);
                                let s = s_u[9 * g + 3 * ki + kj];
                                let ne = norm_err(a, b, s);
                                assert!(
                                    ne <= 4.0,
                                    "grad U component {}{} breaks |a - b| \
                                     <= 4 eps S: normalised error {ne}, \
                                     S = {s:e}, {at}",
                                    ['x', 'y', 'z'][ki],
                                    ['x', 'y', 'z'][kj]
                                );
                                line_nu = line_nu.max(ne);
                            }
                            line_compared_u += 9;
                            owned_total += 2;
                            owned_bitwise += usize::from(psi_eq) + usize::from(u_eq);
                        } else {
                            // GHOST: BIT FOR BIT the value the OWNING rank
                            // computed for the same global cell.
                            let owner = d.cell_part[g] as usize;
                            let op = &d.parts[owner];
                            let j = op.global_cell[..op.mesh.n_cells]
                                .iter()
                                .position(|&x| x as usize == g)
                                .unwrap_or_else(|| {
                                    panic!("global cell {g} is owned by no part's owned cells, {at}")
                                });
                            let (ogp, ogu) = got[owner]
                                .as_ref()
                                .unwrap_or_else(|e| panic!("rank {owner} failed: {e}"));
                            let want = [ogp[j].x, ogp[j].y, ogp[j].z];
                            let got_c = [gp[i].x, gp[i].y, gp[i].z];
                            for (k, (a, b)) in got_c.into_iter().zip(want).enumerate() {
                                assert_eq!(
                                    a.to_bits(),
                                    b.to_bits(),
                                    "grad psi ghost component {k} is not the owner's bits, {at}"
                                );
                            }
                            let want = comps(ogu[j]);
                            let got_c = comps(gu[i]);
                            for (k, (a, b)) in got_c.into_iter().zip(want).enumerate() {
                                assert_eq!(
                                    a.to_bits(),
                                    b.to_bits(),
                                    "grad U ghost component {k} is not the owner's bits, {at}"
                                );
                            }
                            ghosts += 2;
                        }
                    }
                }
                assert!(
                    ghosts > 0,
                    "{name} P={pcount} {} compared no ghost cell - the gate \
                     passed on nothing",
                    method.as_str()
                );
                eprintln!(
                    "  [139.2] {name} P={pcount} {:>10}: max ulp grad psi \
                     {line_psi}, max ulp grad U {line_u}, normalised error \
                     psi {line_npsi:.3}, U {line_nu:.3}, owned bitwise \
                     {owned_bitwise}/{owned_total}, above 4 ulp psi \
                     {line_above_psi}/{line_compared_psi}, U \
                     {line_above_u}/{line_compared_u}, ghosts compared \
                     {ghosts}",
                    method.as_str()
                );
                max_psi = max_psi.max(line_psi);
                max_u = max_u.max(line_u);
                above_psi += line_above_psi;
                above_u += line_above_u;
                compared_psi += line_compared_psi;
                compared_u += line_compared_u;
                nmax_psi = nmax_psi.max(line_npsi);
                nmax_u = nmax_u.max(line_nu);
                ghosts_total += ghosts;
            }
        }
    }
    assert!(ghosts_total > 0, "no ghost cell was compared anywhere");
    let build = if cfg!(feature = "single") { "single" } else { "f64" };
    let total = compared_psi + compared_u;
    eprintln!(
        "  [139.2] global maxima: grad psi {max_psi} ulp, grad U {max_u} ulp, \
         normalised error psi {nmax_psi:.3}, U {nmax_u:.3}, {ghosts_total} \
         ghost cell-payloads compared"
    );
    if max_psi > 4 || max_u > 4 {
        eprintln!(
            "  [139.2] MISSED: Gate 139.2 4 ulp per component: {build} build: \
             grad psi max {max_psi} ulp ({above_psi} of {compared_psi} \
             components above 4), grad U max {max_u} ulp ({above_u} of \
             {compared_u} above 4), {total} owned components compared; \
             cause: the Green-Gauss sum of a processor-face cell is added in \
             another order and its non-owning face weight is \
             weight_from_offsets(b, a), and the sum's cancellation amplifies \
             the rounding (see the conditioning check)"
        );
    } else {
        eprintln!(
            "  [139.2] PASS: Gate 139.2 4 ulp per component: {build} build: \
             grad psi max {max_psi} ulp, grad U max {max_u} ulp, {total} \
             owned components compared, normalised error psi {nmax_psi:.3}, \
             U {nmax_u:.3}"
        );
    }
}

/// What one rank brings home from [`a_processor_patch_reads_the_owner_value`]:
/// the exchanged fields' internal values and boundary values, and the
/// reference evaluation's boundary values.
struct RankState {
    psi: Vec<Scalar>,
    u: Vec<Vec3>,
    bf_psi: Vec<Scalar>,
    bf_u: Vec<Vec3>,
    bf_psi_ref: Vec<Scalar>,
    bf_u_ref: Vec<Vec3>,
}

/// A processor patch reads the owner's value (SPEC-LIT 139.2). After
/// `exchange_field` every ghost holds the whole-mesh value at its global
/// cell bit for bit, and on every face of every processor patch the
/// exchanged field's `bf` is the reference evaluation's - the reference
/// being the SAME part mesh whose whole `f`, ghosts included, was written
/// from the host with the whole-mesh values and evaluated without any
/// exchange.
#[test]
fn a_processor_patch_reads_the_owner_value() {
    if gpu().is_none() {
        eprintln!("  [139.2] no GPU: a_processor_patch_reads_the_owner_value skipped");
        return;
    }
    let mut faces_total = 0usize;
    let mut ghosts_total = 0usize;
    for (name, hm) in meshes() {
        let n = hm.n_cells;
        let (psis, us) = smooth_fields(&hm);
        for pcount in 2..=4usize {
            for method in partitioners(n, pcount) {
                let d = Decomposition::build(&hm, pcount, &method).expect("cut");
                let got = ThreadComm::run(pcount, |comm| -> Result<RankState> {
                    let gpu = Gpu::new(0)?;
                    let part = &d.parts[comm.rank()];
                    let m = GpuMesh::upload_with_halo(&gpu, &part.mesh, part.n_halo)?;
                    let mut ctx = RankCtx::new(&gpu, comm, part, &m)?;
                    let fk = FieldKernels::new(&gpu)?;

                    // The exchanged fields: owned values only, halo ZERO.
                    let mut psi = GpuScalarField::zeros(&gpu, &m, "psi")?;
                    let mut u = GpuVectorField::zeros(&gpu, &m, "U")?;
                    gpu.write(&mut psi.f, &local_of(part, &psis, 0.0))?;
                    gpu.write(&mut u.f, &local_of(part, &us, Vec3::ZERO))?;
                    ctx.exchange_field(&gpu, &fk, &mut psi, &m)?;
                    ctx.exchange_field_vector(&gpu, &fk, &mut u, &m)?;

                    // The reference: whole-mesh values on OWNED AND GHOST
                    // cells of the same part mesh, then the plain boundary
                    // evaluation - no exchange anywhere.
                    let mut hpsi = vec![0.0; part.n_local()];
                    let mut hu = vec![Vec3::ZERO; part.n_local()];
                    for (i, hp) in hpsi.iter_mut().enumerate() {
                        let g = part.global_cell[i] as usize;
                        *hp = psis[g];
                        hu[i] = us[g];
                    }
                    let mut rpsi = GpuScalarField::zeros(&gpu, &m, "psi")?;
                    let mut ru = GpuVectorField::zeros(&gpu, &m, "U")?;
                    gpu.write(&mut rpsi.f, &hpsi)?;
                    gpu.write(&mut ru.f, &hu)?;
                    correct_boundary_conditions(&gpu, &fk, &mut rpsi, &m)?;
                    correct_boundary_conditions_vector(&gpu, &fk, &mut ru, &m)?;
                    gpu.sync()?;
                    Ok(RankState {
                        psi: gpu.download(&psi.f)?,
                        u: gpu.download(&u.f)?,
                        bf_psi: gpu.download(&psi.bf)?,
                        bf_u: gpu.download(&u.bf)?,
                        bf_psi_ref: gpu.download(&rpsi.bf)?,
                        bf_u_ref: gpu.download(&ru.bf)?,
                    })
                })
                .expect("spmd run");

                let mut line_faces = 0usize;
                let mut line_ghosts = 0usize;
                for (r, part) in d.parts.iter().enumerate() {
                    let st = got[r]
                        .as_ref()
                        .unwrap_or_else(|e| panic!("rank {r} failed: {e}"));
                    for i in part.mesh.n_cells..part.n_local() {
                        let g = part.global_cell[i] as usize;
                        let at = format!(
                            "{name} P={pcount} {} rank {r} ghost {i} (global {g})",
                            method.as_str()
                        );
                        assert_eq!(
                            st.psi[i].to_bits(),
                            psis[g].to_bits(),
                            "psi ghost is not the owner's bits, {at}"
                        );
                        let got_v = [st.u[i].x, st.u[i].y, st.u[i].z];
                        let want_v = [us[g].x, us[g].y, us[g].z];
                        for (k, (a, b)) in got_v.into_iter().zip(want_v).enumerate() {
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "U ghost component {k} is not the owner's bits, {at}"
                            );
                        }
                        line_ghosts += 1;
                    }
                    for &(pi, _q) in &part.proc_patches {
                        let patch = &part.mesh.patches[pi];
                        for f in patch.start..patch.start + patch.size {
                            let at = format!(
                                "{name} P={pcount} {} rank {r} face {f} of {}",
                                method.as_str(),
                                patch.name
                            );
                            assert_eq!(
                                st.bf_psi[f].to_bits(),
                                st.bf_psi_ref[f].to_bits(),
                                "psi bf on a processor face is not the reference's, {at}"
                            );
                            let got_v = [st.bf_u[f].x, st.bf_u[f].y, st.bf_u[f].z];
                            let want_v = [
                                st.bf_u_ref[f].x,
                                st.bf_u_ref[f].y,
                                st.bf_u_ref[f].z,
                            ];
                            for (k, (a, b)) in got_v.into_iter().zip(want_v).enumerate() {
                                assert_eq!(
                                    a.to_bits(),
                                    b.to_bits(),
                                    "U bf component {k} on a processor face is not the \
                                     reference's, {at}"
                                );
                            }
                            line_faces += 1;
                        }
                    }
                }
                assert!(
                    line_faces > 0,
                    "{name} P={pcount} {} checked no processor face - the \
                     gate passed on nothing",
                    method.as_str()
                );
                eprintln!(
                    "  [139.2] {name} P={pcount} {:>10}: {line_faces} \
                     processor faces read the owner, {line_ghosts} ghosts \
                     filled from the owners",
                    method.as_str()
                );
                faces_total += line_faces;
                ghosts_total += line_ghosts;
            }
        }
    }
    eprintln!(
        "  [139.2] processor faces checked {faces_total}, ghosts filled \
         {ghosts_total}"
    );
    assert!(faces_total > 0, "no processor face was checked anywhere");
}

/// The refusal table (SPEC-LIT 139.2): `RankCtx::new`'s rows, in order, each
/// `Error::Config` starting `distflow: ` and each happening BEFORE any
/// collective or device allocation, plus `SpmdHalo`'s own refusal of a field
/// without halo room. Rows 1, 2 and 3 run on
/// `ThreadComm::world(2)` handles on THIS thread - no collective is reached,
/// so no handle is ever left waiting.
#[test]
fn a_rank_ctx_refuses_by_name() {
    let Some(gpu) = gpu() else {
        eprintln!("  [139.2] no GPU: a_rank_ctx_refuses_by_name skipped");
        return;
    };
    let (m, _) = build_case(CaseKind::Big, 5, 4, 3).expect("mesh");
    let d2 = Decomposition::build(&m, 2, &PartitionMethod::Linear).expect("cut");
    assert!(
        d2.parts.iter().all(|p| p.n_halo > 0),
        "the 2-part cut left a part with no halo"
    );
    let w2 = ThreadComm::world(2).expect("world");

    // Row 1: the handle is another rank's.
    let plain0 = GpuMesh::upload(&gpu, &d2.parts[0].mesh).expect("upload");
    let msg = RankCtx::new(&gpu, &w2[1], &d2.parts[0], &plain0)
        .err()
        .expect("row 1 must refuse")
        .to_string();
    assert!(msg.starts_with("distflow: "), "{msg}");
    assert!(msg.contains("rank 1"), "{msg}");
    assert!(msg.contains("part 0"), "{msg}");

    // Row 3: the mesh was uploaded without its halo room, so its every cell
    // buffer is n_cells long and the processor faces would read past them.
    let msg = RankCtx::new(&gpu, &w2[0], &d2.parts[0], &plain0)
        .err()
        .expect("row 3 must refuse")
        .to_string();
    assert!(msg.starts_with("distflow: "), "{msg}");
    assert!(msg.contains("n_halo 0"), "{msg}");
    assert!(
        msg.contains(&format!("n_halo {}", d2.parts[0].n_halo)),
        "{msg}"
    );
    assert!(msg.contains("upload_with_halo"), "{msg}");

    // Row 2: another part's mesh. The 2-part Linear cut of 60 cells is
    // 30/30, so the row is reached with an uneven explicit cut instead:
    // part 0 holds 10 cells and part 1 holds 50, and part 1's mesh, with its
    // own halo room, is handed in with part 0.
    let uneven: Vec<Label> = (0..m.n_cells).map(|c| Label::from(c >= 10)).collect();
    let du = Decomposition::build(&m, 2, &PartitionMethod::Explicit(uneven)).expect("cut");
    let n0 = du.parts[0].mesh.n_cells;
    let n1 = du.parts[1].mesh.n_cells;
    assert_ne!(n0, n1, "the uneven cut must give two different counts");
    let halo1 = GpuMesh::upload_with_halo(&gpu, &du.parts[1].mesh, du.parts[1].n_halo)
        .expect("upload");
    let msg = RankCtx::new(&gpu, &w2[0], &du.parts[0], &halo1)
        .err()
        .expect("row 2 must refuse")
        .to_string();
    assert!(msg.starts_with("distflow: "), "{msg}");
    assert!(msg.contains(&format!("{n1} cells")), "{msg}");
    assert!(msg.contains(&format!("has {n0}")), "{msg}");

    // `SpmdHalo`'s own refusal: `exchange_field` on a field built on a mesh
    // WITHOUT halo room, through a context whose mesh HAS it. Inside
    // `ThreadComm::run` because `RankCtx::new`'s reduction allgather is a
    // collective; the buffer-length refusal itself fires BEFORE any message
    // is sent, so both ranks leave with a name and nobody waits - which is
    // why this cannot be set up from `world(2)` handles on this one thread
    // (the allgather would block it) and why it is not a deadlock here.
    let msgs = ThreadComm::run(2, |comm| -> Result<String> {
        let gpu = Gpu::new(0)?;
        let part = &d2.parts[comm.rank()];
        let m_halo = GpuMesh::upload_with_halo(&gpu, &part.mesh, part.n_halo)?;
        let m_plain = GpuMesh::upload(&gpu, &part.mesh)?;
        let mut ctx = RankCtx::new(&gpu, comm, part, &m_halo)?;
        let fk = FieldKernels::new(&gpu)?;
        let mut psi = GpuScalarField::zeros(&gpu, &m_plain, "psi")?;
        match ctx.exchange_field(&gpu, &fk, &mut psi, &m_halo) {
            Ok(()) => Err(Error::Config(
                "distflow: exchange_field accepted a field whose f has no \
                 halo room"
                    .to_string(),
            )),
            Err(e) => Ok(e.to_string()),
        }
    })
    .expect("spmd run");
    for (r, msg) in msgs.iter().enumerate() {
        let msg = msg
            .as_ref()
            .unwrap_or_else(|e| panic!("rank {r} failed: {e}"));
        assert!(msg.starts_with("halo: "), "rank {r}: {msg}");
        assert!(msg.contains("must be"), "rank {r}: {msg}");
    }
    println!(
        "  [139.2] SpmdHalo refused the halo-less field on both ranks by name"
    );
}





