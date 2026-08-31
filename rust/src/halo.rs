// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The ghost-cell exchange: the launcher for `cuda/halo.cu`.
//!
//! SPEC-LIT §71.4. [`crate::decompose`] decides *what* crosses each cut and in
//! what order; this file moves it. The whole operation is
//!
//! ```text
//! pack   sendBuf[k] = psi[sendIndex[k]]              one gather kernel per part
//! move   sendBuf[send_offset[i] ..] -> psi_q[n_cells + recv_offset[j] ..]
//!        one device-to-device copy per ordered pair of neighbours
//! ```
//!
//! and there is no unpack, because the halo is ordered so that each
//! neighbour's cells are one contiguous run.
//!
//! # Why it is written as a copy and not as a gather across parts
//!
//! Every part in this build lives in one process on one device, so the
//! exchange could read the neighbour's field array directly and skip the pack
//! entirely. It deliberately does not. The pack-then-copy shape is what
//! `ncclSend`/`ncclRecv` replaces line for line, and reading another rank's
//! memory is the one thing a distributed run cannot do; writing it the short
//! way now would mean discovering that later. `cudarc`'s `memcpy_dtod` already
//! dispatches to `cuMemcpyPeerAsync` when the two buffers belong to different
//! contexts, so the same call is also what a one-process, several-device run
//! will use.
//!
//! # What it cannot change
//!
//! Nothing here does arithmetic. The pack is a gather - one thread per send
//! slot, one address written by exactly one thread - and the copy is the
//! identity on every byte. So a ghost cell holds, bit for bit, the value its
//! owner holds, for every partition and every part count. That is the
//! premise the bitwise gate of §71.8 rests on, and
//! `an_exchanged_halo_is_bit_for_bit_the_owner` asserts it directly rather
//! than inferring it from a solve.
//!
//! Provenance: ORIGINAL - the exchange plan and its execution. No external
//! source; `PROVENANCE.md`, *GPU plumbing and tooling - original*.
//! No GPL-licensed source was consulted.

use cudarc::driver::{CudaFunction, DeviceRepr, PushKernelArg};

use crate::decompose::Decomposition;
use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::{Label, Scalar, Vec3};

/// Entry points of `cuda/halo.cu`, resolved once.
struct HaloKernels {
    pack_scalar: CudaFunction,
    pack_vector: CudaFunction,
    pack_label: CudaFunction,
}

/// One part's half of the plan, on the device.
struct HaloPlan {
    n_cells: usize,
    n_halo: usize,
    /// The parts this one exchanges with, ascending.
    nbr: Vec<Label>,
    /// `[nbr.len() + 1]` into `send_index` and into the send buffer.
    send_offset: Vec<usize>,
    /// `[nbr.len() + 1]` halo-relative receive slices.
    recv_offset: Vec<usize>,
    send_index: DevBuf<Label>,
    n_send: usize,
    /// For each neighbour `i`, the position of THIS part in
    /// `plans[nbr[i]].nbr`. Resolved once at set-up so the exchange itself
    /// does no searching.
    back: Vec<usize>,
}

/// The exchange, with its own send buffers.
///
/// Built once, at set-up. Nothing in a time loop allocates, which is why all
/// three payload buffers exist from the start even if a given run only ever
/// exchanges scalars: the halo is a few per cent of the cells, so the unused
/// two cost about a byte per cell between them.
pub struct HaloExchange {
    k: HaloKernels,
    plans: Vec<HaloPlan>,
    buf_scalar: Vec<DevBuf<Scalar>>,
    buf_vector: Vec<DevBuf<Vec3>>,
    buf_label: Vec<DevBuf<Label>>,
}

impl HaloExchange {
    pub fn new(gpu: &Gpu, dec: &Decomposition) -> Result<Self> {
        let ks = KernelSet::new(gpu, crate::kernels::HALO)?;
        let k = HaloKernels {
            pack_scalar: ks.func("haloPackScalar")?,
            pack_vector: ks.func("haloPackVector")?,
            pack_label: ks.func("haloPackLabel")?,
        };

        let mut plans = Vec::with_capacity(dec.n_parts);
        let mut buf_scalar = Vec::with_capacity(dec.n_parts);
        let mut buf_vector = Vec::with_capacity(dec.n_parts);
        let mut buf_label = Vec::with_capacity(dec.n_parts);

        for part in &dec.parts {
            let nn = part.nbr_parts.len();
            if part.send_offset.len() != nn + 1 || part.recv_offset.len() != nn + 1 {
                return Err(Error::Config(format!(
                    "halo: part {} lists {nn} neighbours but its send/recv \
                     offsets are {}/{} long, not {}",
                    part.part,
                    part.send_offset.len(),
                    part.recv_offset.len(),
                    nn + 1
                )));
            }
            let n_send = part.send_index.len();
            if n_send != *part.send_offset.last().unwrap_or(&0) as usize {
                return Err(Error::Config(format!(
                    "halo: part {}'s send list holds {n_send} entries, its \
                     offsets end at {}",
                    part.part,
                    part.send_offset.last().copied().unwrap_or(0)
                )));
            }

            let mut back = Vec::with_capacity(nn);
            for &q in &part.nbr_parts {
                let qi = q as usize;
                let pos = dec.parts[qi]
                    .nbr_parts
                    .iter()
                    .position(|&r| r as usize == part.part)
                    .ok_or_else(|| {
                        Error::Config(format!(
                            "halo: part {} sends to part {qi}, which does not \
                             list it as a neighbour",
                            part.part
                        ))
                    })?;
                back.push(pos);
            }

            // A zero-length allocation is not something to hand to the driver,
            // and a part with no neighbours - P = 1, or an island - is a real
            // case. One element, never read.
            let alloc = n_send.max(1);
            buf_scalar.push(gpu.zeros::<Scalar>(alloc)?);
            buf_vector.push(gpu.zeros::<Vec3>(alloc)?);
            buf_label.push(gpu.zeros::<Label>(alloc)?);

            plans.push(HaloPlan {
                n_cells: part.mesh.n_cells,
                n_halo: part.n_halo,
                nbr: part.nbr_parts.clone(),
                send_offset: part.send_offset.iter().map(|&x| x as usize).collect(),
                recv_offset: part.recv_offset.iter().map(|&x| x as usize).collect(),
                send_index: gpu.upload(if part.send_index.is_empty() {
                    &[0 as Label][..]
                } else {
                    &part.send_index[..]
                })?,
                n_send,
                back,
            });
        }

        Ok(Self {
            k,
            plans,
            buf_scalar,
            buf_vector,
            buf_label,
        })
    }

    pub fn n_parts(&self) -> usize {
        self.plans.len()
    }

    /// Length every field buffer of part `p` must have.
    pub fn n_local(&self, p: usize) -> usize {
        self.plans[p].n_cells + self.plans[p].n_halo
    }

    /// Fill every part's halo of a scalar field.
    pub fn scalar(&mut self, gpu: &Gpu, fields: &mut [DevBuf<Scalar>]) -> Result<()> {
        let (k, plans, bufs) = (&self.k.pack_scalar, &self.plans, &mut self.buf_scalar);
        run(gpu, k, plans, bufs, fields)
    }

    /// The same for a vector field.
    pub fn vector(&mut self, gpu: &Gpu, fields: &mut [DevBuf<Vec3>]) -> Result<()> {
        let (k, plans, bufs) = (&self.k.pack_vector, &self.plans, &mut self.buf_vector);
        run(gpu, k, plans, bufs, fields)
    }

    /// The same for a label field - the integer masks a coupled face's
    /// neighbour is tested for, `lduSetValues`'s `isFixed` being the one this
    /// tree has today.
    pub fn label(&mut self, gpu: &Gpu, fields: &mut [DevBuf<Label>]) -> Result<()> {
        let (k, plans, bufs) = (&self.k.pack_label, &self.plans, &mut self.buf_label);
        run(gpu, k, plans, bufs, fields)
    }
}

/// Pack, then copy. One implementation for all three payloads, because the
/// exchange is a property of the plan and not of what is being moved.
fn run<T: DeviceRepr>(
    gpu: &Gpu,
    pack: &CudaFunction,
    plans: &[HaloPlan],
    bufs: &mut [DevBuf<T>],
    fields: &mut [DevBuf<T>],
) -> Result<()> {
    if fields.len() != plans.len() {
        return Err(Error::Config(format!(
            "halo: {} field buffers for {} parts",
            fields.len(),
            plans.len()
        )));
    }
    for (p, plan) in plans.iter().enumerate() {
        let want = plan.n_cells + plan.n_halo;
        if fields[p].len() < want {
            return Err(Error::Config(format!(
                "halo: part {p}'s field buffer holds {} elements, but the part \
                 has {} owned cells and {} halo cells and every field on it \
                 must be {want} long",
                fields[p].len(),
                plan.n_cells,
                plan.n_halo
            )));
        }
    }

    // ---- 1. pack -----------------------------------------------------------
    for (p, plan) in plans.iter().enumerate() {
        if plan.n_send == 0 {
            continue;
        }
        let ns = plan.n_send as Label;
        let nc = plan.n_cells as Label;
        unsafe {
            gpu.stream()
                .launch_builder(pack)
                .arg(&mut bufs[p])
                .arg(&fields[p])
                .arg(&plan.send_index)
                .arg(&ns)
                .arg(&nc)
                .launch(cfg_for(plan.n_send))?;
        }
    }

    // ---- 2. move -----------------------------------------------------------
    // One contiguous copy per ordered pair. `memcpy_dtod` is a
    // device-to-device copy inside one context and `cuMemcpyPeerAsync` across
    // two, so this same loop is what a one-process, several-device run runs.
    for (p, plan) in plans.iter().enumerate() {
        for (i, &q) in plan.nbr.iter().enumerate() {
            let qi = q as usize;
            let (s0, s1) = (plan.send_offset[i], plan.send_offset[i + 1]);
            if s1 <= s0 {
                continue;
            }
            let j = plan.back[i];
            let base = plans[qi].n_cells;
            let (r0, r1) = (
                base + plans[qi].recv_offset[j],
                base + plans[qi].recv_offset[j + 1],
            );
            if r1 - r0 != s1 - s0 {
                return Err(Error::Config(format!(
                    "halo: part {p} sends {} cells to part {qi}, which has room \
                     for {}",
                    s1 - s0,
                    r1 - r0
                )));
            }
            let src = bufs[p].slice(s0..s1);
            let mut dst = fields[qi].slice_mut(r0..r1);
            gpu.stream().memcpy_dtod(&src, &mut dst)?;
        }
    }

    Ok(())
}

// ==========================================================================
//  Tests - and the gate SPEC-LIT §71 exists to pass
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decompose::tests::{boxes, round_robin};
    use crate::decompose::{Decomposition, PartitionMethod};
    use crate::device::cfg_for;
    use crate::ldu::{GpuLduMatrix, HostLduMatrix};
    use crate::ldu_ops::{self, LduKernels};
    use crate::mesh::{GpuMesh, HostMesh};
    use crate::solver::{self, SolverKernels};
    use cudarc::driver::PushKernelArg;

    /// Every device test needs a card. Returning `None` makes the test pass
    /// vacuously on a machine without one, which is the convention the rest of
    /// the crate follows.
    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    /// The part counts and partitions every gate below runs over.
    ///
    /// The round robin is there because it is the worst partition possible -
    /// cell `c` to part `c % P` cuts nearly every face, so nearly every row
    /// has a term that moved out of `upper` and into `boundary_coeffs`. A neat
    /// partition leaves most rows untouched and proves much less.
    fn plans(m: &HostMesh) -> Vec<(usize, PartitionMethod)> {
        let mut out = Vec::new();
        for p in 1..=4 {
            out.push((p, PartitionMethod::Hilbert));
            out.push((p, PartitionMethod::Linear));
            out.push((p, round_robin(m.n_cells, p)));
        }
        out
    }

    /// A deterministic field: a function of the GLOBAL cell id, so that
    /// splitting it is a permutation and not a computation.
    fn field(c: usize) -> Scalar {
        0.25 + 0.0625 * ((c * 37) % 23) as Scalar
    }

    /// A Laplacian plus a `ddt`, assembled on the host from the mesh's own
    /// metrics, with a scattering of pinned cells.
    ///
    /// Host-assembled deliberately. The *assembly* kernels are not yet
    /// partition-invariant - SPEC-LIT §70.5 lists the sixteen that are not and
    /// says why ordering alone cannot fix them - so a gate that re-assembled
    /// on each part would be testing something this unit does not claim. What
    /// it claims is that a matrix which already exists survives being cut:
    /// ONE matrix, assembled once, distributed by `split_matrix`.
    fn laplacian(m: &HostMesh, rdt: Scalar) -> HostLduMatrix {
        let mut a = HostLduMatrix::zeros(m);
        for f in 0..m.n_internal_faces {
            let g = m.mag_sf[f] * m.delta_coeffs[f];
            a.upper[f] = g;
            a.lower[f] = g;
            a.diag[m.owner[f] as usize] -= g;
            a.diag[m.neighbour[f] as usize] -= g;
        }
        for bf in 0..m.n_boundary_faces {
            let g = m.b_mag_sf[bf] * m.b_delta_coeffs[bf];
            a.internal_coeffs[bf] = -g;
            a.boundary_coeffs[bf] = if m.b_nbr_cell[bf] >= 0 { -g } else { -g * 0.375 };
        }
        for c in 0..m.n_cells {
            let vd = m.v[c] * rdt;
            a.diag[c] -= vd;
            a.source[c] -= vd * field(c);
        }
        // Something to pin, chosen by GLOBAL cell id so that the same cells
        // are pinned however the mesh is cut - SPEC-LIT §71.7. Every part
        // owns some and some parts neighbour one across a cut, which is the
        // case `lduSetValues` needs the halo for.
        for c in (0..m.n_cells).step_by(7) {
            a.is_fixed[c] = 1;
            a.fixed_value[c] = 0.5 + 0.125 * (c % 5) as Scalar;
        }
        a
    }

    /// Everything one part needs to be driven. `psi` is deliberately NOT here:
    /// the exchange takes a slice of field buffers, which is exactly the shape
    /// a distributed run has, so the fields live in their own vector.
    struct Rig {
        mesh: GpuMesh,
        a: GpuLduMatrix,
        n_cells: usize,
        apsi: DevBuf<Scalar>,
        r: DevBuf<Scalar>,
        z: DevBuf<Scalar>,
        rdiag: DevBuf<Scalar>,
        one: DevBuf<Scalar>,
    }

    impl Rig {
        fn new(gpu: &Gpu, hm: &HostMesh, ha: &HostLduMatrix) -> Result<Self> {
            Ok(Self {
                mesh: GpuMesh::upload(gpu, hm)?,
                a: ha.upload(gpu)?,
                n_cells: hm.n_cells,
                apsi: gpu.zeros(hm.n_cells)?,
                r: gpu.zeros(hm.n_cells)?,
                z: gpu.zeros(hm.n_cells)?,
                rdiag: gpu.zeros(hm.n_cells)?,
                one: gpu.upload(&[1.0 as Scalar])?,
            })
        }
    }

    /// `relax`, then `set_values`, then the fold - the order `src/ldu_ops.rs`
    /// documents, and the order the assembly of any real equation uses.
    fn prepare(
        gpu: &Gpu,
        lk: &LduKernels,
        sk: &SolverKernels,
        rig: &mut Rig,
        psi: &DevBuf<Scalar>,
        alpha: Scalar,
    ) -> Result<()> {
        ldu_ops::relax(gpu, lk, &mut rig.a, &rig.mesh, psi, alpha)?;
        ldu_ops::set_values(gpu, lk, &mut rig.a, &rig.mesh)?;
        ldu_ops::add_boundary_contributions(gpu, lk, &mut rig.a, &rig.mesh)?;
        let n = rig.n_cells as Label;
        unsafe {
            gpu.stream()
                .launch_builder(&sk.invert_diag)
                .arg(&mut rig.rdiag)
                .arg(&rig.a.diag)
                .arg(&n)
                .launch(cfg_for(rig.n_cells))?;
        }
        Ok(())
    }

    /// One Jacobi sweep, `psi += (b - A psi)/diag`.
    ///
    /// Every step but the product is elementwise on one cell's own values, so
    /// the only thing in the sweep a decomposition can move is `A psi` - which
    /// is exactly what §70's row ordering and this section's halo exist to
    /// hold still.
    fn sweep(
        gpu: &Gpu,
        sk: &SolverKernels,
        rig: &mut Rig,
        psi: &mut DevBuf<Scalar>,
    ) -> Result<()> {
        let n = rig.n_cells;
        let nl = n as Label;
        solver::amul(gpu, sk, &mut rig.apsi, psi, &rig.a, &rig.mesh)?;
        solver::vec_sub(gpu, sk, &mut rig.r, &rig.a.source, &rig.apsi, n)?;
        unsafe {
            gpu.stream()
                .launch_builder(&sk.precond_jacobi)
                .arg(&mut rig.z)
                .arg(&rig.r)
                .arg(&rig.rdiag)
                .arg(&nl)
                .launch(cfg_for(n))?;
            gpu.stream()
                .launch_builder(&sk.axpy)
                .arg(&mut *psi)
                .arg(&rig.z)
                .arg(&rig.one)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    /// `x` and `y` are the same number. `-0.0` and `0.0` count as the same:
    /// `set_values` writes `+0.0` into a boundary coefficient where the whole
    /// mesh writes `0.0` into `upper` and the split negates it, and the sign
    /// of a zero coefficient reaches no answer - `sum -= 0.0*psi` and
    /// `sum += (-0.0)*psi` are both `sum`. Everywhere else this is bit
    /// equality.
    fn same(x: Scalar, y: Scalar) -> bool {
        x.to_bits() == y.to_bits() || (x == 0.0 && y == 0.0)
    }

    // ----------------------------------------------------------------------
    //  The exchange, on its own
    // ----------------------------------------------------------------------

    /// The premise everything else rests on: a ghost cell holds, bit for bit,
    /// the value its owner holds. No arithmetic happens in the halo path, so
    /// this is not an approximation and no tolerance is accepted.
    #[test]
    fn an_exchanged_halo_is_bit_for_bit_the_owner() {
        let Some(gpu) = gpu() else { return };
        for cyclic in [false, true] {
            let m = boxes([4, 3, 2], cyclic);
            let global: Vec<Scalar> = (0..m.n_cells).map(field).collect();
            for (np, method) in plans(&m) {
                let d = Decomposition::build(&m, np, &method).expect("cut");
                let mut ex = HaloExchange::new(&gpu, &d).expect("exchange");
                let mut fields: Vec<DevBuf<Scalar>> = (0..np)
                    .map(|p| {
                        gpu.upload(&d.split_field(p, &global).expect("split"))
                            .expect("upload")
                    })
                    .collect();
                ex.scalar(&gpu, &mut fields).expect("exchange");
                gpu.sync().expect("sync");
                for (p, part) in d.parts.iter().enumerate() {
                    let got = gpu.download(&fields[p]).expect("download");
                    for i in 0..part.n_local() {
                        assert_eq!(
                            got[i].to_bits(),
                            global[part.global_cell[i] as usize].to_bits(),
                            "P={np} {} part {p} local cell {i}",
                            method.as_str()
                        );
                    }
                }
            }
        }
    }

    /// The same plan carries a vector and a label payload. `is_fixed` is the
    /// label one this tree has today - `lduSetValues` tests it at a coupled
    /// face's NEIGHBOUR, which on a part is a ghost cell.
    #[test]
    fn a_vector_and_a_label_halo_cross_the_cut_too() {
        let Some(gpu) = gpu() else { return };
        let m = boxes([4, 3, 2], true);
        let vecs: Vec<Vec3> = (0..m.n_cells)
            .map(|c| Vec3::new(field(c), -field(c), 2.0 * field(c)))
            .collect();
        let labels: Vec<Label> = (0..m.n_cells).map(|c| (c as Label) * 3 - 1).collect();
        for (np, method) in plans(&m) {
            let d = Decomposition::build(&m, np, &method).expect("cut");
            let mut ex = HaloExchange::new(&gpu, &d).expect("exchange");

            let mut vf: Vec<DevBuf<Vec3>> = d
                .parts
                .iter()
                .map(|p| {
                    let mut h = vec![Vec3::ZERO; p.n_local()];
                    for (i, hi) in h.iter_mut().enumerate().take(p.mesh.n_cells) {
                        *hi = vecs[p.global_cell[i] as usize];
                    }
                    gpu.upload(&h).expect("upload")
                })
                .collect();
            let mut lf: Vec<DevBuf<Label>> = (0..np)
                .map(|p| {
                    gpu.upload(&d.split_labels(p, &labels).expect("split"))
                        .expect("upload")
                })
                .collect();
            ex.vector(&gpu, &mut vf).expect("exchange");
            ex.label(&gpu, &mut lf).expect("exchange");
            gpu.sync().expect("sync");

            for (p, part) in d.parts.iter().enumerate() {
                let gv = gpu.download(&vf[p]).expect("download");
                let gl = gpu.download(&lf[p]).expect("download");
                for i in 0..part.n_local() {
                    let g = part.global_cell[i] as usize;
                    assert_eq!(gv[i], vecs[g], "vector, P={np} part {p} cell {i}");
                    assert_eq!(gl[i], labels[g], "label, P={np} part {p} cell {i}");
                }
            }
        }
    }

    /// A field buffer only `n_cells` long is the single most likely mistake in
    /// a distributed build, so it is refused by name with both numbers in the
    /// message rather than read past the end.
    #[test]
    fn a_field_buffer_without_room_for_the_halo_is_refused_by_name() {
        let Some(gpu) = gpu() else { return };
        let m = boxes([4, 3, 2], false);
        let d = Decomposition::build(&m, 3, &round_robin(m.n_cells, 3)).expect("cut");
        let mut ex = HaloExchange::new(&gpu, &d).expect("exchange");
        let mut fields: Vec<DevBuf<Scalar>> = d
            .parts
            .iter()
            .map(|p| gpu.zeros::<Scalar>(p.mesh.n_cells).expect("alloc"))
            .collect();
        let msg = ex
            .scalar(&gpu, &mut fields)
            .expect_err("must refuse")
            .to_string();
        assert!(msg.contains("halo cells"), "{msg}");
        assert!(msg.contains("must be"), "{msg}");
    }

    // ----------------------------------------------------------------------
    //  THE GATE
    // ----------------------------------------------------------------------

    /// SPEC-LIT §71.8, and the reason §70 was done first.
    ///
    /// One matrix and one field, cut into 1, 2, 3 and 4 parts by three
    /// different partitioners, run in one process through the whole
    /// `relax -> set_values -> fold -> six Jacobi sweeps` pipeline with a halo
    /// exchange before every product, and gathered back. **Every number must
    /// be bit-for-bit the undecomposed answer.** Not "agrees to 1e-13": equal.
    ///
    /// The matrix is compared as well as the field, so that a failure names
    /// the stage that lost the bits rather than leaving the whole pipeline as
    /// the suspect.
    #[test]
    fn a_decomposed_run_is_bitwise_the_undecomposed_run() {
        let Some(gpu) = gpu() else { return };
        let lk = LduKernels::new(&gpu).expect("ldu kernels");
        let sk = SolverKernels::new(&gpu).expect("solver kernels");
        const SWEEPS: usize = 6;
        const ALPHA: Scalar = 0.5;

        for (dims, cyclic) in [([4usize, 3, 2], false), ([5, 4, 3], true)] {
            let m = boxes(dims, cyclic);
            let a0 = laplacian(&m, 1.5);
            let psi_in: Vec<Scalar> = (0..m.n_cells).map(field).collect();

            // ---- the undecomposed run -------------------------------------
            let mut serial = Rig::new(&gpu, &m, &a0).expect("rig");
            let mut spsi = gpu.upload(&psi_in).expect("upload");
            prepare(&gpu, &lk, &sk, &mut serial, &spsi, ALPHA).expect("prepare");
            for _ in 0..SWEEPS {
                sweep(&gpu, &sk, &mut serial, &mut spsi).expect("sweep");
            }
            gpu.sync().expect("sync");
            let want_a = HostLduMatrix::download(&gpu, &serial.a).expect("download");
            let want_psi = gpu.download(&spsi).expect("download");

            for (np, method) in plans(&m) {
                let d = Decomposition::build(&m, np, &method).expect("cut");
                // A gate that passed because nothing was actually cut would be
                // worse than no gate. For every P above 1 the cut must have
                // moved terms out of `upper` and put cells in a halo, or the
                // bitwise agreement below means nothing.
                if np > 1 {
                    assert!(d.n_cut_faces > 0, "P={np} {} cut no face", method.as_str());
                    assert!(
                        d.parts.iter().all(|p| p.n_halo > 0),
                        "P={np} {} left a part with no halo",
                        method.as_str()
                    );
                }
                let mut ex = HaloExchange::new(&gpu, &d).expect("exchange");
                let split: Vec<HostLduMatrix> = (0..np)
                    .map(|p| d.split_matrix(&m, p, &a0).expect("split"))
                    .collect();
                let mut rigs: Vec<Rig> = (0..np)
                    .map(|p| Rig::new(&gpu, &d.parts[p].mesh, &split[p]).expect("rig"))
                    .collect();

                // `is_fixed` and `fixed_value` reach the halo through the
                // EXCHANGE, not from the host's copy of the whole mesh: a
                // distributed build has no such copy, and taking the short cut
                // here would leave the label exchange untested on the one
                // kernel that needs it.
                let mut isf: Vec<DevBuf<Label>> = (0..np)
                    .map(|p| gpu.upload(&split[p].is_fixed).expect("upload"))
                    .collect();
                let mut fv: Vec<DevBuf<Scalar>> = (0..np)
                    .map(|p| gpu.upload(&split[p].fixed_value).expect("upload"))
                    .collect();
                ex.label(&gpu, &mut isf).expect("exchange");
                ex.scalar(&gpu, &mut fv).expect("exchange");
                for (rig, (i, v)) in rigs.iter_mut().zip(isf.into_iter().zip(fv)) {
                    rig.a.is_fixed = i;
                    rig.a.fixed_value = v;
                }

                let mut psi: Vec<DevBuf<Scalar>> = (0..np)
                    .map(|p| {
                        gpu.upload(&d.split_field(p, &psi_in).expect("split"))
                            .expect("upload")
                    })
                    .collect();
                ex.scalar(&gpu, &mut psi).expect("exchange");
                for (p, rig) in rigs.iter_mut().enumerate() {
                    prepare(&gpu, &lk, &sk, rig, &psi[p], ALPHA).expect("prepare");
                }
                gpu.sync().expect("sync");

                // ---- the matrix, stage by stage ---------------------------
                let label = format!("P={np} {} {dims:?}{}", method.as_str(), if cyclic { " cyclic" } else { "" });
                for (p, rig) in rigs.iter().enumerate() {
                    let got = HostLduMatrix::download(&gpu, &rig.a).expect("download");
                    let part = &d.parts[p];
                    let pm = &part.mesh;
                    for i in 0..pm.n_cells {
                        let g = part.global_cell[i] as usize;
                        assert!(
                            same(got.diag[i], want_a.diag[g]),
                            "{label}: part {p} diag at global cell {g}: {} vs {}",
                            got.diag[i],
                            want_a.diag[g]
                        );
                        assert!(
                            same(got.source[i], want_a.source[g]),
                            "{label}: part {p} source at global cell {g}: {} vs {}",
                            got.source[i],
                            want_a.source[g]
                        );
                    }
                    for f in 0..pm.n_internal_faces {
                        let g = pm.global_face[f] as usize;
                        assert!(same(got.upper[f], want_a.upper[g]), "{label} upper {g}");
                        assert!(same(got.lower[f], want_a.lower[g]), "{label} lower {g}");
                    }
                    for bf in 0..pm.n_boundary_faces {
                        let g = pm.global_face[pm.n_internal_faces + bf] as usize;
                        if g >= m.n_internal_faces {
                            let src = g - m.n_internal_faces;
                            assert!(
                                same(got.internal_coeffs[bf], want_a.internal_coeffs[src]),
                                "{label} internalCoeffs {src}"
                            );
                            assert!(
                                same(got.boundary_coeffs[bf], want_a.boundary_coeffs[src]),
                                "{label} boundaryCoeffs {src}"
                            );
                        } else {
                            let owns = d.cell_part[m.owner[g] as usize] == p as Label;
                            let want = if owns { want_a.upper[g] } else { want_a.lower[g] };
                            assert!(
                                same(got.boundary_coeffs[bf], -want),
                                "{label}: the cut face {g} carries {} but the \
                                 whole mesh has {want}",
                                got.boundary_coeffs[bf]
                            );
                        }
                    }
                }

                // ---- the run ----------------------------------------------
                for _ in 0..SWEEPS {
                    ex.scalar(&gpu, &mut psi).expect("exchange");
                    for (p, rig) in rigs.iter_mut().enumerate() {
                        sweep(&gpu, &sk, rig, &mut psi[p]).expect("sweep");
                    }
                }
                gpu.sync().expect("sync");

                let per_part: Vec<Vec<Scalar>> = psi
                    .iter()
                    .map(|b| gpu.download(b).expect("download"))
                    .collect();
                let got_psi = d.gather_field(&per_part).expect("gather");
                for c in 0..m.n_cells {
                    assert_eq!(
                        got_psi[c].to_bits(),
                        want_psi[c].to_bits(),
                        "{label}: cell {c} after {SWEEPS} sweeps is {} \
                         decomposed and {} whole",
                        got_psi[c],
                        want_psi[c]
                    );
                }
            }
        }
    }

    /// The other implementation of the row sum. `solver::amul` is what every
    /// Krylov iteration calls and `ldu_ops::amul` is what the energy,
    /// conjugate and residual paths call; they walk the same map but they are
    /// two kernels, and §70 converted both, so the gate has to exercise both.
    #[test]
    fn the_other_amul_survives_the_cut_too() {
        let Some(gpu) = gpu() else { return };
        let lk = LduKernels::new(&gpu).expect("ldu kernels");
        let m = boxes([5, 4, 3], true);
        let mut a0 = laplacian(&m, 1.5);
        // No pinning here: this test is about the product alone.
        a0.is_fixed.iter_mut().for_each(|x| *x = 0);
        let psi_in: Vec<Scalar> = (0..m.n_cells).map(field).collect();

        let smesh = GpuMesh::upload(&gpu, &m).expect("mesh");
        let mut serial = a0.upload(&gpu).expect("upload");
        ldu_ops::add_boundary_contributions(&gpu, &lk, &mut serial, &smesh).expect("fold");
        let spsi = gpu.upload(&psi_in).expect("upload");
        let mut sapsi = gpu.zeros::<Scalar>(m.n_cells).expect("alloc");
        ldu_ops::amul(&gpu, &lk, &mut sapsi, &spsi, &serial, &smesh).expect("amul");
        gpu.sync().expect("sync");
        let want = gpu.download(&sapsi).expect("download");

        for (np, method) in plans(&m) {
            let d = Decomposition::build(&m, np, &method).expect("cut");
            let mut ex = HaloExchange::new(&gpu, &d).expect("exchange");
            let mut psi: Vec<DevBuf<Scalar>> = (0..np)
                .map(|p| {
                    gpu.upload(&d.split_field(p, &psi_in).expect("split"))
                        .expect("upload")
                })
                .collect();
            ex.scalar(&gpu, &mut psi).expect("exchange");
            let mut out = Vec::new();
            for (p, ppsi) in psi.iter().enumerate().take(np) {
                let pm = &d.parts[p].mesh;
                let gm = GpuMesh::upload(&gpu, pm).expect("mesh");
                let mut pa = d
                    .split_matrix(&m, p, &a0)
                    .expect("split")
                    .upload(&gpu)
                    .expect("upload");
                ldu_ops::add_boundary_contributions(&gpu, &lk, &mut pa, &gm).expect("fold");
                let mut apsi = gpu.zeros::<Scalar>(pm.n_cells).expect("alloc");
                ldu_ops::amul(&gpu, &lk, &mut apsi, ppsi, &pa, &gm).expect("amul");
                gpu.sync().expect("sync");
                out.push(gpu.download(&apsi).expect("download"));
            }
            let got = d.gather_field(&out).expect("gather");
            for c in 0..m.n_cells {
                assert_eq!(
                    got[c].to_bits(),
                    want[c].to_bits(),
                    "lduAmul, P={np} {}: cell {c} is {} decomposed and {} whole",
                    method.as_str(),
                    got[c],
                    want[c]
                );
            }
        }
    }
}
