// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The SPMD halo exchange: a rank per part, five steps (SPEC-LIT §138.3).
//!
//! §71.4's exchange has every part in one process and moves the packed cells
//! with a device-to-device copy. In an SPMD run a rank holds one part and
//! cannot read another rank's memory, so the move goes through the host and a
//! [`Comm`](crate::comm::Comm): pack with §71.4's own kernel, copy the send
//! buffer to page-locked host memory, hand the bytes to a `Comm`, copy the
//! whole receive buffer into the halo, and synchronise. There is still no
//! unpack and no arithmetic: the halo is ordered by (owning part, global cell
//! id), so the neighbours' runs in `nbr` order ARE the halo, and a ghost cell
//! holds its owner's value bit for bit.
//!
//! Two synchronisations, both explicit, because this crate runs cudarc with
//! event tracking off: a page-locked buffer does not wait for the copy that
//! fills it, and the stream must be synchronised after the device-to-host
//! copy before the host reads the bytes, and after the host-to-device copy
//! before the next exchange may overwrite the buffer. An SPMD exchange is
//! therefore host round trips by construction and is refused inside a CUDA
//! graph by name.
//!
//! One plan (`PartMesh` + the rank count), four payloads (`Scalar`, `Vec3`,
//! `Tensor`, `Label`), one tag ([`HALO_TAG`]). An exchange allocates nothing
//! on the device.
//!
//! Provenance: ORIGINAL - the SPMD form is this project's own design
//! (SPEC-LIT §138.3, PAR-02); it uses the exchange plan of §71.4 unchanged.
//! No external source. No GPL-licensed source was consulted.

use cudarc::driver::{
    CudaFunction, DeviceRepr, PinnedHostSlice, PushKernelArg, ValidAsZeroBits,
};

use crate::comm::Comm;
use crate::decompose::PartMesh;
use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::halo::LevelHalo;
use crate::{Label, Scalar, Tensor, Vec3};

/// The one tag every halo message travels under (SPEC-LIT §138.3).
///
/// Every rank runs the same program and messages are FIFO within an envelope,
/// so consecutive exchanges of different fields under one tag cannot overtake
/// each other.
pub const HALO_TAG: u32 = 71;

/// One payload's send buffers: on the device, and two page-locked host
/// buffers, allocated once in [`SpmdHalo::new`]. An exchange allocates
/// nothing.
struct Lane<T> {
    /// The packed cells, `n_send` long (one element when `n_send` is zero).
    dev: DevBuf<T>,
    /// What step 2 fills and step 3 reads. `alloc_pinned`, which sets the
    /// write-combined flag: cudarc 0.19.9, which `Cargo.lock` pins, offers
    /// only `alloc_pinned` (SPEC-LIT §138.3).
    send: PinnedHostSlice<T>,
    /// What step 3 fills and step 4 moves into the halo.
    recv: PinnedHostSlice<T>,
}

/// What this rank sends, to whom, and where its halo lands - the one part's
/// half of the plan of §71.4, unchanged.
struct Plan {
    part: usize,
    n_ranks: usize,
    n_cells: usize,
    n_halo: usize,
    /// The parts this one exchanges with, ascending.
    nbr: Vec<Label>,
    /// `[nbr.len() + 1]` into the send buffer.
    send_offset: Vec<usize>,
    /// `[nbr.len() + 1]` halo-relative receive slices.
    recv_offset: Vec<usize>,
    n_send: usize,
    send_index: DevBuf<Label>,
}

/// The SPMD halo exchange of one part: the rank's half of §71.4's plan, plus
/// the host round trip of §138.3.
///
/// Built once per rank, at set-up. Nothing in an exchange allocates on the
/// device. Every failure is `Error::Config` starting `halo: `. `Send`, so a
/// rank can move it into its thread.
pub struct SpmdHalo {
    plan: Plan,
    pack_scalar: CudaFunction,
    pack_vector: CudaFunction,
    pack_tensor: CudaFunction,
    pack_label: CudaFunction,
    lane_scalar: Lane<Scalar>,
    lane_vector: Lane<Vec3>,
    lane_tensor: Lane<Tensor>,
    lane_label: Lane<Label>,
}

impl SpmdHalo {
    /// Build the rank's plan from its part and the rank count. Rank `r` holds
    /// part `r` (SPEC-LIT §138.3).
    pub fn new(gpu: &Gpu, part: &PartMesh, n_ranks: usize) -> Result<Self> {
        Self::from_level(gpu, &LevelHalo::of_part(part), n_ranks)
    }

    /// Build the rank's plan for one level: the fine level's
    /// ([`LevelHalo::of_part`], which is [`Self::new`]) or a coarse level's
    /// of the distributed GAMG (SPEC-LIT §141.10). Rank `r` holds part `r`.
    pub fn from_level(gpu: &Gpu, part: &LevelHalo, n_ranks: usize) -> Result<Self> {
        // ---- the refusals of §138.3, in the table's order ------------------
        let p = part.part;
        if p >= n_ranks {
            return Err(Error::Config(format!(
                "halo: part {p} is not below the {n_ranks} ranks this plan \
                 was built for"
            )));
        }
        for &q in &part.nbr_parts {
            if q < 0 || q as usize >= n_ranks {
                return Err(Error::Config(format!(
                    "halo: part {p}'s neighbour {q} is not below the \
                     {n_ranks} ranks this plan was built for"
                )));
            }
        }
        let nn = part.nbr_parts.len();
        if part.send_offset.len() != nn + 1 || part.recv_offset.len() != nn + 1 {
            return Err(Error::Config(format!(
                "halo: part {p} lists {nn} neighbours but its send/recv \
                 offsets are {}/{} long, not {}",
                part.send_offset.len(),
                part.recv_offset.len(),
                nn + 1
            )));
        }
        let n_send = part.send_index.len();
        if n_send != *part.send_offset.last().unwrap_or(&0) as usize {
            return Err(Error::Config(format!(
                "halo: part {p}'s send list holds {n_send} entries, its \
                 offsets end at {}",
                part.send_offset.last().copied().unwrap_or(0)
            )));
        }
        // Row 4: the offsets' VALUES. A plan whose offsets do not start at
        // 0, whose offsets decrease, or whose receive offsets do not end at
        // the part's halo count would panic inside an exchange's step 3, in
        // a rank thread; it is refused here, before anything is allocated
        // or uploaded (SPEC-LIT §138.3).
        if part.send_offset[0] != 0 {
            return Err(Error::Config(format!(
                "halo: part {p}'s send offsets must start at 0; their first \
                 entry is {}",
                part.send_offset[0]
            )));
        }
        if part.recv_offset[0] != 0 {
            return Err(Error::Config(format!(
                "halo: part {p}'s receive offsets must start at 0; their \
                 first entry is {}",
                part.recv_offset[0]
            )));
        }
        for (name, offs) in [
            ("send", &part.send_offset[..]),
            ("receive", &part.recv_offset[..]),
        ] {
            for i in 1..offs.len() {
                if offs[i] < offs[i - 1] {
                    return Err(Error::Config(format!(
                        "halo: part {p}'s {name} offsets decrease: entry \
                         {} is {}, entry {i} is {}",
                        i - 1,
                        offs[i - 1],
                        offs[i]
                    )));
                }
            }
        }
        let recv_end = part.recv_offset[nn];
        if recv_end != part.n_halo as Label {
            return Err(Error::Config(format!(
                "halo: part {p}'s receive offsets end at {recv_end}, but \
                 the part has {} halo cells",
                part.n_halo
            )));
        }

        part.check_send_index()?;

        // ---- the pack kernels and the plan, loaded once --------------------
        let ks = KernelSet::new(gpu, crate::kernels::HALO)?;
        let plan = Plan {
            part: p,
            n_ranks,
            n_cells: part.n_cells,
            n_halo: part.n_halo,
            nbr: part.nbr_parts.clone(),
            send_offset: part.send_offset.iter().map(|&x| x as usize).collect(),
            recv_offset: part.recv_offset.iter().map(|&x| x as usize).collect(),
            n_send,
            // A zero-length upload is not something to hand to the driver;
            // the placeholder is never read, because step 1 is skipped at
            // `n_send == 0` - the same move the in-process form makes.
            send_index: gpu.upload(if part.send_index.is_empty() {
                &[0 as Label][..]
            } else {
                &part.send_index[..]
            })?,
        };

        let lane_scalar = lane::<Scalar>(gpu, n_send, part.n_halo)?;
        let lane_vector = lane::<Vec3>(gpu, n_send, part.n_halo)?;
        let lane_tensor = lane::<Tensor>(gpu, n_send, part.n_halo)?;
        let lane_label = lane::<Label>(gpu, n_send, part.n_halo)?;

        Ok(Self {
            plan,
            pack_scalar: ks.func("haloPackScalar")?,
            pack_vector: ks.func("haloPackVector")?,
            pack_tensor: ks.func("haloPackTensor")?,
            pack_label: ks.func("haloPackLabel")?,
            lane_scalar,
            lane_vector,
            lane_tensor,
            lane_label,
        })
    }

    /// The part, which is the rank this plan belongs to.
    pub fn part(&self) -> usize {
        self.plan.part
    }

    /// The rank count the plan was built with.
    pub fn n_ranks(&self) -> usize {
        self.plan.n_ranks
    }

    /// The part's own cells.
    pub fn n_cells(&self) -> usize {
        self.plan.n_cells
    }

    /// The ghost cells the neighbours fill.
    pub fn n_halo(&self) -> usize {
        self.plan.n_halo
    }

    /// Length every field buffer on this rank must have.
    pub fn n_local(&self) -> usize {
        self.plan.n_cells + self.plan.n_halo
    }

    /// Fill this rank's halo of a scalar field.
    pub fn scalar(&mut self, gpu: &Gpu, comm: &dyn Comm, field: &mut DevBuf<Scalar>) -> Result<()> {
        exchange(
            gpu,
            comm,
            field,
            &self.pack_scalar,
            &self.plan,
            &mut self.lane_scalar,
        )
    }

    /// The same for a vector field.
    pub fn vector(&mut self, gpu: &Gpu, comm: &dyn Comm, field: &mut DevBuf<Vec3>) -> Result<()> {
        exchange(
            gpu,
            comm,
            field,
            &self.pack_vector,
            &self.plan,
            &mut self.lane_vector,
        )
    }

    /// The same for a tensor field - nine scalars moved as one struct, the
    /// gradient of a vector field being the one a distributed flow needs it
    /// for (SPEC-LIT §71.4).
    pub fn tensor(&mut self, gpu: &Gpu, comm: &dyn Comm, field: &mut DevBuf<Tensor>) -> Result<()> {
        exchange(
            gpu,
            comm,
            field,
            &self.pack_tensor,
            &self.plan,
            &mut self.lane_tensor,
        )
    }

    /// The same for a label field - the integer masks a coupled face's
    /// neighbour is tested for.
    pub fn label(&mut self, gpu: &Gpu, comm: &dyn Comm, field: &mut DevBuf<Label>) -> Result<()> {
        exchange(
            gpu,
            comm,
            field,
            &self.pack_label,
            &self.plan,
            &mut self.lane_label,
        )
    }
}

/// One payload's three buffers, allocated once. A zero length is an error for
/// the driver, so a part with no sends or no halo gets one element it is
/// never sent, received or copied - a part with no neighbours, `P = 1` or an
/// island, is a real case.
fn lane<T: DeviceRepr + ValidAsZeroBits + Default + Clone>(
    gpu: &Gpu,
    n_send: usize,
    n_halo: usize,
) -> Result<Lane<T>> {
    let dev = gpu.zeros::<T>(n_send.max(1))?;
    // SAFETY: cudarc offers no safe route to page-locked host memory, so this
    // is unsafe for the one reason `alloc_pinned` is - the memory arrives
    // uninitialised - and the very next statement overwrites every element
    // with zero. `alloc_pinned` sets the write-combined flag: cudarc 0.19.9,
    // which `Cargo.lock` pins, offers only `alloc_pinned`. Write-combined
    // memory is slow for the host to read, and the host reads the send buffer
    // once, front to back, in step 3: that costs time and changes no byte,
    // and the send buffer moves to `alloc_pinned_with_flags` when the lock
    // file reaches cudarc 0.19.10. The slice is private to this `Lane` and
    // never aliased; `PinnedHostSlice` frees it in its own `Drop`.
    let mut send = unsafe { gpu.ctx().alloc_pinned::<T>(n_send.max(1))? };
    send.as_mut_slice()?.fill(T::default());
    let mut recv = unsafe { gpu.ctx().alloc_pinned::<T>(n_halo.max(1))? };
    recv.as_mut_slice()?.fill(T::default());
    Ok(Lane { dev, send, recv })
}

/// The bytes of a payload slice: its in-memory representation, native
/// endianness, `size_of::<T>()` each (SPEC-LIT §138.3).
fn bytes<T>(s: &[T]) -> &[u8] {
    // SAFETY: the four payloads are `#[repr(C)]` with no padding -
    // `types::tests::layout_matches_device` pins that - so the bytes ARE the
    // values, step 2 wrote them, and step 3 is the identity on them; `u8` has
    // alignment 1, so any byte count the slice's own length gives is fine.
    unsafe {
        std::slice::from_raw_parts(s.as_ptr().cast::<u8>(), std::mem::size_of_val(s))
    }
}

/// The same view, writable, for the receive side.
fn bytes_mut<T>(s: &mut [T]) -> &mut [u8] {
    // SAFETY: as `bytes`; a byte write through this view lands in the same
    // payload element, which the `Comm` contract fills before step 4 reads it.
    unsafe {
        std::slice::from_raw_parts_mut(
            s.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(s),
        )
    }
}

/// One exchange of one payload: the five steps of SPEC-LIT §138.3. One
/// implementation for all four payloads, because the exchange is a property
/// of the plan and not of what is being moved.
fn exchange<T: DeviceRepr + ValidAsZeroBits>(
    gpu: &Gpu,
    comm: &dyn Comm,
    field: &mut DevBuf<T>,
    pack: &CudaFunction,
    plan: &Plan,
    ln: &mut Lane<T>,
) -> Result<()> {
    // ---- the checks of §138.3's table, rows 4-7 in order, BEFORE anything
    // is launched or sent ----------------------------------------------------
    if gpu.is_capturing() {
        return Err(Error::Config(
            "halo: an SPMD halo exchange cannot run inside a CUDA-graph \
             capture: it copies the packed cells to the host, hands them to \
             a Comm and copies the halo back, synchronising the stream twice \
             - host round trips, and there is nothing to record. Alternative: \
             capture the device work BETWEEN two exchanges, or, in one \
             process, use HaloExchange, whose pack and copy are device-only \
             (SPEC-LIT 81.3, 138.3)"
                .to_string(),
        ));
    }
    let size = comm.size();
    if size != plan.n_ranks {
        return Err(Error::Config(format!(
            "halo: the world has {size} ranks, but the plan was built for \
             {} ranks",
            plan.n_ranks
        )));
    }
    let rank = comm.rank();
    if rank != plan.part {
        return Err(Error::Config(format!(
            "halo: this handle is rank {rank}, but the plan belongs to \
             part {}",
            plan.part
        )));
    }
    let n_local = plan.n_cells + plan.n_halo;
    if field.len() < n_local {
        return Err(Error::Config(format!(
            "halo: rank {rank}'s field buffer holds {} elements, but the \
             part has {} owned cells and {} halo cells and every field on it \
             must be {n_local} long",
            field.len(),
            plan.n_cells,
            plan.n_halo
        )));
    }

    // ---- 1. pack -----------------------------------------------------------
    // §71.4's kernel, unchanged: sendBuf[k] = psi[sendIndex[k]], a gather.
    if plan.n_send > 0 {
        let ns = plan.n_send as Label;
        let nc = plan.n_cells as Label;
        unsafe {
            gpu.stream()
                .launch_builder(pack)
                .arg(&mut ln.dev)
                .arg(&*field)
                .arg(&plan.send_index)
                .arg(&ns)
                .arg(&nc)
                .launch(cfg_for(plan.n_send))?;
        }
    }

    // ---- 2. device -> host, then the stream is synchronised ----------------
    // THE TRAP (§138.3, "Why two synchronisations"): this crate turns cudarc's
    // event tracking off, so nothing waits for this copy on its own. The host
    // must not read the bytes before it ends.
    if plan.n_send > 0 {
        let src = ln.dev.slice(0..plan.n_send);
        gpu.stream().memcpy_dtoh(&src, &mut ln.send)?;
        gpu.sync()?;
    }

    // ---- 3. the Comm, every neighbour in both lists, empty included --------
    // No handshake: what r sends to s is as long as what s expects from r,
    // because both lists are built from the same sorted halo list. The
    // offsets count ELEMENTS; the Comm moves BYTES, so every slice is
    // elements times the payload's size.
    let sz = std::mem::size_of::<T>();
    let mut sends: Vec<(usize, &[u8])> = Vec::with_capacity(plan.nbr.len());
    {
        let all = bytes(ln.send.as_slice()?);
        for (i, &q) in plan.nbr.iter().enumerate() {
            let (a, b) = (plan.send_offset[i] * sz, plan.send_offset[i + 1] * sz);
            sends.push((q as usize, &all[a..b]));
        }
    }
    let mut recvs: Vec<(usize, &mut [u8])> = Vec::with_capacity(plan.nbr.len());
    {
        // `rest` advances neighbour by neighbour, so each split is taken at
        // the offset RELATIVE to where the previous one ended.
        let mut done = 0usize;
        let mut rest = bytes_mut(ln.recv.as_mut_slice()?);
        for (i, &q) in plan.nbr.iter().enumerate() {
            let (a, b) = (plan.recv_offset[i] * sz, plan.recv_offset[i + 1] * sz);
            let (_, tail) = rest.split_at_mut(a - done);
            let (mid, tail2) = tail.split_at_mut(b - a);
            recvs.push((q as usize, mid));
            rest = tail2;
            done = b;
        }
    }
    comm.exchange(HALO_TAG, &sends, &mut recvs)?;

    // ---- 4. host -> halo, ONE copy; then ---- 5. synchronised --------------
    // The halo is ordered by (owning part, global cell id), so the receive
    // buffer IS the halo and the copy lands directly in place: there is no
    // unpack. The second wait, because the copy reads the host buffer
    // asynchronously and the next exchange must not overwrite it first
    // (§138.3, "Why two synchronisations").
    if plan.n_halo > 0 {
        let src = ln.recv.as_slice()?;
        let mut dst = field.slice_mut(plan.n_cells..plan.n_cells + plan.n_halo);
        gpu.stream().memcpy_htod(src, &mut dst)?;
        gpu.sync()?;
    }
    Ok(())
}

// ==========================================================================
//  Tests - the library half of §138.3's gate
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comm::ThreadComm;
    use crate::decompose::tests::{boxes, round_robin};
    use crate::decompose::{Decomposition, PartitionMethod};
    use crate::halo::HaloExchange;

    /// Every device test needs a card. Returning `None` makes the test pass
    /// vacuously on a machine without one, the crate's convention.
    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    /// A deterministic field: a function of the GLOBAL cell id, so splitting
    /// it is a permutation and not a computation.
    fn field(c: usize) -> Scalar {
        0.25 + 0.0625 * ((c * 37) % 23) as Scalar
    }

    /// The nine components, row-major, of the test tensor at cell `c`:
    /// component `k` is `(k+1) * field(c)`, negated when `k` is odd.
    fn tensor_at(c: usize) -> Tensor {
        let s = field(c);
        let v: [Scalar; 9] = std::array::from_fn(|k| {
            let x = (k + 1) as Scalar * s;
            if k % 2 == 1 {
                -x
            } else {
                x
            }
        });
        Tensor {
            xx: v[0], xy: v[1], xz: v[2],
            yx: v[3], yy: v[4], yz: v[5],
            zx: v[6], zy: v[7], zz: v[8],
        }
    }

    fn comps(t: Tensor) -> [Scalar; 9] {
        [
            t.xx, t.xy, t.xz, t.yx, t.yy, t.yz, t.zx, t.zy, t.zz,
        ]
    }

    /// The local buffer of one part: its owned cells from the global field,
    /// its halo ZERO - no value above is zero, so a forgotten exchange
    /// cannot pass.
    fn local_of<T: Copy>(part: &PartMesh, global: &[T], zero: T) -> Vec<T> {
        let mut h = vec![zero; part.n_local()];
        for (i, hi) in h.iter_mut().enumerate().take(part.mesh.n_cells) {
            *hi = global[part.global_cell[i] as usize];
        }
        h
    }

    /// SPEC-LIT §138.3's library gate: a cyclic box, `P = 2..4`, three
    /// partitioners, `Scalar` and `Tensor`. Every local cell of every rank is
    /// bit for bit the owner's value AND the in-process `HaloExchange`
    /// result, and at least one of the nine cuts splits a cyclic couple -
    /// which puts a cell in a halo through the couple alone.
    #[test]
    fn an_spmd_halo_crosses_a_cut_cyclic_couple() {
        let Some(gpu0) = gpu() else { return };
        let m = boxes([5, 4, 3], true);
        let scalars: Vec<Scalar> = (0..m.n_cells).map(field).collect();
        let tensors: Vec<Tensor> = (0..m.n_cells).map(tensor_at).collect();
        let mut any_couple = false;
        for np in 2..=4usize {
            for method in [
                PartitionMethod::Hilbert,
                PartitionMethod::Linear,
                round_robin(m.n_cells, np),
            ] {
                let d = Decomposition::build(&m, np, &method).expect("cut");
                any_couple |= d.n_cut_couples > 0;

                // The in-process reference, all four ghost halves filled on
                // the main thread's card.
                let mut ex = HaloExchange::new(&gpu0, &d).expect("exchange");
                let mut sf: Vec<DevBuf<Scalar>> = d
                    .parts
                    .iter()
                    .map(|p| gpu0.upload(&local_of(p, &scalars, 0.0)).expect("upload"))
                    .collect();
                let mut tf: Vec<DevBuf<Tensor>> = d
                    .parts
                    .iter()
                    .map(|p| {
                        gpu0.upload(&local_of(p, &tensors, Tensor::ZERO))
                            .expect("upload")
                    })
                    .collect();
                ex.scalar(&gpu0, &mut sf).expect("exchange");
                ex.tensor(&gpu0, &mut tf).expect("exchange");
                gpu0.sync().expect("sync");
                let ref_s: Vec<Vec<Scalar>> = sf
                    .iter()
                    .map(|b| gpu0.download(b).expect("download"))
                    .collect();
                let ref_t: Vec<Vec<Tensor>> = tf
                    .iter()
                    .map(|b| gpu0.download(b).expect("download"))
                    .collect();

                // The SPMD run: one thread per rank, each with its own card
                // handle on device 0. A rank's error must fail the test with
                // its text, so the closure returns `Result`.
                let got = ThreadComm::run(np, |comm| -> crate::Result<(Vec<Scalar>, Vec<Tensor>)> {
                    let gpu = Gpu::new(0)?;
                    let part = &d.parts[comm.rank()];
                    let mut halo = SpmdHalo::new(&gpu, part, np)?;
                    let mut s = gpu.upload(&local_of(part, &scalars, 0.0))?;
                    let mut t = gpu.upload(&local_of(part, &tensors, Tensor::ZERO))?;
                    halo.scalar(&gpu, comm, &mut s)?;
                    halo.tensor(&gpu, comm, &mut t)?;
                    gpu.sync()?;
                    Ok((gpu.download(&s)?, gpu.download(&t)?))
                })
                .expect("spmd run");

                for (r, part) in d.parts.iter().enumerate() {
                    let (gs, gt) = got[r]
                        .as_ref()
                        .unwrap_or_else(|e| panic!("rank {r} failed: {e}"));
                    for i in 0..part.n_local() {
                        let g = part.global_cell[i] as usize;
                        let at = format!(
                            "P={np} {} rank {r} local {i} (global {g})",
                            method.as_str()
                        );
                        assert_eq!(
                            gs[i].to_bits(),
                            scalars[g].to_bits(),
                            "scalar vs owner, {at}"
                        );
                        assert_eq!(
                            gs[i].to_bits(),
                            ref_s[r][i].to_bits(),
                            "scalar vs in-process, {at}"
                        );
                        let (a, b) = (comps(gt[i]), comps(tensor_at(g)));
                        let w = comps(ref_t[r][i]);
                        for k in 0..9 {
                            assert_eq!(
                                a[k].to_bits(),
                                b[k].to_bits(),
                                "tensor component {k} vs owner, {at}"
                            );
                            assert_eq!(
                                a[k].to_bits(),
                                w[k].to_bits(),
                                "tensor component {k} vs in-process, {at}"
                            );
                        }
                    }
                }
            }
        }
        assert!(
            any_couple,
            "no cut of the nine split a cyclic couple - the couple-only \
             halo case went untested"
        );
    }
}

