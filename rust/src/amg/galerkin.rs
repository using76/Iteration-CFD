// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The Galerkin coarse operator and the transfers - AMG-02.
//!
//! The gather lists of one coarsening - the CSR of source codes (141.17)
//! over the coarse slots - their host twin, and the device driver of the
//! four kernels of `cuda/amg.cu`: the product (141.14)-(141.16), the
//! restriction (141.13) and the two prolongations (141.11)/(141.12).
//! Every list is summed in its list order, in an f64 accumulator, and
//! rounded once to `Scalar`, so the device answer is the host twin's bit
//! for bit in both builds (SPEC-LIT §141.2).
//!
//! Provenance: ORIGINAL - written from Notay, ETNA 37 (2010) 123-146,
//! eqs. (2.1)-(2.3) (SPEC-LIT §141.2). No GPL-licensed source was consulted.

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::amg::aggregate::{Aggregation, LevelView, INTERIOR, UNCOUPLED};
use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::ldu::GpuLduMatrix;
use crate::mesh::HostMesh;
use crate::{Label, Scalar};

/// The kinds of the source code (141.17): `code = 4 * index + kind`.
pub const KIND_DIAG: Label = 0;
/// The internal face's `upper` entry.
pub const KIND_UPPER: Label = 1;
/// The internal face's `lower` entry.
pub const KIND_LOWER: Label = 2;
/// The coupled boundary coefficient, entering negated per (141.2).
pub const KIND_BOUNDARY: Label = 3;

/// The largest index (141.17) can carry, exclusive: 2^29.
pub const MAX_INDEX: usize = 1 << 29;

/// A level with `2^29` or more cells, faces or boundary faces does not fit
/// the source code of (141.17) and is refused by name (SPEC-LIT §141.2).
fn check_fits(n_fine: usize, n_faces: usize, n_bfaces: usize) -> Result<()> {
    for (name, n) in [
        ("cells", n_fine),
        ("faces", n_faces),
        ("boundary faces", n_bfaces),
    ] {
        if n >= MAX_INDEX {
            return Err(Error::Mesh(format!(
                "amg::galerkin: the level has {} {}, which does not fit the \
                 source code (141.17): the largest index it can carry is \
                 2^29 - 1, so a level of 2^29 or more is refused",
                n, name
            )));
        }
    }
    Ok(())
}

/// The gather lists of one coarsening (host). Built once per aggregation;
/// the coefficients are re-gathered from the lists whenever the matrix
/// changes (SPEC-LIT §141.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GalerkinPlan {
    pub n_fine: usize,
    pub n_fine_faces: usize,
    pub n_fine_bfaces: usize,
    pub n_coarse: usize,
    pub n_coarse_faces: usize,
    /// Fine cell -> aggregate, as the aggregation numbered it.
    pub cell_agg: Vec<Label>,
    /// CSR offsets of the members, `[n_coarse + 1]`.
    pub agg_offset: Vec<Label>,
    /// Members, ascending within each aggregate.
    pub agg_cells: Vec<Label>,
    /// `= min(I, J)`, ascending by `(owner, neighbour)`.
    pub coarse_owner: Vec<Label>,
    /// `= max(I, J)`, ascending by `(owner, neighbour)`.
    pub coarse_neighbour: Vec<Label>,
    /// `[n_coarse + 2 n_coarse_faces + 1]`: diagonals, uppers, lowers.
    pub slot_offset: Vec<Label>,
    /// Codes of (141.17), in the list order of §141.2.
    pub slot_src: Vec<Label>,
}

impl GalerkinPlan {
    /// The lists of `agg`, whose `owner` are the level's face owners and
    /// whose `b_face_cells` are its boundary-face cells (empty on a coarse
    /// level). Lengths are checked against `agg` (SPEC-LIT §141.2).
    pub fn new(agg: &Aggregation, owner: &[Label], b_face_cells: &[Label]) -> Result<Self> {
        check_fits(agg.n_fine, agg.face_coarse.len(), agg.bface_coarse.len())?;
        if owner.len() != agg.face_coarse.len() {
            return Err(Error::Mesh(format!(
                "amg::galerkin: length mismatch: owner has {} entries, the \
                 aggregation has {} internal faces",
                owner.len(),
                agg.face_coarse.len()
            )));
        }
        if b_face_cells.len() != agg.bface_coarse.len() {
            return Err(Error::Mesh(format!(
                "amg::galerkin: length mismatch: b_face_cells has {} entries, \
                 the aggregation has {} boundary faces",
                b_face_cells.len(),
                agg.bface_coarse.len()
            )));
        }
        for (f, &o) in owner.iter().enumerate() {
            if o < 0 || o as usize >= agg.n_fine {
                return Err(Error::Mesh(format!(
                    "amg::galerkin: owner[{}] = {} out of 0..{}",
                    f, o, agg.n_fine
                )));
            }
        }
        for (bf, &c) in b_face_cells.iter().enumerate() {
            if c < 0 || c as usize >= agg.n_fine {
                return Err(Error::Mesh(format!(
                    "amg::galerkin: b_face_cells[{}] = {} out of 0..{}",
                    bf, c, agg.n_fine
                )));
            }
        }
        Self::build(agg, owner, b_face_cells)
    }

    /// The lists, in the order §141.2 states: a diagonal slot reads its
    /// members' `diag[c]` in ascending `c`, then its INTERIOR internal
    /// faces in ascending `f` (each `upper[f]` then `lower[f]`), then its
    /// INTERIOR coupled faces in ascending `bf`; an upper or lower slot
    /// reads its faces in ascending `f`, then its coupled faces in
    /// ascending `bf`. Uncoupled boundary faces appear nowhere, so every
    /// fine entry lands in exactly one list, exactly once.
    fn build(agg: &Aggregation, owner: &[Label], b_face_cells: &[Label]) -> Result<Self> {
        let (nc, ncf) = (agg.n_coarse, agg.n_coarse_faces());
        let n_slots = nc + 2 * ncf;
        let mut lists: Vec<Vec<Label>> = (0..n_slots).map(|_| Vec::new()).collect();
        for g in 0..nc {
            let l = &mut lists[g];
            for k in agg.agg_offset[g]..agg.agg_offset[g + 1] {
                l.push(4 * agg.agg_cells[k as usize] + KIND_DIAG);
            }
        }
        for (f, &fc) in agg.face_coarse.iter().enumerate() {
            let io = agg.cell_agg[owner[f] as usize];
            if fc == INTERIOR {
                lists[io as usize].push(4 * f as Label + KIND_UPPER);
                lists[io as usize].push(4 * f as Label + KIND_LOWER);
            } else {
                let fu = if agg.face_flip[f] { KIND_LOWER } else { KIND_UPPER };
                let fl = if agg.face_flip[f] { KIND_UPPER } else { KIND_LOWER };
                lists[nc + fc as usize].push(4 * f as Label + fu);
                lists[nc + ncf + fc as usize].push(4 * f as Label + fl);
            }
        }
        for (bf, &bc) in agg.bface_coarse.iter().enumerate() {
            if bc == UNCOUPLED {
                continue;
            }
            let code = 4 * bf as Label + KIND_BOUNDARY;
            if bc == INTERIOR {
                let i = agg.cell_agg[b_face_cells[bf] as usize];
                lists[i as usize].push(code);
            } else if agg.bface_flip[bf] {
                lists[nc + ncf + bc as usize].push(code);
            } else {
                lists[nc + bc as usize].push(code);
            }
        }
        let mut slot_offset = vec![0 as Label; n_slots + 1];
        for s in 0..n_slots {
            slot_offset[s + 1] = slot_offset[s] + lists[s].len() as Label;
        }
        let slot_src: Vec<Label> = lists.into_iter().flatten().collect();
        Ok(Self {
            n_fine: agg.n_fine,
            n_fine_faces: agg.face_coarse.len(),
            n_fine_bfaces: agg.bface_coarse.len(),
            n_coarse: nc,
            n_coarse_faces: ncf,
            cell_agg: agg.cell_agg.clone(),
            agg_offset: agg.agg_offset.clone(),
            agg_cells: agg.agg_cells.clone(),
            coarse_owner: agg.coarse_owner.clone(),
            coarse_neighbour: agg.coarse_neighbour.clone(),
            slot_offset,
            slot_src,
        })
    }

    /// The finest level: `owner` and `b_face_cells` straight from the mesh.
    pub fn for_mesh(agg: &Aggregation, m: &HostMesh) -> Result<Self> {
        Self::new(agg, &m.owner, &m.b_face_cells)
    }

    /// `n_coarse + 2 n_coarse_faces`, the number of slots.
    pub fn n_slots(&self) -> usize {
        self.n_coarse + 2 * self.n_coarse_faces
    }

    /// Whether any coupled boundary face reaches the coarse off-diagonals
    /// or diagonals, so the gather needs `boundary_coeffs` at all.
    pub fn has_boundary_sources(&self) -> bool {
        self.slot_src.iter().any(|&c| c % 4 == KIND_BOUNDARY)
    }

    /// Host twin of `amgGalerkin` (SPEC-LIT §141.2): each slot's list is
    /// summed in list order into an f64 accumulator and rounded once to
    /// `Scalar`. With boundary sources, `boundary_coeffs` is exactly
    /// `n_fine_bfaces` long; without them, it is empty or exactly
    /// `n_fine_bfaces` long - the matrix's own array, whose uncoupled
    /// entries nothing reads - and any other length is refused.
    pub fn coarse_host(
        &self,
        diag: &[Scalar],
        upper: &[Scalar],
        lower: &[Scalar],
        boundary_coeffs: &[Scalar],
    ) -> Result<HostLevel> {
        if diag.len() != self.n_fine {
            return Err(Error::Config(format!(
                "amg::galerkin: length mismatch: diag has {} entries, the \
                 level has {} cells",
                diag.len(),
                self.n_fine
            )));
        }
        if upper.len() != self.n_fine_faces {
            return Err(Error::Config(format!(
                "amg::galerkin: length mismatch: upper has {} entries, the \
                 level has {} faces",
                upper.len(),
                self.n_fine_faces
            )));
        }
        if lower.len() != self.n_fine_faces {
            return Err(Error::Config(format!(
                "amg::galerkin: length mismatch: lower has {} entries, the \
                 level has {} faces",
                lower.len(),
                self.n_fine_faces
            )));
        }
        if self.has_boundary_sources() {
            if boundary_coeffs.len() != self.n_fine_bfaces {
                return Err(Error::Config(format!(
                    "amg::galerkin: length mismatch: boundary_coeffs has {} \
                     entries, the level has {} boundary faces, and the plan \
                     has boundary sources for it",
                    boundary_coeffs.len(),
                    self.n_fine_bfaces
                )));
            }
        } else if boundary_coeffs.len() != self.n_fine_bfaces
            && !boundary_coeffs.is_empty()
        {
            return Err(Error::Config(format!(
                "amg::galerkin: length mismatch: boundary_coeffs has {} \
                 entries, the level has {} boundary faces, and the plan has \
                 no boundary source for any of them, so it must be empty or \
                 exactly their number",
                boundary_coeffs.len(),
                self.n_fine_bfaces
            )));
        }
        Ok(self.gather(diag, upper, lower, boundary_coeffs))
    }

    /// The sums themselves: one list in list order, one f64 accumulator,
    /// one cast - the exact arithmetic the kernel runs.
    fn gather(
        &self,
        diag: &[Scalar],
        upper: &[Scalar],
        lower: &[Scalar],
        b: &[Scalar],
    ) -> HostLevel {
        let mut hd = vec![0 as Scalar; self.n_coarse];
        let mut hu = vec![0 as Scalar; self.n_coarse_faces];
        let mut hl = vec![0 as Scalar; self.n_coarse_faces];
        for s in 0..self.n_slots() {
            let mut acc = 0.0f64;
            let lo = self.slot_offset[s] as usize;
            let hi = self.slot_offset[s + 1] as usize;
            for k in lo..hi {
                let code = self.slot_src[k];
                let idx = (code / 4) as usize;
                acc += match code % 4 {
                    KIND_DIAG => diag[idx] as f64,
                    KIND_UPPER => upper[idx] as f64,
                    KIND_LOWER => lower[idx] as f64,
                    _ => -(b[idx] as f64),
                };
            }
            let v = acc as Scalar;
            if s < self.n_coarse {
                hd[s] = v;
            } else if s < self.n_coarse + self.n_coarse_faces {
                hu[s - self.n_coarse] = v;
            } else {
                hl[s - self.n_coarse - self.n_coarse_faces] = v;
            }
        }
        HostLevel {
            n_cells: self.n_coarse,
            owner: self.coarse_owner.clone(),
            neighbour: self.coarse_neighbour.clone(),
            diag: hd,
            upper: hu,
            lower: hl,
        }
    }

    /// Host twin of `amgRestrict`, (141.13): the segmented sum over the
    /// aggregate-sorted members, f64, one rounding.
    pub fn restrict_host(&self, rf: &[Scalar]) -> Result<Vec<Scalar>> {
        if rf.len() != self.n_fine {
            return Err(Error::Config(format!(
                "amg::galerkin: length mismatch: rf has {} entries, the \
                 level has {} cells",
                rf.len(),
                self.n_fine
            )));
        }
        let mut rc = vec![0 as Scalar; self.n_coarse];
        for g in 0..self.n_coarse {
            let mut acc = 0.0f64;
            let lo = self.agg_offset[g] as usize;
            let hi = self.agg_offset[g + 1] as usize;
            for k in lo..hi {
                acc += rf[self.agg_cells[k] as usize] as f64;
            }
            rc[g] = acc as Scalar;
        }
        Ok(rc)
    }

    /// Host twin of `amgProlong`, (141.11): `(P e_c)_c = e_c[agg(c)]`.
    pub fn prolong_host(&self, ec: &[Scalar]) -> Result<Vec<Scalar>> {
        if ec.len() != self.n_coarse {
            return Err(Error::Config(format!(
                "amg::galerkin: length mismatch: ec has {} entries, the \
                 coarse level has {} cells",
                ec.len(),
                self.n_coarse
            )));
        }
        Ok(self.cell_agg.iter().map(|&g| ec[g as usize]).collect())
    }

    /// Host twin of `amgProlongAdd`, (141.12): `x_c += e_c[agg(c)]`, the
    /// add in `Scalar`, as the kernel does.
    pub fn prolong_add_host(&self, ec: &[Scalar], xf: &mut [Scalar]) -> Result<()> {
        if ec.len() != self.n_coarse {
            return Err(Error::Config(format!(
                "amg::galerkin: length mismatch: ec has {} entries, the \
                 coarse level has {} cells",
                ec.len(),
                self.n_coarse
            )));
        }
        if xf.len() != self.n_fine {
            return Err(Error::Config(format!(
                "amg::galerkin: length mismatch: xf has {} entries, the \
                 level has {} cells",
                xf.len(),
                self.n_fine
            )));
        }
        for (c, g) in self.cell_agg.iter().enumerate() {
            xf[c] += ec[*g as usize];
        }
        Ok(())
    }
}

/// One coarse level on the host: LDU arrays, no boundary faces - every
/// coupled face of the finest level was folded into it (SPEC-LIT §141.1).
#[derive(Debug, Clone, PartialEq)]
pub struct HostLevel {
    pub n_cells: usize,
    pub owner: Vec<Label>,
    pub neighbour: Vec<Label>,
    pub diag: Vec<Scalar>,
    pub upper: Vec<Scalar>,
    pub lower: Vec<Scalar>,
}

impl HostLevel {
    pub fn n_faces(&self) -> usize {
        self.owner.len()
    }

    /// The level as [`LevelView`] with empty boundary slices, which is what
    /// `aggregate_level` reads to coarsen it again.
    pub fn view(&self) -> LevelView<'_> {
        LevelView {
            n_cells: self.n_cells,
            owner: &self.owner,
            neighbour: &self.neighbour,
            diag: &self.diag,
            upper: &self.upper,
            lower: &self.lower,
            b_face_cells: &[],
            b_nbr_cell: &[],
            boundary_coeffs: &[],
        }
    }

    /// The level on the device. Buffers are `max(len, 1)` long (a
    /// zero-length device buffer is not allocated in this crate);
    /// `n_cells`/`n_faces` are the truth.
    pub fn upload(&self, gpu: &Gpu) -> Result<GpuLevel> {
        let lab1 = [0 as Label];
        let sc1 = [0 as Scalar];
        Ok(GpuLevel {
            n_cells: self.n_cells,
            n_faces: self.n_faces(),
            owner: gpu.upload(if self.owner.is_empty() { &lab1 } else { &self.owner })?,
            neighbour: gpu.upload(if self.neighbour.is_empty() {
                &lab1
            } else {
                &self.neighbour
            })?,
            diag: gpu.upload(if self.diag.is_empty() { &sc1 } else { &self.diag })?,
            upper: gpu.upload(if self.upper.is_empty() { &sc1 } else { &self.upper })?,
            lower: gpu.upload(if self.lower.is_empty() { &sc1 } else { &self.lower })?,
        })
    }
}

/// The same level on the device. Buffers are allocated `max(len, 1)` long
/// (a zero-length device buffer is not allocated in this crate);
/// `n_cells`/`n_faces` are the truth.
pub struct GpuLevel {
    pub n_cells: usize,
    pub n_faces: usize,
    pub owner: DevBuf<Label>,
    pub neighbour: DevBuf<Label>,
    pub diag: DevBuf<Scalar>,
    pub upper: DevBuf<Scalar>,
    pub lower: DevBuf<Scalar>,
}

impl GpuLevel {
    /// The device bytes of the level (SPEC-LIT (141.47)).
    pub fn device_bytes(&self) -> crate::amg::single::DeviceBytes {
        use crate::amg::single::DeviceBytes as B;
        B::of_label(&self.owner)
            + B::of_label(&self.neighbour)
            + B::of_float(&self.diag)
            + B::of_float(&self.upper)
            + B::of_float(&self.lower)
    }

    /// The level back on the host, exactly `n` entries even when a buffer
    /// is the padded one.
    pub fn download(&self, gpu: &Gpu) -> Result<HostLevel> {
        let mut owner = gpu.download(&self.owner)?;
        let mut neighbour = gpu.download(&self.neighbour)?;
        let mut diag = gpu.download(&self.diag)?;
        let mut upper = gpu.download(&self.upper)?;
        let mut lower = gpu.download(&self.lower)?;
        owner.truncate(self.n_faces);
        neighbour.truncate(self.n_faces);
        diag.truncate(self.n_cells);
        upper.truncate(self.n_faces);
        lower.truncate(self.n_faces);
        Ok(HostLevel {
            n_cells: self.n_cells,
            owner,
            neighbour,
            diag,
            upper,
            lower,
        })
    }
}

/// The lists on the device plus the four resolved kernels (SPEC-LIT
/// §141.2). Holds the plan's counts; the lists themselves are built once
/// per hierarchy and the coefficients are re-gathered on the device.
pub struct GpuGalerkin {
    slot_offset: DevBuf<Label>,
    slot_src: DevBuf<Label>,
    agg_offset: DevBuf<Label>,
    agg_cells: DevBuf<Label>,
    cell_agg: DevBuf<Label>,
    owner: DevBuf<Label>,
    neighbour: DevBuf<Label>,
    galerkin: CudaFunction,
    restrict: CudaFunction,
    prolong: CudaFunction,
    prolong_add: CudaFunction,
    n_fine: usize,
    n_fine_faces: usize,
    n_fine_bfaces: usize,
    n_coarse: usize,
    n_coarse_faces: usize,
    n_slots: usize,
    has_boundary_sources: bool,
}

impl GpuGalerkin {
    /// Upload the lists and resolve the four entry points of
    /// `cuda/amg.cu`, once.
    pub fn new(gpu: &Gpu, plan: &GalerkinPlan) -> Result<Self> {
        let lab1 = [0 as Label];
        let k = KernelSet::new(gpu, crate::kernels::AMG)?;
        Ok(Self {
            slot_offset: gpu.upload(if plan.slot_offset.is_empty() {
                &lab1
            } else {
                &plan.slot_offset
            })?,
            slot_src: gpu.upload(if plan.slot_src.is_empty() {
                &lab1
            } else {
                &plan.slot_src
            })?,
            agg_offset: gpu.upload(if plan.agg_offset.is_empty() {
                &lab1
            } else {
                &plan.agg_offset
            })?,
            agg_cells: gpu.upload(if plan.agg_cells.is_empty() {
                &lab1
            } else {
                &plan.agg_cells
            })?,
            cell_agg: gpu.upload(if plan.cell_agg.is_empty() {
                &lab1
            } else {
                &plan.cell_agg
            })?,
            owner: gpu.upload(if plan.coarse_owner.is_empty() {
                &lab1
            } else {
                &plan.coarse_owner
            })?,
            neighbour: gpu.upload(if plan.coarse_neighbour.is_empty() {
                &lab1
            } else {
                &plan.coarse_neighbour
            })?,
            galerkin: k.func("amgGalerkin")?,
            restrict: k.func("amgRestrict")?,
            prolong: k.func("amgProlong")?,
            prolong_add: k.func("amgProlongAdd")?,
            n_fine: plan.n_fine,
            n_fine_faces: plan.n_fine_faces,
            n_fine_bfaces: plan.n_fine_bfaces,
            n_coarse: plan.n_coarse,
            n_coarse_faces: plan.n_coarse_faces,
            n_slots: plan.n_slots(),
            has_boundary_sources: plan.has_boundary_sources(),
        })
    }

    /// The device bytes of the lists (SPEC-LIT (141.47)).
    pub fn device_bytes(&self) -> crate::amg::single::DeviceBytes {
        use crate::amg::single::DeviceBytes as B;
        B::of_label(&self.slot_offset)
            + B::of_label(&self.slot_src)
            + B::of_label(&self.agg_offset)
            + B::of_label(&self.agg_cells)
            + B::of_label(&self.cell_agg)
            + B::of_label(&self.owner)
            + B::of_label(&self.neighbour)
    }

    /// A device buffer is at least the level's length (a field may be
    /// halo-extended); short is refused by name.
    fn check_len(name: &str, got: usize, want: usize) -> Result<()> {
        if got < want {
            return Err(Error::Config(format!(
                "amg::galerkin: length mismatch: {name} has {got} entries, \
                 the level needs {want}"
            )));
        }
        Ok(())
    }

    /// A coarse level with the plan's addressing and zero coefficients.
    /// Setup: it allocates - the addressing is copied device-to-device and
    /// the coefficients are zeroed - so it can fail and it runs outside a
    /// capture (SPEC-LIT §81.3).
    pub fn new_level(&self, gpu: &Gpu) -> Result<GpuLevel> {
        let n1 = self.n_coarse.max(1);
        let nf1 = self.n_coarse_faces.max(1);
        Ok(GpuLevel {
            n_cells: self.n_coarse,
            n_faces: self.n_coarse_faces,
            owner: self.owner.try_clone()?,
            neighbour: self.neighbour.try_clone()?,
            diag: gpu.zeros(n1)?,
            upper: gpu.zeros(nf1)?,
            lower: gpu.zeros(nf1)?,
        })
    }

    /// `amgGalerkin`: gather every slot's list on the device. Only a launch
    /// - no allocation, no copy, no sync, so it captures into a graph
    /// (SPEC-LIT §141.2). `boundary_coeffs = None` only if the plan has no
    /// boundary source.
    pub fn coarse(
        &self,
        gpu: &Gpu,
        diag: &DevBuf<Scalar>,
        upper: &DevBuf<Scalar>,
        lower: &DevBuf<Scalar>,
        boundary_coeffs: Option<&DevBuf<Scalar>>,
        out: &mut GpuLevel,
    ) -> Result<()> {
        Self::check_len("diag", diag.len(), self.n_fine)?;
        Self::check_len("upper", upper.len(), self.n_fine_faces)?;
        Self::check_len("lower", lower.len(), self.n_fine_faces)?;
        let b = match boundary_coeffs {
            Some(b) => b,
            None if !self.has_boundary_sources => diag,
            None => {
                return Err(Error::Config(
                    "amg::galerkin: the plan has boundary sources, so \
                     boundary_coeffs is required - pass the coupled faces' \
                     coefficients or None is refused"
                        .to_string(),
                ));
            }
        };
        if self.has_boundary_sources {
            Self::check_len("boundary_coeffs", b.len(), self.n_fine_bfaces)?;
        }
        if out.n_cells != self.n_coarse || out.n_faces != self.n_coarse_faces {
            return Err(Error::Mesh(format!(
                "amg::galerkin: the coarse level is {}/{} cells/faces but \
                 the plan's is {}/{}",
                out.n_cells, out.n_faces, self.n_coarse, self.n_coarse_faces
            )));
        }
        Self::check_len("coarse diag", out.diag.len(), self.n_coarse)?;
        Self::check_len("coarse upper", out.upper.len(), self.n_coarse_faces)?;
        Self::check_len("coarse lower", out.lower.len(), self.n_coarse_faces)?;
        if self.n_slots == 0 {
            return Ok(());
        }
        let nl = self.n_coarse as Label;
        let nfl = self.n_coarse_faces as Label;
        let f = self.galerkin.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut out.diag)
                .arg(&mut out.upper)
                .arg(&mut out.lower)
                .arg(diag)
                .arg(upper)
                .arg(lower)
                .arg(b)
                .arg(&self.slot_offset)
                .arg(&self.slot_src)
                .arg(&nl)
                .arg(&nfl)
                .launch(cfg_for(self.n_slots))?;
        }
        Ok(())
    }

    /// The product from a fine matrix: its `diag`/`upper`/`lower` and its
    /// coupled `boundary_coeffs`.
    pub fn coarse_of_matrix(
        &self,
        gpu: &Gpu,
        a: &GpuLduMatrix,
        out: &mut GpuLevel,
    ) -> Result<()> {
        self.coarse(
            gpu,
            &a.diag,
            &a.upper,
            &a.lower,
            Some(&a.boundary_coeffs),
            out,
        )
    }

    /// The product from a fine level: no boundary faces to fold.
    pub fn coarse_of_level(&self, gpu: &Gpu, fine: &GpuLevel, out: &mut GpuLevel) -> Result<()> {
        self.coarse(gpu, &fine.diag, &fine.upper, &fine.lower, None, out)
    }

    /// `amgRestrict`, (141.13): the segmented sum over each aggregate's
    /// members. Only a launch.
    pub fn restrict(&self, gpu: &Gpu, rf: &DevBuf<Scalar>, rc: &mut DevBuf<Scalar>) -> Result<()> {
        Self::check_len("rf", rf.len(), self.n_fine)?;
        Self::check_len("rc", rc.len(), self.n_coarse)?;
        if self.n_coarse == 0 {
            return Ok(());
        }
        let nl = self.n_coarse as Label;
        let f = self.restrict.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(rc)
                .arg(rf)
                .arg(&self.agg_offset)
                .arg(&self.agg_cells)
                .arg(&nl)
                .launch(cfg_for(self.n_coarse))?;
        }
        Ok(())
    }

    /// `amgProlong`, (141.11): one gather per fine cell. Only a launch.
    pub fn prolong(&self, gpu: &Gpu, ec: &DevBuf<Scalar>, ef: &mut DevBuf<Scalar>) -> Result<()> {
        Self::check_len("ec", ec.len(), self.n_coarse)?;
        Self::check_len("ef", ef.len(), self.n_fine)?;
        if self.n_fine == 0 {
            return Ok(());
        }
        let nl = self.n_fine as Label;
        let f = self.prolong.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(ef)
                .arg(ec)
                .arg(&self.cell_agg)
                .arg(&nl)
                .launch(cfg_for(self.n_fine))?;
        }
        Ok(())
    }

    /// `amgProlongAdd`, (141.12): the add in `Scalar`. Only a launch.
    pub fn prolong_add(
        &self,
        gpu: &Gpu,
        ec: &DevBuf<Scalar>,
        xf: &mut DevBuf<Scalar>,
    ) -> Result<()> {
        Self::check_len("ec", ec.len(), self.n_coarse)?;
        Self::check_len("xf", xf.len(), self.n_fine)?;
        if self.n_fine == 0 {
            return Ok(());
        }
        let nl = self.n_fine as Label;
        let f = self.prolong_add.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(xf)
                .arg(ec)
                .arg(&self.cell_agg)
                .arg(&nl)
                .launch(cfg_for(self.n_fine))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blockgen::{BlockSpec, GradedAxis, build_mesh};
    use crate::capture::{Snapshot, capture_replays_bitwise};
    use crate::ldu::{GpuLduMatrix, HostLduMatrix};
    use crate::mesh::GpuMesh;

    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    /// A uniform 6x6x6 box, cyclic on x, y and z.
    fn box6_cyclic() -> HostMesh {
        let axis = |n: usize| GradedAxis {
            lo: 0.0,
            hi: n as Scalar,
            n,
            expansion: 1.0,
            two_sided: false,
        };
        let mut b = BlockSpec {
            x: axis(6),
            y: axis(6),
            z: axis(6),
            windows: Vec::new(),
            patch_name: BlockSpec::default().patch_name,
            patch_type: ["wall"; 6].map(String::from),
            cyclic: Vec::new(),
        };
        for a in [0usize, 1, 2] {
            b.set_cyclic_axis(a).expect("cyclic axis");
        }
        build_mesh(&b).expect("box mesh")
    }

    /// The non-dyadic matrix of (141.18): `A(a, b)` for an edge `a < b` is
    /// `-(1/10 + ((11a + 3b) mod 7)/3)`, `A(b, a)` is
    /// `-(1/10 + ((5a + b) mod 3)/7)`, `diag[c] = 20 + c/9`, each evaluated
    /// in f64 and then cast, and `boundary_coeffs[bf] = -A(c, d)` on a
    /// coupled face.
    fn nondyadic(m: &HostMesh) -> HostLduMatrix {
        let a_of = |c: usize, d: usize| -> f64 {
            if c < d {
                -(0.1 + ((11 * c + 3 * d) % 7) as f64 / 3.0)
            } else {
                -(0.1 + ((5 * d + c) % 3) as f64 / 7.0)
            }
        };
        let mut a = HostLduMatrix::zeros(m);
        for (c, d) in a.diag.iter_mut().enumerate() {
            *d = (20.0 + c as f64 / 9.0) as Scalar;
        }
        for f in 0..m.n_internal_faces {
            let (o, nb) = (m.owner[f] as usize, m.neighbour[f] as usize);
            a.upper[f] = a_of(o, nb) as Scalar;
            a.lower[f] = a_of(nb, o) as Scalar;
        }
        for bf in 0..m.n_boundary_faces {
            let d = m.b_nbr_cell[bf];
            if d < 0 {
                continue;
            }
            a.boundary_coeffs[bf] = (-(a_of(
                m.b_face_cells[bf] as usize,
                d as usize,
            ))) as Scalar;
        }
        a
    }

    /// The registry's gate (`src/capture/registry.rs`, row
    /// `src/amg/galerkin.rs`): the Galerkin product and the three
    /// transfers capture into a CUDA graph and replay bitwise (SPEC-LIT
    /// 81, 141.2).
    #[test]
    fn galerkin_replays_under_capture() {
        let Some(gpu) = gpu() else { return };

        let report = capture_replays_bitwise(
            &gpu,
            "the AMG Galerkin product and transfers (SPEC-LIT 141.2)",
            || {
                let hm = box6_cyclic();
                let a = nondyadic(&hm);
                let agg = crate::amg::aggregate(&hm, &a)?;
                let plan = GalerkinPlan::for_mesh(&agg, &hm)?;
                let gal = GpuGalerkin::new(&gpu, &plan)?;
                let m = GpuMesh::upload(&gpu, &hm)?;
                let mut mat = GpuLduMatrix::new(&gpu, &m)?;
                gpu.write(&mut mat.diag, &a.diag)?;
                gpu.write(&mut mat.upper, &a.upper)?;
                gpu.write(&mut mat.lower, &a.lower)?;
                gpu.write(&mut mat.boundary_coeffs, &a.boundary_coeffs)?;
                let level = gal.new_level(&gpu)?;
                let rf: Vec<Scalar> = (0..hm.n_cells)
                    .map(|c| (0.1 * c as f64 + ((7 * c + 3) % 11) as f64 / 13.0) as Scalar)
                    .collect();
                let rf = gpu.upload(&rf)?;
                let rc = gpu.zeros(plan.n_coarse)?;
                let ef = gpu.zeros(plan.n_fine)?;
                let xf = gpu.zeros(plan.n_fine)?;
                Ok((gal, mat, level, rf, rc, ef, xf))
            },
            |(gal, mat, level, rf, rc, ef, xf): &mut (
                GpuGalerkin,
                GpuLduMatrix,
                GpuLevel,
                DevBuf<Scalar>,
                DevBuf<Scalar>,
                DevBuf<Scalar>,
                DevBuf<Scalar>,
            )| {
                gal.coarse_of_matrix(&gpu, mat, &mut *level)?;
                gal.restrict(&gpu, rf, &mut *rc)?;
                gal.prolong(&gpu, rc, &mut *ef)?;
                gal.prolong_add(&gpu, rc, &mut *xf)?;
                Ok(())
            },
            |(_gal, _mat, level, _rf, rc, ef, xf): &(
                GpuGalerkin,
                GpuLduMatrix,
                GpuLevel,
                DevBuf<Scalar>,
                DevBuf<Scalar>,
                DevBuf<Scalar>,
                DevBuf<Scalar>,
            )| {
                let mut state: Snapshot = Vec::new();
                let mut d = gpu.download(&level.diag)?;
                let mut u = gpu.download(&level.upper)?;
                let mut l = gpu.download(&level.lower)?;
                d.truncate(level.n_cells);
                u.truncate(level.n_faces);
                l.truncate(level.n_faces);
                state.push(("amg coarse diag", d));
                state.push(("amg coarse upper", u));
                state.push(("amg coarse lower", l));
                state.push(("amg rc", gpu.download(rc)?));
                state.push(("amg ef", gpu.download(ef)?));
                state.push(("amg xf", gpu.download(xf)?));
                Ok(state)
            },
        )
        .expect("SPEC-LIT 81.7: the AMG Galerkin product and transfers \
                 must capture and replay bitwise");
        println!("  AMG galerkin: {report}");
    }
}
