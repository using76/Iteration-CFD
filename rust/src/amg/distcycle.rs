// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The distributed V-cycle over SPMD ranks (SPEC-LIT §141.11, AMG-07b).
//!
//! [`DistGamg`] runs (141.5.1) on one rank's part of every level of the
//! [`DistHierarchy`] of §141.10. The levels are decomposed, so a level's
//! product is the part's own rows plus its processor faces, with the level's
//! halo filled first by the level's own [`SpmdHalo`] (141.42). The
//! restriction and the prolongation are the part's own [`GpuGalerkin`],
//! because no aggregate spans a cut. The smoothers are §141.3's on the part's
//! rows; the global quantities of their setup - Gershgorin's bound and, for
//! Chebyshev, the Lanczos estimate - are taken over the whole level, every
//! sum a per-rank partial added in rank order (141.43). The coarsest level is
//! gathered once at setup with [`gather_level`] and factored on every rank;
//! every cycle allgathervs its right-hand side and solves it redundantly, and
//! each rank keeps its own entries (141.44).
//!
//! The cycle is a different preconditioner on every cut - the aggregation is
//! block-local - so `partition_invariant(Gamg)` stays `false`
//! (`distsolve::SpmdWorkspace`). Every exchange and every allgather is a host
//! round trip, so the cycle inside a CUDA-graph capture is refused by name.
//! Multi-GPU runs are deferred by the user: every rank is a `ThreadComm`
//! thread with its own `Gpu` on device 0.
//!
//! Provenance: ORIGINAL - the decomposition of the V-cycle of Saad,
//! Iterative Methods for Sparse Linear Systems, 2nd ed. (2003), §13.4.3
//! Algorithm 13.3 (SPEC-LIT §141.5) over the parts of §141.10 is this
//! project's own design. No multigrid code of any licence was read. No
//! GPL-licensed source was consulted.

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::amg::coarse::{CoarseMethod, CoarseOperator, CoarseSetup, GpuCoarse};
use crate::amg::cycle::GamgConfig;
use crate::amg::dist::{gather_level, vote, DistHierarchy, DistLevel};
use crate::amg::galerkin::{GpuGalerkin, GpuLevel};
use crate::amg::smooth::{
    tridiag_max_eig, GpuLevelRows, GpuSmoother, HostRows, LevelRows, SmootherKind, SmootherSetup,
    LANCZOS_BREAKDOWN, LANCZOS_START, LANCZOS_STEPS,
};
use crate::comm::Comm;
use crate::decompose::PartMesh;
use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::halo::{LevelHalo, SpmdHalo};
use crate::ldu::{GpuLduMatrix, HostLduMatrix};
use crate::ldu_ops::{self, LduKernels};
use crate::mesh::topology::{RF_BOUNDARY, RF_OWNS};
use crate::mesh::GpuMesh;
use crate::{Label, Scalar};

/// The tag of the setup's f64 halo exchange (the distributed Lanczos of
/// (141.43)); the cycle's exchanges travel under `halo::HALO_TAG`.
pub const LANCZOS_TAG: u32 = 141;

fn refuse<T>(msg: String) -> Result<T> {
    Err(Error::Config(format!("amg::distcycle: {msg}")))
}

// ==========================================================================
//  Rows, products and global sums of a decomposed level
// ==========================================================================

/// One part's rows of a decomposed level in f64 (SPEC-LIT §141.11): the
/// diagonal, then per row its internal faces in ascending face id (`upper`
/// on the owner's side, `lower` on the neighbour's) and its coupled boundary
/// faces in ascending face id (`-boundary_coeffs`, column `b_nbr_cell`, which
/// may be a ghost `>= n_cells`). Its [`HostRows::amul`] is (141.42) in f64 on
/// a vector of `n_cells + n_halo` entries.
pub fn level_rows(l: &DistLevel) -> Result<HostRows> {
    l.check()?;
    let n = l.n_cells;
    let mut lists: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for f in 0..l.owner.len() {
        let (o, nb) = (l.owner[f], l.neighbour[f]);
        if o < 0 || o as usize >= n || nb < 0 || nb as usize >= n {
            return refuse(format!(
                "part {}: face {f} joins cells {o} and {nb}, not both in 0..{n}",
                l.part
            ));
        }
        lists[o as usize].push((nb as usize, l.upper[f] as f64));
        lists[nb as usize].push((o as usize, l.lower[f] as f64));
    }
    for bf in 0..l.b_face_cells.len() {
        let d = l.b_nbr_cell[bf];
        if d >= 0 {
            lists[l.b_face_cells[bf] as usize].push((d as usize, -(l.boundary_coeffs[bf] as f64)));
        }
    }
    let mut r = HostRows {
        n_cells: n,
        diag: l.diag.iter().map(|&d| d as f64).collect(),
        offset: vec![0usize; n + 1],
        col: Vec::new(),
        val: Vec::new(),
    };
    for (c, row) in lists.into_iter().enumerate() {
        for (col, val) in row {
            r.col.push(col);
            r.val.push(val);
        }
        r.offset[c + 1] = r.col.len();
    }
    Ok(r)
}

/// `x`'s halo filled from the owners, in f64, over the level's own plan: the
/// host twin of `SpmdHalo::scalar` for the setup arithmetic (141.43). `x`
/// holds `n_cells + n_halo` entries. A collective of the level's neighbours.
fn exchange_f64(comm: &dyn Comm, h: &LevelHalo, x: &mut [f64]) -> Result<()> {
    let n = h.n_cells;
    if x.len() != h.n_local() {
        return refuse(format!(
            "part {}: an f64 exchange of {} entries for {} owned and {} halo cells",
            h.part,
            x.len(),
            n,
            h.n_halo
        ));
    }
    let nn = h.nbr_parts.len();
    let mut packed: Vec<Vec<u8>> = Vec::with_capacity(nn);
    for i in 0..nn {
        let (a, b) = (h.send_offset[i] as usize, h.send_offset[i + 1] as usize);
        let mut buf = Vec::with_capacity(8 * (b - a));
        for &c in &h.send_index[a..b] {
            buf.extend_from_slice(&x[c as usize].to_le_bytes());
        }
        packed.push(buf);
    }
    let mut got: Vec<Vec<u8>> = (0..nn)
        .map(|i| vec![0u8; 8 * (h.recv_offset[i + 1] - h.recv_offset[i]) as usize])
        .collect();
    {
        let sends: Vec<(usize, &[u8])> = h
            .nbr_parts
            .iter()
            .zip(&packed)
            .map(|(&q, b)| (q as usize, &b[..]))
            .collect();
        let mut recvs: Vec<(usize, &mut [u8])> = h
            .nbr_parts
            .iter()
            .zip(got.iter_mut())
            .map(|(&q, b)| (q as usize, &mut b[..]))
            .collect();
        comm.exchange(LANCZOS_TAG, &sends, &mut recvs)?;
    }
    for (i, buf) in got.iter().enumerate() {
        let at = n + h.recv_offset[i] as usize;
        for (k, ch) in buf.chunks_exact(8).enumerate() {
            x[at + k] = f64::from_le_bytes(ch.try_into().expect("8 bytes"));
        }
    }
    Ok(())
}

/// Every rank's `x`, in rank order (one allgather).
fn allgather_f64(comm: &dyn Comm, x: f64) -> Result<Vec<f64>> {
    let mut all = vec![0u8; 8 * comm.size()];
    comm.allgather(&x.to_le_bytes(), &mut all)?;
    Ok(all
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().expect("8 bytes")))
        .collect())
}

/// The sum of every rank's partial, added from 0 in rank order, so every
/// rank holds the same bits (141.43).
fn allsum(comm: &dyn Comm, x: f64) -> Result<f64> {
    Ok(allgather_f64(comm, x)?.iter().fold(0.0f64, |s, &v| s + v))
}

/// §141.3's Lanczos on the whole decomposed level (141.43): Saad Algorithm
/// 6.15 on `S = sigma |D|^-1/2 A |D|^-1/2`, the start vector by GLOBAL key,
/// every inner product a rank-ordered [`allsum`], every product (141.42) in
/// f64 after an [`exchange_f64`]. `steps` is capped at the whole level's
/// cell count `n_global`. A collective: every rank runs the same steps.
pub fn dist_lanczos(
    comm: &dyn Comm,
    l: &DistLevel,
    rows: &HostRows,
    sigma: f64,
    n_global: usize,
    steps: usize,
) -> Result<(f64, usize)> {
    let n = rows.n_cells;
    if n != l.n_cells {
        return refuse(format!(
            "part {}: the rows have {n} cells, the level {}",
            l.part, l.n_cells
        ));
    }
    let max_steps = steps.min(n_global);
    if max_steps == 0 {
        return Ok((0.0, 0));
    }
    let sqrt_d: Vec<f64> = rows.diag.iter().map(|&d| d.abs().sqrt()).collect();
    let mut v: Vec<f64> = (0..n)
        .map(|c| {
            let x = (l.cell_key[c] as i64 + 1) as f64 * LANCZOS_START;
            x - x.trunc() - 0.5
        })
        .collect();
    let norm = allsum(comm, v.iter().map(|&x| x * x).sum::<f64>())?.sqrt();
    for x in v.iter_mut() {
        *x /= norm;
    }
    let mut alphas: Vec<f64> = Vec::with_capacity(max_steps);
    let mut betas: Vec<f64> = Vec::with_capacity(max_steps - 1);
    let mut v_prev = vec![0.0f64; n];
    let mut vs = vec![0.0f64; l.halo.n_local()];
    let mut beta_j = 0.0f64;
    let mut j_run = 0usize;
    for j in 1..=max_steps {
        for c in 0..n {
            vs[c] = v[c] / sqrt_d[c];
        }
        exchange_f64(comm, &l.halo, &mut vs)?;
        let av = rows.amul(&vs);
        let mut w: Vec<f64> = (0..n)
            .map(|c| sigma * av[c] / sqrt_d[c] - beta_j * v_prev[c])
            .collect();
        let alpha_j = allsum(comm, w.iter().zip(&v).map(|(&a, &b)| a * b).sum())?;
        for (wc, &vc) in w.iter_mut().zip(&v) {
            *wc -= alpha_j * vc;
        }
        let beta_next = allsum(comm, w.iter().map(|&x| x * x).sum::<f64>())?.sqrt();
        alphas.push(alpha_j);
        j_run = j;
        let mut t = 0.0f64;
        for &a in &alphas {
            t = t.max(a.abs());
        }
        for &b in &betas {
            t = t.max(b.abs());
        }
        if j == max_steps || beta_next <= LANCZOS_BREAKDOWN * t {
            break;
        }
        betas.push(beta_next);
        for c in 0..n {
            v_prev[c] = v[c];
            v[c] = w[c] / beta_next;
        }
        beta_j = beta_next;
    }
    let theta = tridiag_max_eig(&alphas, &betas)?;
    Ok((theta, j_run))
}

/// One level's smoother setup on this rank (SPEC-LIT §141.11): §141.3's
/// setup on the part's [`level_rows`] - `m` and `sigma` the part's own,
/// `sigma` checked equal on every rank - with Gershgorin's bound the whole
/// level's maximum and, for Chebyshev, the interval from [`dist_lanczos`].
/// A collective; the upload is the caller's, inside its own vote.
fn level_smoother(
    comm: &dyn Comm,
    l: &DistLevel,
    kind: SmootherKind,
    n_global: usize,
) -> Result<SmootherSetup> {
    let local =
        level_rows(l).and_then(|r| SmootherSetup::without_interval(&r, kind).map(|s| (r, s)));
    vote(comm, local.as_ref().map(|_| 0))?;
    let (rows, mut s) = local?;
    let sigmas = allgather_f64(comm, s.sigma)?;
    if sigmas.iter().any(|&x| x != sigmas[0]) {
        return refuse(format!(
            "the parts' diagonals sum to different signs {sigmas:?}; a level whose rows do not \
             all share one sign is indefinite, and its smoother is refused (SPEC-LIT 141.3)"
        ));
    }
    let g = allgather_f64(comm, s.gershgorin)?
        .iter()
        .fold(0.0f64, |m, &v| m.max(v));
    if matches!(kind, SmootherKind::Chebyshev { .. }) {
        let (theta, ran) = dist_lanczos(comm, l, &rows, s.sigma, n_global, LANCZOS_STEPS)?;
        s.set_interval(theta, ran, g)?;
    } else {
        s.gershgorin = g;
    }
    Ok(s)
}

// ==========================================================================
//  A coarse level on the device
// ==========================================================================

/// Upload a slice, or one default entry for an empty one (a zero-length
/// buffer is never handed to the driver).
fn up<T: cudarc::driver::DeviceRepr + Copy + Default>(gpu: &Gpu, v: &[T]) -> Result<DevBuf<T>> {
    if v.is_empty() {
        gpu.upload(&[T::default()])
    } else {
        gpu.upload(v)
    }
}

/// One part's coarse level on the device, in `lduAmul`'s arguments: the LDU
/// coefficients, the processor faces, and a merged row map - per cell its
/// internal faces in ascending face id (owner side [`RF_OWNS`], neighbour
/// side 0), then its processor faces in ascending face id ([`RF_BOUNDARY`]
/// `|` [`RF_OWNS`]).
struct DevLevel {
    n_cells: usize,
    diag: DevBuf<Scalar>,
    upper: DevBuf<Scalar>,
    lower: DevBuf<Scalar>,
    bc: DevBuf<Scalar>,
    owner: DevBuf<Label>,
    neighbour: DevBuf<Label>,
    b_nbr: DevBuf<Label>,
    rf_offset: DevBuf<Label>,
    rf_face: DevBuf<Label>,
    rf_flags: DevBuf<Label>,
}

impl DevLevel {
    fn new(gpu: &Gpu, l: &DistLevel) -> Result<Self> {
        l.check()?;
        let n = l.n_cells;
        let mut rows: Vec<Vec<(Label, Label)>> = vec![Vec::new(); n];
        for f in 0..l.owner.len() {
            let (o, nb) = (l.owner[f], l.neighbour[f]);
            if o < 0 || o as usize >= n || nb < 0 || nb as usize >= n {
                return refuse(format!(
                    "part {}: face {f} joins cells {o} and {nb}, not both in 0..{n}",
                    l.part
                ));
            }
            rows[o as usize].push((f as Label, RF_OWNS));
            rows[nb as usize].push((f as Label, 0));
        }
        for bf in 0..l.b_face_cells.len() {
            rows[l.b_face_cells[bf] as usize].push((bf as Label, RF_BOUNDARY | RF_OWNS));
        }
        let mut offset = vec![0 as Label; n + 1];
        let mut face = Vec::new();
        let mut flags = Vec::new();
        for (c, row) in rows.into_iter().enumerate() {
            for (f, fl) in row {
                face.push(f);
                flags.push(fl);
            }
            offset[c + 1] = face.len() as Label;
        }
        Ok(Self {
            n_cells: n,
            diag: up(gpu, &l.diag)?,
            upper: up(gpu, &l.upper)?,
            lower: up(gpu, &l.lower)?,
            bc: up(gpu, &l.boundary_coeffs)?,
            owner: up(gpu, &l.owner)?,
            neighbour: up(gpu, &l.neighbour)?,
            b_nbr: up(gpu, &l.b_nbr_cell)?,
            rf_offset: up(gpu, &offset)?,
            rf_face: up(gpu, &face)?,
            rf_flags: up(gpu, &flags)?,
        })
    }

    /// (141.42) by `lduAmul`, `x`'s halo already filled. Only a launch.
    fn amul(
        &self,
        gpu: &Gpu,
        f: &CudaFunction,
        y: &mut DevBuf<Scalar>,
        x: &DevBuf<Scalar>,
    ) -> Result<()> {
        if self.n_cells == 0 {
            return Ok(());
        }
        let nl = self.n_cells as Label;
        unsafe {
            gpu.stream()
                .launch_builder(f)
                .arg(&mut *y)
                .arg(x)
                .arg(&self.diag)
                .arg(&self.upper)
                .arg(&self.lower)
                .arg(&self.bc)
                .arg(&self.owner)
                .arg(&self.neighbour)
                .arg(&self.b_nbr)
                .arg(&self.rf_offset)
                .arg(&self.rf_face)
                .arg(&self.rf_flags)
                .arg(&nl)
                .launch(cfg_for(self.n_cells))?;
        }
        Ok(())
    }
}

// ==========================================================================
//  The coarsest level, gathered and solved on every rank
// ==========================================================================

fn put_u64(out: &mut Vec<u8>, x: u64) {
    out.extend_from_slice(&x.to_le_bytes());
}

fn put_labels(out: &mut Vec<u8>, v: &[Label]) {
    put_u64(out, v.len() as u64);
    for &x in v {
        out.extend_from_slice(&(x as i64).to_le_bytes());
    }
}

fn put_scalars(out: &mut Vec<u8>, v: &[Scalar]) {
    put_u64(out, v.len() as u64);
    for &x in v {
        // Exact in both builds: an f32 widens to f64 without rounding.
        out.extend_from_slice(&(x as f64).to_le_bytes());
    }
}

/// A coarse level as bytes, every field `gather_level` and `check` read.
fn encode(l: &DistLevel) -> Vec<u8> {
    let mut o = Vec::new();
    put_u64(&mut o, l.part as u64);
    put_u64(&mut o, l.n_cells as u64);
    put_labels(&mut o, &l.owner);
    put_labels(&mut o, &l.neighbour);
    put_scalars(&mut o, &l.diag);
    put_scalars(&mut o, &l.upper);
    put_scalars(&mut o, &l.lower);
    put_labels(&mut o, &l.b_face_cells);
    put_labels(&mut o, &l.b_nbr_cell);
    put_scalars(&mut o, &l.boundary_coeffs);
    put_labels(&mut o, &l.cell_key);
    let flat: Vec<Label> = l.bface_key.iter().flat_map(|&(a, b)| [a, b]).collect();
    put_labels(&mut o, &flat);
    put_u64(&mut o, l.halo.part as u64);
    put_u64(&mut o, l.halo.n_cells as u64);
    put_u64(&mut o, l.halo.n_halo as u64);
    put_labels(&mut o, &l.halo.nbr_parts);
    put_labels(&mut o, &l.halo.send_offset);
    put_labels(&mut o, &l.halo.send_index);
    put_labels(&mut o, &l.halo.recv_offset);
    o
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn u64(&mut self) -> Result<u64> {
        let Some(c) = self.b.get(self.at..self.at + 8) else {
            return refuse(format!("a gathered level ends at byte {}", self.b.len()));
        };
        self.at += 8;
        Ok(u64::from_le_bytes(c.try_into().expect("8 bytes")))
    }

    fn usize(&mut self) -> Result<usize> {
        Ok(self.u64()? as usize)
    }

    fn labels(&mut self) -> Result<Vec<Label>> {
        let n = self.usize()?;
        (0..n).map(|_| Ok(self.u64()? as i64 as Label)).collect()
    }

    fn scalars(&mut self) -> Result<Vec<Scalar>> {
        let n = self.usize()?;
        (0..n)
            .map(|_| Ok(f64::from_bits(self.u64()?) as Scalar))
            .collect()
    }
}

fn decode(b: &[u8]) -> Result<DistLevel> {
    let mut r = Reader { b, at: 0 };
    let part = r.usize()?;
    let n_cells = r.usize()?;
    let owner = r.labels()?;
    let neighbour = r.labels()?;
    let diag = r.scalars()?;
    let upper = r.scalars()?;
    let lower = r.scalars()?;
    let b_face_cells = r.labels()?;
    let b_nbr_cell = r.labels()?;
    let boundary_coeffs = r.scalars()?;
    let cell_key = r.labels()?;
    let flat = r.labels()?;
    let bface_key = flat.chunks_exact(2).map(|c| (c[0], c[1])).collect();
    let halo = LevelHalo {
        part: r.usize()?,
        n_cells: r.usize()?,
        n_halo: r.usize()?,
        nbr_parts: r.labels()?,
        send_offset: r.labels()?,
        send_index: r.labels()?,
        recv_offset: r.labels()?,
    };
    if r.at != b.len() || flat.len() % 2 != 0 {
        return refuse(format!(
            "a gathered level of {} bytes was read to byte {}",
            b.len(),
            r.at
        ));
    }
    let l = DistLevel {
        part,
        n_cells,
        owner,
        neighbour,
        diag,
        upper,
        lower,
        b_face_cells,
        b_nbr_cell,
        boundary_coeffs,
        cell_key,
        bface_key,
        halo,
    };
    l.check()?;
    Ok(l)
}

/// The coarsest level, gathered on every rank (141.44): `gid[q][c]` is the
/// whole level's id of rank `q`'s coarsest cell `c`, the factor and the
/// device level of `gather_level`'s level, and the whole-level vectors.
struct Coarsest {
    rank: usize,
    gid: Vec<Vec<usize>>,
    setup_method: CoarseMethod,
    gpu_c: GpuCoarse,
    level: GpuLevel,
    rows: GpuLevelRows,
    host_b: Vec<Scalar>,
    b: DevBuf<Scalar>,
    x: DevBuf<Scalar>,
    n_gathers: u64,
}

impl Coarsest {
    /// A collective: one allgatherv of the parts' coarsest levels.
    fn new(gpu: &Gpu, comm: &dyn Comm, mine: &DistLevel, permissive: bool) -> Result<Self> {
        let all = comm.allgatherv(&encode(mine))?;
        let built = (|| -> Result<_> {
            let levels: Vec<DistLevel> = all.iter().map(|b| decode(b)).collect::<Result<_>>()?;
            let refs: Vec<&DistLevel> = levels.iter().collect();
            let gl = gather_level(&refs)?;
            let mut keys: Vec<Label> = levels
                .iter()
                .flat_map(|l| l.cell_key[..l.n_cells].iter().copied())
                .collect();
            keys.sort_unstable();
            let gid = levels
                .iter()
                .map(|l| {
                    l.cell_key[..l.n_cells]
                        .iter()
                        .map(|k| {
                            keys.binary_search(k).map_err(|_| {
                                Error::Config(format!(
                                    "amg::distcycle: key {k} is owned by no part"
                                ))
                            })
                        })
                        .collect::<Result<Vec<usize>>>()
                })
                .collect::<Result<Vec<_>>>()?;
            let setup = CoarseSetup::general(&HostRows::of_level(&gl)?, permissive)?;
            let gpu_c = GpuCoarse::new(gpu, &setup)?;
            let level = gl.upload(gpu)?;
            let rows = GpuLevelRows::new(gpu, &LevelRows::of_level(&gl)?)?;
            let n = gl.n_cells;
            Ok(Self {
                rank: comm.rank(),
                gid,
                setup_method: setup.method,
                gpu_c,
                level,
                rows,
                host_b: vec![0 as Scalar; n],
                b: gpu.zeros(n.max(1))?,
                x: gpu.zeros(n.max(1))?,
                n_gathers: 0,
            })
        })();
        vote(comm, built.as_ref().map(|_| 0))?;
        built
    }

    /// `x_L = C(b_L)` of (141.44): `b_l` holds this rank's coarsest entries,
    /// `x_l` receives them. One allgatherv.
    fn solve(
        &mut self,
        gpu: &Gpu,
        comm: &dyn Comm,
        b_l: &DevBuf<Scalar>,
        x_l: &mut DevBuf<Scalar>,
    ) -> Result<()> {
        let sz = std::mem::size_of::<Scalar>();
        let n_own = self.gid[self.rank].len();
        let mine = gpu.download(b_l)?;
        let mut bytes = Vec::with_capacity(sz * n_own);
        for v in &mine[..n_own] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let all = comm.allgatherv(&bytes)?;
        self.n_gathers += 1;
        for (q, buf) in all.iter().enumerate() {
            let ids = &self.gid[q];
            if buf.len() != sz * ids.len() {
                return refuse(format!(
                    "rank {q} sent {} bytes of its coarsest right-hand side, its {} cells need {}",
                    buf.len(),
                    ids.len(),
                    sz * ids.len()
                ));
            }
            for (k, ch) in buf.chunks_exact(sz).enumerate() {
                self.host_b[ids[k]] = Scalar::from_le_bytes(ch.try_into().expect("Scalar bytes"));
            }
        }
        gpu.write(&mut self.b, &self.host_b)?;
        self.gpu_c.solve(
            gpu,
            CoarseOperator::Level {
                level: &self.level,
                rows: &self.rows,
            },
            &self.b,
            &mut self.x,
        )?;
        let x = gpu.download(&self.x)?;
        let own: Vec<Scalar> = self.gid[self.rank].iter().map(|&g| x[g]).collect();
        let mut dst = x_l.slice_mut(0..n_own);
        gpu.stream().memcpy_htod(&own, &mut dst)?;
        gpu.sync()?;
        Ok(())
    }
}

// ==========================================================================
//  The cycle
// ==========================================================================

/// The kernels the cycle launches outside the smoothers and transfers.
struct Kernels {
    ldu: LduKernels,
    ldu_amul: CudaFunction,
    residual: CudaFunction,
}

/// Which operator a level's product applies.
#[derive(Clone, Copy)]
enum Op<'a> {
    Fine(&'a GpuMesh, &'a GpuLduMatrix),
    Coarse(&'a DevLevel),
}

/// (141.42): `x`'s halo through the level's exchange, then the product.
fn product(
    gpu: &Gpu,
    comm: &dyn Comm,
    halo: &mut SpmdHalo,
    op: Op<'_>,
    k: &Kernels,
    y: &mut DevBuf<Scalar>,
    x: &mut DevBuf<Scalar>,
    n_ex: &mut u64,
) -> Result<()> {
    halo.scalar(gpu, comm, x)?;
    *n_ex += 1;
    match op {
        Op::Fine(mesh, a) => ldu_ops::amul(gpu, &k.ldu, y, x, a, mesh),
        Op::Coarse(d) => d.amul(gpu, &k.ldu_amul, y, x),
    }
}

/// `amgResidual`: `r = b - y` over `n` owned cells. Only a launch.
fn residual(
    gpu: &Gpu,
    f: &CudaFunction,
    r: &mut DevBuf<Scalar>,
    b: &DevBuf<Scalar>,
    y: &DevBuf<Scalar>,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    unsafe {
        gpu.stream()
            .launch_builder(f)
            .arg(&mut *r)
            .arg(b)
            .arg(y)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// The distributed V-cycle on one rank (SPEC-LIT §141.11): the rank's part
/// of the hierarchy of §141.10, its smoothers, transfers, per-level halos,
/// the redundant coarsest solve, and every buffer a cycle touches, allocated
/// in [`DistGamg::new`].
pub struct DistGamg {
    cfg: GamgConfig,
    rank: usize,
    n_ranks: usize,
    hier: DistHierarchy,
    /// `[L + 1]` the whole level's cell count, finest first.
    n_global: Vec<usize>,
    /// `[L + 1]` this rank's owned cells per level.
    sizes: Vec<usize>,
    /// Levels `0 .. L-1`: the exchange and the smoother of each.
    halos: Vec<SpmdHalo>,
    smoothers: Vec<GpuSmoother>,
    /// Levels `1 .. L-1` at `l - 1`: the coarse operator.
    ops: Vec<DevLevel>,
    /// Steps `0 .. L-1`: the part-local restriction and prolongation.
    gals: Vec<GpuGalerkin>,
    coarsest: Coarsest,
    k: Kernels,
    /// Per level `0 ..= L`; level `l < L` sized owned + halo, level `L`
    /// owned. `x_l[0]` and `b_l[0]` are never used: level 0 is the caller's.
    b_l: Vec<DevBuf<Scalar>>,
    x_l: Vec<DevBuf<Scalar>>,
    r_l: Vec<DevBuf<Scalar>>,
    y_l: Vec<DevBuf<Scalar>>,
    n_exchanges: u64,
}

impl DistGamg {
    /// Build this rank's part of the distributed hierarchy and its cycle
    /// (SPEC-LIT §141.11). `part` is the rank's part (rank `r` holds part
    /// `r`) and `a` its matrix as `Decomposition::split_matrix` cut it and
    /// the solve receives it - folded, as [`crate::amg::Gamg::new`] takes
    /// it. A collective: every rank calls it at the same point with the same
    /// `cfg` and `permissive`; a step refused on one rank is refused on
    /// every rank by a vote. A hierarchy without a coarse level (the whole
    /// mesh at most `n_cells_coarsest` cells, or `max_levels = 1`) is
    /// refused by name, as is any part without a cell on some level.
    pub fn new(
        gpu: &Gpu,
        comm: &dyn Comm,
        part: &PartMesh,
        a: &HostLduMatrix,
        cfg: &GamgConfig,
        permissive: bool,
    ) -> Result<Self> {
        let pre: Result<()> = if cfg.n_pre + cfg.n_post == 0 {
            refuse(
                "n_pre + n_post is 0; a cycle without any smoothing sweeps does nothing at all, \
                 so nPreSweeps + nPostSweeps = 0 is refused"
                    .to_string(),
            )
        } else if part.part != comm.rank() {
            refuse(format!("rank {} was given part {}", comm.rank(), part.part))
        } else {
            Ok(())
        };
        vote(comm, pre.as_ref().map(|_| 0))?;
        pre?;
        let hier = DistHierarchy::build_spmd(gpu, comm, part, a, &cfg.hierarchy)?;
        let l_count = hier.n_coarse_levels();
        if l_count == 0 {
            return refuse(
                "the hierarchy has no coarse level (the whole mesh is at most n_cells_coarsest \
                 cells, or max_levels is 1); a distributed cycle needs one, so solve such a \
                 system on one rank (SPEC-LIT 141.11)"
                    .to_string(),
            );
        }
        let sizes: Vec<usize> = hier.levels.iter().map(|l| l.n_cells).collect();
        let empty = sizes.iter().position(|&n| n == 0);
        vote(
            comm,
            match empty {
                None => Ok(0),
                Some(l) => Err(format!("part {} has no cell on level {l}", comm.rank())),
            },
        )?;
        let mut n_global = Vec::with_capacity(l_count + 1);
        for &n in &sizes {
            n_global.push(vote(comm, Ok::<usize, String>(n))?);
        }
        let mut setups = Vec::with_capacity(l_count);
        for l in 0..l_count {
            setups.push(level_smoother(
                comm,
                &hier.levels[l],
                cfg.hierarchy.smoother,
                n_global[l],
            )?);
        }
        let coarsest = Coarsest::new(gpu, comm, &hier.levels[l_count], permissive)?;

        let local = (|| -> Result<_> {
            let smoothers = setups
                .iter()
                .map(|s| GpuSmoother::new(gpu, s))
                .collect::<Result<Vec<_>>>()?;
            let halos = (0..l_count)
                .map(|l| SpmdHalo::from_level(gpu, &hier.levels[l].halo, comm.size()))
                .collect::<Result<Vec<_>>>()?;
            let ops = (1..l_count)
                .map(|l| DevLevel::new(gpu, &hier.levels[l]))
                .collect::<Result<Vec<_>>>()?;
            let gals = hier
                .steps
                .iter()
                .map(|s| GpuGalerkin::new(gpu, &s.plan))
                .collect::<Result<Vec<_>>>()?;
            let set = KernelSet::new(gpu, crate::kernels::AMG)?;
            let k = Kernels {
                ldu: LduKernels::new(gpu)?,
                ldu_amul: KernelSet::new(gpu, crate::kernels::LDU)?.func("lduAmul")?,
                residual: set.func("amgResidual")?,
            };
            let len = |l: usize| -> usize {
                if l < l_count {
                    hier.levels[l].halo.n_local().max(1)
                } else {
                    sizes[l].max(1)
                }
            };
            let bufs = |gpu: &Gpu| -> Result<Vec<DevBuf<Scalar>>> {
                (0..=l_count).map(|l| gpu.zeros(len(l))).collect()
            };
            Ok((
                smoothers,
                halos,
                ops,
                gals,
                k,
                bufs(gpu)?,
                bufs(gpu)?,
                bufs(gpu)?,
                bufs(gpu)?,
            ))
        })();
        vote(comm, local.as_ref().map(|_| 0))?;
        let (smoothers, halos, ops, gals, k, b_l, x_l, r_l, y_l) = local?;
        Ok(Self {
            cfg: *cfg,
            rank: comm.rank(),
            n_ranks: comm.size(),
            hier,
            n_global,
            sizes,
            halos,
            smoothers,
            ops,
            gals,
            coarsest,
            k,
            b_l,
            x_l,
            r_l,
            y_l,
            n_exchanges: 0,
        })
    }

    /// The config the cycle was built with.
    pub fn config(&self) -> &GamgConfig {
        &self.cfg
    }

    /// The rank (= part) the cycle was built on.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The world size the cycle was built for.
    pub fn n_ranks(&self) -> usize {
        self.n_ranks
    }

    /// The rank's part of the hierarchy.
    pub fn hierarchy(&self) -> &DistHierarchy {
        &self.hier
    }

    /// This rank's owned cells per level, finest first.
    pub fn sizes(&self) -> &[usize] {
        &self.sizes
    }

    /// The whole level's cell count per level, finest first.
    pub fn global_sizes(&self) -> &[usize] {
        &self.n_global
    }

    /// Owned fine cells.
    pub fn n_fine(&self) -> usize {
        self.sizes[0]
    }

    /// Owned + halo fine cells, the length a field must have.
    pub fn n_fine_local(&self) -> usize {
        self.hier.levels[0].halo.n_local()
    }

    /// How the gathered coarsest level is solved (§141.4, §141.5).
    pub fn coarse_method(&self) -> CoarseMethod {
        self.coarsest.setup_method
    }

    /// Whether `n_pre == n_post`, so the cycle from zero is a symmetric
    /// operator and PCG may take it (§141.5).
    pub fn is_symmetric_cycle(&self) -> bool {
        self.cfg.n_pre == self.cfg.n_post
    }

    /// Halo exchanges the cycles have made so far.
    pub fn n_exchanges(&self) -> u64 {
        self.n_exchanges
    }

    /// Coarsest-level allgathervs the cycles have made so far, one per cycle.
    pub fn n_coarse_gathers(&self) -> u64 {
        self.coarsest.n_gathers
    }

    /// One V-cycle, (141.5.1) with (141.42) and (141.44), from the caller's
    /// `x` (owned + halo, the halo refilled here) against the caller's `b`
    /// (owned). `mesh` and `a` are the rank's part as the solve holds it,
    /// the operator the hierarchy was built from. A collective; refused by
    /// name inside a CUDA-graph capture.
    pub fn cycle(
        &mut self,
        gpu: &Gpu,
        comm: &dyn Comm,
        mesh: &GpuMesh,
        a: &GpuLduMatrix,
        x: &mut DevBuf<Scalar>,
        b: &DevBuf<Scalar>,
    ) -> Result<()> {
        let n = self.sizes[0];
        if gpu.is_capturing() {
            return refuse(
                "a distributed cycle was called inside a CUDA-graph capture; every level's halo \
                 exchange and the coarsest allgatherv cross the host, and a graph cannot record \
                 them (SPEC-LIT 81.3, 141.11)"
                    .to_string(),
            );
        }
        if comm.rank() != self.rank || comm.size() != self.n_ranks {
            return refuse(format!(
                "the cycle was built on rank {} of {}, this handle is rank {} of {}",
                self.rank,
                self.n_ranks,
                comm.rank(),
                comm.size()
            ));
        }
        if a.n_cells != n || mesh.n_cells != n {
            return refuse(format!(
                "the operator has {} cells and the mesh {}, the hierarchy was built for {n}",
                a.n_cells, mesh.n_cells
            ));
        }
        if x.len() < self.n_fine_local() || b.len() < n {
            return refuse(format!(
                "x holds {} and b {} values; the part needs {} (owned + halo) and {n}",
                x.len(),
                b.len(),
                self.n_fine_local()
            ));
        }
        let l_count = self.hier.n_coarse_levels();
        let Self {
            cfg,
            halos,
            smoothers,
            ops,
            gals,
            coarsest,
            k,
            b_l,
            x_l,
            r_l,
            y_l,
            sizes,
            n_exchanges,
            ..
        } = self;
        // The descent: pre-smooth, product, residual, restrict, zero the
        // correction (141.5.1).
        for l in 0..l_count {
            let op = if l == 0 {
                Op::Fine(mesh, a)
            } else {
                Op::Coarse(&ops[l - 1])
            };
            {
                let (xl, bl): (&mut DevBuf<Scalar>, &DevBuf<Scalar>) = if l == 0 {
                    (&mut *x, b)
                } else {
                    (&mut x_l[l], &b_l[l])
                };
                let halo = &mut halos[l];
                smoothers[l].smooth_with(gpu, sizes[l], xl, bl, cfg.n_pre, |y, xv| {
                    product(gpu, comm, &mut *halo, op, k, y, xv, &mut *n_exchanges)
                })?;
                product(
                    gpu,
                    comm,
                    &mut halos[l],
                    op,
                    k,
                    &mut y_l[l],
                    xl,
                    n_exchanges,
                )?;
                residual(gpu, &k.residual, &mut r_l[l], bl, &y_l[l], sizes[l])?;
            }
            gals[l].restrict(gpu, &r_l[l], &mut b_l[l + 1])?;
            gpu.fill_zero(&mut x_l[l + 1])?;
        }
        // The coarsest level, gathered and solved on every rank (141.44).
        coarsest.solve(gpu, comm, &b_l[l_count], &mut x_l[l_count])?;
        // The ascent: prolong-add, post-smooth.
        for l in (0..l_count).rev() {
            let op = if l == 0 {
                Op::Fine(mesh, a)
            } else {
                Op::Coarse(&ops[l - 1])
            };
            let (lo, hi) = x_l.split_at_mut(l + 1);
            let (xl, bl): (&mut DevBuf<Scalar>, &DevBuf<Scalar>) = if l == 0 {
                (&mut *x, b)
            } else {
                (&mut lo[l], &b_l[l])
            };
            gals[l].prolong_add(gpu, &hi[0], xl)?;
            let halo = &mut halos[l];
            smoothers[l].smooth_with(gpu, sizes[l], xl, bl, cfg.n_post, |y, xv| {
                product(gpu, comm, &mut *halo, op, k, y, xv, &mut *n_exchanges)
            })?;
        }
        Ok(())
    }

    /// The preconditioner application of §141.5 on this rank: `z = 0;
    /// cycle(z, r)`. `z` holds owned + halo values. A collective.
    pub fn precondition(
        &mut self,
        gpu: &Gpu,
        comm: &dyn Comm,
        mesh: &GpuMesh,
        a: &GpuLduMatrix,
        z: &mut DevBuf<Scalar>,
        r: &DevBuf<Scalar>,
    ) -> Result<()> {
        gpu.fill_zero(z)?;
        self.cycle(gpu, comm, mesh, a, z, r)
    }
}
