// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The distributed hierarchy's levels: block-local aggregation, coarse
//! processor faces and coarse halo plans (SPEC-LIT §141.10, AMG-07a).
//!
//! A decomposed level is one part's LDU matrix whose coupled boundary faces
//! may point past `n_cells`, into the halo. Each part aggregates its own
//! cells by §141.1 with every such processor face read as UNCOUPLED, so no
//! aggregate crosses a cut. The coarse level of a part is then the Galerkin
//! product of §141.2 on its own cells, plus one coarse processor face for
//! every pair (own aggregate, neighbour's aggregate) that a fine processor
//! face joins, plus the halo plan that moves a coarse field across the cut.
//!
//! Every list is summed in the order of a GLOBAL key (§141.10): the key of a
//! fine cell is its global cell id and of a fine face its global face slot,
//! and the key of an aggregate is its smallest member's key. Gathered, the
//! parts' coarse levels are therefore the Galerkin product of the gathered
//! fine level under the union of the block-local aggregations, bit for bit,
//! which is this unit's gate.
//!
//! Host code only. The part-local product is an ordinary [`GalerkinPlan`],
//! so its device gather is `GpuGalerkin` unchanged. The distributed cycle
//! on these levels is AMG-07b's, in `amg::distcycle` (SPEC-LIT §141.11).
//!
//! Provenance: ORIGINAL - the decomposition of Notay's aggregation
//! (ETNA 37 (2010), SPEC-LIT §141.1) into blocks is this project's own
//! design. No external source. No GPL-licensed source was consulted.

use crate::amg::aggregate::{aggregate_level, Aggregation, LevelView};
use crate::amg::galerkin::{GalerkinPlan, HostLevel};
use crate::amg::hierarchy::HierarchyConfig;
use crate::comm::Comm;
use crate::decompose::{Decomposition, PartMesh};
use crate::device::Gpu;
use crate::error::{Error, Result};
use crate::halo::{LevelHalo, SpmdHalo};
use crate::ldu::HostLduMatrix;
use crate::{Label, Scalar};

/// The second half of a fine face key, which is one number (SPEC-LIT
/// §141.10).
pub const NO_KEY: Label = -1;

fn refuse<T>(msg: String) -> Result<T> {
    Err(Error::Config(format!("amg::dist: {msg}")))
}

/// One part's level of the distributed hierarchy (SPEC-LIT §141.10).
///
/// Cells `0..n_cells` are owned, `n_cells .. n_cells + halo.n_halo` are
/// ghosts. A boundary face with `b_nbr_cell >= n_cells` is a processor face:
/// its `boundary_coeffs` enters the operator as `-boundary_coeffs ·
/// psi[b_nbr_cell]`, the sign of (141.2). Every other boundary face is the
/// part's own (uncoupled, or a cyclic couple inside the part); a coarse
/// level has only processor faces.
#[derive(Debug, Clone, PartialEq)]
pub struct DistLevel {
    pub part: usize,
    pub n_cells: usize,
    pub owner: Vec<Label>,
    pub neighbour: Vec<Label>,
    pub diag: Vec<Scalar>,
    pub upper: Vec<Scalar>,
    pub lower: Vec<Scalar>,
    pub b_face_cells: Vec<Label>,
    pub b_nbr_cell: Vec<Label>,
    pub boundary_coeffs: Vec<Scalar>,
    /// `[n_cells + n_halo]` the global key of every local cell, owned
    /// strictly ascending, halo strictly ascending within each owning part.
    pub cell_key: Vec<Label>,
    /// `[n_boundary_faces]` the global key of every boundary face; only the
    /// processor faces' keys are read.
    pub bface_key: Vec<(Label, Label)>,
    /// The exchange plan of this level.
    pub halo: LevelHalo,
}

/// One part's block-local aggregation of one level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockAggregation {
    /// §141.1 on the part's cells, processor faces read as UNCOUPLED.
    pub agg: Aggregation,
    /// `[n_coarse]` the key of each aggregate: its smallest member's key.
    pub root: Vec<Label>,
}

/// The lists of one part's coarsening.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DistCoarsening {
    /// The part-local Galerkin lists: diagonals, uppers and lowers of the
    /// coarse faces inside the part (§141.2).
    pub plan: GalerkinPlan,
    /// `[n_coarse_processor_faces + 1]` CSR over [`Self::proc_src`].
    pub proc_offset: Vec<Label>,
    /// The fine processor faces of each coarse processor face, ascending in
    /// their key (SPEC-LIT (141.40)).
    pub proc_src: Vec<Label>,
}

impl DistLevel {
    /// The finest level of a part: the part's mesh and its matrix as
    /// `Decomposition::split_matrix` cut it, folded as every solve receives
    /// it (`diag` after `add_boundary_contributions`, §141.2).
    pub fn fine(part: &PartMesh, a: &HostLduMatrix) -> Result<Self> {
        let pm = &part.mesh;
        let (n, n_if, n_bf) = (pm.n_cells, pm.n_internal_faces, pm.n_boundary_faces);
        if a.diag.len() != n
            || a.upper.len() != n_if
            || a.lower.len() != n_if
            || a.boundary_coeffs.len() != n_bf
        {
            return refuse(format!(
                "part {}: the matrix has {}/{}/{}/{} diag/upper/lower/boundary entries, the part \
                 {n}/{n_if}/{n_if}/{n_bf}",
                part.part,
                a.diag.len(),
                a.upper.len(),
                a.lower.len(),
                a.boundary_coeffs.len()
            ));
        }
        if part.global_cell.len() != part.n_local() || pm.global_face.len() != n_if + n_bf {
            return refuse(format!(
                "part {}: global_cell has {} entries (want {}), global_face {} (want {})",
                part.part,
                part.global_cell.len(),
                part.n_local(),
                pm.global_face.len(),
                n_if + n_bf
            ));
        }
        let level = Self {
            part: part.part,
            n_cells: n,
            owner: pm.owner.clone(),
            neighbour: pm.neighbour.clone(),
            diag: a.diag.clone(),
            upper: a.upper.clone(),
            lower: a.lower.clone(),
            b_face_cells: pm.b_face_cells.clone(),
            b_nbr_cell: pm.b_nbr_cell.clone(),
            boundary_coeffs: a.boundary_coeffs.clone(),
            cell_key: part.global_cell.clone(),
            bface_key: (0..n_bf)
                .map(|bf| (pm.global_face[n_if + bf], NO_KEY))
                .collect(),
            halo: LevelHalo::of_part(part),
        };
        level.check()?;
        Ok(level)
    }

    /// Ghost cells of the level.
    pub fn n_halo(&self) -> usize {
        self.halo.n_halo
    }

    /// Boundary faces whose neighbour is a ghost cell.
    pub fn n_proc_faces(&self) -> usize {
        let n = self.n_cells as Label;
        self.b_nbr_cell.iter().filter(|&&d| d >= n).count()
    }

    /// The shape rules of §141.10, each refused by name.
    pub fn check(&self) -> Result<()> {
        let p = self.part;
        let (n, nf, nbf, nh) = (
            self.n_cells,
            self.owner.len(),
            self.b_face_cells.len(),
            self.halo.n_halo,
        );
        if self.diag.len() != n
            || self.neighbour.len() != nf
            || self.upper.len() != nf
            || self.lower.len() != nf
            || self.b_nbr_cell.len() != nbf
            || self.boundary_coeffs.len() != nbf
            || self.bface_key.len() != nbf
            || self.cell_key.len() != n + nh
        {
            return refuse(format!(
                "part {p}: length mismatch among diag/neighbour/upper/lower/b_nbr_cell/\
                 boundary_coeffs/bface_key/cell_key for {n} cells, {nf} faces, {nbf} boundary \
                 faces, {nh} halo cells"
            ));
        }
        if self.halo.part != p || self.halo.n_cells != n {
            return refuse(format!(
                "part {p}: the halo plan is part {}'s with {} cells, the level has {n}",
                self.halo.part, self.halo.n_cells
            ));
        }
        if self.cell_key[..n].windows(2).any(|w| w[0] >= w[1]) {
            return refuse(format!(
                "part {p}: the owned cell keys are not strictly ascending, so the local order is \
                 not the global order (SPEC-LIT §141.10)"
            ));
        }
        let hp = self.halo.halo_parts()?;
        for h in 1..nh {
            if hp[h] == hp[h - 1] && self.cell_key[n + h] <= self.cell_key[n + h - 1] {
                return refuse(format!(
                    "part {p}: halo keys {} and {} of part {} are not strictly ascending",
                    self.cell_key[n + h - 1],
                    self.cell_key[n + h],
                    hp[h]
                ));
            }
        }
        for (bf, (&c, &d)) in self.b_face_cells.iter().zip(&self.b_nbr_cell).enumerate() {
            if c < 0 || c as usize >= n {
                return refuse(format!("part {p}: b_face_cells[{bf}] = {c} out of 0..{n}"));
            }
            if d >= 0 && d as usize >= n + nh {
                return refuse(format!(
                    "part {p}: b_nbr_cell[{bf}] = {d} is past the {nh} halo cells"
                ));
            }
        }
        Ok(())
    }

    /// Block-local aggregation (SPEC-LIT §141.10): §141.1 on the part's own
    /// cells, every processor face read as UNCOUPLED, so no aggregate spans
    /// parts.
    pub fn aggregate(&self) -> Result<BlockAggregation> {
        self.check()?;
        let n = self.n_cells as Label;
        let masked: Vec<Label> = self
            .b_nbr_cell
            .iter()
            .map(|&d| if d >= n { -1 } else { d })
            .collect();
        let agg = aggregate_level(&LevelView {
            n_cells: self.n_cells,
            owner: &self.owner,
            neighbour: &self.neighbour,
            diag: &self.diag,
            upper: &self.upper,
            lower: &self.lower,
            b_face_cells: &self.b_face_cells,
            b_nbr_cell: &masked,
            boundary_coeffs: &self.boundary_coeffs,
        })?;
        // Members are ascending and owned keys are ascending, so the first
        // member holds the smallest key.
        let root = (0..agg.n_coarse)
            .map(|g| self.cell_key[agg.agg_cells[agg.agg_offset[g] as usize] as usize])
            .collect();
        Ok(BlockAggregation { agg, root })
    }

    /// The coarse level and its lists, given every halo cell's aggregate on
    /// its owning part (`halo_agg`, that part's local id) and that
    /// aggregate's key (`halo_root`), both `[n_halo]` (SPEC-LIT §141.10).
    pub fn coarsen(
        &self,
        ba: &BlockAggregation,
        halo_agg: &[Label],
        halo_root: &[Label],
    ) -> Result<(DistCoarsening, DistLevel)> {
        self.check()?;
        let p = self.part;
        let (n, nh) = (self.n_cells, self.n_halo());
        if ba.agg.n_fine != n || ba.root.len() != ba.agg.n_coarse {
            return refuse(format!(
                "part {p}: the aggregation is of {} cells with {} roots for {} aggregates, the \
                 level has {n} cells",
                ba.agg.n_fine,
                ba.root.len(),
                ba.agg.n_coarse
            ));
        }
        if halo_agg.len() != nh || halo_root.len() != nh {
            return refuse(format!(
                "part {p}: {} halo aggregates and {} halo roots for {nh} halo cells",
                halo_agg.len(),
                halo_root.len()
            ));
        }
        let hp = self.halo.halo_parts()?;

        // The part-local product, (141.14)-(141.16).
        let plan = GalerkinPlan::new(&ba.agg, &self.owner, &self.b_face_cells)?;
        let local =
            plan.coarse_host(&self.diag, &self.upper, &self.lower, &self.boundary_coeffs)?;
        let n_c = local.n_cells;

        // Every fine processor face: (own aggregate, (part, remote aggregate,
        // its root), fine key, face).
        let mut pf: Vec<(Label, (Label, Label, Label), (Label, Label), usize)> = Vec::new();
        for bf in 0..self.b_face_cells.len() {
            let d = self.b_nbr_cell[bf];
            if d < n as Label {
                continue;
            }
            let h = d as usize - n;
            if halo_agg[h] < 0 {
                return refuse(format!("part {p}: halo cell {h} has aggregate {}", halo_agg[h]));
            }
            let i = ba.agg.cell_agg[self.b_face_cells[bf] as usize];
            pf.push((i, (hp[h], halo_agg[h], halo_root[h]), self.bface_key[bf], bf));
        }

        // The coarse halo: unique (owning part, remote aggregate), ascending.
        let mut slots: Vec<(Label, Label, Label)> = pf.iter().map(|x| x.1).collect();
        slots.sort_unstable();
        slots.dedup();
        for w in slots.windows(2) {
            if w[0].0 == w[1].0 && w[0].1 == w[1].1 {
                return refuse(format!(
                    "part {p}: aggregate {} of part {} arrives with two keys, {} and {}",
                    w[0].1, w[0].0, w[0].2, w[1].2
                ));
            }
            if w[0].0 == w[1].0 && w[0].2 >= w[1].2 {
                return refuse(format!(
                    "part {p}: part {}'s aggregates {} and {} are not in key order",
                    w[0].0, w[0].1, w[1].1
                ));
            }
        }
        let n_hc = slots.len();
        let mut nbr_parts: Vec<Label> = slots.iter().map(|s| s.0).collect();
        nbr_parts.dedup();
        let mut recv_offset = vec![0 as Label; nbr_parts.len() + 1];
        let mut send_offset = vec![0 as Label; nbr_parts.len() + 1];
        let mut send_index: Vec<Label> = Vec::new();
        for (k, &q) in nbr_parts.iter().enumerate() {
            recv_offset[k + 1] =
                recv_offset[k] + slots.iter().filter(|s| s.0 == q).count() as Label;
            let mut send: Vec<Label> = pf.iter().filter(|x| x.1 .0 == q).map(|x| x.0).collect();
            send.sort_unstable();
            send.dedup();
            send_index.extend_from_slice(&send);
            send_offset[k + 1] = send_index.len() as Label;
        }

        // The coarse processor faces (own aggregate, halo slot), ascending,
        // each with its fine faces in key order.
        let mut faces: Vec<(Label, usize, (Label, Label), usize)> = Vec::with_capacity(pf.len());
        for x in &pf {
            let hc = slots
                .binary_search_by(|s| (s.0, s.1).cmp(&(x.1 .0, x.1 .1)))
                .map_err(|_| Error::Config(format!("amg::dist: part {p}: lost a halo slot")))?;
            faces.push((x.0, hc, x.2, x.3));
        }
        faces.sort_unstable();
        let mut proc_offset: Vec<Label> = vec![0];
        let mut proc_src: Vec<Label> = Vec::with_capacity(faces.len());
        let mut b_face_cells: Vec<Label> = Vec::new();
        let mut b_nbr_cell: Vec<Label> = Vec::new();
        let mut boundary_coeffs: Vec<Scalar> = Vec::new();
        let mut bface_key: Vec<(Label, Label)> = Vec::new();
        let mut t = 0usize;
        while t < faces.len() {
            let (i, hc) = (faces[t].0, faces[t].1);
            // (141.40): one f64 accumulator, in key order, one rounding.
            let mut acc = 0.0f64;
            let mut last: Option<(Label, Label)> = None;
            while t < faces.len() && faces[t].0 == i && faces[t].1 == hc {
                if last == Some(faces[t].2) {
                    return refuse(format!(
                        "part {p}: two processor faces share the key {:?}",
                        faces[t].2
                    ));
                }
                last = Some(faces[t].2);
                let bf = faces[t].3;
                acc += -(self.boundary_coeffs[bf] as f64);
                proc_src.push(bf as Label);
                t += 1;
            }
            proc_offset.push(proc_src.len() as Label);
            let a_ih = acc as Scalar;
            let (ri, rh) = (ba.root[i as usize], slots[hc].2);
            b_face_cells.push(i);
            b_nbr_cell.push((n_c + hc) as Label);
            boundary_coeffs.push(-a_ih);
            bface_key.push((ri.min(rh), ri.max(rh)));
        }

        let mut cell_key = ba.root.clone();
        cell_key.extend(slots.iter().map(|s| s.2));
        let coarse = DistLevel {
            part: p,
            n_cells: n_c,
            owner: local.owner,
            neighbour: local.neighbour,
            diag: local.diag,
            upper: local.upper,
            lower: local.lower,
            b_face_cells,
            b_nbr_cell,
            boundary_coeffs,
            cell_key,
            bface_key,
            halo: LevelHalo {
                part: p,
                n_cells: n_c,
                n_halo: n_hc,
                nbr_parts,
                send_offset,
                send_index,
                recv_offset,
            },
        };
        coarse.check()?;
        Ok((
            DistCoarsening {
                plan,
                proc_offset,
                proc_src,
            },
            coarse,
        ))
    }
}

impl BlockAggregation {
    /// Each local cell's aggregate and that aggregate's root, `[n_cells +
    /// n_halo]` with zero ghosts: the two fields the halo exchange fills to
    /// give [`DistLevel::coarsen`] its `halo_agg` and `halo_root`.
    pub fn cell_fields(&self, level: &DistLevel) -> Result<(Vec<Label>, Vec<Label>)> {
        if self.agg.n_fine != level.n_cells {
            return refuse(format!(
                "part {}: the aggregation is of {} cells, the level has {}",
                level.part, self.agg.n_fine, level.n_cells
            ));
        }
        let nl = level.n_cells + level.n_halo();
        let mut a = vec![0 as Label; nl];
        let mut r = vec![0 as Label; nl];
        for c in 0..level.n_cells {
            let g = self.agg.cell_agg[c];
            a[c] = g;
            r[c] = self.root[g as usize];
        }
        Ok((a, r))
    }
}

/// Every part's `(halo_agg, halo_root)` read directly off the other parts,
/// for an in-process hierarchy: the halo cell's key is found among its
/// owner's owned keys (SPEC-LIT §141.10).
pub fn halo_aggregates(
    levels: &[DistLevel],
    aggs: &[BlockAggregation],
) -> Result<Vec<(Vec<Label>, Vec<Label>)>> {
    if levels.len() != aggs.len() {
        return refuse(format!(
            "{} levels and {} aggregations",
            levels.len(),
            aggs.len()
        ));
    }
    let mut out = Vec::with_capacity(levels.len());
    for (p, l) in levels.iter().enumerate() {
        if l.part != p {
            return refuse(format!("level {p} is part {}'s", l.part));
        }
        let hp = l.halo.halo_parts()?;
        let mut ha = Vec::with_capacity(l.n_halo());
        let mut hr = Vec::with_capacity(l.n_halo());
        for (h, &q) in hp.iter().enumerate() {
            let key = l.cell_key[l.n_cells + h];
            let ql = levels.get(q as usize).ok_or_else(|| {
                Error::Config(format!("amg::dist: part {p}'s neighbour {q} has no level"))
            })?;
            let c = ql.cell_key[..ql.n_cells].binary_search(&key).map_err(|_| {
                Error::Config(format!(
                    "amg::dist: part {p}'s halo key {key} is not owned by part {q}"
                ))
            })?;
            let g = aggs[q as usize].agg.cell_agg[c];
            ha.push(g);
            hr.push(aggs[q as usize].root[g as usize]);
        }
        out.push((ha, hr));
    }
    Ok(out)
}

/// The rank's `(halo_agg, halo_root)` by two label exchanges over the
/// level's own halo plan (SPEC-LIT §141.10). A collective: every rank calls
/// it at the same point, rank `r` holding part `r`.
pub fn spmd_halo_aggregates(
    gpu: &Gpu,
    comm: &dyn Comm,
    level: &DistLevel,
    ba: &BlockAggregation,
) -> Result<(Vec<Label>, Vec<Label>)> {
    let mut halo = SpmdHalo::from_level(gpu, &level.halo, comm.size())?;
    let (a, r) = ba.cell_fields(level)?;
    let mut da = gpu.upload(&a)?;
    let mut dr = gpu.upload(&r)?;
    halo.label(gpu, comm, &mut da)?;
    halo.label(gpu, comm, &mut dr)?;
    let a = gpu.download(&da)?;
    let r = gpu.download(&dr)?;
    let n = level.n_cells;
    Ok((a[n..].to_vec(), r[n..].to_vec()))
}

/// The coarse levels of every part gathered into one whole level, numbered
/// by key (the (141.8) order of the union aggregation), with each coarse
/// face's upper from its lower-numbered row and its lower from the other
/// (SPEC-LIT §141.10). Every boundary face must be a processor face, so a
/// finest level is refused.
pub fn gather_level(levels: &[&DistLevel]) -> Result<HostLevel> {
    let mut keys: Vec<Label> = Vec::new();
    for (p, l) in levels.iter().enumerate() {
        l.check()?;
        if l.part != p {
            return refuse(format!("level {p} is part {}'s", l.part));
        }
        if l.n_proc_faces() != l.b_face_cells.len() {
            return refuse(format!(
                "part {p}: gather_level takes a coarse level; this one has {} boundary faces \
                 that are not processor faces",
                l.b_face_cells.len() - l.n_proc_faces()
            ));
        }
        keys.extend_from_slice(&l.cell_key[..l.n_cells]);
    }
    keys.sort_unstable();
    if keys.windows(2).any(|w| w[0] == w[1]) {
        return refuse("two parts own a cell with the same key".to_string());
    }
    let gid = |k: Label| -> Result<Label> {
        keys.binary_search(&k)
            .map(|i| i as Label)
            .map_err(|_| Error::Config(format!("amg::dist: key {k} is owned by no part")))
    };
    let n = keys.len();
    let mut diag = vec![0 as Scalar; n];
    // (row, column) -> value, every entry of the operator off the diagonal.
    let mut ent: Vec<((Label, Label), Scalar)> = Vec::new();
    for l in levels {
        for c in 0..l.n_cells {
            diag[gid(l.cell_key[c])? as usize] = l.diag[c];
        }
        for f in 0..l.owner.len() {
            let o = gid(l.cell_key[l.owner[f] as usize])?;
            let nb = gid(l.cell_key[l.neighbour[f] as usize])?;
            ent.push(((o, nb), l.upper[f]));
            ent.push(((nb, o), l.lower[f]));
        }
        for bf in 0..l.b_face_cells.len() {
            let i = gid(l.cell_key[l.b_face_cells[bf] as usize])?;
            let h = gid(l.cell_key[l.b_nbr_cell[bf] as usize])?;
            ent.push(((i, h), -l.boundary_coeffs[bf]));
        }
    }
    ent.sort_by_key(|e| e.0);
    if ent.windows(2).any(|w| w[0].0 == w[1].0) {
        return refuse("an entry of the gathered operator comes from two places".to_string());
    }
    let mut pairs: Vec<(Label, Label)> = ent.iter().map(|e| (e.0 .0.min(e.0 .1), e.0 .0.max(e.0 .1))).collect();
    pairs.sort_unstable();
    pairs.dedup();
    let find = |r: Label, c: Label| -> Result<Scalar> {
        ent.binary_search_by_key(&(r, c), |e| e.0)
            .map(|i| ent[i].1)
            .map_err(|_| {
                Error::Config(format!(
                    "amg::dist: entry ({r}, {c}) is missing while ({c}, {r}) is present"
                ))
            })
    };
    let mut upper = Vec::with_capacity(pairs.len());
    let mut lower = Vec::with_capacity(pairs.len());
    for &(i, j) in &pairs {
        upper.push(find(i, j)?);
        lower.push(find(j, i)?);
    }
    Ok(HostLevel {
        n_cells: n,
        owner: pairs.iter().map(|p| p.0).collect(),
        neighbour: pairs.iter().map(|p| p.1).collect(),
        diag,
        upper,
        lower,
    })
}

/// One part's hierarchy: `levels[0]` is the finest, `aggs[k]` and `steps[k]`
/// take `levels[k]` to `levels[k + 1]` (SPEC-LIT §141.10).
#[derive(Debug, Clone, PartialEq)]
pub struct DistHierarchy {
    pub levels: Vec<DistLevel>,
    pub aggs: Vec<BlockAggregation>,
    pub steps: Vec<DistCoarsening>,
}

fn check_config(cfg: &HierarchyConfig) -> Result<()> {
    if cfg.n_cells_coarsest == 0 {
        return refuse("n_cells_coarsest is 0; a hierarchy keeps at least one cell".to_string());
    }
    if cfg.max_levels == 0 {
        return refuse("max_levels is 0; the finest level counts as one".to_string());
    }
    Ok(())
}

impl DistHierarchy {
    /// Coarse levels below the finest.
    pub fn n_coarse_levels(&self) -> usize {
        self.levels.len() - 1
    }

    /// Every part's hierarchy in one process. The stopping rule is the
    /// serial one (§141.3) on the WHOLE level's cell count, so every part
    /// has the same number of levels: stop at `n_cells_coarsest` or
    /// `max_levels`, and drop a coarsening that does not shrink the whole
    /// level. `mats[p]` is part `p`'s folded matrix.
    pub fn build_in_process(
        dec: &Decomposition,
        mats: &[HostLduMatrix],
        cfg: &HierarchyConfig,
    ) -> Result<Vec<Self>> {
        check_config(cfg)?;
        if mats.len() != dec.n_parts {
            return refuse(format!("{} matrices for {} parts", mats.len(), dec.n_parts));
        }
        let mut hs: Vec<Self> = Vec::with_capacity(dec.n_parts);
        for (part, a) in dec.parts.iter().zip(mats) {
            hs.push(Self {
                levels: vec![DistLevel::fine(part, a)?],
                aggs: Vec::new(),
                steps: Vec::new(),
            });
        }
        loop {
            let n_l: usize = hs.iter().map(|h| h.levels.last().map_or(0, |l| l.n_cells)).sum();
            if n_l <= cfg.n_cells_coarsest || hs[0].levels.len() >= cfg.max_levels {
                break;
            }
            let lv: Vec<DistLevel> = hs.iter().map(|h| h.levels.last().cloned()).collect::<Option<_>>()
                .ok_or_else(|| Error::Config("amg::dist: a part has no level".to_string()))?;
            let ag: Vec<BlockAggregation> =
                lv.iter().map(|l| l.aggregate()).collect::<Result<_>>()?;
            let halos = halo_aggregates(&lv, &ag)?;
            let mut next = Vec::with_capacity(hs.len());
            for ((l, a), (ha, hr)) in lv.iter().zip(&ag).zip(&halos) {
                next.push(l.coarsen(a, ha, hr)?);
            }
            let n_c: usize = next.iter().map(|x| x.1.n_cells).sum();
            if n_c >= n_l {
                break;
            }
            for ((h, a), (s, c)) in hs.iter_mut().zip(ag).zip(next) {
                h.aggs.push(a);
                h.steps.push(s);
                h.levels.push(c);
            }
        }
        Ok(hs)
    }

    /// The rank's hierarchy, SPMD: rank `r` holds part `r` and its folded
    /// matrix. A collective with the same stopping rule as
    /// [`Self::build_in_process`]; a step that fails on any rank fails on
    /// every rank, by a vote before the next collective.
    pub fn build_spmd(
        gpu: &Gpu,
        comm: &dyn Comm,
        part: &PartMesh,
        a: &HostLduMatrix,
        cfg: &HierarchyConfig,
    ) -> Result<Self> {
        let fine = check_config(cfg).and_then(|_| DistLevel::fine(part, a));
        let n0 = vote(comm, fine.as_ref().map(|l| l.n_cells))?;
        let mut h = Self {
            levels: vec![fine?],
            aggs: Vec::new(),
            steps: Vec::new(),
        };
        let mut n_l = n0;
        loop {
            if n_l <= cfg.n_cells_coarsest || h.levels.len() >= cfg.max_levels {
                break;
            }
            let lv = h.levels.last().cloned().ok_or_else(|| {
                Error::Config("amg::dist: the hierarchy has no level".to_string())
            })?;
            let ag = lv.aggregate();
            vote(comm, ag.as_ref().map(|_| 0))?;
            let ag = ag?;
            let (ha, hr) = spmd_halo_aggregates(gpu, comm, &lv, &ag)?;
            let next = lv.coarsen(&ag, &ha, &hr);
            let n_c = vote(comm, next.as_ref().map(|x| x.1.n_cells))?;
            let (s, c) = next?;
            if n_c >= n_l {
                break;
            }
            h.aggs.push(ag);
            h.steps.push(s);
            h.levels.push(c);
            n_l = n_c;
        }
        Ok(h)
    }
}

/// Every rank's `(ok, count)`; the summed count when all are ok, and the
/// same refusal on every rank otherwise.
pub(crate) fn vote<E: std::fmt::Display>(
    comm: &dyn Comm,
    mine: std::result::Result<usize, E>,
) -> Result<usize> {
    let mut b = [0u8; 16];
    if let Ok(n) = &mine {
        b[..8].copy_from_slice(&1u64.to_le_bytes());
        b[8..].copy_from_slice(&(*n as u64).to_le_bytes());
    }
    let mut all = vec![0u8; 16 * comm.size()];
    comm.allgather(&b, &mut all)?;
    let mut sum = 0usize;
    for (q, c) in all.chunks_exact(16).enumerate() {
        let ok = u64::from_le_bytes(c[..8].try_into().expect("8 bytes"));
        if ok != 1 {
            return match mine {
                Err(e) => refuse(format!("rank {}: {e}", comm.rank())),
                Ok(_) => refuse(format!("rank {q} failed a hierarchy step")),
            };
        }
        sum += u64::from_le_bytes(c[8..].try_into().expect("8 bytes")) as usize;
    }
    Ok(sum)
}
