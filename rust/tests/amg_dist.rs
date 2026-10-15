// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT §141.10's gates (AMG-07a) - the decomposed levels of the GAMG
//! hierarchy, in BOTH builds.
//!
//! An integration test because the library's own test target does not
//! compile under `--features single`. Class D of docs/17 §5.1: every
//! comparison is `to_bits()` or `==` on ids. The matrices are the
//! non-dyadic (141.18) and the §2 laplacian, so a sum taken in another
//! order would show in the bits.
//!
//! Provenance: ORIGINAL - the gates of this project's own design (SPEC-LIT
//! §141.10). No external source. No GPL-licensed source was consulted.

use ofgpu::amg::{
    gather_level, Aggregation, BlockAggregation, DistHierarchy, DistLevel, GalerkinPlan,
    HierarchyConfig, HostHierarchy, HostLevel, LevelView,
};
use ofgpu::blockgen::{build_mesh, BlockSpec, GradedAxis};
use ofgpu::comm::{Comm, ThreadComm};
use ofgpu::decompose::{Decomposition, PartitionMethod};
use ofgpu::device::Gpu;
use ofgpu::halo::{HaloExchange, LevelHalo, SpmdHalo};
use ofgpu::ldu::HostLduMatrix;
use ofgpu::mesh::HostMesh;
use ofgpu::{Label, Result, Scalar};

#[allow(clippy::useless_conversion)]
fn sc(x: f64) -> Scalar {
    x as Scalar
}

/// A uniform box, every patch a wall, the axes of `cyclic` cyclic. Cells
/// are numbered `i + nx*(j + ny*k)`.
fn box_mesh(nx: usize, ny: usize, nz: usize, cyclic: &[usize]) -> HostMesh {
    let axis = |n: usize| GradedAxis {
        lo: 0.0,
        hi: n as Scalar,
        n,
        expansion: 1.0,
        two_sided: false,
    };
    let mut b = BlockSpec {
        x: axis(nx),
        y: axis(ny),
        z: axis(nz),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["wall"; 6].map(String::from),
        cyclic: Vec::new(),
    };
    for &a in cyclic {
        b.set_cyclic_axis(a).expect("cyclic axis");
    }
    build_mesh(&b).expect("box mesh")
}

/// `A(c, d)` of (141.18).
fn a_of(c: usize, d: usize) -> f64 {
    if c < d {
        -(0.1 + ((11 * c + 3 * d) % 7) as f64 / 3.0)
    } else {
        -(0.1 + ((5 * d + c) % 3) as f64 / 7.0)
    }
}

/// The non-dyadic matrix (141.18), with `boundary_coeffs = -A(c, d)` on a
/// coupled face.
fn nondyadic(m: &HostMesh) -> HostLduMatrix {
    let mut a = HostLduMatrix::zeros(m);
    for (c, d) in a.diag.iter_mut().enumerate() {
        *d = sc(20.0 + c as f64 / 9.0);
    }
    for f in 0..m.n_internal_faces {
        let (o, nb) = (m.owner[f] as usize, m.neighbour[f] as usize);
        a.upper[f] = sc(a_of(o, nb));
        a.lower[f] = sc(a_of(nb, o));
    }
    for bf in 0..m.n_boundary_faces {
        let d = m.b_nbr_cell[bf];
        if d >= 0 {
            a.boundary_coeffs[bf] = sc(-a_of(m.b_face_cells[bf] as usize, d as usize));
        }
    }
    a
}

/// The §2 laplacian, folded (`internal_coeffs` in `diag`).
fn poisson_folded(m: &HostMesh) -> HostLduMatrix {
    let mut a = HostLduMatrix::zeros(m);
    for f in 0..m.n_internal_faces {
        let g = m.mag_sf[f] * m.delta_coeffs[f];
        a.upper[f] = g;
        a.lower[f] = g;
        a.diag[m.owner[f] as usize] -= g;
        a.diag[m.neighbour[f] as usize] -= g;
    }
    for bf in 0..m.n_boundary_faces {
        let c = m.b_face_cells[bf] as usize;
        a.diag[c] -= m.b_mag_sf[bf] * m.b_delta_coeffs[bf];
    }
    a
}

fn cases() -> Vec<(&'static str, HostMesh, HostLduMatrix)> {
    let a = box_mesh(9, 7, 5, &[]);
    let b = box_mesh(8, 6, 5, &[0, 1, 2]);
    let c = box_mesh(12, 10, 8, &[]);
    let (aa, ab, ac) = (nondyadic(&a), nondyadic(&b), poisson_folded(&c));
    vec![
        ("9x7x5 (141.18)", a, aa),
        ("8x6x5 cyclic xyz (141.18)", b, ab),
        ("12x10x8 laplacian", c, ac),
    ]
}

fn methods(n: usize, p: usize) -> Vec<(&'static str, PartitionMethod)> {
    vec![
        ("hilbert", PartitionMethod::Hilbert),
        ("linear", PartitionMethod::Linear),
        (
            "round robin",
            PartitionMethod::Explicit((0..n).map(|c| ((c / 3) % p) as Label).collect()),
        ),
    ]
}

fn cfg() -> HierarchyConfig {
    HierarchyConfig {
        n_cells_coarsest: 4,
        ..Default::default()
    }
}

fn bits(x: &[Scalar]) -> Vec<u64> {
    x.iter().map(|v| v.to_bits() as u64).collect()
}

fn same_level(got: &HostLevel, want: &HostLevel, what: &str) {
    assert_eq!(got.n_cells, want.n_cells, "{what}: n_cells");
    assert_eq!(got.owner, want.owner, "{what}: owner");
    assert_eq!(got.neighbour, want.neighbour, "{what}: neighbour");
    assert_eq!(bits(&got.diag), bits(&want.diag), "{what}: diag bits");
    assert_eq!(bits(&got.upper), bits(&want.upper), "{what}: upper bits");
    assert_eq!(bits(&got.lower), bits(&want.lower), "{what}: lower bits");
}

/// The fine level gathered back from the parts: every coefficient put back
/// at its global slot, a cut face's `-boundary_coeffs` at the side it came
/// from.
fn gather_fine(d: &Decomposition, m: &HostMesh, mats: &[HostLduMatrix]) -> HostLduMatrix {
    let mut g = HostLduMatrix::zeros(m);
    let nif = m.n_internal_faces;
    for (p, part) in d.parts.iter().enumerate() {
        let (pm, a) = (&part.mesh, &mats[p]);
        for c in 0..pm.n_cells {
            g.diag[part.global_cell[c] as usize] = a.diag[c];
        }
        for f in 0..pm.n_internal_faces {
            let gf = pm.global_face[f] as usize;
            g.upper[gf] = a.upper[f];
            g.lower[gf] = a.lower[f];
        }
        for bf in 0..pm.n_boundary_faces {
            let gs = pm.global_face[pm.n_internal_faces + bf] as usize;
            if gs >= nif {
                g.boundary_coeffs[gs - nif] = a.boundary_coeffs[bf];
            } else if d.cell_part[m.owner[gs] as usize] == p as Label {
                g.upper[gs] = -a.boundary_coeffs[bf];
            } else {
                g.lower[gs] = -a.boundary_coeffs[bf];
            }
        }
    }
    g
}

/// The union of the parts' block-local aggregations of level `k`, as a
/// whole-level map numbered by key, and the part of every whole aggregate.
fn union_map(hs: &[DistHierarchy], k: usize) -> (Vec<Label>, Vec<usize>) {
    let mut fine_keys: Vec<Label> = Vec::new();
    let mut roots: Vec<(Label, usize)> = Vec::new();
    for (p, h) in hs.iter().enumerate() {
        let l = &h.levels[k];
        fine_keys.extend_from_slice(&l.cell_key[..l.n_cells]);
        roots.extend(h.aggs[k].root.iter().map(|&r| (r, p)));
    }
    fine_keys.sort_unstable();
    roots.sort_unstable();
    let mut map = vec![-1 as Label; fine_keys.len()];
    for h in hs {
        let (l, ba): (&DistLevel, &BlockAggregation) = (&h.levels[k], &h.aggs[k]);
        for c in 0..l.n_cells {
            let gc = fine_keys.binary_search(&l.cell_key[c]).expect("fine key");
            let r = ba.root[ba.agg.cell_agg[c] as usize];
            let gi = roots.binary_search_by_key(&r, |x| x.0).expect("root");
            map[gc] = gi as Label;
        }
    }
    assert!(map.iter().all(|&g| g >= 0), "every cell is in an aggregate");
    (map, roots.iter().map(|x| x.1).collect())
}

/// **The gate.** Every level of a P-part hierarchy, gathered, is bit for bit
/// the Galerkin product of the gathered level above it under the union of
/// the block-local aggregations, in the serial list order; no aggregate
/// spans parts. Non-vacuous: cross-part coarse faces exist, and some of
/// their slots and of the diagonal slots sum three or more sources.
#[test]
fn gathered_coarse_levels_are_the_galerkin_product_of_the_gathered_fine_level() {
    let mut levels_checked = 0usize;
    let mut cross_faces = 0usize;
    let mut cross_slots_3 = 0usize;
    for (name, m, a) in cases() {
        for p in 2..=4usize {
            for (mname, method) in methods(m.n_cells, p) {
                let what0 = format!("{name}, P={p}, {mname}");
                let d = Decomposition::build(&m, p, &method).expect("cut");
                assert!(d.n_cut_faces > 0, "{what0}: nothing was cut");
                let mats: Vec<HostLduMatrix> = (0..p)
                    .map(|q| d.split_matrix(&m, q, &a).expect("split"))
                    .collect();
                let g0 = gather_fine(&d, &m, &mats);
                assert_eq!(bits(&g0.diag), bits(&a.diag), "{what0}: gathered diag");
                assert_eq!(bits(&g0.upper), bits(&a.upper), "{what0}: gathered upper");
                assert_eq!(bits(&g0.lower), bits(&a.lower), "{what0}: gathered lower");
                assert_eq!(
                    bits(&g0.boundary_coeffs),
                    bits(&a.boundary_coeffs),
                    "{what0}: gathered boundary_coeffs"
                );

                let hs = DistHierarchy::build_in_process(&d, &mats, &cfg()).expect("hierarchy");
                let nl = hs[0].levels.len();
                assert!(hs.iter().all(|h| h.levels.len() == nl), "{what0}: level counts");
                assert!(nl >= 2, "{what0}: only {nl} level, so nothing is gated");
                let mut gathered: Option<HostLevel> = None;
                for k in 0..nl - 1 {
                    let what = format!("{what0}, level {k} -> {}", k + 1);
                    let (map, agg_part) = union_map(&hs, k);
                    // No aggregate spans parts: every member of a whole
                    // aggregate lives on the part that owns its root.
                    let cell_part: Vec<usize> = match &gathered {
                        None => d.cell_part.iter().map(|&q| q as usize).collect(),
                        Some(_) => {
                            let mut v: Vec<(Label, usize)> = Vec::new();
                            for (q, h) in hs.iter().enumerate() {
                                let l = &h.levels[k];
                                v.extend(l.cell_key[..l.n_cells].iter().map(|&x| (x, q)));
                            }
                            v.sort_unstable();
                            v.iter().map(|x| x.1).collect()
                        }
                    };
                    for (c, &g) in map.iter().enumerate() {
                        assert_eq!(cell_part[c], agg_part[g as usize], "{what}: aggregate {g} spans parts");
                    }
                    let (agg, plan, want) = match &gathered {
                        None => {
                            let view = LevelView {
                                n_cells: m.n_cells,
                                owner: &m.owner,
                                neighbour: &m.neighbour,
                                diag: &a.diag,
                                upper: &a.upper,
                                lower: &a.lower,
                                b_face_cells: &m.b_face_cells,
                                b_nbr_cell: &m.b_nbr_cell,
                                boundary_coeffs: &a.boundary_coeffs,
                            };
                            let agg = Aggregation::from_map(&view, map).expect("from_map");
                            let plan = GalerkinPlan::for_mesh(&agg, &m).expect("plan");
                            let want = plan
                                .coarse_host(&a.diag, &a.upper, &a.lower, &a.boundary_coeffs)
                                .expect("coarse");
                            (agg, plan, want)
                        }
                        Some(gk) => {
                            let agg = Aggregation::from_map(&gk.view(), map).expect("from_map");
                            let plan = GalerkinPlan::new(&agg, &gk.owner, &[]).expect("plan");
                            let want =
                                plan.coarse_host(&gk.diag, &gk.upper, &gk.lower, &[]).expect("coarse");
                            (agg, plan, want)
                        }
                    };
                    let parts: Vec<&DistLevel> = hs.iter().map(|h| &h.levels[k + 1]).collect();
                    let got = gather_level(&parts).expect("gather");
                    same_level(&got, &want, &what);
                    levels_checked += 1;
                    for f in 0..agg.n_coarse_faces() {
                        let (i, j) = (agg.coarse_owner[f] as usize, agg.coarse_neighbour[f] as usize);
                        if agg_part[i] == agg_part[j] {
                            continue;
                        }
                        cross_faces += 1;
                        for s in [agg.n_coarse + f, agg.n_coarse + agg.n_coarse_faces() + f] {
                            if plan.slot_offset[s + 1] - plan.slot_offset[s] >= 3 {
                                cross_slots_3 += 1;
                            }
                        }
                    }
                    gathered = Some(got);
                }
            }
        }
    }
    println!(
        "AMG-07a gate: {levels_checked} gathered levels bitwise, {cross_faces} cross-part coarse \
         faces, {cross_slots_3} of their slots sum >= 3 sources"
    );
    assert!(levels_checked >= 2 * 27, "levels checked: {levels_checked}");
    assert!(cross_faces > 0 && cross_slots_3 > 0, "the gate is vacuous");
}

/// One part is the serial hierarchy: the same aggregations, plans and levels,
/// bit for bit.
#[test]
fn one_part_is_the_serial_hierarchy() {
    for (name, m, a) in cases() {
        let d = Decomposition::build(&m, 1, &PartitionMethod::Linear).expect("cut");
        let mats = vec![d.split_matrix(&m, 0, &a).expect("split")];
        let hs = DistHierarchy::build_in_process(&d, &mats, &cfg()).expect("hierarchy");
        let h = HostHierarchy::build(&m, &a, &cfg()).expect("serial");
        let dh = &hs[0];
        assert_eq!(dh.n_coarse_levels(), h.n_coarse_levels(), "{name}: levels");
        assert!(h.n_coarse_levels() >= 3, "{name}: {} levels", h.n_coarse_levels());
        for k in 0..h.n_coarse_levels() {
            assert_eq!(dh.aggs[k].agg, h.aggs[k], "{name}: aggregation {k}");
            assert_eq!(dh.steps[k].plan, h.plans[k], "{name}: plan {k}");
            let got = gather_level(&[&dh.levels[k + 1]]).expect("gather");
            same_level(&got, &h.levels[k], &format!("{name}: level {}", k + 1));
        }
    }
}

fn f_of(key: Label) -> Scalar {
    sc(1.0 + 0.5 * key as f64)
}

fn field_of(l: &DistLevel) -> Vec<Scalar> {
    let mut v = vec![sc(0.0); l.n_cells + l.n_halo()];
    for c in 0..l.n_cells {
        v[c] = f_of(l.cell_key[c]);
    }
    v
}

fn halo_is_owner(l: &DistLevel, got: &[Scalar], what: &str) -> usize {
    for c in 0..l.n_cells + l.n_halo() {
        assert_eq!(
            got[c].to_bits(),
            f_of(l.cell_key[c]).to_bits(),
            "{what}: local cell {c} (key {})",
            l.cell_key[c]
        );
    }
    l.n_halo()
}

/// The SPMD build (a `ThreadComm` rank per part, each with its own `Gpu` on
/// device 0) is the in-process build, and every coarse level's halo plan,
/// SPMD and in-process, brings each ghost its owner's value.
#[test]
fn an_spmd_build_is_the_in_process_build_and_its_coarse_halos_carry_the_owners_value() {
    let gpu0 = match Gpu::new(0) {
        Ok(g) => g,
        Err(_) => return,
    };
    let mut ghosts = 0usize;
    for (name, m, a) in cases().into_iter().take(2) {
        for p in 2..=4usize {
            let what0 = format!("{name}, P={p}");
            let d = Decomposition::build(&m, p, &PartitionMethod::Hilbert).expect("cut");
            let mats: Vec<HostLduMatrix> =
                (0..p).map(|q| d.split_matrix(&m, q, &a).expect("split")).collect();
            let hs = DistHierarchy::build_in_process(&d, &mats, &cfg()).expect("hierarchy");
            assert!(hs[0].n_coarse_levels() >= 2, "{what0}: levels");

            // In process: one HaloExchange per level from the level plans.
            for k in 0..hs[0].levels.len() {
                let plans: Vec<LevelHalo> = hs.iter().map(|h| h.levels[k].halo.clone()).collect();
                let mut ex = HaloExchange::from_levels(&gpu0, &plans).expect("exchange");
                let mut fs: Vec<_> = hs
                    .iter()
                    .map(|h| gpu0.upload(&field_of(&h.levels[k])).expect("upload"))
                    .collect();
                ex.scalar(&gpu0, &mut fs).expect("scalar");
                for (q, h) in hs.iter().enumerate() {
                    let got = gpu0.download(&fs[q]).expect("download");
                    ghosts += halo_is_owner(&h.levels[k], &got, &format!("{what0}, in-process level {k} part {q}"));
                }
            }

            let got = ThreadComm::run(p, |comm| -> Result<(DistHierarchy, Vec<Vec<Scalar>>)> {
                let gpu = Gpu::new(0)?;
                let r = comm.rank();
                let h = DistHierarchy::build_spmd(&gpu, comm, &d.parts[r], &mats[r], &cfg())?;
                let mut fields = Vec::new();
                for l in &h.levels {
                    let mut halo = SpmdHalo::from_level(&gpu, &l.halo, comm.size())?;
                    let mut f = gpu.upload(&field_of(l))?;
                    halo.scalar(&gpu, comm, &mut f)?;
                    fields.push(gpu.download(&f)?);
                }
                Ok((h, fields))
            })
            .expect("spmd run");
            for (r, res) in got.iter().enumerate() {
                let (h, fields) = res.as_ref().unwrap_or_else(|e| panic!("{what0}: rank {r}: {e}"));
                assert_eq!(h, &hs[r], "{what0}: rank {r}'s hierarchy");
                for (k, l) in h.levels.iter().enumerate() {
                    assert_eq!(bits(&l.diag), bits(&hs[r].levels[k].diag), "{what0}: diag bits");
                    assert_eq!(
                        bits(&l.boundary_coeffs),
                        bits(&hs[r].levels[k].boundary_coeffs),
                        "{what0}: processor coefficient bits"
                    );
                    ghosts += halo_is_owner(l, &fields[k], &format!("{what0}, SPMD level {k} rank {r}"));
                }
            }
        }
    }
    println!("AMG-07a halo: {ghosts} ghost values compared");
    assert!(ghosts > 0, "no ghost was compared");
}

fn refused<T: std::fmt::Debug>(r: Result<T>, needle: &str) {
    let e = r.expect_err("must be refused").to_string();
    assert!(e.contains(needle), "message {e:?} lacks {needle:?}");
}

/// The refusals of §141.10, by name.
#[test]
fn a_malformed_decomposed_level_is_refused_by_name() {
    let m = box_mesh(6, 5, 4, &[]);
    let a = nondyadic(&m);
    let d = Decomposition::build(&m, 2, &PartitionMethod::Linear).expect("cut");
    let a0 = d.split_matrix(&m, 0, &a).expect("split");
    let l = DistLevel::fine(&d.parts[0], &a0).expect("fine");
    let ba = l.aggregate().expect("aggregate");
    refused(l.coarsen(&ba, &[], &[]), "amg::dist: part 0: 0 halo aggregates");
    let mut bad = l.clone();
    bad.cell_key.swap(0, 1);
    refused(bad.aggregate(), "not strictly ascending");
    refused(gather_level(&[&l]), "gather_level takes a coarse level");
    let mut short = a0.clone();
    short.diag.pop();
    refused(DistLevel::fine(&d.parts[0], &short), "amg::dist: part 0: the matrix has");
    let mut bad = l.clone();
    bad.b_nbr_cell[0] = (l.n_cells + l.n_halo()) as Label;
    refused(bad.check(), "past the");
    // A map not numbered by (141.8) is not an aggregation.
    let view = LevelView {
        n_cells: m.n_cells,
        owner: &m.owner,
        neighbour: &m.neighbour,
        diag: &a.diag,
        upper: &a.upper,
        lower: &a.lower,
        b_face_cells: &m.b_face_cells,
        b_nbr_cell: &m.b_nbr_cell,
        boundary_coeffs: &a.boundary_coeffs,
    };
    let mut map = vec![0 as Label; m.n_cells];
    map[0] = 1;
    refused(Aggregation::from_map(&view, map), "not numbered by (141.8)");
    let mut cfg0 = cfg();
    cfg0.n_cells_coarsest = 0;
    let mats: Vec<HostLduMatrix> = (0..2).map(|q| d.split_matrix(&m, q, &a).expect("split")).collect();
    refused(DistHierarchy::build_in_process(&d, &mats, &cfg0), "n_cells_coarsest is 0");
    // A level plan that sends a cell it does not own.
    if let Ok(gpu) = Gpu::new(0) {
        let mut plans: Vec<LevelHalo> = d.parts.iter().map(LevelHalo::of_part).collect();
        plans[1].send_index[0] = plans[1].n_cells as Label;
        refused(HaloExchange::from_levels(&gpu, &plans).map(|_| ()), "is not an owned cell");
    }
}
