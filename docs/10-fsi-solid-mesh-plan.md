# 10 — Fluid–structure interaction, the solid mesh, and the geometry a case starts from

**Status:** adopted 2026-09-12. Part II of the solver programme. Part I is `docs/09-thermal-structural-plan.md`
(heat transfer and thermal deformation); this document keeps Part I's content, re-numbers its sections where
the tree forbids the old numbers (§D), adds the mesh that moves, FSI, the solid mesh and the geometry tools,
and splits all of it into coding units sized for a 256k-context coder (§I).

Planned and reviewed by Opus 5; coded by GLM-5.3-Flash one unit at a time; every unit verified and committed
by an Opus 5 supervisor. No GPL/LGPL/AGPL source is consulted at any point (Gmsh and pymeshlab are run as
separate programs by the Python tools, as `tools/mesh` already does — `docs/06-mesh-oss-2024.md` §"외부 도구").

The facts this plan rests on were read from the tree on 2026-09-12 and are kept beside the briefs
(`facts-solver-solid.md`, `facts-solver-mesh-time.md`, `facts-mesh-tools.md`, `facts-automesher.md`,
`facts-gui-server.md`, `facts-fsi-sources.md`); a brief cites them by section so a coder never re-derives them.

## A. What the product must do

A user, or the assistant acting for them, opens a geometry that has fluid and solid parts; names the solids
and edits them where needed; meshes the fluid and each solid as its own region with a conformal interface, or
brings an existing mesh; runs the fluid alone (every existing driver, unchanged), or conjugate heat, or thermal
stress, or fluid–structure interaction on a mesh that moves; and sees the solid's stress and deformed shape
beside the fluid's fields, with the same viewer, the same assistant, the same run monitor. Later, a body that
moves too far for a deforming mesh is carried by an overset mesh. All of it on the GPU-resident solver this
repository already has, and all of it gated.

## B. The decisions that make it tractable (each with the fact that forced it)

1. **One mesh contract for everything: the region layout** (§C) — one complete polyMesh per region and one
   manifest naming the regions and their conformal interface patch pairs. This is already the crate's own
   multi-region model (`cht::ThermalMesh::build` concatenates separate `HostMesh`es and pairs patches by face
   centroid, SPEC-LIT §47.4); `cellZones` are read by nothing and written by nothing, by design
   (`sources.rs:40`). Every producer (the Gmsh tools, the automesher, the splitter for an imported mesh, the
   GUI) writes this layout; every consumer reads it. Nothing else is shared between the streams.
2. **The solid is solved on its own region mesh, not through masked kernels.** Every `fv`/`ldu`/`solver`
   kernel is launched over a whole `GpuMesh`; a "RegionView" with restricted assembly has nothing to build on,
   and the cheapest faithful restriction is the region's own `HostMesh`, which `LoweredChtCase.meshes` already
   holds (facts-solver-solid §11.2). The conjugate temperature stays on the concatenated mesh; a solid region
   reads its `T` as the slice `cells()` of that field (concatenation preserves order). §93's per-region
   residual is a ranged reduction over that slice, not a new matrix.
3. **The solid solver is Part I's §95 with the prototype's design made binding — and it is a COMPACT-BODY
   solver.** TS-0 measured that bare Picard on a body with a free surface diverges at ν = 0.45 and 0.49 and
   contracts at 0.87 (not the derived 0.625) at ν = 0.2; Aitken Δ² plus a traction condition solved at the face
   with three sub-passes converges at every ν tried (71 outer iterations at 0.45). **S6b then measured the shape
   TS-0 never entered** (`docs/09` §F.1b): the end-loaded cantilever stalls at an observed contraction of 0.95–1.00
   from about 2.5:1 upward under Aitken, and at 10:1 under everything tried — bare Picard, Aitken, Anderson at
   depths 3/5/10, and the implicit coefficient of the split enlarged 1.5/2/4× — on three meshes and at ν = 0.2 and
   0.3. Three things are therefore binding. **Anderson acceleration at depth five is the outer loop's relaxation**
   (Walker & Ni 2011, DOI 10.1137/10078356X; the same mixing over an interface is IQN-ILS, Degroote, Bathe &
   Vierendeels, DOI 10.1016/j.compstruc.2008.11.013): it beats Aitken everywhere Aitken finishes (30 outer
   iterations at ν = 0.45 against 71) and converges 2.5:1 and 5:1, which Aitken cannot reach. The face-solved
   traction stays mandatory. And **a bending-dominated slender body is refused by name** — slenderness above 5,
   computed from the volume-weighted covariance of the cell centres — with the block-coupled matrix (Cardiff,
   Tuković, Jasak & Ivanković 2016, DOI 10.1016/j.compstruc.2016.07.004) named as the route. Gate 95-A is not
   written; the cantilever is the refusal's test; release 1 stands on 95-B, 95-C, 95-F and S7's thick cylinder.
   The host prototype (`src/solid/prototype.rs`) remains the reference the device kernels are diffed against.
4. **Mesh motion is ALE with the space conservation law, on the buffers the mesh already has.** The device
   geometry recompute exists as four kernels producing all sixteen arrays (`cuda/meshgeom.cu`), but re-uploads
   its inputs every call; `GpuMesh` buffers are `pub DevBuf` and kernels index them directly, so the ALE step
   is a resident variant of that recompute writing in place, plus a volume history the ddt kernels do not
   have today (one `V` for all levels). Every convective consumer of `phi` funnels through six `fv.rs`
   functions and one turbulence chokepoint, so the relative flux is one buffer handed to them. Interior motion
   is inverse-distance weighting first (no solve, no point adjacency needed beyond the face→point CSR).
5. **FSI is partitioned, in one executable, on conformal interfaces first.** Fluid step → traction on the
   interface faces (a patch traction integrator that does not exist yet and also yields drag/lift) → solid
   step → interface displacement → Aitken, then IQN-ILS → mesh motion → repeat to an interface tolerance.
   Steady FSI1 before the dynamic cases; the fluid-only CFD1–CFD3 rows of the same benchmark gate the fluid on
   the same mesh before any coupling. Small strain is a stated limit: CSM1/CSM3/FSI2 deflect 6–8 cm on a 35 cm
   flap and their references are St. Venant–Kirchhoff, so those are band statements with the expected sign of
   the linear model's error written first; FSI1 (0.8 mm) is the clean gate; finite strain is named for later.
6. **The solid mesh comes from the Gmsh pipeline first, the automesher second.** `step_mesh.py` cuts every
   solid out of the fluid today (`occ.cut`, one volume enforced); keeping the declared solids and fragmenting
   gives conformal shared faces for free (probed: 2 volumes, shared faces meshed once). A one-cell-thick hex
   mesh is reachable (2-D fragment + extrude with recombine), which is how the Turek–Hron mesh is built. The
   automesher's own judgement (facts-automesher §8) is that the smallest change is "castellate and snap one
   mesh, split into region meshes after snap, layers per region" — it is scheduled after its uncommitted driver
   work is landed, not instead of the Gmsh route.
7. **Geometry is read, listed, edited and written by a separate Python tool on Gmsh's OpenCASCADE kernel**,
   run as a subprocess by the GUI server the way the mesher is. Facts that shape it: there is no
   `occ.exportShapes`, export is `gmsh.write` by extension; physical names survive only in `.xao`/`.msh`, so the
   tool keeps a JSON sidecar of names and materials beside the STEP; a STEP round trip duplicates a shared face,
   so import is followed by `removeAllDuplicates`/fragment; IGES imports as surfaces only.
8. **The GUI's geometry server (SRV3) must be rebuilt** — it was destroyed uncommitted on 2026-09-11 by a
   worktree removal that followed `gui/node_modules` junctions, and the geometry tab never ran. Its brief,
   coder report and captured diffs survive and are the spec. Rule from that loss: no worktree ever has
   `gui/node_modules` junctioned in; every unit's output is committed before the next dispatch.
9. **The existing drivers are not touched.** They read a region's polyMesh as they read any polyMesh; the
   moving-mesh flux enters the shared operators behind a `phi_rel` buffer that IS `phi` on a static mesh; the
   static gate suite is re-run bit-for-bit after every ALE unit.
10. **Every unit ends in a gate the coder does not write the verdict of.** Closed forms first (patch test,
    free expansion, cantilever order, thick cylinder, bimetal, SCL, piston, Euler–Bernoulli frequency), then the
    published benchmarks, with §94's observed order and GCI beside every multi-mesh number.

## C. The region layout (the contract every stream builds to)

```
<case>/mesh/
  regions.json
  fluid/polyMesh/{points,faces,owner,neighbour,boundary}
  flap/polyMesh/{...}
```

`regions.json`:

```json
{
  "version": 1,
  "units": "m",
  "regions": [
    { "name": "fluid", "kind": "fluid", "polyMesh": "fluid/polyMesh" },
    { "name": "flap",  "kind": "solid", "polyMesh": "flap/polyMesh", "material": "steel" }
  ],
  "interfaces": [
    { "regions": ["fluid", "flap"], "patches": ["fluid_to_flap", "flap_to_fluid"],
      "faces": 240, "tolerance": 1e-9 }
  ],
  "source": { "tool": "step_mesh", "version": "…", "geometry": "…", "config": "…" }
}
```

Rules (each is a test somewhere; the unit that owns the test is named):

* **R1** Every region's polyMesh is complete and standalone: `owner[f] < neighbour[f]`, internal faces
  ordered by owner then neighbour, patches contiguous — what `io::polymesh::build_host_mesh` requires. Every
  existing driver reads `fluid/polyMesh` unchanged. (M4 checker; S11 reads it.)
* **R2** An interface is a pair of boundary patches, one per region, with the same number of faces, and the
  k-th face of one is the k-th face of the other with opposite winding (centroid within `tolerance`, normals
  opposed, areas equal). Pairing is by index; §47.4's centroid hash stays the check and the refusal. (M4, S11.)
* **R3** Interface patch names are `<this>_to_<other>`; patch `type` is `patch`. (M4.)
* **R4** `kind` is `fluid` or `solid`. A solid region carries one `material` name; a solid region that holds
  two bonded materials (a bimetal strip) is ONE region with a `materials` map per `cellZones`-like list in the
  case, not two regions — bonded solids share cells' faces internally and the harmonic/series coefficient of
  §46.2's shape is written at the bond faces (S8). (S9, S11.)
* **R5** A region's cell numbering is its own; the manifest never refers to global indices.
* **R6** Paths are relative to the manifest's directory. (S11.)
* **R7** A single polyMesh with `cellZones` becomes this layout through `ofgpu-regions split`; faces between
  two zones become the interface pair, ordered identically on both sides. (S11.)
* **R8** A case names the manifest (`"mesh": {"regions": "mesh/regions.json"}`) or lists regions explicitly;
  both lower to the same `Vec<RegionInput>` plus interface requests; the explicit form wins on conflict and the
  conflict is named. (S11.)

## D. Section numbers (allocated when written; SPEC-LIT §0 rule 6 and the xref audit)

The tree fixes two things Part I did not know: `§92` is the automesher on branch `feat/automesher` and is
merged into this branch first (S0), so `§93` is the first vacant number here; and **`§99` is reserved by the
citation audit** (`xref::tests::a_reserved_number_has_no_heading`, SPEC-LIT §80.3) as the invented address the
registry tests cite, so it may never carry a real section. Part I's numbers therefore shift after §98:

| § | content | Part I said |
|---|---|---|
| 93 | per-region residual; points retained through lowering; VTU with real points, PointData, `Tensor`; the small refusals | 93 |
| 94 | observed order, GCI, `uncertainty` on `GateReport` | 94 |
| 95 | the thermo-elastic solid, on its region mesh; Anderson(5) the measured relaxation; ν > 0.45 and slenderness > 5 refused by name (S6b) | 95 |
| 96 | the case (`mechanics`, `stress` mode, `output` for `ofgpu-cht`, `docs/schema/cht-1.json`), pair tests | 96 |
| 97 | the imported region, the region layout, `ofgpu-regions` | 97 |
| 98 | a face that exchanges heat with something not meshed | 98 |
| 100 | properties that are functions, sources that vary, viscous dissipation | 99 |
| 101 | the turbulent conjugate interface | 100 |
| 102 | the buoyant wall | 101 |
| 103 | time in a conjugate run | 102 |
| 104 | the gate registry in the library, the report | 103 |
| 105 | ALE motion and the space conservation law | 104 (Track B) |
| 106 | fluid–structure interaction (a: steady conformal; b: dynamic solid; c: IQN-ILS; d: non-conformal mapping) | — |
| 107 | overset | 106 |
| 108 | a mesh fit for VOF | 105 |
| 109 | **the block-coupled matrix** — the route both of §95's measured refusals name, and what §106 waits on (S6b) | — |

`docs/09-thermal-structural-plan.md` is amended by S0 with this table and is otherwise unchanged.

## E. The programme by stream

**Stream S — solver** (Rust/CUDA). Worktree `Iteration-CFD-solver`, branch `feat/thermal-structural`.
Strictly sequential: one crate, one `target/`, one GPU. Unit shape: GLM codes from its brief → the Opus
supervisor reads the whole diff, builds, runs the unit's tests and the two audits (`xref`,
`provenance_audit`) and the capture-registry test, fixes small things, commits with the house message. A unit
that fails review is re-briefed once with the findings; a second failure stops the chain and is reported.

**Stream M — mesh and geometry tools** (Python, and the automesher's Rust). Worktree `Iteration-CFD-mesh`
on branch `feat/automesher` (the branch that already holds §92; the uncommitted driver work in the main tree is
landed there first by A0). Runs in parallel with S. Verify = the tools' selftests and the region checker; the
solver-side acceptance of its meshes is S10/S11's gate.

**Stream G — GUI and server** (TypeScript). The existing trees (`Iteration-CFD` on `feat/iteration-gui-link`,
`iteration-gui`). Starts only when the three GUI workflows now running have finished and committed (a gate
agent watches their journals and a clean tree). Sequential after that; server units first.

**Context rule for every GLM brief:** the reading list is line-ranged and sums to under 6,000 lines; new code
under 1,200 lines; one concern; every tool input under 50 lines; verify commands fixed in the brief; the
supervisor, not the coder, commits.

## F. Workflows

| workflow | units | when |
|---|---|---|
| **WF-A** | S0 · S1–S3 (§93) · S4 (§94) · S5–S8 (§95) · S9 (§96) · S10–S11 (§97) ‖ A0 · M1–M5 ‖ G0 · G1–G5 | now |
| **WF-B** | S12–S14 (§105 ALE) · S15–S18 (§106 FSI) · S19–S20 (§98) · S21–S22 (§100) ‖ M6 (automesher regions) ‖ G6 (FSI monitor, moving-mesh playback) | after WF-A's review |
| **WF-C** | S23 (§101) · S24 (§102) · S25 (§103) · S26 (§104) · S27–S29 (§107) · S30 (§108) ‖ G7 | after WF-B's review |

## G. Risks, and what is measured first

0. **MEASURED, and it changed the plan (S6b, 2026-09-13): the segregated loop does not do bending.** The risk
   this list did not carry. `docs/09` §F.1b's sweep is the measurement and decision 3 above is the consequence:
   Anderson at depth five replaces Aitken, a slenderness above 5 is refused by name, Gate 95-A is not written, and
   the block-coupled matrix moves from "named for later" to **the section §106 waits on**. Nothing in Stage 1's
   compact-body gates changes; everything slender does.

1. **GLM writing CUDA against a 26,000-line spec.** Every kernel unit ships a scatter-shaped host mirror in
   `reference.rs` (or reuses the prototype's) and the supervisor diffs device against host to 1e-12 before
   anything else; briefs give kernel signatures and the `KERNEL_UNITS` / capture-registry / PROVENANCE
   bookkeeping as a checklist.
2. **Small strain against large-deflection references** (CSM1/CSM3/FSI2/FSI3). The section states the expected
   direction of the linear model's error before the run; FSI1 and the Euler–Bernoulli closed form are the
   pass/fail gates; finite strain is a named later section.
3. **Added mass at density ratio 1** (FSI3; Causin, Gerbeau & Nobile 2005 closed form). IQN-ILS is its own
   unit; the section computes the closed-form threshold on the actual mesh and refuses an explicit scheme
   below it.
4. **Time accuracy on a moving mesh** (Farhat, Geuzaine & Grandmont 2001). Gate 105-B measures the order
   through §94 before FSI is built on it.
5. **The Gmsh-tet solid** is first order in stress at the interface layer; §94's GCI stands beside every solid
   gate; the automesher route (hex, layers both sides) is the second producer.
6. **Three GUI workflows still running; SRV3 lost.** G0 gates; G1 rebuilds from the surviving spec; no
   worktree junctions.
7. **Build contention.** S and A0/M6 build the same crate in different worktrees: `CARGO_BUILD_JOBS=3`, and A0
   runs once, cold, before M1.

## H. What this plan does not do

Monolithic FSI; contact; finite strain (named as the section after §106); plasticity; fracture;
participating-media radiation; AMI/sliding interfaces (overset is the route for relative motion); topological
mesh change; two-way thermo-elastic coupling; a native CAD kernel; Fluent export of multi-zone meshes (the
Fluent writer is tet-only and single-zone; named, not built).

## I. Unit catalogue — WF-A

Every unit below becomes one GLM brief in the house template (reading list with line ranges, "what must hold"
table, files owned, equations with sources, fixed verify commands, report). The fact sheets are the source of
line numbers; the brief writer re-checks them against the tree at dispatch time. "Reads" names files and the
fact-sheet section that carries their line ranges. All Stream S paths are under `rust/`.

### Stream S

**S0 (Opus, no GLM) — land the ground.** In `Iteration-CFD-solver`: (1) commit the TS-0 prototype as it stands
after `cargo test --release --lib -- solid::prototype xref provenance_audit` is green (fix counts if not);
(2) `git merge feat/automesher` (the committed §92; resolve NOTICE/PROVENANCE counts and the SPEC-LIT tail,
rebuild, run `automesher` tests + `xref`); (3) amend `docs/09` with §D's renumbering and a "§F.1a measured"
subsection carrying the TS-0 sweep table verbatim (from `scratchpad/sweep3.log`); (4) commit each step.

**S1 — §93a: the per-region residual.** Reads: `solver.rs` `device_norm_factor` 1149–1193, `convergence_test`
1204–1231, `solve_pcg` 1575–1672 (facts-solver-solid §3); `cht.rs` `ThermalRegion` 418–437, `correct`
1707–1745, `run_case` 1845–2007 (§4); `reference.rs` `norm_factor`/`residual` 1387–1416 (§5); SPEC-LIT §8.4
728–741, §13.4.2 1314–1391, §59.9 11633–11649. Writes: `solver.rs` (`device_norm_factor_ranged(gpu,k,w,psi,a,m,
offset,n)` and `residual_ranged`, one new kernel `norm_factor1_ranged` in `cuda/solver.cu`), `cht.rs`
(`ConjugateHeat::correct` returns per-region `SolverPerformance` beside the global one; `ChtSolution.region_residuals`),
`reference.rs` mirrors, `bin/cht.rs` (prints one residual line per region; refuses `converged` unless every region
met the tolerance; the §13.4.2 start-up block lists each region's row-scale ratio to the largest). Must hold:
ranged norm over the full range equals the global norm to the bit; on a two-region slab the per-region residuals
differ from the global by the predicted row-scale ratio; report-only sweep over the existing `cht::tests`
recorded in the report. Gate 93-B (silicon on mould compound, k ratio 200:1, Carslaw & Jaeger ch. I, the 1-D
series) registered via `Checks::report` with both numbers (per-region 1e-8 vs global 1e-6 nominal). Verify:
`cargo test --release --lib -- cht solver xref provenance_audit` + `ofgpu-validate` conjugate section.

**S2 — §93b: points kept, VTU with real points, `PointData`, `Tensor`.** Reads: `io/output_types.rs` (all,
53 lines), `io/vtu.rs` 31–110, 192–300, 350–364, 401–470 (§6.2), `io/writer.rs` 128–240, 382–390, `nvdb.rs`
184–190, `vdb.rs` 253–259, `io/polymesh.rs` 50–62, `blockgen.rs` 1493–1515 (`raw_mesh`), `io/case_cht.rs`
488–521, 1309–1391 (`build_region_mesh`), `cht.rs` 561–707 (`ThermalMesh::build`), `types.rs` 189–275;
PyFR's VTU writer (`reference/pyfr`, BSD-3, documentation only) for the appended-binary and 6→9 tensor
convention. Writes: `FieldValues::Tensor(&[Tensor])` handled in all five match sites (USD/VDB: write the six
components as scalars named `.xx …`); `vtu::write_vtu_points(path, raw: &PolyMeshRaw, cell: &[OutputField],
point: &[OutputField], time)` — real polyhedra sharing points, `<PointData>` block, tensors expanded 6→9;
`cell_to_point(raw, m, cell_values) -> point_values` inverse-distance over the cells sharing a point (host, in
`io/vtu.rs` or a new `io/pointfield.rs`); `LoweredChtCase.raw: Vec<PolyMeshRaw>` kept per region (block regions
through `blockgen::raw_mesh` + `build_host_mesh`); `ThermalMesh` gains `points: Vec<Vec3>` and `faces` by
concatenation (offsets) so a concatenated VTU can be written. Must hold: the old CellData path is bitwise
unchanged for existing callers; a linear field cell→point→cell round trip is exact on a uniform block; the
VTU is read back by a Python reader (`vtk` if installed, else a 40-line XML+appended parser in the test) with
the right counts and a symmetric 9-tensor. Verify: `cargo test --release --lib -- io cht xref provenance_audit`.

**S3 — §93c: the refusals, the SPEC-LIT section, Gate 93-A.** Reads: `bin/lowmach.rs` banner/write lines
(grep `println!` near 2010–2060), `energy.rs` `GasState` (grep), `decompose.rs` (patch walk; grep
`PatchKind::Interface`), `psychro.rs` `refuse_condensation`, SPEC-LIT §47.2 6315–6423, §13.4 1124–1485, §59.9,
§0 11–72, S1's and S2's diffs (`git log -2 -p --stat`). Writes: `GasState::mach(...)` and its print/refusal
in `ofgpu-lowmach` (threshold `M > 0.3`, §26's low-Mach premise; warn under `-permissive`); `decompose.rs`
refuses a partition cutting an `Interface` face naming §47.2 consequence 2; `psychro::refuse_condensation`
names a condition a case can express; Gate 93-A: the assembly of one region alone equals, bit for bit, that
region's rows of the two-region concatenated assembly (`diag/upper/lower/source` compared after
`ThermalMesh::build` of the pair vs a single-region build) registered via `Checks::report`; SPEC-LIT §93
written in the house style (S1+S2+S3 content, equations, DESIGN marks, gates, refusals, §13.4 contract).
Verify: `cargo test --release --lib -- lowmach decompose psychro cht xref provenance_audit` + `ofgpu-validate`.

**S4 — §94: observed order and reported uncertainty.** Reads: `bin/validate.rs` 116–316 (`Verdict`, `How`,
`GateReport`, `Checks`), the manufactured-solution section after 2440 (grep `manufactured`), §60.5 and §79.12's
three-mesh code (grep `7.12`), SPEC-LIT §46.7 6224–6239, §60.5, §69 13789–14080, §79.12; sources: Roache 1994
(DOI 10.1115/1.2910291), Celik et al. 2008 (10.1115/1.2960953), Eça & Hoekstra 2014 (10.1016/j.jcp.2014.01.006),
Roache 2002 MMS (10.1115/1.1436090). Writes: `src/vv.rs` (observed order incl. non-integer `r` fixed point,
Richardson extrapolate, `GCI_fine`, the least-squares estimator for oscillatory sequences, `E = S − D`,
`u_val`), `GateReport.uncertainty: Option<Uncertainty>` and the §69 rule (a multi-mesh gate without it fails to
register; a single-mesh gate declares itself), gates 94-A (Gate 46-B: anisotropic MMS `k = 1:10:100`, three
meshes r=2, `p ∈ [1.9,2.1]`), 94-B (MMS across a 100:1 interface with `R_c` on a mesh sheared so the interface
is 20° off-normal; `p` reported, fail if `< 0.9`), 94-C (retrofit §60.5/§79.12: `E ± u_val` printed and added to
their SPEC-LIT text). SPEC-LIT §94. Verify: `cargo test --release --lib -- vv cht xref provenance_audit` +
`ofgpu-validate`.

**S5 — §95a: the displacement operator on the device, diffed against the prototype.** Reads: the prototype
`src/solid/prototype.rs` 128–149 (traction design), 165–214, 227–263, 276–333, 449–513, 517–620, 620–760
(assembly, `update_boundary_gradient`, `update_ref_grad`, `evaluate_boundary`), `src/solid.rs`; `fv.rs`
1416–1489 (`fvm_laplacian`), 1547–1646, 1951–2057 (`fvc_grad_vector_scheme`), 1786–1816 (`fvm_su`);
`momentum.rs` 394–440, 948–983, 1447–1596, 1845–1890 (the private component helpers — copy, do not export);
`cuda/momentum.cu` 137–164, 481–512; `ldu.rs` 13–70; `ldu_ops.rs` 179–208, 295, 364; `field.rs` 900–1013;
`field_ops.rs` 126–171, 447–474; `reference.rs` 30–47, 106–241, 290–315, 772–820; `build.rs` 32–60, 285–301;
`capture/registry.rs` 30–60, 97–100, 132, 307–360; `cht/tests.rs` 19–53, 1883–1930 (capture-gate template);
SPEC-LIT §1 73–98, §2.4 155–171, §3.2 329–346, §4 402–429, §81 (the gate rule paragraph only). Writes:
`src/solid/{mod.rs (Lamé + validate moved from the prototype's Material), displacement.rs, bc.rs}`,
`cuda/solid.cu` (`solidVecComponent`, `solidSetComponent`, `solidDivSigmaExp` — the deferred gather
`Σ_f [μ ∇uᵀ + λ tr(∇u) I − (μ+λ)∇u]·Sf` over cell faces with linear face interpolation of the cell gradient
tensor and the boundary gradient on boundary faces, `solidThermalLoad` = `−(3λ+2μ)α V_P (∇T)_P`,
`solidTractionRefGrad` = the prototype's `update_ref_grad`, `solidBoundaryTraction` source `t|Sf|`), host
mirrors in `reference.rs` (scatter-shaped) or reuse of the prototype's functions where they already are the
mirror, `build.rs` `KERNEL_UNITS += "solid.cu"`, capture-registry row `Stance::Gate("solid::tests::the_displacement_iteration_replays_bitwise")`
with that test (PBiCGStab + `fixed_iters`, template cht/tests.rs:1883). One `GpuLduMatrix`, three sources, DIC
preconditioner rebuilt per component (accepted; say so in the doc comment). Must hold: on the prototype's
20³ block with `fixed_minus_x()` and ΔT, ONE Picard application on the device equals the prototype's
`apply_map` to 1e-12 in every component (device vs host); Gate 95-B free expansion (`σ ≤ 1e-12` relative,
`ε_ii` to 1e-14) on the device; Gate 95-C linear-displacement patch test on the jittered block `≤ 1e-12|u|`;
the capture gate replays bitwise. Verify: `cargo test --release --lib -- solid reference capture xref
provenance_audit`.

**S6 — §95b: the outer loop, Aitken, the cantilever, the refusals.** Reads: prototype 847–1050
(`apply_map`, `run`, `aitken_omega`), `OuterReport` 408–431, tests 1257–1360; S5's `src/solid/**`;
`vv.rs` (S4); `blockgen.rs` 99–194 (`GradedAxis`, `BlockSpec`, `PatchWindow`); `validate.rs` 116–316,
546–645, 2330–2440 (how a section registers); SPEC-LIT §8.2 715–719, §21 1964–1994, §46.4 6124–6201 (the
"measured, not asserted" precedent); Küttler & Wall 2008 (DOI 10.1007/s00466-008-0255-5). Writes:
`src/solid/outer.rs` (Picard + Aitken Δ² on the increment, three boundary sub-passes, stop on the increment
norm over six decades or `max_outer`, `predicted = (μ+λ)/(2μ+λ)` printed beside the observed geometric-mean
ratio every run, divergence detection by name), `src/solid/mod.rs` refusals (ν outside (−1, 0.5); ν above the
measured edge 0.45 → block coupling named as the route, Cardiff et al. 2016 DOI 10.1016/j.compstruc.2016.07.004;
finite strain, plasticity, contact, fracture, inertia, orthotropic on a non-aligned mesh, two-way coupling with
`δ` printed, displacement reaching the fluid `max|u|/h_min` above 0.1 naming §105), Gate 95-A end-loaded
cantilever on three graded blocks r=2 with a `PatchWindow` for the load strip: `σ_xx`, `σ_xy`, tip deflection
(Timoshenko & Goodier, *Theory of Elasticity*, 3rd ed. 1970 — book), observed `p ≥ 1.9` on `u`, `≥ 0.9` on
cell-centre stress, both through `vv.rs` and reported; Gate 95-F the contraction table at ν = 0.2/0.3/0.45
predicted vs observed (the device twin of the sweep). Must hold: outer loop on the device reproduces the
prototype's iteration count ±1 and observed ratio to 1e-6 on the 20³ case; refusals fire by name (tests).
Verify: `cargo test --release --lib -- solid xref provenance_audit` + `ofgpu-validate` solid section.

**S7 — §95c: stress, von Mises, principal, point displacement, the thick cylinder.** Reads: S5/S6 code;
`types.rs` 189–275; S2's `cell_to_point` and tensor output; `walldistance.rs` 455–500 (the quarter-annulus
point-mapped mesh — the pattern for a structured annulus fixture); `blockgen.rs` 213 (`set_cyclic_axis`),
1513 (`raw_mesh`); Boley & Weiner *Theory of Thermal Stresses* ch. 9 / Timoshenko & Goodier (book) for the
thick-walled cylinder under a radial temperature field. Writes: `src/solid/stress.rs` + kernels `solidStress`
(σ from ∇u and T, 6 components), `solidVonMises`, `solidPrincipal` (trigonometric closed form of the cubic on
the deviator — say why over Jacobi), hydrostatic/deviatoric split, `|u|`; `u` on points through S2; an
annulus fixture `solid::fixtures::annulus(nr, nθ, nz, r_in, r_out)` by point-mapping `raw_mesh` with the θ
cyclic pair; Gate 95-D: the cylinder under a steady radial `T(r)` CHAINED through §46's conduction (the only
gate that tests the coupling): `σ_r, σ_θ, σ_z` within 1 % on the finest of three meshes, GCI beside it. Must
hold: von Mises of a uniaxial state equals |σ| exactly; principal stresses of a diagonal tensor are its
diagonal to 1e-14; the stress of the free-expansion case is zero to 1e-12. Verify: `cargo test --release
--lib -- solid xref provenance_audit` + `ofgpu-validate`.

**S8 — §95d: two materials bonded, the bimetal, the section.** Reads: S5–S7; `cht.rs` 1048–1230
(`Conduction::build` — the series face coefficient at a jump, the pattern), SPEC-LIT §46.2 6035–6057;
Tuković, Ivanković & Karač 2013 (DOI 10.1002/nme.4390) for traction continuity; Timoshenko 1925 *JOSA* 11,
233 (DOI 10.1364/JOSA.11.000233) for the bimetal curvature. Writes: per-cell `μ, λ, α, T_ref` (a `materials`
map over cell lists within one solid region — R4), face coefficient `(2μ+λ)_f` as the series value at a jump
(harmonic in the two one-sided values), the deferred term evaluated one-sided with the traction made continuous
(the Tuković form), thermal load per cell with its own `α`; Gate 95-E bimetal strip under uniform ΔT: curvature
within 2 %, AND the second leg showing that linearly interpolating `(3λ+2μ)α` at the bond does not converge
under refinement (printed); §95 written complete in SPEC-LIT (equations with Demirdžić & Muzaferija 1994 DOI
10.1002/nme.1620372110 and 1995 DOI 10.1016/0045-7825(95)00800-G; Jasak & Weller 2000 DOI
10.1002/(SICI)1097-0207(20000520)48:2<267::AID-NME884>3.0.CO;2-Q; Demirdžić & Martinović 1993 DOI
10.1016/0045-7825(93)90085-C; the measured contraction table; the refusal list with the measured threshold;
95-G NAFEMS written only if the primary is in `reference/`, else the §60.5 Gate 5-style disclosure).
Verify: `cargo test --release --lib -- solid xref provenance_audit` + `ofgpu-validate`.

**S9 — §96: what a thermo-elastic case says.** Reads: `io/case_cht.rs` 87–384, 448–521, 583–700 (`lower`),
`io/case_cht/tests.rs` 25–60, 320–349 (pair-test shape); `case_json.rs` 2611–2614, 3350–3366, 3835–3840 (the
schema trio); `bin/cht.rs` (all, 395 lines); `bin/common/mod.rs` 683–704; `io/output_plan.rs` 142–168,
300–311, 461–530, 691–830; `cases/dieStack.cht.jsonc`; SPEC-LIT §13.4.1 1146–1313, §44 5627–5890, §47.14
6871–6962; the GUI contract `../Iteration-CFD/gui/shared/src/registry.ts` 328–343 (read only). Writes:
`mechanics` block per solid region (`E, nu, alpha, T_ref, rho`, per-patch mechanical BCs: fixed per component,
traction vector, symmetry, free; solver: tolerance, max outer, relaxation), run mode `stress` (one-way, after
the thermal solve converges — the per-region verdict of §93), `output: Option<JsonOutput>` on `ChtCase`
routed through `OutputPlan::from_json` + `OutputPipeline::from_plan` into `ofgpu-cht` (VTU per region with
`T`, `u` cell+point, `sigma` 9, `vonMises`, `sigma_principal`, `|u|`; CSV behind `-csv`), the banner (material
table, refusals passed, predicted/observed contraction, `δ`, peak von Mises with cell and position),
`docs/schema/cht-1.json` with the emit/drift/write test trio, `usage()` extended (the GUI sync test reads it),
`cases/dieStack.cht.jsonc` + `mechanics` (silicon, solder, copper — property sources in the header), new
`cases/bimetalStrip.stress.jsonc`; Gate 96-A ten pair tests (`alpha, E, nu, T_ref`, a traction, a fixed
displacement, symmetry vs free, tolerance, relaxation, the bond treatment); refusals by name (`E ≤ 0`, ν range
and threshold, `alpha` without `T_ref`, `mechanics` on a fluid region, `ddtScheme` on displacement, `stress`
on an unconverged thermal solve, an `output` field the run did not compute); SPEC-LIT §96; `docs/11-thermal-stress.md`
user narrative; `rust/README.md` driver table. Verify: `cargo test --release --lib -- io solid cht xref
provenance_audit`; `ofgpu-cht` end to end on both cases with VTU; the VTU opened by the S2 test reader.

**S10 — §97a: the imported region.** Reads: `io/case_cht.rs` 154–185, 1309–1391; `io/polymesh.rs` 50–62,
72–80, 236–260, 835–856 (the writer); `io/msh.rs` 162–216, 241–273 (volume tags discarded — say so);
`cht.rs` 561–707; `validate.rs` 594–645 (`make_mesh` writes/reads a polyMesh); SPEC-LIT §46.5 6202–6210,
§47.4 6489–6556, §47.14. Writes: `ChtRegionMesh` as an untagged enum `Block{..} | PolyMesh{polyMesh: String}`
(path relative to the case file; `deny_unknown_fields` on both), `build_region_mesh` dispatch keeping
`PolyMeshRaw` (S2), `docs/schema/cht-1.json` regenerated; refusals: a patch named in the case but not in the
mesh, a mesh path outside the case directory, an `.msh` with several volumes (named: use `ofgpu-regions`,
S11). Gate 97-A: `cases/dieStack.cht.jsonc` with every region written out through `write_poly_mesh_raw` and
read back gives a converged field bit-for-bit identical to the block run (registered, single-mesh by name).
SPEC-LIT §97 first half. Verify: `cargo test --release --lib -- io cht xref provenance_audit` + `ofgpu-validate`.

**S11 — §97b: the region layout and `ofgpu-regions`.** Reads: §C of this document; `io/polymesh.rs` (all
reader paths), `io/msh.rs` 353–420, `cht.rs` 744–943 (`couple`, the pairing and its refusals), `mesh/geometry.rs`
724 (`cell_regions`), `bin/convert_mesh.rs` 514–706 (region pass), `Cargo.toml` `[[bin]]` block, the GUI sync
test contract `../Iteration-CFD/gui/server/src/registry/sync.test.ts` 20–62 (read only: a new bin needs a
`fn usage()` literal). Writes: `src/io/regions.rs` (`RegionsManifest` serde with `deny_unknown_fields`, `load`
→ `Vec<(RegionInput-like owned mesh, PolyMeshRaw)>` + `Vec<InterfaceRequest>`, R1/R2/R3/R6 checks with named
refusals), `ChtCase.mesh: Option<{regions: String}>` (R8), `cellZones` reader in `polymesh.rs`
(`read_cell_zones(dir) -> Vec<(String, Vec<Label>)>`), the splitter (`split_by_zones(raw, zones) -> (Vec<(String,
PolyMeshRaw)>, interfaces)` — faces between two zones become the pair, ordered identically, R7), new bin
`ofgpu-regions` with `split <polyMesh> <outDir>`, `check <manifest>` (prints the pairing report), `list`; Gate
97-B: split a two-zone block, load through the manifest, run dieStack-shaped conduction → bit-for-bit the
concatenated block run. SPEC-LIT §97 complete (the layout as the contract, R1–R8). Verify: `cargo test
--release --lib -- io cht xref provenance_audit`, `cargo test --release --bin ofgpu-regions`, `ofgpu-validate`;
plus M4/M5's Turek–Hron layout loaded by `ofgpu-regions check` if present (report, not gate).

### Stream M (worktree `Iteration-CFD-mesh`, branch `feat/automesher`)

**A0 (Opus, no GLM) — land the automesher driver work.** Create the worktree on `feat/automesher`; from the
main tree take `git diff -- rust tools NOTICE` plus the untracked automesher files (`rust/src/automesher/driver.rs`,
`tools/automesher/*`) as a patch and apply it here (never `git` in the main tree beyond `diff`/`status`); cold
`cargo build --release` with `CARGO_BUILD_JOBS=3`; `cargo test --release --lib -- automesher xref
provenance_audit`; fix what the move broke (counts, paths); commit in the house style. Then merge `main` into
`feat/automesher` so M1–M5 see the site-mesh tools. Report what the driver stage does and does not pass.

**M1 — the geometry tool, read side.** New `tools/geom/geom_tool.py` (+ `README.md`, `selftest.py`):
`info <file> [--json out]` lists solids (`tag, name, volume, bbox, centroid, n_faces, closed`) for STEP/STP,
BREP, IGES (surfaces only — say so), STL (discrete); `export <file> --out <path>` by extension via
`gmsh.write` (STEP/BREP/STL/XAO), with the mm convention `export_fluid.py:40–44` uses; a JSON sidecar
`<stem>.geom.json` (names, materials, op history) read/written beside the file; `Geometry.OCCImportLabels=1`
probed for STEP names, tags as the fallback key. Reads: `tools/mesh/step_mesh.py` 443–519 (import stage),
`tools/mesh/diag/export_fluid.py` (all), `tools/mesh/selftest.py` (all — the test pattern), facts-mesh-tools
§3. Must hold: a two-box STEP built in the selftest lists two solids with the right volumes to 1e-9 relative;
export→import round trip preserves the solid count and volumes; a shared face after a STEP round trip is
de-duplicated (`removeAllDuplicates`) and reported.

**M2 — the geometry tool, edit side.** `edit <file> --ops ops.json --out <file>`: `rename, delete,
translate, rotate, scale, mirror, fuse, cut, intersect, fragment, box, cylinder, sphere, set_material`;
each op validated by name; the sidecar carries the op history and names survive through tag renumbering by
centroid/volume matching after booleans (say the rule). Selftest: box minus cylinder volume to 1e-9 of the
closed form; fragment of two overlapping boxes gives three solids whose volumes sum to the union; a rename
survives export/import through the sidecar.

**M3 — multi-region meshing in `step_mesh.py`.** A `regions` config block: `{"solids": [{"tag": 2, "name":
"flap", "kind": "solid"}], "interface_names": true}`; declared solids are NOT cut but kept and
`occ.fragment`ed with the fluid (conformal shared faces), each volume its own physical group (`fluid`,
`<name>`), each shared surface a physical group `<a>_to_<b>`, per-region size fields (`sizes.regions.<name>`);
classification of the fluid's outer faces unchanged; `Mesh.SaveAll=0` still. Reads: `step_mesh.py`
88–120 (DEFAULTS), 194–439 (config), 817–1084 (cut stage), 1132–1274 (classify), 1461–1580 (groups+fields),
1836–1967 (mesh stage), 2663–2734 (write); README 103–229. Must hold: selftest's building declared as a
region yields two volumes in the `.msh` with `$PhysicalNames` `fluid`, `building`, `fluid_to_building`; the
single-fluid path is bitwise unchanged (same tet count and quality on the existing selftest).

**M4 — the layout converter and checker.** New `tools/mesh/regions_from_msh.py <mesh.msh> <outDir>`:
reads MSH 4.1 (element blocks carry `entityDim, entityTag`; volume physical names), splits cells by volume,
renumbers per region, builds faces with OpenFOAM ordering (R1: internal faces sorted by owner then neighbour,
`owner < neighbour`, boundary faces by patch), turns each shared face into a boundary face of BOTH regions
with opposite winding and IDENTICAL order (R2), names them `<a>_to_<b>` (R3), writes each region's polyMesh
with a new Python writer (`tools/mesh/polymesh_write.py`: points at 17 significant digits, the five files,
`boundary` with `type`/`nFaces`/`startFace`) and `regions.json` (§C); `--fluent` refused by name (tet-only,
single zone). `tools/mesh/regions_check.py <regions.json>`: R1–R3, R6 checks, pairing report (max centroid
distance, area mismatch, normal dot), exit 1 on violation. Reads: `rust/src/io/msh.rs` 162–216, 353–420 (what
the Rust reader expects — for the msh format only), `rust/src/io/polymesh.rs` 72–201, 236–300 (what
`build_host_mesh` requires), `tools/mesh/diag/polymesh_to_fluent.py` 35–72 (the Python polyMesh reader
pattern), facts-mesh-tools §4. Must hold: on M3's two-region selftest mesh the layout passes the checker; the
fluid region's polyMesh converts identically (cell count, patch names) to what `ofgpu-convert-mesh` produces
from the fluid-only `.msh`; a hex one-cell-thick mesh (probe `th.msh` recipe) round-trips with `empty`
front/back patches.

**M5 — benchmark recipes.** `tools/mesh/examples/turek_hron.py` builds the FSI benchmark geometry in gmsh
(channel 2.5×0.41, cylinder centre (0.2,0.2) r=0.05, flap 0.35×0.02 from the cylinder — subtract the cylinder
from the flap before fragmenting so no lens remains, facts-mesh-tools §3 note), 2-D fragment → extrude one
cell with recombine → hex, physical groups `fluid`, `flap`, `fluid_to_flap`, `inlet`, `outlet`, `wall_top`,
`wall_bottom`, `cylinder`, `flap_fixed` (the flap's face on the cylinder), `empty_front`, `empty_back`; three
refinement levels; writes the layout through M4 and a case skeleton `cases/turekHron/{fsi1,cfd1}.jsonc` in
the S9 shape (materials from facts-fsi-sources §1.3, boundary conditions from §1.2 incl. the parabolic inlet
profile and the 2 s ramp). Also `examples/bimetal_strip.py`, `examples/thick_cylinder.py`,
`examples/cantilever.py` (solid-only layouts for S7/S8's gates, hex). Must hold: every layout passes the
checker; the T–H fluid mesh at level 1 has the cylinder resolved by ≥ 40 faces; `regions_check` prints zero
pairing error.

### Stream G (after G0; `Iteration-CFD/gui` + `iteration-gui`)

**G0 (gate)** — wait until the three GUI workflows' journals carry a `result` for every `started` unit, `git
status --short` is clean in both trees, and no `claude`/`node` process from those workflows holds `gui/`.

**G1 — the geometry server, rebuilt and extended.** From `scratchpad/units/ig/IG-SRV3.md`, the SRV3 coder
report in `logs/SRV3.log` and `lost-gui-work/hunks.md`: `server/src/formats/stl.ts` (+ OBJ), `tools/geometry.ts`
(`geometry_open/info/save`, LRU 4, 512 MB), `shared/src/geometry.ts`, routes `/api/geometry/*`, `PIPELINES`
(`geom-tool`, `mesh-step`, `regions-from-msh`) + `dispatchPipeline` (quote EVERY argv entry — the SRV4 finding),
new tools `geometry_import_step {path}` (M1 `info` + per-solid STL export → one geometry with one solid per
tag), `geometry_edit {path, ops}` (M2), `mesh_regions {step, config}` (M3+M4 → a layout), `regions_check`.
Tests: unit + `npm run test -w server` green; secrets/paths policy as before.

**G2 — the dataset model grows.** `shared/src/viewerDataset.ts` `FieldInfo.components: 1|3|6|9`,
`location: 'cell'|'point'`, `SurfaceInfo.pointOfVertex`; `formats/vtu.ts` PointData + tensors; `formats/foam.ts`
6/9 comps; `datasets/manifest.ts`, `service.ts`, `worker.ts`, `stats.ts` (`vonMises`, components, principal),
`viewerCommands.ts` `setWarp {field, scale}`; multi-region discovery (`regions/<name>/` and `regions.json`),
`open {path, region}`; `registry/schema.ts` serves `cht-1.json`; `formats/casejsonc.ts` reads `regions[]`,
`interfaces[]`, `mechanics`; `tools/case.ts` picks the schema by file name; `/api/case/patches?region`;
`registry.ts` `ofgpu-cht` entry updated to S9's usage (sync test) and `ofgpu-regions` added; `mock/solver.ts`
`runCht` writes a tiny VTU with `u`/`vonMises` in demo mode. Proved on a hand-written VTU before S9 lands.

**G3 — the Geometry tab.** Per `IG-G2.md` plus STEP: `src/components/geometry/**` (picker, solid list with
name/material/keep, transform form composed to the 4×4, boolean dialog, op history), `viewer/geometrySource.ts`
feeding `SurfaceLayer` one patch per solid, store slice, `Sidebar` row, tab kind; every action a `ui.command`
(`geometry_open, geometry_part, geometry_transform, geometry_boolean, geometry_save, geometry_import_step`)
mirrored in `protocol.ts`, `tools.ts` summariser, `prompt.ts`, `ai-drive.ts`.

**G4 — regions in the mesh dialog, the case editor, and the results.** Mesh dialog "Regions" table (name,
kind, source, status, cells) → `mesh_regions`; Mesh tab region selector; Properties: `materials`, `bc` per
region, `interfaces`, `mechanics` editor with solid presets (fixed, free/traction, symmetry) from the served
`cht-1.json`; Post panel: region/dataset picker, tensor components, Warp {field, scale} row, deformed-shape
toggle, residual chart per region (S1's lines → `residuals.ts` style); ui.commands `post_warp`, `open_result
{region}`; `ColorBars` per tab as before.

**G5 — end to end.** Playwright: open the two-box STEP from M1's selftest, list, rename, cut, save; mesh
regions from it; open `cases/dieStack.cht.jsonc`, edit `mechanics`, run in demo, open the result, warp,
colour by von Mises, cut; then the same by the assistant through `ai-drive.ts` with `CFD_LLM=zai`; findings
fixed; both repos committed (never pushed from `Iteration-CFD`; `iteration-gui` pushed only when the user says).

## J. Unit catalogue — WF-B and WF-C (titles and gates; briefs written when WF-A is reviewed)

**§106 waits on the block-coupled matrix, and the block matrix is the next section written (S6b, 2026-09-13).**
The Turek–Hron flap is 350 mm long and 20 mm thick — **17.5:1**. `docs/09` §F.1b measured the segregated
displacement loop stalling at an observed contraction of 0.95–1.00 on a 10:1 cantilever on three meshes, at
ν = 0.2 and 0.3, with bare Picard, with Aitken, with Anderson at depths 3, 5 and 10, and with the implicit
coefficient of the split enlarged three ways; §95 refuses a slenderness above 5 by name for that reason. A solid
solver that refuses the flap cannot carry S15–S18, whatever the fluid side does. So the order changes: **§109, the
block-coupled matrix of Cardiff, Tuković, Jasak & Ivanković (2016, DOI 10.1016/j.compstruc.2016.07.004), is
written before §106** — three components in one matrix, a 3×3 coefficient per face, which needs a second matrix
format beside SPEC-LIT §1's one-entry-per-face LDU and is therefore a unit of its own, not an option on the
existing solve. Gate 95-A, the end-loaded cantilever at observed order through §94, is what §109 has to earn, and
it is written with §109 and not before. S12–S14 (§105, ALE and the space conservation law) are **unaffected** —
they are fluid and mesh work and do not touch the solid loop — and may proceed in parallel. S15–S18 do not start
until §109's cantilever passes.


**S12 §105a** resident geometry (points `DevBuf`, face CSR, scratch resident; `GpuGeometry::recompute_in_place(gm)`),
swept-volume kernel per face (fan about `x_avg`, same order as `face_geometry`), volume history `v0/v00`, ALE
ddt kernels `tsDdtGeneralV`; Gate 105-A SCL: `V^{n+1} − V^n − dt Σ_f s φ_mesh,f` to 1e-12 on a box with
prescribed interior sinusoidal motion, uniform flow uniform to round-off over 100 steps. **S13 §105b** `phi_rel`
buffer and its routing (momentum, `FlowState`, `assemble_pressure`, inlet-outlet, LTS, energy `phi_conv`),
moving-wall `ref_value = u_mesh,f` with `fr = 1` (not `movingWallVelocity`), IDW smoother (Luke, Collins & Blades
2012, DOI 10.1016/j.jcp.2011.09.021), `motion` case block; Gate 105-B piston `u = x ẋ_w/x_w`, time order via
§94 (`euler` 1±0.1, `backward` 2±0.1); the static suite bitwise. **S14 §105c** Gate 105-C: Turek–Hron CFD3 on
the M5 mesh with prescribed interior wobble (walls fixed): drag/lift statistics unchanged within 0.5 % of the
static-mesh run; CFD1/CFD2 steady drag/lift vs 14.2929/1.11905 and 136.700/10.5343 within 2 % with GCI (the
fluid regression that makes FSI1 credible). **S15 §106a** patch traction integrator (`t = −p n + μ(∇u+∇uᵀ)·n`
per face, patch force sum = drag/lift, also used by S14), `ofgpu-fsi` driver: conformal transfer by index
(R2), Aitken on the interface displacement, mesh motion from the interface, steady FSI1; Gate 106-A FSI1
`u_x(A) = 2.270e-5, u_y(A) = 8.209e-4, F_D = 14.294, F_L = 0.7637` within 2 % (the small-strain case).
**S16 §106b** dynamic solid: Newmark-β (β=1/4, γ=1/2; Newmark 1959) with generalized-α named (Chung &
Hulbert 1993, DOI 10.1115/1.2900803); Gate 106-B Euler–Bernoulli cantilever `f_1 = (1.875²/2π) sqrt(EI/ρAL⁴)`
within 1 %, second-order time convergence via §94; CSM1 static and CSM3 transient reported as bands with the
linear-model caveat stated first (refs −66.10 mm, −63.6 ± 65.2 mm at 1.0995 Hz). **S17 §106c** IQN-ILS
(Degroote, Bathe & Vierendeels 2009, DOI 10.1016/j.compstruc.2008.11.013) and the added-mass condition
(Causin, Gerbeau & Nobile 2005, DOI 10.1016/j.cma.2004.12.005) computed on the mesh and refused for an
explicit scheme; Gate 106-C FSI2 and FSI3 bands from the FEATFLOW `.point` series (last-period mean ± amplitude
[Hz]); iteration-count ordering IQN-ILS < Aitken < constant printed. **S18 §106d** non-conformal mapping:
nearest-projection then RBF (Beckert & Wendland 2001, DOI 10.1016/S1270-9638(01)01118-3), transpose map for
forces; Gate 106-D linear-field reproduction to round-off and force conservation; FSI1 on a non-matching solid
mesh within the conformal result's tolerance. **S19–S20 §98**, **S21–S22 §100** as Part I (numbers shifted).
**M6** automesher regions per facts-automesher §8.1 A–K (in `Iteration-CFD-mesh`). **G6** FSI monitor
(interface residual, coupling iterations), moving-mesh playback (per-step points).

**WF-C:** S23 §101, S24 §102, S25 §103 (with 103-D duty cycle over S9's stress mode), S26 §104, S27–S29 §107
overset (static two-box Poiseuille; moving body with mass deficit printed; cylinder on a component grid vs
CFD1), S30 §108 VOF-fit mesh (interface-tracking adapt criterion; dam-break at each level; Martin & Moyce at
half the cost), G7 overset/VOF editors.
