# Tranche 17 — the completion plan (docs/17, adopted 2026-10-06)

Assembled by Opus 5.5 on 2026-10-06 from the seven stream plans in `plan17/` (COMP, CHEMRAD, PAR, PREC, MESHIO, STUDIO,
VAL) and the memory note `tranche-17-completion.md`. Nothing in any repository was edited. The assembler ran read-only
`git` and `grep` commands on the three work trees to settle the facts in §3, and a scheduling script (scratchpad
`plan17/asm/`) that checks every dependency against the wave plan and sums the lane hours.

---

## 1. The user's order, verbatim

> opus 5.5로 리뷰하고 glm으로 코딩해줘, 유동: 압축성 천음속 부스네스큐, 연성: 일단제외, 화학복사: 모두구현, 병렬:모구 구현, 격자: gmsh는 라이센스문제로 안되는 문제없는 사면체로 구축필요, 경계조건: 모두구현, 입력형식: 모두구현, CAD: 제외, 2번 미완성부분 옥트리격자 VBD USD출력 해결해줘, Isaacsim 볼륨도 구현해주고, 단정밀도 지원하게 해줘(중요함), 스튜디오: 제대로 구현해줘, 자동격자 설정: 모두구현 필요, CAD루프 일단제외, 검증게이트쪽 모두 구현해줘, 4. 10분규칙 나중에 일단 이대로,

Working rule from the same order and the standing memory notes: Opus 5.5 plans, writes contract-only briefs and
reviews; GLM-5.3-Flash writes every implementation, one unit at a time per tree, through the `tools/glm-code.sh`
wrapper (memory `glm-writes-the-implementation`, `session-roles-fable-glm`). Decisions are taken by recommendation
unless they change a gate, a numerics path that already exists, a licence position, or need hardware or money.

## 2. Scope

| area (the user's words) | in scope | stream | out of scope / deferred |
|---|---|---|---|
| 유동: 압축성 천음속 부스네스큐 | a pressure-based all-speed compressible solver (`ofgpu-compressible`), transonic steady LTS, compressible BCs, ρ-weighted SA; a Boussinesq body force beside the density-ratio form | COMP (CMP-) | density-based HLLC (contingency, needs the user); ρ-weighted k-ω SST / k-ε (backlog); compressible reacting flow |
| 연성 | — | — | **FSI excluded** ("일단제외") |
| 화학복사: 모두구현 | NASA-7 thermo, CHEMKIN-II, finite-rate kinetics, equilibrium, Rosenbrock on the device, mixture-averaged transport, EDM/EDC/PaSR, P1, fvDOM with scattering, WSGG, S2S outside CHT, radiation in `ofgpu-lowmach`/`ofgpu-buoyant` | CHEMRAD (CHR-) | RADCAL narrow band, analytic Jacobian, warp-per-cell (named, not built) |
| 병렬: 모두 구현 | `Comm` with ThreadComm and a dynamically loaded MPICH-ABI MPI, SPMD halo/reduction/Krylov, a decomposed SIMPLE/PISO with energy, scalars and RANS, decomposed I/O, measured scaling, a native GAMG, AMGX exercised | PAR (PAR-, AMG-, AMX-) | real multi-GPU scaling numbers (needs hardware); device aggregation for ALE/AMR |
| 격자: gmsh 없이 사면체 | a native conforming-Delaunay tetrahedral mesher with prism layers inside `ofgpu-automesher` (`kind: "tet"`), no GPL/LGPL/AGPL anywhere on the path | MESHIO (TET-) | STEP input (needs OCC, LGPL; CAD excluded); polyhedral dual (§92.5) |
| 경계조건: 모두구현 | the true axisymmetric wedge and the whole public BC catalogue (Function1, wall/slip, inlets, outlets, buoyant pressure, thermal, atmospheric, recycled and mapped, synthetic LES inflow, cyclic jumps, static AMI by imprint, compressible, VOF, radiation, species) | PREC (BC-), plus COMP, CHEMRAD | `coded*`/`#codeStream`, `cyclicACMI`, sliding AMI, v2f, wave models: refused by name with a reason |
| 입력형식: 모두구현 | case input (JSONC and dictionary routes) for species, chemistry, combustion, radiation (P1/fvDOM/S2S), parcels, compressible thermo, GAMG settings, wedge, lattice output | CHEMRAD, COMP, PAR, PREC, MESHIO | — |
| CAD / CAD루프 | — | — | **excluded**; the gmsh STEP route in `tools/geom` is not touched (§11 item 6) |
| 2번 미완성: 옥트리 격자 VDB/USD 출력 | `.vdb`/`.nvdb`/`.usda` from any polyMesh through a setup-time VoxelMap; true boundary polygons in `.usda`; `.vdb` version 224 | MESHIO (VIO-) | — |
| Isaac Sim 볼륨 | an Isaac profile (extinction-scaled `density`, IndeX and path-traced scene variants), a generic probe, and a measured "visible" verdict on the drone solve | MESHIO (VIO-) | — |
| 단정밀도 (중요함) | f32 as a **supported** build: f64 accumulation, f64 host geometry and guards, mixed-precision iterative refinement, delta-form transients, local origin, f64 moving geometry and parcel positions, a precision class on every gate row, the f32 test debt paid | PREC (F32-) and a rule on every stream (§5.1) | making `single` the default precision (needs the user) |
| 스튜디오: 제대로 | every `gui_control` command applied or refused by name; Mesh dialog, Boundary editor, project tree, right panels; walls-first surfaces on non-Cartesian meshes; exact unstructured slices, probes, iso-surfaces, streamlines; STL booleans through Manifold (Apache-2.0) | STUDIO (STU-) | the STEP boolean path (CAD) |
| 자동격자 설정: 모두구현 | every autonomy gate pursued honestly, the prior and the optimiser re-enabled only from a passing, non-vacuous gate file, seed 3 spent only on a pre-registered go | STUDIO (AUT-) | relaxing any locked constant |
| 검증게이트쪽 모두 | every open or missed gate given its measuring unit (95-A, 95-G, TG0, T3A, SA plate, Gate 5, 68-C, 78-D, Gnielinski, §110 A/B/C, Martin & Moyce), plus a reference gate for every new physics area | VAL (VAL-) and every stream | — |
| 4. 10분규칙 | **unchanged** ("나중에 일단 이대로"): every unit verifies in ≤ 10 min; longer runs are recorded runs (R) launched by the supervisor | all | changing the rule |

## 3. Facts the assembler verified (and the inconsistency they settle)

| fact | evidence (read-only, 2026-10-06) |
|---|---|
| The real `main` is **`8ab1c4a`**. The local `main` ref is stale at `77785cd` (2026-09-18), 244 commits behind | `git rev-parse origin/main` = `8ab1c4a`; `git rev-list --left-right --count 77785cd...8ab1c4a` = `0 244` |
| `Iteration-CFD-solver` `feat/core-2` = `68d454c` is an **ancestor** of `8ab1c4a` (0 ahead, 189 behind) | `git merge-base --is-ancestor 68d454c 8ab1c4a` true; `rev-list --left-right --count` = `0 189`. PREC's "55 commits ahead of main" was measured against the stale local `main`; it is wrong against `8ab1c4a` |
| `Iteration-CFD-gui` `feat/gui-2` = `53765b3` is also an ancestor of `8ab1c4a` (0 ahead, 45 behind) | `rev-list --left-right --count feat/gui-2...8ab1c4a` = `0 45`. No stream plan said this; the gui tree must be fast-forwarded too |
| `Iteration-CFD-mesh` `feat/mesh-2` = `8ab1c4a`, clean | `git status --porcelain` empty |
| All four checkouts share one `.git` (worktrees of `Iteration-CFD`) | `git rev-parse --git-common-dir` |
| The last SPEC-LIT section on `8ab1c4a` is **§115** | `grep '^## 11[0-9]' rust/SPEC-LIT.md` → §111-§115 |
| `build.rs` lists every CUDA unit in one `KERNEL_UNITS` line | `rust/build.rs:32`; every new `.cu` appends to the same line |
| `WriteCtx { … }` is built at 19 sites in 9 files, 6 of them drivers | `grep -rn "WriteCtx *{"`; this puts VIO-03 in the solver lane |
| The mechanics case lowering lives in `io/case_cht.rs` (`lower_mechanics`, `:2484`) | grep; this decides VAL-11's wave |

## 4. Conventions

### 4.1 Unit ids

One scheme for the whole tranche: `XXX-NN[x]`, with a two-digit number and an optional split letter. Prefixes:

| prefix | stream | area |
|---|---|---|
| CMP | COMP | Boussinesq and compressible |
| CHR | CHEMRAD | chemistry, radiation, their case input, parcels input, species BCs |
| PAR | PAR | transports, distributed flow, decomposed I/O, scaling, studio launch |
| AMG | PAR | native GAMG (was `PAR-G*`) |
| AMX | PAR | AMGX (was `PAR-X*`) |
| F32 | PREC | single precision |
| BC | PREC | wedge and BC catalogue |
| TET | MESHIO | tetrahedral mesher |
| VIO | MESHIO | volume output and Isaac |
| STU | STUDIO | studio |
| AUT | STUDIO | autonomy gates |
| VAL | VAL | open and missed gates |

The renames are listed in Appendix A. `CHR-S1/S2/A1/A2` became `CHR-00a/00b/00c/00d`, `PAR-U1` became `PAR-17`, and the
single-digit ids gained a leading zero. Marks: **(T2)** = PAR Tier 2, droppable; **(R)** = the unit carries a recorded
or supervisor run outside its 10-minute verification.

### 4.2 Lanes and trees

A **lane** is a tree and its branch. Lanes are assigned **by file ownership, not by stream**, because the solver
crate is shared by five streams (§7.4):

| lane | tree | branch | carries |
|---|---|---|---|
| **S** | `Iteration-CFD-solver` | `feat/core-2` | every unit that edits a hot shared solver file: F32, BC, Boussinesq, compressible and chemistry integration, distributed flow, VIO driver wiring, most VAL in-place work late in the tranche |
| **M** | `Iteration-CFD-mesh` | `feat/mesh-2` | TET, VIO lattice and Isaac, AUT, the self-contained compressible module (`compressible/`), the VAL probes, oracle, plate harness and replay records |
| **G** | `Iteration-CFD-gui` | `feat/gui-2` | STU and every stream's gui unit, plus the self-contained Rust modules `chem/`, `radiation/`, `comm/`, `amg/`, `pressure/amgx.rs` |

One unit runs at a time in a lane: the GLM wrapper edits the working tree and runs `cargo` in it, so two units in one
tree would build each other's half-written files.

### 4.3 SPEC-LIT section allocation

Every stream proposed provisional numbers, and they collided (COMP §116-§122 and CHEMRAD §120-§129 against MESHIO
§116/§117, and the labels §P1-§P4 in both PAR and PREC). The allocation below is final. A unit writes only inside its
range, and §80's xref audit refuses a citation to a missing section, so each range's skeleton lands before code cites
it. Cite top-level section numbers; the audit does not see paragraph labels such as (92.14.1) (memory
`am-u7a-driver-tranche`).

| § | content | written by | lane |
|---|---|---|---|
| 116 | The native tetrahedral mesher | TET-00 (and every TET unit) | M |
| 117 | Volume output from any mesh, and the Isaac Sim profile | VIO-00 | M |
| 118 | Mixed precision: what is f64 in the f32 build, and why (accumulation, geometry and guards, IR, delta form, local origin, moving geometry, parcels; the chemistry-in-double scope from CHR-07b; the opt-in f32 GAMG preconditioner from AMG-06b) | F32-02 creates it; F32-03..18, CHR-07b, AMG-06b add subsections | S (CHR-07b's subsection is written in G) |
| 119 | The true wedge: a rotational self-couple | BC-03, BC-04 | S |
| 120 | The BC catalogue | BC-01, then each BC unit | S |
| 121 | Boussinesq buoyancy | CMP-01 | S |
| 122-127 | The compressible formulation (gas, momentum, pressure, energy and loop, LTS, BCs, gates) | CMP-05 | M |
| 128-133 | Chemistry: NASA-7, CHEMKIN-II, kinetics, equilibrium, Rosenbrock, transport, the reacting low-Mach divergence | CHR-00a | G |
| 134-137 | Turbulence-chemistry interaction, P1, fvDOM, WSGG, radiation coupling, case blocks | CHR-00b | G |
| 138 | SPMD transports (`Comm`, ThreadComm, MPI) | PAR-01, PAR-06 | G |
| 139 | Distributed flow | PAR-10c, PAR-14 | S |
| 140 | Measured scaling | PAR-16 | S |
| 141 | Native GAMG and AMGX | AMG-01, AMX-01 | G |
| 142 | Gate 95-A diagnosis and the GMRES outer loop | VAL-10, VAL-11 | M |
| 143 | The 1-D RANS oracle and Wilcox 2006 | VAL-20, VAL-21, VAL-22a | M |
| 144 | The flat-plate harness, Blasius, T3A leg 4, ZPG | VAL-30..34 | M |
| 145 | Martin & Moyce | VAL-03, VAL-60 | G |
| 146 | Gate 78-D's measured leg | VAL-05, VAL-70 | G |
| 147 | Gate 68-C's coupled leg | VAL-71, VAL-72b | S |
| 148 | Studio post-processing on unstructured meshes | STU-12a, STU-14, STU-15 | G |
| 149-150 | reserved for splits and fixes found during the tranche | supervisor | — |

Amendments to existing sections stay in place (§9 untouched; §23.2 AUT-02c; §25 CHR-11; §29.3 AUT-15; §32.4 VAL-40;
§44.1 VIO-00; §50.9 unchanged by CHR-17; §56.11 VAL-34; §60.5 VAL-80; §68.12 VAL-72b; §71.2/§73.6 PAR-08; §78.10
VAL-70; §88.10/§90.10 VAL-33; §92.4 TET-00; §92.14 AUT-03; §95.11 VAL-14; §109.6-7 VAL-12; §110.4-5 VAL-02, VAL-50..54;
§112.3-5 F32 units; §115.6 VAL-06).

### 4.4 Effort and hours

The planners' S/M/L are kept. For scheduling, one unit's wall time in its lane (brief, GLM run, review, both builds and
its tests) is taken as **S = 1.0 h, S-M = 1.25 h, M = 1.5 h, M-L = 2.0 h, L = 2.5 h**, and VAL-13 as 3 × M = 4.5 h.
These are the planners' GLM times plus about 20 minutes of Opus and verification. §9 applies a rework factor on top.

---

## 5. Rules every unit respects (cross-stream contracts)

These bind every unit in every stream. A brief quotes the ones that apply; a review checks them.

### 5.1 Single precision (the user marked it IMPORTANT)

1. **Both builds compile, every Rust unit:** `cargo build --release` and
   `cargo build --release --features single --target-dir target/single`. A unit that adds a feature also compiles the
   combined builds (`"mpi single"`, `"amgx single"`). The unit's own tests pass in both builds.
2. **The f64 build stays bitwise** unless the unit's own opt-in is selected. From W1's F32-01 onward, every Rust unit
   runs `tools/f64_identity.py` (SASS hash per cubin, and a row-by-row diff of the touched `ofgpu-validate -sections`).
   Before F32-01 lands, the unit's module tests and the existing golden hashes are the check.
3. **No new `#[cfg_attr(feature = "single", ignore)]` in tranche 17** (PREC decision 11, adopted for all streams; it
   overrides VAL's "carry §112.3's attribute"). A new test either passes in both builds with a precision-classed f32
   arm, or its subject is f64 itself and it is `#[cfg(not(feature = "single"))]` and named in §112.3.
4. **Every new gate row and test tolerance has a precision class** (PREC §2.2): P (physics, same tolerance in f32), A
   (accumulated identity, evaluated in f64, same tolerance), D (discrete or bitwise, same), R (round-off twin below f32
   unit round-off, `check_roundoff`, tol32 per D-F32-1). **No stream invents an f32 tolerance.** The planners' ad-hoc
   f32 numbers are replaced by the class rule: CMP-04's symmetry "1e-4 in f32", PAR-10b's "1e-4 (f32)", PAR-10c's
   "1e-3 (f32)" and AMX-01's "1e-4 (dFFI)". Under D-F32-1, R rows get `tol32 = max(tol64, min(tol64·2^29, 1e-5))`. A
   row that misses is a finding, never a widened band.
5. **f64 where round-off is the subject:** reductions accumulate in `ofacc`/`Acc` once F32-02 lands. Geometry,
   predicates, host twins and oracles are f64; TET does all its geometry in `[f64; 3]`. Device floors use the
   precision's `OFGPU_*_TINY` and `SCALAR_FLOOR`, never a `1e-300` literal (§112 scan test). Pressure-like unknowns
   are deviations: the compressible gauge `p` about `p_op`, `p_rgh`, `p'`.
6. **Chemistry computes in `double` in both builds** (CHR-07b). `typedef double ofchem` lives in `cuda/chem.cu`, not in
   `ofgpu_device.cuh`, so CHEMRAD never edits the header F32-02 owns. §118 records it as a scope of the one
   mixed-precision section. It is not "the first mixed-precision module", because F32-02 lands first.
7. **Local origin (F32-10, W2):** after W2 every consumer of absolute position goes through the shifted-coordinate
   API. That covers writers, probes, the VoxelMap box, parcel injection, the gate-mesh generators, `fixedProfile`,
   the atmospheric `z`, `prgh`'s `h_ref` and the surge-front sampler. The shift is 0 in the f64 build.
8. **`PolyMeshRaw` points become f64 (`DVec3`) in F32-03.** In the f64 build this is the identity. TET-11 and VIO-01
   build against it after the W1 sync. TET-11's "f32 flip refused" check moves to the f32 cast it guards.

### 5.2 Boundary conditions

1. Every new BC, from any stream, is a §4 triple rewrite with no new evaluation branch. Its name is registered in
   BC-01's catalogue with a status (implemented, alias, planned unit, owned by a stream, or refused with a reason), and
   is taken from the public user guides by name only.
2. Time and coordinate profiles go through BC-06's device Function1, so §81's capture gate holds.
3. **Ownership:**
   - The compressible kinds belong to COMP (CMP-16/17/18): `totalPressure` with `gamma`, `totalTemperature`,
     `supersonicInlet`, the compressible mass-flow inlet, `waveTransmissive`, `characteristicFarfield`/`farfield` and
     the supersonic outlet switch.
   - BC-20 is reduced to `inletOutletTotalTemperature`, `supersonicFreestream`, `freestreamPressure` and
     `fixedPressureCompressibleDensity`, built on `compressible/bc.rs`.
   - The generic Orlanski `advective` outlet belongs to BC-09. It is written with the wave speed as a face-field
     input, so CMP-17 reuses it with `w = u_n + c`.
   - Radiation wall conditions live inside the radiation module (CHR-18/20/21). CHR-23 registers their four names.
   - Species BCs go to the new CHR-29.
4. **The wedge:** after W2 every axisymmetric gate (BC-05, CHR-16 Flame D, VAL-23 pipe) uses BC-04's true wedge. A
   3-D sector or quarter-pipe fallback is used only if BC-04 fails its own gate, and then the user decides.

### 5.3 Gates

1. **Never loosen a gate**: reduce a gate's size, never its tolerance. Every new physics area has an open-reference
   gate, and SPEC-LIT §10/§22 forbid code-to-code answer keys.
2. **New gate code lives in `src/bin/validate_gates/<gate>.rs`** (VAL D10, extended to every stream). `validate.rs`
   gains only an `enter_gate` call in its append-only dispatch block. In-place edits to existing `validate.rs` code
   have one owner per wave (§7.4).
3. A gate whose reference cannot be obtained is NAMED and NOT RUN. It is never replaced by a weaker check under the
   same name.
4. Any change to the numerics of an existing path is the user's decision (memory
   `no-solver-numerics-changes-for-convergence`). That covers PAR Phase-2 conversions that move a serial bit,
   VAL-13's implicit stencil, a KaysCrawford default, and an f32 chemistry path.

### 5.4 Licence and provenance

No GPL, LGPL or AGPL source is read or linked. That rules out OpenFOAM, solids4foam, SU2, gmsh, TetGen, CGAL, Netgen,
OCC, MeshLab, Cork and FeenoX. `docs/02`'s GAMG rows and `docs/01`'s formula columns are not read either. Permissive
material may be read and is recorded: Shewchuk's predicates.c, Geogram, OpenVDB, OpenUSD docs, MS-MPI's `mpi.h`,
AMGX, Manifold, FDS and NASA/NIST/OSTI reports. Every new file carries the house header with "No GPL-licensed source
was consulted." and a PROVENANCE row. Every DOI is checked by the unit that cites it.

### 5.5 Process

- One unit per lane at a time. Commit per unit (memory `no-worktree-node-modules-junctions`). The bug is shown by a
  failing test first.
- Verification stays at ≤ 10 min, with `-sections`, module filters and the `--target-dir` split.
- **R** and **L** runs are launched by the supervisor in a visible console from Bash `cmd //c start`, and stopped by
  PID or name only (memories `launch-consoles-from-bash-not-powershell`, `kill-processes-by-name-only`).
- **PAR-09's allocation audit** (W4) and later: a new `zeros(n_cells)` in the audited SIMPLE, scalar and energy paths
  either allocates `n_cells + n_halo` or joins the allow list with a reason. `compressible/`, `chem/` and
  `radiation/` are outside the audit, because their drivers refuse decomposition by name. PAR-09 keeps the public
  field-constructor signatures.
- **The schema `docs/schema/case-1.json` is generated in lane S only and never hand-merged.** After each sync it is
  regenerated and the gui copy is re-synced (CMP-29, CHR-28, BC-22 and `schema.test.ts`).
- Counts (the PROVENANCE/NOTICE file count, §112.3's ignore count, README counts) are recomputed by their audit tests
  after a merge, never merged by hand.

---

## 6. Cross-stream dependencies

| from (provides) | to (needs) | what crosses | handled by |
|---|---|---|---|
| F32-01 (W1 S) | every Rust unit from W1 on; BC-05 | `-sections`, the f64 identity tool | lands first in S |
| F32-02 (W1 S) | CHR-07b, AMG-*, every new reduction | `ofacc`, and §118's existence | W1; CHR-07b is G W3 |
| F32-03 (W1 S) | TET-11, VIO-01, every host mesh consumer | f64 `PolyMeshRaw` points, `HostGeom64` | dep added; S1 sync |
| F32-06/07 (W1, W3 S) | AMG-06a, PAR-10b/10c f32 rows, every gate's P/A class in f32 | iterative refinement on every Krylov path, including SPMD | AMG-06a and F32-07 deps added |
| F32-10 (W2 S) | VIO-01..09, VAL-60, the CMP gate meshes, BC-13/14 | the local origin | rule 5.1.7 |
| F32-13/14 (W5 S) | — | classify pre-existing rows only; new rows carry classes from birth | run after M's in-place `validate.rs` work (W2-W4) |
| BC-01 (W1 S) | CMP-16/17/18, BC-20, CHR-23, CHR-29, BC-22 | the catalogue and its status test | W1 |
| BC-04/05 (W2 S) | CHR-16, VAL-23 | the true wedge | deps added |
| BC-06 (W1 S) | BC-07..21 | Function1 | W1 |
| BC-09 (W1 S) | CMP-17 | the Robin-form advective kernel | dep added |
| CMP-01..03 (W1 S) | PAR-11 | Boussinesq | dep added |
| CMP-15..18 (W4 S) | BC-20, CMP-19..25, CMP-29 | the compressible driver and BCs | W4, then W5 |
| CHR-07b, 08 (W3 G) | CHR-09, CHR-12a (S W4) | the device chemistry and transport | S3 sync |
| CHR-12b (W4 S) | CHR-14, 15 (G W5) | the reacting driver | S4 sync |
| CHR-23 (W5 S) | VAL-54 (M W6), CHR-28 (G W6) | radiation in `ofgpu-lowmach` | S5 sync |
| CHR-25/26 (W3, W5 S) | VAL-72b | parcels from the case and in the driver; **VAL-72a merged into CHR-26** | — |
| PAR-01 (W1 G) | PAR-02..07 | `Comm` | G W1-W3 |
| PAR-04b (W2 G) | F32-07 (W3 S), AMG-07a | SPMD Krylov | S2 sync |
| AMG-04a (W2 G) | AMG-04b, 05 (W3 S), AMX-02 | GAMG cycles | S2 sync |
| VIO-02 (W2 M) | VIO-03 (W3 S) | VoxelMap sampling | S2 sync |
| VIO-06 (W3 S), VIO-10 (W5 M) | VIO-11 (W5 M) | `ofgpu-sample volume`, the probe | — |
| TET-12 (W5 M) | TET-17 (W6 G) | the `kind: tet` summary | S5 sync |
| AUT-15 (W2 S) | AUT-16 (W6 M) | y⁺ JSON | — |
| VAL-04/05/03 keys (G) | VAL-33, 70, 60 | answer keys | — |
| solver lane `lowmach.rs` | CMP-03, CMP-15 (messages), CHR-12b, 23, 24, 26, VIO-05a, AUT-15, VAL-41 | one file, nine units | all in lane S, serialised |
| solver lane `case_json.rs` and schema | CMP-03, CMP-14, CHR-12a, 23, 25, BC-05, AMG-04b, VIO-04 | one file | all in lane S, serialised |
| solver lane `field.rs`/`field.cu`/`field_setup.rs` | BC-01..21, CMP-16/17/18, CHR-29, PAR-09, PAR-10a | one family of files | all in lane S, serialised |
| solver lane `energy.rs` | CHR-10a/10b/11, BC-12, PAR-09, PAR-11 (and CMP-15's message) | — | CHEMRAD's energy edits land in W3-W4, before PAR-09 |

---

## 7. The wave plan

### 7.1 Shape

Seven waves. W0 is the supervisor's setup. W1-W6 each run the three lanes in parallel, file-disjoint by the ownership
table in §7.4, and end in a **sync**, S1-S6, after which all three branches point at the same commit. The script in
`plan17/asm/assemble.py` checked every dependency of all 261 units against this plan (0 violations): a dependency is
either in an earlier wave, or earlier in the same lane in the same wave. `filecheck.py` found no hot file owned by two
lanes in one wave, except the append-only files that §7.3 merges by rule.

### 7.2 W0 — supervisor setup (no GLM)

1. **CMP-00 for both lagging trees.** `git -C Iteration-CFD-solver merge --ff-only 8ab1c4a` and
   `git -C Iteration-CFD-gui merge --ff-only 8ab1c4a`. Both are verified ancestors (§3). Briefs say `8ab1c4a` or
   `origin/main`, never `main`, until the user refreshes the stale local ref.
2. **Baselines per tree, recorded in docs/17:**
   - both builds compile;
   - the lib and bin test counts and the counted single-ignore number (the §112.3 bold count);
   - the gui server, shared and web vitest counts, today's numbers rather than memory's 286+32 of 2026-09-11;
   - `ofgpu-validate` 1021/1026 with its five failing rows;
   - the L1a golden hashes (95bfea4);
   - the clippy drift baseline (memory `clippy-toolchain-drift`).
3. **Write docs/17** from this file in the solver tree, carrying the §4.3 allocation, so every brief can cite it.
4. **Ask the user** the §11 questions in one message, because their answers are needed by W3-W5, not W1.
5. Target dirs `target/` and `target/single` in each tree. No `gui/node_modules` junction in the solver or mesh tree.

### 7.3 The sync, S1-S6 (about 2 h each)

1. Each lane finishes its current unit and commits. The trees are clean.
2. In the solver tree: `git merge feat/mesh-2`, then `git merge feat/gui-2`. These are local branches of the shared
   `.git`, and nothing is pushed. Resolve by rule:
   - **SPEC-LIT:** sections in number order;
   - **PROVENANCE, NOTICE, `KERNEL_UNITS`, `lib.rs` mod lines, `Cargo.toml` `[[bin]]` and the `validate.rs` dispatch
     block:** union;
   - **`docs/schema/case-1.json`:** regenerate;
   - **counts:** recompute through their audit tests.
3. Verify the merged commit:
   - both builds;
   - the provenance, ignore-count, xref §80 and registry-sync audit tests;
   - the module tests of every module changed in the wave, in both builds, because F32 units change f32 results
     under other lanes' tests;
   - the gui suites;
   - `f64_identity.py` against the previous sync's validate JSON on the sections the wave touched.
4. `git -C Iteration-CFD-mesh merge --ff-only feat/core-2` and the same for the gui tree. All three lanes now start
   the next wave from one commit.
5. **No push and no merge to `main`.** Both are the user's (§11 item 8).

### 7.4 Hot-file ownership per wave

A file is edited by one lane per wave. Unlisted new files belong to the unit that creates them.

| file / area | W1 | W2 | W3 | W4 | W5 | W6 |
|---|---|---|---|---|---|---|
| `field.rs`, `field.cu`, `field_setup.rs`, `momentum.*`, `simple.*`, `energy.*`, `fv.*`, `ldu.cu`, `mesh.rs`, `mesh/geometry.rs`, `io/polymesh.rs`, `timescheme.*`, `turbulence.*`, `species.rs`, `scalar_transport.rs` | S | S | S | S | S | S |
| `io/case_json.rs`, `io/case.rs`, `io/dict.rs`, `docs/schema/` | S | S | S | S (and G for AMG-06b's `case.rs` key) | S | S |
| `io/case_cht.rs` | S | S | S | **M** (VAL-11) | S | S |
| every driver in `src/bin/` except `par.rs`, `compressible.rs`, `sample.rs` | S | S | S | S | S | S |
| `solver.rs`, `solver.cu`, `precon.rs` | S | **G** (AMG-04a) | — | **G** (AMG-04c) | S | S |
| `halo.*`, `exactsum.*`, `distsolve*.rs`, `device.rs` | G (new `comm/`) | **G** | S (F32-07 `distsolve`), G (`device.rs`) | **G** (AMG-07a/b) | S | S |
| `decompose.rs` | **G** (PAR-08) | — | — | — | — | S (PAR-13a) |
| `pressure/*.rs` | G (`amgx.rs`) | S | S | — | — | — |
| `io/writer.rs`, `io/vdb.rs`, `io/nvdb.rs`, `io/output_plan.rs` | S | S | S | S | — | — |
| `io/voxelmap.rs` (new) | — | **M** | S | S | — | — |
| `io/usda.rs` | — | **M** (VIO-07) | — | S (VIO-09) | — | — |
| `bin/sample.rs` | — | — | S (VIO-06) | — | — | — |
| `models/coupled.rs`, `models/registry.rs` | S (CMP-02) | — | — | **M** (VAL-22c) | — | S (F32-16/17) |
| `models/spalart_allmaras.rs`, `cuda/sa.cu` | — | — | — | — | **M** (CMP-25) | S |
| `solid/*` | **M** (VAL-10) | — | **M** (VAL-12/14) | **M** (VAL-11) | S (F32-15) | S |
| `blockgen.rs` | S (BC-02) | — | **M** (VAL-30) | — | — | — |
| `parcels*`, `cuda/parcel*.cu` | — | S (F32-12) | — | S (VAL-71) | S (CHR-26, VAL-72b) | — |
| `compressible/`, `cuda/compressible.cu` | M | M | M | **S** (CMP-15..18) | M | M |
| `chem/`, `cuda/chem.cu` | G | G | G | **S** (CHR-09/10b `chem/source.rs`) | G (CHR-14/15) | — |
| `radiation.rs`, `radiation/`, `cuda/radiation.cu` | — | G | G | — | S (read only; CHR-23/24 add `io/case_rad.rs`) | — |
| `tetmesh/`, `automesher/` | M | M | M | M | M | M, except automesher **tests** S (F32-16) |
| `vv/rans1d.rs`, `models/k_omega_2006.rs`, `cuda/k_omega_2006.cu` | M | — | — | M | — | — |
| `validate.rs`, in-place code | S (F32-01) | M (VAL-01) | M | M (VAL-11, 40, 51-53, 80) | S (F32-13/14) | S (CHR-27) |
| `validate.rs` dispatch block | union | union | union | union | union | union |
| `tools/autonomy/`, `tools/geom/stl_repair.py`, `tools/isaac/`, `tools/drone/` | M | M | M | M | M | M |
| `gui/**` | G | G | G | G | G | G |
| `README*`, `docs/GUIDEBOOK*` | — | — | S (VIO-04's GUIDEBOOK lines) | — | — | S (CMP-28, then VAL-82) |

### 7.5 The waves

Hours are the §4.4 lane hours. "Wave length" is the simulated length of the wave: its longest lane plus the 2 h sync.
Within a lane, units run in the order listed.

| wave | lane S (solver tree) | lane M (mesh tree) | lane G (gui tree) | wave length (sim.) |
|---|---|---|---|---|
| **W1** | 17 units, 25.5 h: F32-01, BC-01, F32-02, F32-03, F32-09, F32-04, CMP-01, CMP-02, CMP-03, F32-06, F32-05, BC-02, BC-03, F32-08, BC-06, BC-07, BC-09 | 19 units, 23.5 h: TET-00, TET-01, TET-02, TET-04a, TET-05, VIO-00, CMP-05, CMP-06, CMP-07, AUT-01, AUT-03, AUT-06, AUT-08, AUT-09, VAL-02, VAL-06, VAL-20, VAL-21, VAL-10 | 18 units, 22.0 h: CHR-00a, CHR-00b, CHR-00c, CHR-00d, CHR-01, CHR-02a, PAR-01, PAR-08, AMG-01, AMX-01, STU-01, STU-10, STU-02, STU-03, STU-04, STU-16, VAL-03, VAL-05 | 27.5 h (ends at 27.5 h) |
| **W2** | 16 units, 25.2 h: CMP-04, BC-04, BC-05, F32-10, F32-11, F32-12, BC-08, BC-10, BC-11, BC-12, BC-13, BC-14, BC-15, BC-17, VIO-08, AUT-15 | 15 units, 26.2 h: TET-03a, TET-03b (R), TET-04b, TET-06a, TET-06b, VIO-01, VIO-02, VIO-07, CMP-08, CMP-09, CMP-10, CMP-26, AUT-02a, VAL-01, BC-18 | 17 units, 25.5 h: CHR-02b, CHR-03, CHR-04, CHR-05, CHR-06, CHR-07a, CHR-17, CHR-18, CHR-19, PAR-02, PAR-03, PAR-04a, PAR-04b, AMG-02, AMG-03a, AMG-03b, AMG-04a | 28.2 h (ends at 55.8 h) |
| **W3** | 15 units, 23.5 h: F32-07, CMP-14, CHR-10a, CHR-25, VIO-03, VIO-04, VIO-06, VIO-05a, VIO-05b, BC-16, BC-19, BC-21, AMG-04b, AMG-05, AMX-02 | 14 units, 20.8 h: TET-07a, TET-07b, TET-08, CMP-11, CMP-12, CMP-13, AUT-02b, AUT-02c, AUT-05, VAL-12, VAL-14, VAL-30, VAL-31, VAL-50 (R) | 13 units, 19.5 h: CHR-07b, CHR-08, CHR-20, CHR-21, CHR-22, PAR-05, PAR-06, PAR-07, VAL-04, STU-05, STU-12a, STU-06a, STU-11 | 25.5 h (ends at 81.2 h) |
| **W4** | 16 units, 25.0 h: CMP-15, CMP-16, CMP-17, CMP-18, BC-20, CHR-09, CHR-10b, CHR-11, CHR-12a, CHR-12b, CHR-29, VIO-09, PAR-09, PAR-10a, PAR-10b, VAL-71 | 18 units, 25.5 h: TET-09, TET-10a, TET-10b, VAL-11, AUT-04, AUT-07, AUT-10, AUT-11, VAL-22a, VAL-22b, VAL-22c, VAL-32 (R), VAL-34 (R), VAL-40 (R), VAL-51 (R), VAL-52 (R), VAL-53 (R), VAL-80 | 16 units, 22.5 h: AMG-06a, AMG-04c (T2), AMG-06b (T2), AMG-07a (T2), AMG-07b (T2), PAR-17, VAL-60, VAL-70, STU-06b, STU-07a, STU-07b, STU-08a, STU-09, STU-12b, STU-13, STU-17 | 27.5 h (ends at 108.8 h) |
| **W5** | 14 units, 24.0 h: PAR-10c, PAR-11, PAR-12, PAR-15, PAR-16, CHR-13 (R), CHR-23, CHR-24, CHR-26, VAL-41, VAL-72b (R), F32-13, F32-14, F32-15 | 15 units, 22.0 h: TET-11, TET-12, TET-13a, TET-15 (R), TET-16, CMP-21, CMP-23, CMP-24, CMP-25, VAL-23 (R), VAL-33, VAL-35 (R), AUT-12 (R), VIO-10, VIO-11 (R) | 13 units, 19.5 h: CMP-19, CMP-20, CMP-22, CHR-14, CHR-15, CHR-16 (R), STU-08b, STU-14, STU-15, STU-18 (R), CMP-29, BC-22, F32-19 | 26.0 h (ends at 134.8 h) |
| **W6** | 10 units, 17.0 h: PAR-13a (T2), PAR-13b (T2), F32-16, F32-17, PAR-14 (T2), CHR-27, CMP-28, VAL-81, F32-18 (R), VAL-82 (R) | 11 units, 20.0 h: TET-13b, TET-13c, TET-14 (R), TET-18 (R), TET-19, CMP-27 (R), VAL-54 (R), VAL-13, AUT-13, AUT-14 (R), AUT-16 (R) | 3 units, 3.5 h: CHR-28, TET-17, STU-19 | 22.0 h (ends at 156.8 h) |

What each wave delivers:

- **W1 — foundations.**
  - S: the f32 core (F32-01 → 02 → 03 → 09 → 04 → 06 → 05 → 08), the catalogue and Function1 (BC-01, 06), the wedge
    mesh and kind (BC-02, 03), Boussinesq through the drivers (CMP-01..03), and the first BC family (BC-07, 09).
  - M: the TET predicates and surface preparation, the compressible contract §122-§127 and exact solutions, the
    autonomy evidence units, and the VAL probes (95-A fixed point, RANS oracle).
  - G: both CHEMRAD contracts and keys, NASA-7 and the CHEMKIN lexer, `Comm`, Hilbert, GAMG aggregation, AMGX by
    `libloading`, the studio census, wall-first surfaces and booleans.
- **W2.**
  - S: Gate CMP-B, the true wedge and its gate, the local origin, f64 moving geometry and parcels, and the BC families.
  - M: Delaunay, the size field and the remesh, the VoxelMap, the compressible thermo, momentum and pressure, the
    C-mesh, AMI imprint.
  - G: kinetics, equilibrium, Rosenbrock and the device ω; P1 and the S_N quadratures; the SPMD halo, reduction and
    Krylov; the GAMG hierarchy, cycles and preconditioner.
- **W3.**
  - S: IR on every backend (F32-07), the compressible and parcel case input, the volume writers in every driver, GAMG
    as a backend (AMG-04b/05) and AMGX in a loop (AMX-02).
  - M: boundary recovery and carve, the compressible energy, loop and LTS, the G-family repairs, the flat-plate
    harness and Blasius.
  - G: device Rosenbrock and transport, fvDOM with scattering, WSGG, MPI, `ofgpu-par`, the mesh draft and the exact
    polyMesh plane cut.
- **W4.**
  - S: `ofgpu-compressible` and its BCs, reacting `ofgpu-lowmach` (CHR-09 → 12b), species BCs, the Isaac profile,
    and the halo audit with the first distributed solves.
  - M: refinement and optimisation, the autonomy rules and the learned layers, Wilcox 2006 on the device, the
    recorded runs for T3A, ZPG, Gnielinski and §110.
  - G: distributed GAMG, the remaining studio dialogs and panels, the unstructured slice and probe.
- **W5.**
  - S: decomposed SIMPLE and RANS, decomposed I/O and measured scaling; radiation, S2S and parcels from the case;
    the flame gate; f32 classification of every validate row.
  - M: tet emission, the `kind: tet` driver and the first layers; the compressible gates (nozzle, ramp, Ringleb) and
    ρ-SA; the pipe and T3A verdicts; the autonomy go/no-go projection (L-run); the Isaac verdict (VIO-11).
  - G: Sod, acoustics, Couette, EDM, PaSR, EDC and Flame D (REDUCED); unstructured iso-surfaces and streamlines; the
    G-LLM live re-measure; the gui units for the compressible, BC and f32 work.
- **W6 — close-out.**
  - S: PAR Phase 2 (T2), the f32 test debt (F32-16/17), the CHEMRAD validate registry, the README counts, VAL-81/82
    and the deferred full f32 validate run (F32-18).
  - M: layers at sharp features, the tet pipe solve, the drone tet run, RAE2822, McCaffrey with radiation, seed 3
    (only on a go) and G-YPLUS.
  - G: the gui registry for chemistry and radiation, the `kind: tet` summary, and the Playwright walk.

### 7.6 GPU and long runs

There is one GPU (RTX 5070 Ti) and three lanes. Unit verification is short and time-shares the card. **R/L runs**
need the card for long periods. They go into the sync windows and nights, and a lane that runs on while one of them is
active takes its host-only or gui units. A unit whose 10-minute verification was measured under contention is
re-measured before it is reported as over budget. VAL-50 prices every §110 run in W3, before any batch is scheduled.

**The R/L backlog:**

| unit | runs |
|---|---|
| VAL-32, 34, 40, 51-53, 23, 33, 35, 54 | 12 + 12 + 4 + 9 + 3 + 3 + 12 + a few + 3 runs |
| VAL-72b | the 68-C 90-stream overnight batch |
| AUT-12, AUT-14, AUT-16 | about 2 h, then hours of CPU, then GPU hours |
| STU-18 | about 20 min live |
| CMP-27 | RAE2822 fine meshes |
| CHR-13, CHR-16 | the full flame sweep and the full Flame D comparison (both need the user, §11) |
| TET-18 | the drone tet mesh |
| F32-18 | the full single-build validate run |
| VIO-11 | Isaac probes, about 6 min each |
| VAL-82 | the full validate run for the README |

The planned total is **10-15 GPU-nights**, to be priced by VAL-50.

---

## 8. Critical path

**The tranche ends when the longest of three chains ends. Each of them occupies all six waves:**

1. **TET (lane M):** TET-00 → 04a → 06a → 06b → 07a → 07b → 08 → 09 → 10a → 10b → 11 → 12 → 13a → 13b → 13c → TET-18.
   It is 16 units and 30.5 h, the longest single-stream chain. 7a, 7b, 9, 10b and 13b/c are the units the planner
   flagged for possible splits. A miss at 7a/7b triggers plan B (isosurface stuffing, about 4 more units). A miss at
   10b triggers TET-10c (exudation, 1 L).
2. **CHEMRAD (lanes G → S → G → S):** CHR-00a → 02a → 02b → 03 → 04 → 06 → 07b (G W1-W3) → 09 → 10b → 11 → 12b (S W4) →
   14 → 15 → 16 (G W5) → CHR-27 (S W6). It is 15 units and 23.5 h, and it crosses lanes three times, so each crossing
   waits for a sync. The planner's chain was 17-18 units long and put CHR-13 on it. CHR-13 is not on it, because
   CHR-14 depends on CHR-12b, not on CHR-13.
3. **F32 (lane S):** F32-01 → 02 → 06 (W1) … → F32-13 (W5) → 16 → 17 → 18 (W6). It is 7 units and 14 h of pure
   dependency, but it is pinned to the end of the tranche by design: classification and the test debt wait until every
   other stream's rows and tests exist.

The other chains have slack:

| stream | chain | length |
|---|---|---|
| COMP | CMP-05 → 08 → 09 → 10 → 11 → 12 (M W1-W3) → 15 → 16 (S W4) → 21 (M W5) → 28 (S W6) | 10 units, 17.5 h. The planner's "13 units" also counted parallel units. The equally long RAE branch is 15 → 25 → 27 |
| PAR | PAR-01 → 02 → 04a (G W1-W2) → 09 → 10a → 10b (S W4) → 10c → 11 (S W5) → 14 (T2, S W6) | 9 units |
| GAMG and AMGX | AMG-01 → 02 → 03a → 03b → 04a (G W1-W2) → 04b → 05 → AMX-02 (S W3) | 8 units |
| BC | BC-02 → 03 → 04 → 05 | 4 units, done by W2 |
| VIO | VIO-00 → 01 → 02 (M) → 03 → 04 → 05a/b (S W3), then VIO-09 (S W4) → VIO-10 → 11 (M W5) | — |
| STU | STU-01 → 05 → 07a → 07b → 08b → 19 | 6 units |
| AUT | AUT-09 → 10 → 11 → 12 (L-run) → 13 → 14 (L-run) → 16 (GPU L-run) | gated by its L-runs, not by coding |
| VAL | the coding chain is VAL-20 → 21 → 22a → 22b → 22c → 23 (conditional on D-VAL-5) | the real tail is the R backlog of §7.6 |

---

## 9. Counts and an honest duration

**Units: 261.** The tables hold 262 rows, because VAL-72a stays as a "merged into CHR-26" row, and CHR-29 is new.

| stream | units | note |
|---|---|---|
| COMP | 30 | CMP-00 is the W0 fast-forward |
| CHEMRAD | 37 | 36 + CHR-29 |
| PAR | 35 | 28 Tier 1 + 7 Tier 2 |
| PREC | 41 | 19 F32 + 22 BC |
| MESHIO | 40 | 27 TET + 13 VIO |
| STUDIO | 41 | 23 STU + 18 AUT. The plan said 22 + 17 = 39, but its own table has 41 |
| VAL | 37 | 38 rows − VAL-72a. That is 30 firm and 8 conditional; the plan said 31 firm. VAL-13 is 3 units if it is taken |

- **Firm core: 244** (261 − 8 VAL conditional − 7 PAR Tier 2 − TET-19 stretch − STU-15 optional).
- Five units are Opus-only contract or reading units: CMP-05, CHR-00a, CHR-00b, VAL-02 and VAL-06. TET-00 and VIO-00
  are Opus text plus a GLM skeleton. One unit is a supervisor step (CMP-00). The rest are GLM-coded.

**Lane hours (§4.4 rates):**

| lane | units | hours |
|---|---|---|
| S | 88 | 140.2 h |
| M | 92 | 138.0 h |
| G | 80 | 112.5 h |
| all | 261 | 390.8 h of unit work |

The wave-gated simulation, with every lane serial and every cross-lane dependency waiting for its sync, ends at
**156.8 h**. That includes 12 h of syncs.

**Honest estimate:**

- **Code units, 3 to 4 weeks of calendar time.**
  - Add 25 % for briefs that are wrong or tests that fail first time. Every tranche since 09-26 found two or three
    brief errors per stream (memories `cad10-…`, `cad12-…`, `am13-…`).
  - That gives **≈ 196 h of lane time**. At about 16 productive hours a day, that is about **12-13 working days**.
  - Then come the waits that cannot be planned away: the weekly subagent limit (memory `weekly-limit-handoff`, a
    9-29 precedent), the user's answers to §11, and GPU contention with the R backlog. 3-4 weeks is the realistic
    range.
- **Every recorded verdict, 4-5 weeks.** That adds the 10-15 GPU-nights of §7.6, which mostly overlap W3-W6, plus the
  tail runs that only start after W6: AUT-14, AUT-16, F32-18's full run and VAL-82.
- **The things that would stretch it further:**
  - TET plan B (+4 units);
  - TET-10c (+1 L);
  - the COMP density-based contingency (+4 units, the user's call);
  - VAL-13 (+3 units, the user's call);
  - Tier 2 PAR failing its golden-hash rule. That is dropped, not extended.

---

## 10. Decisions taken by recommendation

### 10.1 Taken by the assembler

| # | decision | reason |
|---|---|---|
| A1 | Three lanes on the three trees, assigned by **file ownership**, with a sync and a fast-forward of all three branches at every wave end | Five streams share the solver crate. By the planners' own tree columns, the solver tree would carry 175 units and 268 h, against 56 units and 85 h for mesh and 29 units and 39 h for gui. That would put the tranche at 268 h or more of serial work, while file-partitioning moves the self-contained new modules (`compressible/`, `chem/`, `radiation/`, `comm/`, `amg/`, `tetmesh/`, `vv/`, `validate_gates/`) to the other two trees |
| A2 | No fourth worktree | Another two CUDA target dirs, more GPU contention, more merges, and the 09-11 junction incident. Three balanced lanes already bring the wall time near the TET chain's own length |
| A3 | W0 fast-forwards both `feat/core-2` and `feat/gui-2` to `8ab1c4a` | Both are verified ancestors. PREC's "55 ahead" was measured against the stale local `main` |
| A4 | The SPEC-LIT allocation of §4.3 | Every stream collided |
| A5 | Unified ids (§4.1, Appendix A) | Mixed one- and two-digit ids and `PAR-G/X/U` sub-prefixes |
| A6 | BC ownership partition (§5.2.3); BC-20 reduced; CMP-17 reuses BC-09 | COMP and PREC both planned `waveTransmissive`, `totalTemperature` and the reflection gate |
| A7 | Species BCs get a unit, CHR-29 | PREC assigned them to chemistry, and CHEMRAD had none. "경계조건: 모두구현" |
| A8 | VAL-72a merged into CHR-26 | The same scope: two-way parcels in `ofgpu-lowmach` |
| A9 | One mixed-precision section §118, with `ofchem` defined in `chem.cu` | PREC and CHEMRAD each claimed the "first" mixed precision, and both would have edited `ofgpu_device.cuh` |
| A10 | No new single-ignore attributes; every tolerance carries a precision class; the ad-hoc f32 numbers are replaced (§5.1) | The user's priority is f32, and the streams disagreed |
| A11 | New gates in `validate_gates/` for every stream | `validate.rs` is 23 k lines and every stream adds rows |
| A12 | `PolyMeshRaw` points in f64 from F32-03 | TET and VIO want f64 geometry, PREC needs f64 points, and it is the identity in the f64 build |
| A13 | F32-13/14 in W5 and F32-16/17 in W6 | They classify existing rows and tests after every in-place `validate.rs` edit (M, W2-W4) and every new test exists |
| A14 | PAR-13a/b in W6 | f32 golden hashes are stable only after the F32 core (W1-W3) |
| A15 | CHEMRAD's energy edits (CHR-10a/10b/11) before PAR-09 | PAR-09's audit then covers them once |
| A16 | PAR-14's refusal list adds `ofgpu-compressible`, reacting chemistry and P1/fvDOM | COMP decision 14 and CHEMRAD both refuse decomposition |
| A17 | TET quality bars (min dihedral ≥ 10°, max ≤ 165°, layer coverage ≥ 90 %) are adopted as fixed before measurement | They come from published benchmarks. A miss goes to the user, never a lower bar |
| A18 | Isaac τ = 2, studio default precision `double`, the AMGX 2.5.0 DLL from `FDS_GPU_RUST_AmgX2` used for the exercise (rebuild recipe deferred), BC tail units in this tranche and last | These were planner recommendations marked "for the user", but each is reversible and changes no gate |

### 10.2 Taken by the stream planners (adopted)

- **COMP:**
  - pressure-based all-speed solver, in a self-contained `compressible/` module;
  - gauge p about `p_op`;
  - energy in the `c_p T + K` form;
  - ψ* shock capture;
  - LTS;
  - Robin-triple BCs;
  - laminar plus ρ-weighted SA;
  - analytic and experimental keys only;
  - in-memory gate meshes;
  - Boussinesq guard at β·ΔT 0.1;
  - centro-symmetry gate;
  - single-GPU v1.
- **CHEMRAD:**
  - reacting flow in `ofgpu-lowmach`;
  - Lie or Strang splitting with a 0-D reactor;
  - ROS4/RODAS3 with an FD Jacobian;
  - NASA-7 in `chem/thermo.rs`;
  - CHEMKIN-II, without reading Cantera;
  - WD1, BFER2 and JL4 authored here, GRI and DRM19 only after a licence check;
  - P1 on PCG, fvDOM step scheme;
  - S4 default, S8 gated;
  - computed exact references;
  - one lowering per block;
  - parcels in `ofgpu-lowmach` only.
- **PAR:** D1-D13 (SPMD over `Comm`, our own MPICH-ABI binding, host staging, tolerance-equal then bitwise, wall
  distance split, whole-mesh read with a hash vote, unsmoothed pairwise aggregation in LDU form, host aggregation,
  `solver` versus `preconditioner GAMG`, AMGX by `libloading`, default solver unchanged).
- **PREC:** decisions 1-11 (mixed precision rather than pure f32, IR only under `single`, guards keep their limits,
  shadow over compensated state, local origin under `single` only, wedge as a rotational self-couple, Function1 on the
  device, cyclicAMI by imprint, radiation and species BCs owned by their streams, no new single-ignore).
- **MESHIO:**
  - conforming Delaunay with Delaunay refinement;
  - post-inflated prisms;
  - f64 inside `tetmesh`;
  - one CLI;
  - STEP stays out;
  - no new crates;
  - setup-time VoxelMap;
  - two activations and two Isaac profiles;
  - extinction scaled to τ;
  - file version 224;
  - no polyhedral dual.
- **STUDIO:** D-1 to D-11 (seed-2 full scope withdrawn, STEP and gmsh untouched, Manifold for STL booleans, server-side
  unstructured post-processing, wall-first only when walls and non-walls coexist, G-PRIOR tightened, one G-OPT model
  change, G-COST never relaxed, G-FAIL bar stands, hole limit 64 only with proof, AUT-04 on whitelisted knobs).
- **VAL:**
  - measure before fixing, for 95-A and TG0;
  - an opt-in GMRES outer loop;
  - Wilcox 2006 as the TG0 candidate;
  - SA plate against K-S and Coles, not C_d;
  - one plate harness validated on Blasius;
  - undistributable keys fetched, not tracked;
  - Gnielinski sibling rows;
  - Martin & Moyce live;
  - `validate_gates/`;
  - no gate loosened.

---

## 11. Decisions that need the user

Eight items. None blocks W1 or W2. The column says when the answer is needed.

| # | decision | recommendation | needed by |
|---|---|---|---|
| 1 | **D-F32-1:** f32 tolerance for round-off-class (R) rows. Either same ulps as f64, capped at 84 ulps (`tol32 = max(tol64, min(tol64·2^29, 1e-5))`), or `skip` with a printed count | the same-ulps rule. It is never looser in ulps, and the f64 column stays byte-identical | W5 (F32-13). It is one constant either way |
| 2 | **Hardware and installs:** MS-MPI runtime and SDK (admin) to run PAR-06/16 over real MPI; a second GPU or a cloud multi-GPU node for any scaling number | install MS-MPI. Without a second GPU, no efficiency figure is published (§73.6's rule) | W3 (PAR-06) / W5 (PAR-16) |
| 3 | **Numerics changes surfaced by measurement**, each asked only if its trigger fires: D-VAL-5 (Wilcox 2006 on the device only if the oracle closes all three TG0 bands), D-VAL-6 (fourth meshes for 95-A n_y = 32 and the ring nr = 96, the LE10 precedent), D-VAL-7 (implicit tangential stencil), D-VAL-8 (KaysCrawford as the default), any PAR-13 conversion that moves a serial bit, the COMP density-based HLLC contingency, an f32 chemistry path if double chemistry is too slow | D-VAL-5 yes if the oracle closes; D-VAL-6 yes; D-VAL-7 only if VAL-10/12 show an order near 1; D-VAL-8 no; PAR bit moves stop the unit; HLLC only after the fix ladder fails; f32 chemistry no | W3-W5, when triggered |
| 4 | **Reference data and licences:** D-VAL-1 (track the US-government/public-domain Driver-Seegmiller and McCaffrey transcriptions), D-VAL-2 (ERCOFTAC T3A terms: fetched and local only), D-VAL-3 (Martin & Moyce access), D-VAL-4 (Gate 5: O1, O2 or O3), GRI-Mech 3.0 and DRM19 if their readme is ambiguous | D-VAL-1 yes; D-VAL-2 fetched and local; D-VAL-4 O1 plus the VAL-06 search; GRI/DRM19 user-supplied by path if ambiguous | W1-W3 |
| 5 | **Autonomy relocks:** ratify D-6's non-vacuity thresholds for G-PRIOR (a tightening); D-8 if AUT-06 proves G-COST's 1.5× unattainable with layers (keep it failing, or relock); D-9 whether to spend seed 3 on a no-go projection; STU-18's options if GLM grounding is still above 0 | ratify D-6; keep G-COST as written; do not spend seed 3 on a no-go | W4-W6 |
| 6 | **The gmsh exposure in the STEP route:** `tools/geom/geom_tool.py`/`geom_edit.py` `import gmsh` in-process, and PROVENANCE row 455's "run as a separate program" | keep the STEP route as an optional external path outside the product, and let TET-16 correct row 455's wording to the truth. Retiring it is a CAD-scope decision | W5 (TET-16) |
| 7 | **Runs longer than one night:** the DRM19/GRI flame sweep (CHR-13), the full Flame D comparison (CHR-16), the RAE2822 fine meshes (CMP-27), the 68-C 90-stream batch (VAL-72b) if VAL-50 prices it over a night, and AUT-14's seed-3 evaluation | approve the ones VAL-50 prices under two nights | W5-W6 |
| 8 | **Pushing and merging to `main`** (and refreshing the stale local `main` ref) | after W6, from the converged commit | after W6 |

---

## 12. Unit tables

Each table is the stream planner's table, copied with the ids renamed (Appendix A), the SPEC-LIT placeholders replaced
(§4.3), the dependencies made explicit (the planner's original wording follows in parentheses where it carried
prose), and a **lane · wave** column. The assembler's corrections are listed under each table. Where a correction
changes a gate, the change is a tightening or a duplicate removed. No band is widened.

### 12.1 COMP — compressible and transonic flow, Boussinesq (CMP-)

| id | what becomes TRUE | files | gate (quantitative; reference; tests) | depends | eff | lane · wave |
|---|---|---|---|---|---|---|
| **CMP-00** | `feat/core-2` contains `main` `8ab1c4a`, and both builds compile there. Supervisor step, not GLM. | none (fast-forward) | `git merge-base --is-ancestor 8ab1c4a feat/core-2`; both `cargo build` commands compile | — | S | W0 supervisor |
| **CMP-01** | A case can choose a Boussinesq body force `-β(T-T_ref)g` on faces beside the unchanged density-ratio form. | `src/momentum.rs`, `cuda/momentum.cu`, every `BuoyancyCoeffs{}` literal site (§1.1), SPEC-LIT new §121 "Boussinesq buoyancy" | Spiegel & Veronis 1960; Gray & Giorgini 1976, DOI 10.1016/0017-9310(76)90168-X. `momentum::tests::boussinesq_over_density_ratio_is_t_over_tref_when_beta_is_one_over_tref` (exact ratio `T/T_ref` to 4 ulp); `…boussinesq_is_exactly_zero_at_tref`; `…boussinesq_hot_cell_accelerates_up`; `simple::tests::boussinesq_hydrostatic_column_jump_is_h_gz_beta_dt` (analytic face-by-face jump to 1e-12 in f64, 1e-5 in f32); all existing momentum and simple tests unmodified | — | M | S · W1 |
| **CMP-02** | Buoyancy production of k (and ε/ω) uses `G_b = β(ν_t/Pr_t) g·∇T` under Boussinesq. The ratio form keeps its existing kernel unchanged. | `cuda/turbulence.cu`, `src/turbulence.rs`, `src/models/coupled.rs` (`ThermalCtx` gains the form) | Rodi (1987) as §17 cites. `turbulence::tests::boussinesq_gb_equals_ratio_gb_times_t_beta` (round-off); sign tests for stable and unstable stratification | CMP-01 | S | S · W1 |
| **CMP-03** | `physics.buoyancy: "boussinesq"` with `physics.fluid.beta` lowers and runs in `ofgpu-buoyant`, `ofgpu-cht` and `ofgpu-datacentre`. `ofgpu-lowmach` refuses it by name. A run with `β·max|T-T_ref| > 0.1` refuses, or warns under `-permissive`. | `src/io/case_json.rs`, `src/momentum.rs` (`from_case` reads `beta` and `buoyancy`), the three drivers, `src/bin/lowmach.rs`, `docs/schema/case-1.json` (regenerated) | `io::case_json::tests::{boussinesq_lowers_with_beta, boussinesq_without_beta_is_refused_by_name, beta_under_density_ratio_is_refused}`; `momentum::tests::boussinesq_validity_guard_fires_above_0_1`; each driver's pair-test knob; the schema sync test | CMP-01, CMP-02 | M | S · W1 |
| **CMP-04** | Gate CMP-B: de Vahl Davis passes with Boussinesq at β ΔT = 0.03 and is centro-symmetric, and the density-ratio form at the same ΔT is measurably not. | `src/bin/validate.rs` (new rows), `reference/PROVENANCE.md` (rows for Ra 1e3, 1e5 and 1e6 Nu), SPEC-LIT | de Vahl Davis (1983) *IJNMF* 3:249, DOI 10.1002/fld.1650030305, via the open secondary Qi et al. (2013), DOI 10.1186/1556-276X-8-56. Nu within **2 %** of 1.118, 2.243, 4.519 and 8.800 (Ra 1e3, 1e4 and 1e5 on 40², 60² and 80² uniform; Ra 1e6 on a 120² graded mesh); hot and cold wall Nu agree to 1e-6; centro-symmetry residual ≤ **1e-6** (Boussinesq) and ≥ **1e-3** (density ratio, control). Same rows under `--features single` (2 % band; symmetry ≤ 1e-4 in f32, stated as its own row). If Ra 1e6 exceeds 10 min, that row is deferred under rule 4. | CMP-03 | M | S · W2 |
| **CMP-05** | SPEC-LIT carries the compressible formulation (§2.2 items 1-7): every equation numbered and cited, the boundary-condition algebra, the f32 design and the gate table. This is the contract the later briefs quote. Opus writes it. | `rust/SPEC-LIT.md` (section numbers assigned by the docs/17 assembler; assigned §122-§127) | the xref/citation audit (§80) passes; every equation has a DOI/URL | — | M | M · W1 |
| **CMP-06** | Host functions give the isentropic, normal-shock, oblique-shock (θ-β-M, weak branch), area-Mach (sub- and supersonic roots) and shocked-nozzle (shock position from p_b/p0 and A_e/A*) answers. | `src/compressible/mod.rs` (skeleton), `src/compressible/exact.rs` | NACA Report 1135. `exact::tests`: A/A* = 1.6875 at M 2; normal shock at M1 = 2 gives M2 = 0.57735, p2/p1 = 4.5 and p02/p01 = 0.72087; M 2 at θ = 10° gives β = 39.31° (±0.01°) and p2/p1 = 1.7066 (±1e-4); shock-in-nozzle inverse round-trip to 1e-10 | CMP-05 | M | M · W1 |
| **CMP-07** | Host functions give Toro's exact Riemann solution with sampling at (x, t), the Ringleb field at any point, and compressible Couette with viscous heating. | `src/compressible/exact/riemann.rs`, `src/compressible/exact.rs` | Toro (2009) ch. 4 printed star states. Sod (test 1) gives p* = 0.30313 and u* = 0.92745 (5 digits); the 123 problem, the left half of the blast wave and the vacuum check are as printed. Ringleb satisfies the Euler equations pointwise (residual ≤ 1e-10 by finite difference). Couette adiabatic-wall temperature `T_w - T_top = Pr U²/(2c_p)` is exact | CMP-05 | M | M · W1 |
| **CMP-08** | Device kernels compute ψ, ρ from (p + p_op, T), Sutherland μ, κ, c and Mach, and agree with host twins. A 1 Pa gauge perturbation on 1e5 Pa survives in f32. | `src/compressible/thermo.rs`, `cuda/compressible.cu` (new), `build.rs` if the unit list is explicit | Sutherland as stated in NACA 1135 / White. `thermo::tests::{gpu_matches_host_to_4ulp, sutherland_air_at_273_15_is_1_716e_5, ideal_gas_round_trip, f32_gauge_pressure_resolves_1pa}` | CMP-05 | M | M · W2 |
| **CMP-09** | The ρ-weighted momentum equation is assembled from the mass flux with the full Newtonian stress. `rAU`, `HbyA` and `ṁ_HbyA` follow §5.1. | `src/compressible/momentum.rs`, `cuda/compressible.cu` | Moukalled §15-16. `momentum::tests::{uniform_flow_is_a_fixed_point_to_round_off, rho_one_matrix_equals_incompressible_matrix_times_rho (bitwise in f64), explicit_transpose_stress_vanishes_for_solenoidal_linear_u, couette_shear_stress_exact}` | CMP-08 | L | M · W2 |
| **CMP-10** | The all-speed pressure equation, with an upwind or limited ψ_f* convective term, is assembled and solved. It corrects ṁ, U and ρ, and it reduces to ρ × the incompressible equation as ψ → 0. | `src/compressible/pressure.rs`, `cuda/compressible.cu` | Karki & Patankar 1989; Demirdžić et al. 1993; Hafez et al. 1979. `pressure::tests::{psi_zero_reduces_to_incompressible_to_round_off, uniform_state_has_zero_residual, mass_imbalance_after_correction_below_solver_tol}` | CMP-09 | L | M · W2 |
| **CMP-11** | The energy equation in the `c_p T + K` form is solved for T, with ∂p/∂t, the K terms and an optional viscous work term. | `src/compressible/energy.rs`, `cuda/compressible.cu` | Moukalled §16; Ferziger & Perić §11.2. `energy::tests::{uniform_flow_preserves_t, slab_conduction_linear_exact, viscous_work_matches_couette_exact (CMP-07), h0_conserved_along_inviscid_duct_to_1e-6}` | CMP-07, CMP-10 | M | M · W3 |
| **CMP-12** | A `Compressible` solver runs transient PIMPLE (nOuter ≥ 1, nCorr ≥ 1) and steady SIMPLE. Energy is inside the outer loop, old-time levels are refreshed once per step, ρ/T/p bounds violations are counted and printed (never silently clipped), and it reports its mass imbalance and h0 residual. | `src/compressible/mod.rs`, `src/compressible/tests.rs` | `compressible::tests::{one_step_twice_equals_two_steps_state (old-level discipline), quiescent_box_stays_at_rest, bounds_counter_fires_and_names_the_cell}` | CMP-11 | M | M · W3 |
| **CMP-13** | Steady runs can use a pseudo-transient local time step `Δt_P = CFL·V^{1/3}/(|u|+c)`. | `src/compressible/lts.rs`, `cuda/compressible.cu` | Blazek (2015) §6.1.4. `lts::tests::{dt_field_matches_host_formula, steady_fixed_point_independent_of_cfl}` (two CFLs converge to the same field within 1e-8 in f64) | CMP-12 | S | M · W3 |
| **CMP-14** | A compressible case can be stated. In the OpenFOAM directory this is `constant/thermophysicalProperties` (subset: `perfectGas`, `hConst`, `sutherland` or `const`, `Pr`, `pOperating`). In JSONC it is `physics.compressible { gamma, W, cp, Pr, viscosity{model, As, Ts | mu}, pOperating }`, plus patch kinds `farfield` and `supersonicInlet`. Unknown keys are refused by name (§13.4). | `src/io/case_json.rs`, `src/io/case.rs` or a new `src/io/thermo_dict.rs`, `docs/schema/case-1.json` | `io::…::tests::{thermophysical_subset_round_trips, unknown_thermo_key_is_refused_by_name, jsonc_compressible_lowers}`; schema sync test | CMP-08 | M | S · W3 |
| **CMP-15** | `ofgpu-compressible <case>` runs the solver from a JSONC file or an OpenFOAM directory and writes p, U, T, rho, Mach through the existing `output` block (foam, vtu, nvdb, vdb, usda). It refuses decomposition, motion, non-SA turbulence and sources by name. `ofgpu-lowmach`'s M > 0.3 refusal now names `ofgpu-compressible`. | `src/bin/compressible.rs` (new), `Cargo.toml` `[[bin]]`, `src/bin/lowmach.rs` (message only), `src/energy.rs:893` (message only) | Pair tests `compressible::{every_knob_changes_the_run, refusals_name_the_setting}`; a smoke on a 50-cell tube in < 30 s | CMP-12, CMP-14 | L | S · W4 |
| **CMP-16** | The compressible total-pressure (with `gamma`) and `totalTemperature` inlets, the `supersonicInlet` preset and a compressible mass-flow inlet rewrite their triples each iteration. | `src/compressible/bc.rs`, `cuda/compressible.cu`, `src/field.rs` (names), `src/field_setup.rs` | NACA 1135 eqs. 43-45; Carlson 2011. `bc::tests::{total_conditions_satisfy_isentropic_relation_to_round_off, outflow_face_is_zero_gradient, mass_flow_inlet_delivers_mdot_to_1e-10}`; the incompressible `totalPressure` without `gamma` is bitwise unchanged | CMP-15 | M | S · W4 |
| **CMP-17** | A `waveTransmissive` outlet (σ, `lInf`, `fieldInf`) is a per-step Robin rewrite, and a fixed-pressure outlet switches to extrapolation on supersonic faces. | `src/compressible/bc.rs`, `cuda/compressible.cu`, `src/field.rs` | Poinsot & Lele 1992; Rudy & Strikwerda 1980; Selle et al. 2004. **Gate CMP-NR**: a Gaussian acoustic pulse (amplitude 1e-3 p, about 20 cells) leaving a 1-D tube. Measured reflection **≤ 0.05** with `waveTransmissive`, and **≥ 0.9** with fixed p (control); the long-time outlet mean relaxes to p_∞ within 1e-3. f32 the same. `bc::tests::robin_form_matches_advective_update_exactly` | BC-09, CMP-15 | M | S · W4 |
| **CMP-18** | A characteristic far-field (Riemann invariants; sub- and supersonic, in- and outflow branches) is available as `characteristicFarfield` / JSONC `farfield`. | `src/compressible/bc.rs`, `cuda/compressible.cu`, `src/field.rs` | Carlson 2011 §2; Blazek §8.2. `bc::tests::{freestream_preserved_m0_5_and_m1_5 (≤1e-10 f64, ≤1e-5 f32 relative), branch_selection_by_un_and_c, invariants_exact_on_host_twin}` | CMP-15 | M | S · W4 |
| **CMP-19** | **Gate CMP-SOD** passes: the Sod shock tube against the exact solution. | `src/bin/validate.rs`, SPEC-LIT | Sod (1978) *JCP* 27:1, DOI 10.1016/0021-9991(78)90023-2; Toro 2009 (CMP-07). 1-D tube with 200, 400 and 800 cells at t = 0.2 (non-dimensional), PISO, acoustic CFL 0.5. On 800 cells: plateau ρ, u and p within **1 %**, and shock position within **0.5 %** of the length. Observed L1(ρ) order ≥ **0.6** across the three meshes (the discontinuous-solution expectation of 0.6-1; §94 reporting). f64 and f32 rows | CMP-15 | M | G · W5 |
| **CMP-20** | **Gate CMP-AC/LM** passes: a small acoustic pulse travels at c, and at M 0.05 the solver reproduces Ghia's cavity. | `src/bin/validate.rs` | Linear acoustics, `c = √(γ R T)`: speed within **1 %** on 400 cells. Ghia, Ghia & Shin 1982 (`reference/ghia1982`, the existing key): Re 100 at lid M = 0.05, within the **existing incompressible Ghia tolerance** row for row. f64 and f32 | CMP-15 | M | G · W5 |
| **CMP-21** | **Gate CMP-NOZ** passes on a one-cell-across duct generator: subsonic isentropic, supersonic started and shocked regimes. | `src/compressible/gatemesh.rs` (duct), `src/bin/validate.rs` | NACA 1135 via CMP-06. On 200 cells: M(x) within **0.5 %** (subsonic, supersonic); shock position within **1 %** of the length; p02/p01 within **1 %**; exit p within **0.5 %**. f64 and f32 | CMP-06, CMP-13, CMP-16, CMP-17 | M | M · W5 |
| **CMP-22** | **Gate CMP-COU** passes: compressible Couette with viscous heating, exact. | `src/bin/validate.rs` | White §3-3 via CMP-07: adiabatic-wall T within **0.5 %** of `Pr U²/(2c_p)` at U = 200 m/s (constant μ); T profile L∞ within 0.5 %. f64 and f32 | CMP-15 | S | G · W5 |
| **CMP-23** | **Gate CMP-OBL** passes on a 2-D ramp generator: an oblique shock at M 2, θ 10°. | `gatemesh.rs` (ramp), `src/bin/validate.rs` | NACA 1135 θ-β-M via CMP-06: post-shock p2/p1 within **1 %** of 1.7066; shock angle within **0.5°** of 39.31° (fit of the p-jump locus); M2 within 1 % of 1.6405. f64 and f32 | CMP-13, CMP-16, CMP-18 | M | M · W5 |
| **CMP-24** | **Gate CMP-RING** passes: Ringleb transonic smooth flow shows second-order convergence. | `gatemesh.rs` (Ringleb), `src/bin/validate.rs` | Ringleb 1940 / Chiocchia AGARD-AR-211, via CMP-07. Three meshes (h, h/2, h/4), linearUpwind with no limiter: the L2 entropy error's observed order is ≥ **1.5**, reported with §94 uncertainty. f64 only for the order; an f32 row reports its error floor | CMP-13, CMP-18 | M | M · W5 |
| **CMP-25** | SA has a conservative ρ-weighted form, `∂(ρν̃)/∂t + ∇·(ṁν̃) = …` (TMR), selected by the compressible driver. With ρ ≡ 1 its matrix is bitwise the kinematic one. | `src/models/spalart_allmaras.rs`, `cuda/sa.cu`, `src/bin/compressible.rs` | TMR SA page (as cited at `spalart_allmaras.rs:19`). `spalart_allmaras::tests::{rho_one_is_bitwise_kinematic, conservative_form_conserves_rho_nutilda_in_closed_box}`; the existing SA tests unmodified | CMP-15 | L | M · W5 |
| **CMP-26** | An airfoil C-mesh generator (TFI plus Laplace smoothing, y+ first-cell control, one cell thick with `empty` sides) builds the RAE2822 from a reference coordinate file. | `src/compressible/gatemesh.rs` (or `gatemesh/cmesh.rs`), `reference/rae2822/coordinates.csv` plus a `reference/PROVENANCE.md` row with sha256 | Thompson, Warsi & Mastin 1985 (open online); coordinates from AGARD AR-138 Appendix A via the NASA NPARC archive (grc.nasa.gov/WWW/wind/valid/raetaf). `gatemesh::tests::{no_negative_volume, closure_round_off, surface_points_reproduce_coordinates_1e-9, max_non_orthogonality_below_60deg}` | CMP-05 | M | M · W2 |
| **CMP-27** | **Gate CMP-RAE** passes: RAE2822 Case 9 (SA) against experiment. | `src/bin/validate.rs`, `reference/rae2822/cp_case9.csv` plus a PROVENANCE row | Cook, McDonald & Firmin, AGARD AR-138 (1979), experiment, via NPARC; conditions M 0.729, α 2.31°, Re 6.5e6, **to confirm against the NPARC page when the key file is written**. Upper-surface shock x/c within **±0.04** of experiment; Cp RMS deviation on the lower surface ≤ **0.05**; Cl within **6 %** of the experimental value; two meshes with §94 uncertainty. Expected **beyond 10 min**: the row is DEFERRED by rule 4 with a ≤ 10 min coarse smoke (finite, converging residual, shock present) | CMP-13, CMP-18, CMP-25, CMP-26 | M | M · W6 |
| **CMP-28** | f32 and documentation close-out. SPEC-LIT §112.3 lists every compressible floor and every f32 row with both values. `cases/` gains `sodShockTube.jsonc` and a nozzle recipe. README/GUIDEBOOK counts (binaries, SPEC-LIT sections, validate rows) are corrected. `docs/07`'s M > 0.3 note points to the new solver. | `SPEC-LIT.md`, `cases/`, `README.md`, `README.en.md`, `docs/GUIDEBOOK*.md`, `rust/README.md` | the scan test for `1e-300`/`1e300` in `cuda/compressible.cu` passes; `ofgpu-validate --features single` lists every CMP row; README counts match `ls` and `Cargo.toml` | CMP-19, CMP-20, CMP-21, CMP-22, CMP-23, CMP-24, CMP-04 (CMP-19…24, CMP-04) | S | S · W6 |
| **CMP-29** | The studio knows the new solver. `ofgpu-compressible` is in `gui/shared/src/registry.ts` (flags, purpose); `run.ts` accepts its JSONC cases; the compressible BC kinds and `physics.compressible` are in the pick-lists from the re-synced schema; `boussinesq` and `beta` are editable; a Mach-contour preset and a shock-tube template exist. | `gui/shared/src/registry.ts`, `gui/server/src/tools/run.ts`, `gui/server/src/prompts/system.ts`, schema copy, the studio template module, tests | server and shared test suites stay green (baseline: 286 server and 32 shared tests at the 2026-09-11 count; use the current count); `registry.test.ts` includes the new binary; `schema.test.ts` passes on the new schema | CMP-03, CMP-14, CMP-15 | M | G · W5 |

**Assembler corrections:**

1. **CMP-00 is the W0 fast-forward, done for both lagging trees.** The gui tree is also behind `8ab1c4a`, by 45
   commits, and it hosts CMP-29. CMP-01 and CMP-05 therefore have no unit dependency.
2. **SPEC-LIT numbers:** §121 is Boussinesq (CMP-01) and §122-§127 is the compressible formulation (CMP-05).
3. **CMP-04's f32 centro-symmetry row** is an R-class row under §5.1.4, with tol32 = 1e-5 under D-F32-1, not the
   planner's "≤ 1e-4 in f32". CMP-01's 1e-5 and CMP-18's 1e-5 already equal the class rule.
4. **CMP-16/17/18 own the compressible BC kinds** (§5.2.3).
   - CMP-17 depends on BC-09 and reuses its Robin-form advective kernel with `w = u_n + c`.
   - Gate CMP-NR is the tranche's only acoustic reflection gate; BC-20's duplicate is dropped.
   - CMP-16/17/18 register their names in BC-01's catalogue.
5. **CMP-19 to 24 and CMP-27 are files in `src/bin/validate_gates/`**, plus a dispatch line each. CMP-19, 20 and 22
   run in lane G (W5), and CMP-21, 23 and 24 in lane M (W5), because the gate meshes in `gatemesh.rs` are M-owned.
6. **CMP-14's dictionary reader is a new `src/io/thermo_dict.rs`,** not an edit to `io/case.rs`. That keeps
   `case.rs` free for AMG-04b in the same wave.
7. **The critical path is 10 units** (§8), not 13. The first compressible result, CMP-19 (Sod), lands in W5.
8. **PAR-14's refusal list names `ofgpu-compressible`** (decision 14 stands).
9. **CMP-28's README edits run before VAL-82's** in S W6. The README counts are generated, not merged.
10. **The "collision" notes in COMP's §5 risk 5 are resolved by §7.4.** The AUTONOMY, GATES and INPUT streams it names
    are STUDIO's AUT, VAL and CHEMRAD here.

### 12.2 CHEMRAD — chemistry, radiation, their input, parcels input (CHR-)

| id | what becomes TRUE | files | gate (quantitative; reference; tests) | depends | eff | lane · wave |
|---|---|---|---|---|---|---|
| **CHR-00a** | SPEC-LIT §128-§133 exist: NASA-7 thermo, CHEMKIN-II reading, kinetics with falloff/reverse, equilibrium, Rosenbrock, mixture-averaged transport, the reacting low-Mach divergence. Every equation carries a DOI/URL, and the unverified DOIs in this plan are checked or replaced | `rust/SPEC-LIT.md`, `rust/PROVENANCE.md` (draft rows), `docs/17` chemistry part | doc review: `ofgpu` xref/citation audit (§80) passes; every (S128.n-S137.n) label resolves; the GRI-Mech/DRM19 licence decision is recorded with the readme text quoted | — | M | G · W1 |
| **CHR-00b** | SPEC-LIT §134-§137 exist: EDM/EDC/PaSR, P1 with Marshak BC and its slab closed form derived, fvDOM with S_N tables, WSGG, radiation coupling, case-input blocks and refusals | `rust/SPEC-LIT.md` | §80 audit passes; the P1 slab closed form and `1 - 2E3` derivation are written out; the LA-3186 and Heaslet-Warming sources are confirmed or replaced | — | M | G · W1 |
| **CHR-00c** | The chemistry answer keys exist with sha256 rows: NASA B1 extended to 8 species, the IVP test-set Robertson and HIRES reference values, the Konnov 2018 methane-air `S_L` subset (phi 0.7-1.4) | `reference/nasa-glenn2002/table_B1.csv` (ext), `reference/ivptestset2008/*.csv`, `reference/konnov2018/*.csv`, `reference/PROVENANCE.md` | the hash test (existing reference-row checker) green; each row carries URL, licence and sha | CHR-00a | S | G · W1 |
| **CHR-00d** | The radiation and Flame D answer keys exist: Jamaluddin & Smith cube table, Heaslet & Warming scattering slab, Smith 1982 and Bordbar 2014 WSGG coefficient tables, TNF Flame D centreline and x/d = 15, 30, 45 radial subsets | `reference/jamaluddin-smith1988/`, `reference/heaslet-warming1965/`, `reference/wsgg/`, `reference/tnf-flameD/`, `reference/PROVENANCE.md` | the hash test green; a digitised table is labelled DIGITISED with its reading uncertainty (§79.12 precedent) | CHR-00b | S | G · W1 |
| **CHR-01** | A species' cp, h, s and g come from NASA-7 polynomials read from a CHEMKIN THERMO block, and a mixture's cp, h and W_mix come from Y | `src/chem.rs`, `src/chem/thermo.rs`, `src/chem/tests_thermo.rs` | `cp(298.15)` of 8 species within 0.1 % of Table B1 (NASA-7 vs NASA-9 fit difference); `h(298.15) = h_f` within 10 J/mol; the two ranges meet at `T_mid` in cp, h and s to 1e-6 relative; an out-of-range T is refused by name. Tests `thermo_cp_matches_nasa_table_b1`, `thermo_ranges_meet_at_tmid`, `thermo_out_of_range_is_refused` | CHR-00a, CHR-00c | M | G · W1 |
| **CHR-02a** | A CHEMKIN-II mechanism with elementary, third-body and reversible/irreversible reactions parses into a flat, validated `Mechanism`, and every reaction is element-balanced | `src/chem/mechanism.rs`, `src/chem/ck_lexer.rs` | WD one-step and JL four-step files (written from the papers) parse; element balance exact (integer); an unbalanced or unknown-keyword reaction is refused naming the line. Tests `ck_parses_westbrook_dryer`, `ck_refuses_unbalanced`, `ck_units_keywords` (CAL/MOLE, KJOULES/MOLE, MOLECULES) | CHR-00a | M | G · W1 |
| **CHR-02b** | Falloff (LOW/TROE/SRI), DUPLICATE, REV, FORD/RORD parse; GRI-Mech 3.0 and DRM19 load if their licence allows, otherwise from a user path | `src/chem/mechanism.rs`, `cases/mech/{wd1,bfer2,jl4}.ck`, `cases/mech/README.md` | GRI-3.0 loads as 53 species, 325 reactions (counts from `readme30.dat`); DRM19 as 21/84; PLOG/CHEB refused by name. Tests `ck_gri30_counts`, `ck_drm19_counts`, `ck_refuses_plog` | CHR-00a, CHR-02a (02a, S1 (licence)) | M | G · W2 |
| **CHR-03** | Host-twin net production rates `omega_i(T,p,Y)` in f64: Arrhenius, third-body efficiencies, Lindemann and Troe falloff (Gilbert, Luther & Troe, *Ber. Bunsenges. Phys. Chem.* 87 (1983) 169, DOI 10.1002/bbpc.19830870218), reverse via `Kc` in log form, global orders | `src/chem/kinetics.rs` | `sum_i W_i omega_i = 0` to 1e-13 relative over 1000 random states (GRI or JL); element production zero; single-reaction closed forms exact to 1e-14; Troe `F_cent` against a hand-evaluated value. Tests `kin_mass_is_conserved`, `kin_elements_conserved`, `kin_troe_matches_hand_value` | CHR-01, CHR-02b | M | G · W2 |
| **CHR-04** | The equilibrium composition at (T,p) and (h,p) comes from Gibbs minimisation (element potentials, RP-1311), and kinetics agrees with it | `src/chem/equilibrium.rs` | elements to 1e-12; at the CH4-air phi=1 equilibrium (JL/GRI) every reversible reaction has `abs(q_f - q_r)/q_f < 1e-8` (detailed balance identity); T_ad of stoichiometric CH4-air at 298 K / 1 atm against the published value (RP-1311 example if one matches, otherwise a NAMED textbook value with no gate run through it) within 5 K. Tests `eq_elements_conserved`, `eq_detailed_balance_holds`, `eq_tad_methane_air` | CHR-03 | M | G · W2 |
| **CHR-05** | A host-twin Rosenbrock integrator (ROS4, RODAS3; Sandu et al. 1997) with embedded error control and pivoted LU solves stiff problems to the requested tolerance | `src/chem/rosenbrock.rs` | Robertson at t = 1e11 and HIRES at t = 321.8122 within 10*rtol of the test-set reference values at rtol 1e-8; observed order on `y' = lambda y` equals the method order ±0.1. Tests `ros_robertson_testset`, `ros_hires_testset`, `ros_observed_order` | CHR-00a, CHR-00c | M | G · W2 |
| **CHR-06** | A constant-pressure adiabatic 0-D reactor (host twin) ignites and lands on equilibrium | `src/chem/reactor.rs` | H2-air and CH4-air (JL) final state matches CHR-04 (h,p) equilibrium to 0.5 K and 1e-6 in Y; ignition delay changes < 0.5 % between rtol 1e-6 and 1e-9. Tests `reactor_lands_on_equilibrium`, `reactor_delay_converges` | CHR-04, CHR-05 | S | G · W2 |
| **CHR-07a** | Device kernel evaluates `omega` and the FD Jacobian per cell from a flat SoA mechanism, matching the host twin | `cuda/chem.cu`, `src/chem/device.rs`, `build.rs` (kernel list), `src/lib.rs` (kernels const) | 4096 random states: device vs host `omega` 1e-12 relative (f64 build), bitwise replay under capture; f32 build compiles and matches the f64 build to 1e-6 relative (inputs rounded). Tests `chem_device_omega_matches_host`, `chem_device_replays_under_capture` | CHR-03 | M | G · W2 |
| **CHR-07b** | Device Rosenbrock integrates every cell over dt, with chemistry in `double` in both builds (the chemistry scope of §118, mixed precision, recorded) | `cuda/chem.cu`, `src/chem/device.rs`, `cuda/ofgpu_device.cuh` (`ofchem`) | 4096 cells: device vs host after one dt, Y 1e-10 and T 1e-9 K (f64 build); f32 build vs f64 build 1e-5 relative; a reported (not gated) cell-steps/s for JL and GRI in both builds. Tests `chem_device_integrate_matches_host`, `chem_f32_build_matches_f64` | CHR-05, CHR-06, CHR-07a, F32-02 | L | G · W3 |
| **CHR-08** | Mixture-averaged mu, k and D_im (Chapman-Enskog, Neufeld, Wilke, Mathur) and the Lewis-number alternative, on host and device | `src/chem/transport.rs`, `cuda/chem.cu` (transport kernels) | air (N2/O2) mu within 2 % and k within 5 % of `reference/kadoya1985` over 300-1500 K; device vs host 1e-12. Tests `transport_air_viscosity_vs_kadoya`, `transport_device_matches_host` | CHR-00c, CHR-01 | M | G · W3 |
| **CHR-09** | `Species` carries per-species reaction sources through `correct_with_density`'s `su`, the split chemistry step fills them, and §86's budget still closes | `src/species.rs`, `src/chem/source.rs` | a closed adiabatic box with no flow (one-cell-wide periodic) reproduces CHR-06's Y(t) to 1e-6 (Lie split at dt = 1e-6 s, reduced to 200 steps); the §86 species-mass budget closes to round-off; `su = None` is bitwise today's path. Tests `species_box_matches_reactor`, `species_budget_closes_with_chemistry`, `species_without_chemistry_is_bitwise` | CHR-07b | M | S · W4 |
| **CHR-10a** | `Energy` can carry a per-cell `cp(T,Y)` field in its ddt and convection rows; the constant-cp default is bitwise unchanged | `src/energy.rs`, `cuda/energy.cu` | default lowmach channel fields byte-identical to before (hash pin); a uniform-cp field equal to the constant reproduces the constant path to round-off. Tests `energy_cp_field_default_bitwise`, `energy_uniform_cp_field_equals_constant` | CHR-01 | M | S · W3 |
| **CHR-10b** | Heat release `-sum h_i omega_i` (from the split step) and species-enthalpy diffusion enter T's equation via `EnergySources`, and the 0-D box's T(t) matches the reactor | `src/energy.rs`, `src/chem/source.rs`, `cuda/energy.cu` | the box reproduces CHR-06 T(t) to 0.5 K; the energy budget closes (`assembly_budget`) with the heat release metered. Tests `energy_box_matches_reactor`, `energy_budget_meters_heat_release` | CHR-09, CHR-10a | M | S · W4 |
| **CHR-11** | `GasState` uses `W_mix(Y)` for rho, and the §25.1 divergence gains the molar-change term (Day & Bell, *Combust. Theory Model.* 4 (2000) 535, DOI 10.1088/1364-7830/4/4/309; FDS TRG) | `src/energy.rs` (`GasState`), `cuda/energy.cu` | sealed isothermal box with a mole-changing, heat-neutral test reaction: `dp0/dt` matches the closed form exactly (analytic, "no tolerance excuses", §25.2 style) to 1e-12 relative; constant-W default bitwise. Tests `gas_wmix_sealed_box_p0_closed_form`, `gas_constant_w_default_bitwise` | CHR-09, CHR-10b | M | S · W4 |
| **CHR-12a** | JSONC and dictionary blocks `species`, `chemistry`, `combustion` lower to one config with §13.4 refusals; schema regenerated | `src/io/case_json.rs`, `src/io/case_chem.rs`, `docs/schema/case-1.json` | schema-drift test green; every refusal named (unknown species, inert absent, mechanism species missing in thermo, EDM with a detailed mechanism, combustion with `MixingRate::None`); round-trip of an example case. Tests `case_chem_lowers`, `case_chem_refusals`, `schema_matches_emit` | CHR-02b, CHR-08 | M | S · W4 |
| **CHR-12b** | `ofgpu-lowmach` runs a reacting case end to end (laminar finite rate), in the §2.1 order, and writes Y_i, T, rho and HRR | `src/bin/lowmach.rs`, `src/bin/common/*` | pair test: the 0-D box through the case equals the library path bitwise; a reduced 1-D JL flame runs 200 steps without NaN and conserves elements to 1e-10. Tests `lowmach_reacting_case_equals_library`, `lowmach_reacting_elements_conserved` | CHR-11, CHR-12a | M | S · W4 |
| **CHR-13** | A 1-D freely propagating methane-air flame gives S_L within the published band (REDUCED: BFER, phi = 1.0, two meshes) | `cases/flame1d_ch4_bfer.jsonc`, `src/bin/validate.rs` (section) | `S_L = U_in - dx_f/dt` (T = 1500 K isotherm) within ±10 % of Konnov 2018's recommended phi = 1.0 value; two meshes (dx 50 and 25 µm) with §94 uncertainty reported; under 10 min. DEFERRED: phi 0.7-1.4 sweep, DRM19 and GRI-3.0 (±5 %). Test/section `gate_chr13_flame_speed` | CHR-00c, CHR-12b | M | S · W5 |
| **CHR-14** | EDM and PaSR are selectable through `MixingRate` and hold their analytic limits | `src/chem/tci.rs`, `cuda/chem.cu` | EDM: well-stirred box at frozen k, eps, fuel decays as `exp(-A eps t/k)` to 1e-10; PaSR: `tau_mix -> 0` equals laminar bitwise and kappa for a linear reaction is exact to 1e-12; combination with `MixingRate::None` refused. Tests `edm_box_closed_form`, `pasr_limit_is_laminar`, `tci_refuses_no_mixing_rate` | CHR-12b | M | G · W5 |
| **CHR-15** | EDC (1996/2005 by name) integrates the fine structure with the device Rosenbrock over tau* | `src/chem/tci.rs`, `cuda/chem.cu` | for a first-order irreversible one-step test reaction the fine-structure PSR has a closed-form `Y*`, matched to 1e-8; `gamma*`, `tau*` against hand values; the `gamma*^3` cap counted. Tests `edc_psr_closed_form`, `edc_constants_by_variant` | CHR-07b, CHR-14 | M | G · W5 |
| **CHR-16** | The Sandia Flame D case exists and runs REDUCED (coarse, EDC + DRM19 or JL, k-eps with the case-stated C1 = 1.6) | `cases/sandiaFlameD.jsonc`, `src/bin/validate.rs` (section, REDUCED) | REDUCED (<10 min): 2-D axisymmetric coarse mesh, 300 iterations, finite fields, element and mass budgets close to 1e-6. DEFERRED (needs the user to run): centreline mixture fraction and T at x/d = 15, 30, 45 vs TNF within the band set in S2 (proposed: T_max within 10 %, its x/d within ±5) | BC-04, BC-05, CHR-12b, CHR-15 (15, 12b, A2, **external: wedge BC (BC stream)**) | L | G · W5 |
| **CHR-17** | `radiationModel` recognises P1 and fvDOM; absorption (constant, wsgg, wsggGrey) and scatter (none, isotropic, linearAnisotropic) models parse with refusals; `absorptionCoefficient` remains refused for viewFactor only | `src/radiation.rs`, `src/radiation/props.rs` | name-gate tests for every new name; viewFactor with absorption still refused (§50.9 unchanged); unknown names refused with the recognised set. Tests `every_model_name_is_recognised` (extended), `viewfactor_still_refuses_absorption`, `absorption_model_refusals` | CHR-00b | S | G · W2 |
| **CHR-18** | P1 solves G with Marshak walls, gives q_r and registers its source on `EnergySources` | `src/radiation/p1.rs`, `cuda/radiation.cu` | isothermal grey slab (tau = 0.1, 1, 10, eps_w = 1, then 0.5) matches the derived P1 closed form with observed order 2.0 ± 0.1 on two meshes (§94); the P1-vs-exact error is reported, not gated; source sign and linearisation identity to round-off. Tests `p1_slab_closed_form`, `p1_observed_order`, `p1_energy_source_identity` | CHR-17 | M | G · W2 |
| **CHR-19** | Level-symmetric S2-S8 quadratures and the per-direction step-scheme FV row exist, with diffuse grey, symmetry (specular) and open boundaries | `src/radiation/dom_quad.rs`, `src/radiation/dom.rs`, `cuda/radiation.cu` | moments: `sum w = 4 pi`, `sum w mu = 0`, `sum w mu^2 = 4 pi/3` to 1e-14; half-range `sum_{mu>0} w mu = pi` to the set's published value; reflection closure of each set exact. Tests `dom_quadrature_moments`, `dom_reflection_closure` | CHR-17 | M | G · W2 |
| **CHR-20** | fvDOM solves a non-scattering grey medium, sums G and q, and matches the exact slab | `src/radiation/dom.rs`, `cuda/radiation.cu` | slab on a 3-D mesh with symmetry sides: `Psi` vs `1 - 2E3(tau)` for tau = 0.1, 1, 10: S8 within 2 % on the fine mesh, error non-increasing S4 -> S8; under capture bitwise. Tests `dom_slab_exact`, `dom_error_falls_with_order` | CHR-19 | M | G · W3 |
| **CHR-21** | fvDOM with isotropic and linear-anisotropic scattering (source iteration) matches the published slab, and the isothermal cube matches the exact angular integral | `src/radiation/dom.rs`, `src/radiation/tests_cube.rs` | scattering slab: omega = 0.5, tau = 1 vs Heaslet & Warming within 2 % (S8); cube kappa L = 0.1, 1, 10, cold black walls, wall-centre flux vs the in-test adaptive integral: S8 within 3 % and S4 within 6 %; Jamaluddin & Smith table reproduced by that integral to 0.5 % (transcription check). REDUCED to 24^3. Tests `dom_scatter_heaslet_warming`, `dom_cube_exact_integral`, `cube_integral_matches_jamaluddin_smith` | CHR-20 (20, A2) | M | G · W3 |
| **CHR-22** | WSGG (Smith 1982, Bordbar 2014) gives grey-mean and per-grey-gas absorption from `H2O`/`CO2`, and the non-grey loop sums the gases | `src/radiation/wsgg.rs`, `cuda/radiation.cu` | the coefficient tables equal `reference/wsgg` bitwise (hash); homogeneous isothermal slab per-gas DOM and P1 match `sum a_j(1 - 2E3(kappa_j p L))` to the method's own slab error (P1 exactly against its per-gas closed form); WSGG without H2O/CO2 refused. Tests `wsgg_tables_match_reference`, `wsgg_slab_sum_of_grey_gases`, `wsgg_requires_h2o_co2` | CHR-18, CHR-20 (18, 20, A2) | M | G · W3 |
| **CHR-23** | `ofgpu-lowmach` and `ofgpu-buoyant` run P1/fvDOM from the case: source into energy, `q_r` written and added to the wall heat-flux report, adiabatic and fixed-flux walls balanced with `q_r` | `src/io/case_rad.rs`, `src/io/case_json.rs`, `src/bin/lowmach.rs`, `src/bin/buoyant.rs`, `docs/schema/case-1.json` | pair test case == library bitwise; heated sealed box at steady state: `sum_walls q_r A = integral kappa(4 sigma T^4 - G) dV` to exactsum round-off; `radiation.every` honoured. Tests `case_radiation_equals_library`, `radiative_power_balance_closes` | BC-01, CHR-12a, CHR-18, CHR-21, CHR-22 | M | S · W5 |
| **CHR-24** | S2S (viewFactor) is reachable from `ofgpu-buoyant` and `ofgpu-lowmach` cases (`radiation: s2s`, `s2sWall` patches), mirroring §98.7-§98.8; `refuse_enclosure` becomes the implementation it pointed at | `src/bin/buoyant.rs`, `src/bin/lowmach.rs`, `src/io/case_rad.rs` | two infinite parallel plates and concentric cylinders through the driver vs `s2s.rs`'s own closed forms (`parallel_plate_flux`, `concentric_flux`) to the §50 tolerance unchanged; pair test vs the library. Tests `buoyant_s2s_parallel_plates`, `lowmach_s2s_equals_library` | CHR-17 | M | S · W5 |
| **CHR-25** | A `parcels` block (JSONC and `constant/cloudProperties`) lowers to `ParcelControls`, the injectors and `CouplingControls`, with every library field reachable and refusals by name | `src/io/case_parcels.rs`, `src/io/case_json.rs`, `docs/schema/case-1.json` | schema round trip; every `ParcelControls`/`CouplingControls` field set by a test case; `film`/`splash` still refused with their §78 reasons. Tests `case_parcels_lowers_every_field`, `case_parcels_refusals` | CHR-00b | M | S · W3 |
| **CHR-26** | `ofgpu-lowmach` drives parcels from the case: walk and sort, two-way momentum and energy into the sources, and vapour into the named species | `src/bin/lowmach.rs` | a §76 evaporation gate case through the driver equals the library path bitwise (REDUCED); the §68-B "no parcels, no change" pin holds through the driver. Tests `lowmach_parcels_equal_library`, `lowmach_no_parcels_no_change` | CHR-12b, CHR-25 | M | S · W5 |
| **CHR-27** | `ofgpu-validate` carries the CHEMRAD sections (thermo, equilibrium, ODE, slab, cube, WSGG, EDM, flame REDUCED, Flame D REDUCED) in the §69 gate registry, each under its 10-minute budget, in both builds | `src/bin/validate.rs`, `src/bin/validate_key/*` | every section lists its rows and passes in f64; the f32 build reports pass/miss per §112.4 convention; the total CHEMRAD runtime is printed. Test: `ofgpu-validate --section chr` | CHR-13, CHR-14, CHR-16, CHR-21, CHR-22 | S | S · W6 |
| **CHR-28** | The studio's registry knows the species, chemistry, combustion, radiation and parcels blocks from the regenerated schema, and the server suites stay green | `gui/server/src/registry/schema.ts`, `gui/server/src/registry/schema.test.ts`, `gui/server/src/prompts/system.ts` | server and shared tests all green (286 + 32 baseline); a mock-LLM case with a `radiation` block validates. Tests `schema.test.ts`, `routes.test.ts` | CHR-12a, CHR-23, CHR-25 | S | G · W6 |
| **CHR-29** | **[17, added by the assembler]** The species boundary conditions `semiPermeableBaffleMassFraction` and `specieTransfer` exist as §4 triple rewrites on species fields, are registered in BC-01's catalogue, and are refused by name on a case without species | `src/species.rs` (BC hook), `src/field.rs` (names), `src/field/catalogue.rs`, SPEC-LIT §131 subsection | A 1-D two-species slab with a semi-permeable baffle: the species mass flux through the baffle equals the closed form `k_m (Y_P - Y_N)` to 1e-12 relative (f64) and at the class rule in f32; the impermeable species' flux is exactly 0 (bitwise); the §86 species budget closes to round-off. Tests `species_semipermeable_baffle_flux`, `species_bc_refused_without_species`. Reference: Fick's law with a film mass-transfer coefficient (Bird, Stewart & Lightfoot, *Transport Phenomena*, 2nd ed., ch. 22), stated in SPEC-LIT | CHR-09, BC-01 | M | S · W4 |

**Assembler corrections:**

1. **Renamed ids:** `CHR-S1/S2/A1/A2` became `CHR-00a/00b/00c/00d`, and short dependency forms (`S1`, `02b`) are now
   full ids.
2. **SPEC-LIT:** §120-§125 becomes **§128-§133** and §126-§129 becomes **§134-§137**.
3. **The precision policy (CHR-07b) is a scope of §118,** PREC's single mixed-precision section, and it is not the
   "first" mixed-precision module.
   - `typedef double ofchem` lives in `cuda/chem.cu`. CHEMRAD does not edit `cuda/ofgpu_device.cuh` (§5.1.6).
   - CHR-07b depends on F32-02, which creates §118.
4. **CHR-16 depends on BC-04/BC-05,** the true wedge, which lands in W2. The 3-D sector fallback is the user's
   decision, and only if BC-04 fails.
5. **CHR-18/20/21's wall conditions** (Marshak, DOM diffuse grey, symmetry and open) act on G and I inside the
   radiation module. CHR-23 registers `MarshakRadiation`, `MarshakRadiationFixedTemperature`,
   `greyDiffusiveRadiation` and `wideBandDiffusiveRadiation` in BC-01's catalogue (dependency added).
6. **CHR-29 is new:** the species BCs, which PREC assigned to this stream and which had no unit.
7. **CHR-26 absorbs VAL-72a.** Its gate gains VAL-72a's two rows: in the coupled driver, momentum given equals
   momentum gained to 1e-12 relative, and Gate 68-B "no parcels, no change" holds bitwise through the driver.
8. **The critical path is 15 units** (§8). CHR-13 is not on it.
9. **CHR-13, CHR-16 and CHR-27 put their gates in `validate_gates/`.** CHR-27 then only registers the sections in
   §69's registry, and it runs in S W6, when S owns `validate.rs`.
10. **Lanes:**
    - the chemistry core (CHR-00a…08) and the radiation core (CHR-00b, 00d, 17…22) are G W1-W3, new files only;
    - the integration (CHR-09…12b, 23…26, 29) is S W3-W5;
    - the TCI and Flame D (CHR-14, 15, 16) are G W5;
    - the gui registry (CHR-28) is G W6.
11. **The schema** is regenerated in lane S only. CHR-28 re-syncs the gui copy.

### 12.3 PAR — parallel execution (PAR-, AMG-, AMX-)

| id | what becomes TRUE | files | gate (quantitative; reference; tests) | depends | eff | lane · wave |
|---|---|---|---|---|---|---|
| **PAR-01** | N ranks can talk in-process: `Comm` trait + `ThreadComm`, deterministic delivery keyed by (src,dst,tag) | new `src/comm.rs`, `src/comm/thread.rs`, `src/comm/tests.rs`; `lib.rs`; SPEC-LIT §138.1; PROVENANCE row | host-only tests, N = 1..8: `allgather_returns_rank_order`, `allgatherv_ragged_lengths`, `exchange_asymmetric_neighbour_lists_never_deadlock` (one rank with 0 peers, one with 5), `exchange_is_byte_identity` (10^5 random bytes), `bcast_from_every_root`; 1000 repeated runs produce identical bytes. Ref: MPI 4.1 standard §3-§6 semantics (https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report.pdf) | — | S | G · W1 |
| **PAR-02** | a halo can be exchanged across ranks: `HaloExchange` gains an SPMD form (one local part, pack -> D2H pinned -> `Comm::exchange` -> H2D into the contiguous halo slice) | `src/halo.rs` (+`src/halo/spmd.rs`), `cuda/halo.cu` (tensor pack kernel `haloPackTensor`), tests | `an_spmd_halo_is_bit_for_bit_the_in_process_halo`: ThreadComm P = 2,3,4 × {hilbert, linear, round robin} × {scalar, vector, tensor, label}: every ghost byte equals the in-process `HaloExchange` result and the owner; a buffer shorter than `n_local` refused by name (both numbers). Both precisions | PAR-01 | M | G · W2 |
| **PAR-03** | a reduction is global across ranks with the same bits as in-process: `ExactReduction` over `Comm` (anchor allgather, then limb allgather, same one-block combine kernel); `gathered_*` likewise | `src/exactsum.rs` (+`src/exactsum/spmd.rs`), tests | `spmd_exact_reduction_is_the_in_process_one`: sum / sum_mag / dot / norm_factor / max_mag at P = 1..4 under every rotation of rank labels: bitwise equal to in-process `ExactReduction` and to the P = 1 answer; `gathered_*` bitwise equal to in-process gathered. Both precisions | PAR-01 | M | G · W2 |
| **PAR-04a** | one rank solves its part of a PCG: `DistSystem::split_one(rank)` + SPMD `dist_pcg` (existing recurrences; transports from PAR-02/3) | `src/distsolve.rs`, `src/distsolve/spmd.rs`, tests | `an_spmd_pcg_is_the_in_process_pcg`: ThreadComm P = 1..4, 3 partitioners, plain + cyclic box, `Diagonal`/`None`, Exact: field bits **and iteration count** equal to in-process `dist_pcg` (hence to serial, §73.8); debug flag-vote never disagrees | PAR-02, PAR-03 | M | G · W2 |
| **PAR-04b** | SPMD PBiCGStab, and DIC/DILU block-local on SPMD | same | `an_spmd_pbicgstab_is_the_in_process_pbicgstab` as above; DIC/DILU: bitwise equal to in-process block-local (`partition_invariant` still `false`); exchange and reduction counts equal §73.8's `k+3 / 2k+3` and `2k+5 / 4k+4` | PAR-04a | M | G · W2 |
| **PAR-05** | ranks are placed on devices: `rank mod device_count`, `OFGPU_DEVICES` override; per-rank `Gpu` (own stream) safe with event tracking off | `src/device.rs`, `src/comm/placement.rs`, tests | `placement_maps_ranks_to_devices` (pure function, faked counts 1/2/4/8); `four_ranks_on_one_device_run_concurrently`: ThreadComm P = 4 on device 0, the PAR-04a gate still bitwise; `#[ignore = "needs 2 GPUs"] peer_placement_two_devices` written and **not runnable here** | PAR-04a | S | G · W3 |
| **PAR-06** | MPI exists as a transport: `mpi` feature, own dynamic-loaded MPICH-ABI binding (`msmpi.dll` / `libmpi.so.12`), `MpiComm: Comm`; Open MPI refused by name | new `src/comm/mpi.rs`, `Cargo.toml` (`mpi = []`), SPEC-LIT §138.2, NOTICE (MS-MPI, MIT), PROVENANCE | feature off: builds, `MpiComm::open()` refuses by name ("MS-MPI runtime not found; install msmpisetup.exe or use -comm threads"); feature on without the DLL: same refusal, all tests green; `cargo build --features "mpi single"` compiles. With MS-MPI installed (**user action**): `mpiexec -n 4 ofgpu-par -selftest` runs PAR-01's test list over MPI and prints PASS (deferred until install). Refs: MPI 4.1 standard; MS-MPI `mpi.h` constants (github.com/microsoft/Microsoft-MPI, MIT) | PAR-01 | M | G · W3 |
| **PAR-07** | there is a parallel driver: `ofgpu-par <case> -np N -comm threads|mpi` (auto-detects `PMI_RANK`/`PMI_SIZE` under mpiexec), running the §71-§73 gates in SPMD | new `src/bin/par.rs`, `Cargo.toml` `[[bin]]`, tests | `ofgpu-par cases/plume -np 2,3,4 -comm threads`: halo, reduction and Krylov gates PASS with `worst == 0` and the same iteration count, under 3 partitioners and every label rotation, < 10 min; each rank hashes the partition map and the hashes allgather equal | PAR-04b, PAR-05, PAR-06 (PAR-04b, PAR-05 (PAR-06 for `-comm mpi`)) | M | G · W3 |
| **PAR-08** | the Hilbert cut no longer blows up on 2-D and thin meshes: all axes indexed against one common physical scale; a degenerate axis uses the 2-D curve | `src/decompose.rs` (`lattice`, `partition`), SPEC-LIT §71.2/§73.6 re-measured | `channel` P = 8 cut faces <= 1,400 (linear's measured cut); `gb_800000` P = 8 <= 1.5 × 6,800 = 10,200; `plume` P = 4 / 8 halo fraction <= 0.068 / 0.168 (§71.2's published figures may only improve); every bitwise gate in §71-§73 still passes; `the_hilbert_curve_visits_every_point_and_never_jumps` extended to the 2-D curve | — | S | G · W1 |
| **PAR-09** | every cell field in the laminar SIMPLE + scalar + energy path has room for the halo: `GpuMesh::n_halo`; field constructors allocate `n_cells + n_halo`; kernels launch over owned cells only | `src/mesh.rs`, `src/field.rs`, `src/fv.rs`, `src/simple.rs`, `src/momentum.rs`, `src/energy.rs`, `src/scalar_transport.rs`, test | `every_simple_path_buffer_has_room_for_the_halo`: allocation-site audit test (list of allowed `zeros(n_cells)` sites, any new one fails); on a 2-part mesh `compute-sanitizer --tool memcheck` over 3 SIMPLE iterations reports 0 errors (< 2 min); serial golden hashes of `ofgpu-validate`'s laminar subset unchanged (n_halo = 0) | PAR-04a | M | S · W4 |
| **PAR-10a** | a field's processor-patch boundary values and its gradient's halo come from the neighbour rank: `RankCtx` (comm + halo + reducer) and `exchange_field` / `exchange_gradient` (vector, tensor) | `src/distflow.rs` (new), `src/field.rs`, tests | `a_halo_gradient_equals_the_whole_mesh_gradient`: grad of a smooth field on P = 2..4 parts vs whole mesh, `<= 4 ulp` per component (exact equality is Phase 2's job); `correct_boundary` on a processor patch gives the owner's value bitwise | PAR-02, PAR-09 | M | S · W4 |
| **PAR-10b** | the p and momentum solves of a part go through SPMD Krylov: `SpmdPressureBackend: PressureBackend` and momentum routing; all residual norms are global | `src/distflow.rs`, `src/simple.rs` (routing only), `src/momentum.rs`, tests | a part-assembled SPD Poisson solved by `SpmdPressureBackend` at P = 2,4 reaches the serial answer `<= 1e-8` rel (f64) / `<= 1e-4` (f32); reported `initial_residual` identical on every rank | PAR-04b, PAR-10a | M | S · W4 |
| **PAR-10c** | a laminar SIMPLE/PISO case runs decomposed and converges to the serial answer: the exchange points and the global reductions in `Simple` | `src/simple.rs`, `src/distflow.rs`, `ofgpu-par -solve simple`, SPEC-LIT §139 | lid-driven cavity Re = 100, 64^2 (Ghia, Ghia & Shin 1982, DOI 10.1016/0021-9991(82)90058-4), P = 1,2,4 ThreadComm: converged U,p `L_inf` rel diff vs serial `<= 1e-6` (f64) / `<= 1e-3` (f32), with thresholds **fixed before measuring** and tightened in SPEC-LIT if the measurement is far below; centreline u vs Ghia's table within the serial run's own error ± 1e-6; SIMPLE iteration count to `residualControl` within ±2 % of serial; < 10 min | PAR-10b | L | S · W5 |
| **PAR-11** | energy + Boussinesq + passive scalar run decomposed | `src/energy.rs`, `src/scalar_transport.rs`, `src/distflow.rs` | de Vahl Davis (1983, DOI 10.1002/fld.1650030305) Ra = 1e4 cavity, 40^2, P = 4: mean hot-wall Nu within 1 % of 2.243 **and** within 1e-6 rel of the serial run's Nu | CMP-03, PAR-10c (PAR-10c (and FLOW's Boussinesq unit, if it lands first)) | M | S · W5 |
| **PAR-12** | k-epsilon and k-omega SST run decomposed; wall distance computed on the whole mesh and split | `src/turbulence.rs`, `src/walldistance.rs` (call site only), `src/distflow.rs` | `cases/channelPeriodicWF.jsonc`, P = 4: converged U, k, eps/omega within 1e-6 rel of serial; `y` split field bitwise equal to the whole-mesh `y` restricted | PAR-10c | M | S · W5 |
| **PAR-13a** (T2) | the diagonal and laplacian assembly, and both gradients, are partition-invariant: `lduNegSumDiag`+`lduAddBoundaryContributions` merged in global face order; `fvLapNonOrth`, `fvGradScalar`, `fvGradVector` on the merged row with whole-mesh metrics on processor faces | `cuda/ldu.cu`, `cuda/fv.cu`, `src/ldu_ops.rs`, `src/fv.rs`, `src/decompose.rs` (metrics on processor faces) | **serial**: `ofgpu-validate` laminar-subset golden hashes unchanged in f64 **and** f32 (any move stops the unit and goes to the user); **parts**: part-assembled laplacian matrix bitwise equal to `split_matrix` of the whole-mesh one at P = 2,3,4; part gradients bitwise equal to whole-mesh gradients | PAR-10a | L | S · W6 |
| **PAR-13b** (T2) | the convection and flux kernels are partition-invariant: `fvDivBoundedDiag`, `fvDivCorrection`, `fvDivSurface`, `fvReconstruct`, `smpFaceFluxSum` | `cuda/fv.cu`, `cuda/simple.cu`, `src/fv.rs` | as PAR-13a for these kernels | PAR-13a | L | S · W6 |
| **PAR-14** (T2) | a decomposed laminar SIMPLE is **bit for bit** the serial run; every path still unconverted is refused by name in `ofgpu-par` | `src/distflow.rs`, `src/bin/par.rs`, SPEC-LIT §139 refusal list | cavity and `plume` at P = 2,3,4: every cell of U, p, phi bitwise equal to the P = 1 SPMD run after 50 iterations, **same iteration count**, both precisions; refused by name with §13.4.1 pair tests: VOF, parcels, LTS, AMR, ALE, CHT multi-region, FFT backend, CUDA-graph capture | PAR-11, PAR-13b | M | S · W6 |
| **PAR-15** | decomposed I/O: per-rank field write (`processorN/`), reconstruct to the whole mesh (VTU + case fields), restart from decomposed fields | `src/io/` (new `decomposed.rs`), `src/bin/par.rs` | reconstructed fields bitwise equal to the gathered in-memory fields; write -> restart -> one iteration is bitwise equal to an uninterrupted run (P = 2,4); a `processorN` set from a different P refused by name | PAR-10c | M | S · W5 |
| **PAR-16** | scaling is measured, not modelled: `ofgpu-par -bench strong|weak` measures the per-iteration time, the halo exchange latency `L_ex` and the allgather latency `L_ag` for ThreadComm and MS-MPI; weak scaling on generated boxes at fixed cells/rank; JSON + table | `src/bin/par.rs`, `tools/par/scaling.py`, SPEC-LIT §140 (replaces §73.6's free `L` with measured values) | on this machine: `L_ex`, `L_ag` reported as median and 95 % CI over 5 repeats (Hoefler & Belli, SC15, DOI 10.1145/2807591.2807644); **efficiency is refused** whenever ranks > devices (prints "time-shared, not scaling"); the harness runs P = 1..4 in < 10 min. Real strong/weak scaling (Amdahl 1967; Gustafson 1988, DOI 10.1145/42411.42415) **needs multi-GPU hardware** and is a recorded deferred run | PAR-06, PAR-07, PAR-10c (PAR-07, PAR-10c (PAR-06 for MPI)) | M | S · W5 |
| **AMG-01** | an LDU matrix can be coarsened: deterministic pairwise aggregation (two passes, strength `-(a_ij+a_ji)/2`, global-id tie-break) producing the cell->aggregate map, coarse LDU addressing and the fine-face->coarse-face map (interior faces flagged; coupled faces folded) | new `src/amg.rs`, `src/amg/aggregate.rs`, tests; SPEC-LIT §141.1; PROVENANCE | host-only: every cell in exactly one aggregate; mean aggregate size in [3, 5] on 32^3 Poisson (Notay 2010 reports about 4 for double pairwise); coarse graph connected if fine is; 100 runs identical; cyclic box folds its couples. Refs: Notay, ETNA 37 (2010) 123 (open: etna.ricam.oeaw.ac.at/vol.37.2010/pp123-146.dir/pp123-146.pdf); Napov & Notay, SISC 34(2) (2012) A1079, DOI 10.1137/100818509 | — | M | G · W1 |
| **AMG-02** | the Galerkin coarse operator and the transfers run on the device with no atomics: gather-form restrict (segmented sum over aggregate-sorted offsets), prolong (gather), coarse diag/upper/lower via precomputed fine-coefficient lists; host twin | new `cuda/amg.cu`, `src/amg/galerkin.rs`, `build.rs` (kernel list), tests | device coarse matrix bitwise equal to its host twin (same order); `A_c` vs dense `P^T A P` on 6^3 meshes `<= 1e-14` rel (f64) / `<= 1e-5` (f32); `P^T` of the constant vector equals aggregate sizes exactly; run twice -> bitwise identical. Ref: Stüben, JCAM 128 (2001) 281, DOI 10.1016/S0377-0427(00)00516-1 | AMG-01 | M | G · W2 |
| **AMG-03a** | a hierarchy with smoothers exists: levels until `n <= nCellsInCoarsestLevel`; l1-Jacobi, weighted Jacobi, Chebyshev (Lanczos `lambda_max`, 10 steps) | `src/amg/hierarchy.rs`, `src/amg/smooth.rs`, `cuda/amg.cu` | l1-Jacobi error-norm reduction monotone on SPD test matrices (Baker et al., SISC 33(5) (2011) 2864, DOI 10.1137/100798806: l1-Jacobi is convergent for SPD A); Chebyshev degree 2 reduces the high-frequency half of the spectrum by at least the predicted Chebyshev bound (Adams et al., JCP 188 (2003) 593, DOI 10.1016/S0021-9991(03)00194-3) on a 1-D Poisson with known eigenvectors; level count = ceil(log4(n/64)) ± 1 on 64^3 | AMG-02 | M | G · W2 |
| **AMG-03b** | the coarsest level is solved exactly: host dense Cholesky, device one-block triangular solves; singular (pure Neumann) coarse problems projected and pinned | `src/amg/coarse.rs`, `cuda/amg.cu` | coarse solve residual `<= 1e-12` rel (f64) / `<= 1e-5` (f32) on SPD; on a pure-Neumann Poisson the coarse correction has zero mean to 1e-14 and the V-cycle stays bounded over 100 cycles; a non-SPD coarse matrix refused by name (falls back to 50 Jacobi sweeps only under `-permissive`, announced) | AMG-03a | M | G · W2 |
| **AMG-04a** | GAMG preconditions PCG and PBiCGStab and runs standalone: V(ν1,ν2) cycle; `Preconditioner::Gamg`; `LinearSolverKind::Gamg` honoured by `solver::solve` (refusal at `solver.rs:2260` removed) | `src/amg/cycle.rs`, `src/solver.rs`, `src/precon.rs`, tests | 7-point Poisson 32^3 / 64^3 / 128^3, Dirichlet, tol 1e-8: GAMG-PCG iterations `<= 1/3` of DIC-PCG at 128^3 and answer `<= 1e-8` rel of PBiCGStab; standalone GAMG reaches the same tolerance; iteration growth 32^3 -> 128^3 **measured and published** (h-dependence of V-cycle aggregation is expected; G4c gates it); convection-diffusion Pe = 10 asymmetric matrix: GAMG-PBiCGStab converges with iterations `<= 1/2` of DILU-PBiCGStab; < 10 min | AMG-03b | L | G · W2 |
| **AMG-04b** | a case can ask for GAMG and its settings and gets them: `solver GAMG`, `preconditioner GAMG`, `smoother`, `nPreSweeps`, `nPostSweeps`, `nCellsInCoarsestLevel`, `maxLevels`, `cacheAgglomeration`; unsupported values (`GaussSeidel`, `DICGaussSeidel`, `mergeLevels`, `agglomerator faceAreaPair`) refused by name with the alternative; JSONC keys + schema | `src/io/case.rs`, `src/io/case_json.rs`, `src/io/dict.rs`, `docs/schema` (case schema), `src/pressure/mod.rs` (`refuse_gamg` removed) | one §13.4.1 pair test per key (the setting changes the iteration count or the hierarchy); refusal tests per unsupported value; `the_solver_a_case_asks_for_is_the_solver_it_gets` extended to GAMG | AMG-04a | M | S · W3 |
| **AMG-05** | GAMG is a pressure backend and the selector measures it: `GamgBackend` in `default_candidates`, aggregation cached, Galerkin re-gathered per solve; `ofgpu-buoyant -backend gamg` | `src/pressure/gamg.rs` (new), `src/pressure/mod.rs`, `src/bin/buoyant.rs` | selector agreement gate (1e-8 vs PBiCGStab) passes on `plume` and `gb_800000`; set-up / solve time table printed; `plume` 20 steps with GAMG vs PBiCGStab: fields within 1e-8 rel; default backend **unchanged** (`auto` picks by measurement only) | AMG-04b | M | S · W3 |
| **AMG-06a** | GAMG holds in f32: builds and converges under `--features single` | `src/amg/*`, tests | f32: GAMG-PCG reaches the §112 f32 residual floor on 64^3 Poisson with iterations `<= 1.25 ×` the f64 count; no `1e-300`-class floor in `cuda/amg.cu` (the §112 scan test passes) | AMG-04a, F32-07 | S | G · W4 |
| **AMG-04c** (T2) | K-cycle with flexible CG makes aggregation AMG h-independent | `src/amg/cycle.rs`, `src/solver.rs` (FCG) | Poisson 32^3 -> 128^3: K-cycle FCG iteration growth `<= 1.3 ×`. Refs: Notay & Vassilevski, NLAA 15 (2008) 473, DOI 10.1002/nla.542; Notay, SISC 22(4) (2000) 1444, DOI 10.1137/S1064827599362314 | AMG-04a | M | G · W4 |
| **AMG-06b** (T2) | an f64 solve can use an f32 GAMG preconditioner (`gamg.precision single`, default off) | `src/amg/*`, `src/io/case.rs` | f64 PCG + f32 GAMG reaches 1e-10 on 128^3 with iterations `<= 1.15 ×` the all-f64 count and the hierarchy's memory `<= 0.55 ×`; §13.4.1 pair test. Ref: Kronbichler & Ljungkvist, ACM TOPC 6(1) (2019), DOI 10.1145/3322813 | AMG-04b, AMG-06a | M | G · W4 |
| **AMG-07a** (T2) | coarse levels have a halo: block-local aggregation (no aggregate crosses a cut), coarse processor faces, coarse halo plans | `src/amg/dist.rs`, `src/halo.rs` (plan from a level) | coarse level of a P-part hierarchy, gathered, equals the Galerkin product of the gathered fine level bitwise in the same order; no aggregate spans parts | AMG-04a, PAR-04b | M | G · W4 |
| **AMG-07b** (T2) | distributed GAMG-PCG over SPMD; coarsest level allgatherv'd and solved redundantly; `dist_solve` stops refusing GAMG (`distsolve.rs:1349`) | `src/amg/dist.rs`, `src/distsolve.rs` | P = 2,4 ThreadComm: answer `<= 1e-8` rel of serial GAMG, iterations `<= 1.3 ×` serial; `partition_invariant(Gamg) == false` and the answer demonstrably moves under a cut (the §73 honesty pattern) | AMG-07a | L | G · W4 |
| **AMX-01** | `--features amgx` builds on any machine and runs where the DLL is: dynamic load with `libloading` (no import library); DLL from `AMGX_DLL` / `AMGX_DIR` / `PATH`; `OFGPU_REQUIRE_AMGX=1` makes absence a failure; `tools/amgx/build.cmd` recipe (AMGX 2.5.0, CUDA 13.3, sm_120) | `src/pressure/amgx.rs`, `build.rs` (drop `dylib=amgxsh`), `tools/amgx/build.cmd`, NOTICE, SPEC-LIT §141.x | `cargo build --release --features amgx` and `--features "amgx single"` compile with no AMGX present; with the existing 2.5.0 DLL and `OFGPU_REQUIRE_AMGX=1`: the existing amgx tests pass; AMGX vs PBiCGStab on `plume` `<= 1e-8` rel (dDDI) and `<= 1e-4` (dFFI, f32) | — | M | G · W1 |
| **AMX-02** | AMGX runs inside a real time loop and is compared, not just listed: `structure_reuse_levels` to amortise set-up; three-way table PBiCGStab/PCG vs native GAMG vs AMGX | `src/pressure/amgx.rs`, `src/bin/buoyant.rs`, SPEC-LIT table. Ref: Naumov et al., SISC 37(5) (2015) S602, DOI 10.1137/140980260 | `ofgpu-buoyant cases/plume -backend amgx` 20 steps: fields within 1e-8 rel of the PBiCGStab run; set-up time with reuse `<=` half of without (measured); table of iterations, set-up ms and solve ms on `plume` and `gb_800000` published | AMG-05, AMX-01 | M | S · W3 |
| **PAR-17** | the studio can launch a parallel run: registry entry for `ofgpu-par` (`pending: true` first, live when PAR-07 lands), run option "ranks / transport" (`-np`, `-comm`), decomposition report and per-rank residual shown | `gui/shared` registry, `gui/server/src/runs/dispatch.ts`, studio run panel | `registry/sync.test.ts` green (pending entry must not have its Cargo target, then live entry must match `[[bin]]` and its `usage()` flags); a dispatch test builds `ofgpu-par case -np 2 -comm threads`; server 286 + shared 32 baseline stays green | PAR-07 (PAR-07 (live step)) | S | G · W4 |

**Assembler corrections:**

1. **Ids:** `PAR-n` became `PAR-0n`, `PAR-Gn` became `AMG-0n`, `PAR-Xn` became `AMX-0n`, and `PAR-U1` became `PAR-17`.
   **SPEC-LIT:** §P1-§P4 became §138-§141 (PREC also used §P1-§P3).
2. **Lanes:**
   - the transport stack (PAR-01…08), the GAMG core (AMG-01…04a, 06a, 04c, 06b, 07a/b) and AMX-01 are in G;
   - the distributed flow (PAR-09…16), GAMG as a backend (AMG-04b, 05) and AMX-02 are in S.
3. **Added dependencies:**
   - PAR-11 needs CMP-03 (Boussinesq);
   - PAR-07 and PAR-16 need PAR-06;
   - F32-07 needs PAR-04b, so IR wraps the SPMD Krylov too;
   - AMG-06a needs F32-07.
4. **AMG-06a's gate is tightened.** Under IR the f32 build reaches AMG-04a's f64 tolerances. The raw f32 hierarchy's
   residual floor is reported beside that. This replaces "reaches the f32 floor".
5. **f32 numbers** (§5.1.4):
   - PAR-10b's f32 row is A class at its f64 number, because IR is in place from W3.
   - PAR-10c's L∞-versus-serial f32 row is R class (tol32).
   - AMX-01's dFFI row is R class. A miss is reported as an AMGX f32 finding.
6. **PAR-09 keeps the public field-constructor signatures.** `compressible/`, `chem/` and `radiation/` are outside
   its audit, because their drivers refuse decomposition.
7. **PAR-13a/b run in S W6.** The serial f32 golden hashes they must not move are re-pinned at the S3 sync, after
   F32-07/08/10/11/12.
8. **PAR-14's refusal list** adds `ofgpu-compressible`, reacting chemistry and P1/fvDOM (§10.1 A16).
9. **PAR-08 runs in G W1.** `decompose.rs` returns to S for PAR-13a in W6.

### 12.4 PREC — single precision, the wedge, the BC catalogue (F32-, BC-)

| id | what becomes TRUE | files | gate (quantitative; reference; tests) | depends | eff | lane · wave |
|---|---|---|---|---|---|---|
| **F32-01** | `ofgpu-validate -sections a-b` runs only those sections and reports every other section as a `skip` row (never a pass). `tools/f64_identity.py` hashes the SASS of each cubin (`cuobjdump -sass`) and diffs two validate JSONs row by row. | `src/bin/validate.rs`, `src/bin/validate_json/mod.rs`, `tools/f64_identity.py` | `-sections 7` yields exactly 6 check rows and 0 failures, with the remaining 931 accounted for as skips; the §69 registry audit passes; a full run without the flag is byte-identical to before (`validate_section_filter_counts`, `validate_no_flag_is_identity`) | — | S | S · W1 |
| **F32-02** | All device reductions accumulate in `ofacc` (f64 in both builds), with partials `DevBuf<Acc>`. This covers sum, sum_mag, dot, dot2, max, the norm factor and the fused stage-2 kernels (`Divide`/`Beta`/`Ratio`/`Converged`). | `cuda/ofgpu_device.cuh`, `cuda/solver.cu`, `src/solver.rs`, `src/lib.rs` (`Acc`), §118.1 | Failing first, under single: `solver::tests::f32_sum_of_2e20_terms_is_f64_accurate` (Σ of 2^20 terms `1 + k·2^-20`) has relative error ≤ 4·ε64 (Higham γ_n bound; today ~1e-5). f64 SASS of `solver.cu` is unchanged. | F32-01 | M | S · W1 |
| **F32-03** | Host geometry is computed in f64 (points parsed as f64, `geometry::compute` generic) and cast once. Under single, `HostGeom64` and `MeshReport` come from f64. | `src/types.rs` (`DVec3`), `src/io/polymesh.rs`, `src/mesh/geometry.rs`, `src/mesh.rs`, §118.2 | Under single, the closure and mesh rows of sections 1-5 and 12 pass at f64 tolerance (`max_closure_error ≤ 1e-10`); `geometry::tests::f64_geometry_is_bitwise_in_f64_build` hashes every HostMesh array on the validate meshes and finds them identical to before | F32-01 | M | S · W1 |
| **F32-04** | The conduction and CHT guards (`ANISOTROPY_RESIDUAL_LIMIT`, interface conformity 1e-7, area 1e-9) are evaluated on `HostGeom64`, with **limits unchanged**. | `src/cht.rs`, `src/cht/*.rs`, §118.3 | Failing first: `cht::tests::the_axis_aligned_slab_is_accepted_at_f32` (refused today with 1.19e-7 > 1e-10). Validate section 8 under single reports 10/10. The 56 conduction-refusal tests no longer refuse (counted). | F32-03 | M | S · W1 |
| **F32-05** | The remaining sub-round-off guards run in f64: S2S closure and reciprocity (view factors on host in f64), the pressure probe `AGREEMENT_TOL`, psychrometric wet-bulb Newton, `cht/ambient NEWTON_RTOL`, and the fan/porous host algebra. The unit lists all 32 constants of 1e-6 or smaller in §118.3 with a verdict for each. | `src/s2s*.rs`, `src/pressure/mod.rs`, `src/psychro*.rs`, `src/cht/ambient.rs`, `src/fan*.rs`, §118.3 | Validate sections 30-31 under single have no refusals; reciprocity `A_i F_ij = A_j F_ji` to 1e-12 and ΣF = 1 to 1e-12 (§49 gates, unchanged); the 33 "other refusal" tests un-ignored | F32-04 | M | S · W1 |
| **F32-06** | PCG and PBiCGStab, under single, are wrapped in mixed-precision IR: an f64 `x_hi`, an FP64 residual `amul64`, the f64 §8.4 normalisation, an inner relative 1e-4 and the outer stop at the case tolerance. It stays device-resident: refinement steps sit inside the §113 flag cadence. | `cuda/solver.cu` (`solAmul64`, `solResidual64`), `src/solver.rs`, `src/precision.rs` (new), §118.4 | Validate section 6 under single passes at the f64 tolerances (1e-8 vs dense, 1e-13 residual). `solver.rs` tests: the f32 arms `SLACK`/`SOLVE_TOL` tighten to the f64 values and pass (tightening only). Outer IR steps ≤ 6 and total inner iterations ≤ 2× the f64 count on the section-6 systems. | F32-02 | L | S · W1 |
| **F32-07** | IR covers the other backends: the FFT direct solve (§ pressure/fft), the distributed Krylov (§73, whose exact reductions are already associative) and the pressure selector's probe. The native GAMG from the parallel stream is wrapped once it lands. | `src/pressure/*.rs`, `src/distsolve*.rs` | The FFT Poisson row in section 6 at f64 tol under single; `distsolve` part-count invariance at f64 tol (P = 1, 2, 4) | F32-06, PAR-04b | M | S · W3 |
| **F32-08** | Transient solves under single use the delta form against f64 old levels (euler, backward, CN), so a per-step increment below ε32·\|x\| is not lost. | `src/timescheme.rs`, `cuda/timescheme.cu`, `src/solver.rs`, §118.5 | Failing first: `timescheme::tests::f32_slab_heats_by_1e_8_per_step`. A semi-infinite slab is heated at ΔT/step = 1e-8·T for 2000 steps; today T does not move at f32. After: T(x,t) within 1e-3 relative of the erfc solution (Carslaw & Jaeger; open: https://en.wikipedia.org/wiki/Heat_equation), and f32 vs f64 runs within 1e-5. Gate 105-B rows (section 46) at the f64 order bands. | F32-06 | L | S · W1 |
| **F32-09** | Field, phi and restart I/O at f32: the writer emits the shortest round trip (9 significant digits) for `f32`, the reader parses f64 and rounds once, and the restart is bitwise. | `src/io/fields.rs`, `src/io/writer.rs`, `src/restart.rs` | Section 10's phi round trip passes bitwise under single (`io::tests::f32_phi_round_trip_is_bitwise`); f64 output bytes unchanged | F32-01 | S | S · W1 |
| **F32-10** | Under single the mesh is stored about a local f64 origin (the bounding-box centre). gh, selectors, probes, writers, wall distance and parcel injection take the shift. | `src/mesh.rs`, `src/mesh/geometry.rs`, `src/io/writer.rs`, `src/sources.rs`, `src/bin/common/mod.rs` | Failing first: `mesh::tests::f32_translated_cavity_matches`. A Re = 100 cavity at the origin and translated by (2000, 2000, 0) m: max\|ΔU\|/U_lid ≤ 1e-5 (predicted ~1e-3 today); written points equal the input to f32 round trip | F32-03 | M | S · W2 |
| **F32-11** | Under single, moving-mesh geometry (`meshgeom.cu`, `ale.cu`) keeps points, V and swept volumes in f64 on the device, with the hot kernels reading f32 copies. | `cuda/meshgeom.cu`, `cuda/ale.cu`, `src/mesh/ale*.rs`, `src/mesh/gpugeom*.rs` | Gate 105-A SCL rows (section 45) at 1e-12 under single; the FP64 geometry cost per step is ≤ 10 % of one SIMPLE step at 1 M cells (measured, printed) | F32-03 | M | S · W2 |
| **F32-12** | Under single, parcel positions and the face-crossing walk are f64, and the parcel to gas exchange totals accumulate in `ofacc`. | `cuda/parcels.cu`, `cuda/parcelsort.cu`, `cuda/parcelcouple.cu`, `src/parcels*.rs` | Sections 35-40 under single at f64 tol; 0 parcels lost on the 2:1-adapted mesh walk test | F32-02 | M | S · W2 |
| **F32-13** | Every validate row in sections 1-25 carries a class (P/A/R/D). R rows go through `check_roundoff(what, err, tol64)`, which takes tol32 from D-F32-1 or the `skip` alternative (one constant). | `src/bin/validate.rs` (sections 1-25), `src/precision.rs` | Under single, sections 1-25 report every row, every P/A/D row at its f64 tol and every R row at tol32. f64 rows are byte-identical (F32-01 diff). The class counts are printed. | F32-04, F32-06, F32-09 | L | S · W5 |
| **F32-14** | The same for sections 26-51. | `src/bin/validate.rs` (26-51) | Sections 26-51 under single as in F32-13; Gates 95-D/E, 105-A..C, 94-D, 110-C at their f64 bands | F32-05, F32-07, F32-08, F32-11, F32-12, F32-13 | L | S · W5 |
| **F32-15** | The four tests that "did not finish" at f32 finish. Each is diagnosed with a timeout test first (the expected cause is a stop criterion below the f32 floor, which IR and delta form remove). | `src/ale_flow*.rs`, `src/models/k_omega_sst*.rs`, `src/solid*.rs` | Each of the four finishes in ≤ 3× its f64 wall time and passes (P/A at f64 tol, R at tol32) | F32-08 | M | S · W5 |
| **F32-16** | The ignored library tests in io, models, cht, parcels, solid, s2s and automesher run under single: either they pass at f64 tol, or they use `precision::roundoff_tol` (R), or they are `cfg(not(single))` because their subject is f64 itself (each one named in §112.3). | the tests in those modules, `src/types.rs` (counting test) | Ignored count drops by at least 270; the bold number in §112.3 is updated; module-filtered runs < 10 min each | F32-13 | L | S · W6 |
| **F32-17** | The same for the remaining ~22 modules and the 16 binary tests. | the remaining tests | Ignored count at 0 except the named f64-subject tests; `--bins` 272/272 under single | F32-15, F32-16 | L | S · W6 |
| **F32-18** | f32 becomes a **supported configuration**. §112.4 is replaced by the measured 51-section table, §112.5 is rewritten, and §118 is closed. A performance and memory figure is taken. | `SPEC-LIT.md`, `README.md`, `src/bin/bench.rs` | A deferred long run of `ofgpu-validate` under single exits 0 (R rows per D-F32-1). Measured on the 5070 Ti: B/cell slope ≤ 0.65× f64, the cliff ≥ 1.5× of 4.72 M cells, and SIMPLE s/iteration on a 1 M cavity ≥ 1.5× faster. A miss is recorded as a finding, not hidden. | F32-14, F32-17 | M | S · W6 |
| **F32-19** | The studio can run a case at f32: run settings have precision `double`/`single`, the server spawns `target/single/release/<bin>`, and the run header shows the precision. | `gui/shared/src/protocol.ts`, `gui/server/src/*` | Server test: `precision: "single"` spawns the single path; the PIPELINES sync test still holds; 286+32 tests green | F32-06 (F32-18 (or F32-06 for a preview)) | S | G · W5 |
| **BC-01** | §120 holds the catalogue table of §2.4, and a test holds the code to it: every name is implemented, aliased, planned (unit id), owned by another stream, or refused by name with a reason. The permanent refusals get their messages. | `SPEC-LIT.md` §120, `src/field.rs`, `src/field/catalogue.rs` (new, test-only list) | `field::tests::every_catalogue_name_has_a_status`; `coded*` / `#codeStream` / `cyclicACMI` / `v2WallFunction` refuse with a specific reason rather than the generic list | — | S | S · W1 |
| **BC-02** | `blockgen` builds wedge meshes from an (r, z) block: an annulus (`r_in > 0`) and a pipe with axis prisms (collapsed hexes written as 5-face prisms, with no zero-area faces) at any angle θ. Patches are typed `wedge`. | `src/blockgen.rs` (or `src/blockgen/wedge.rs` new) | Volume = θ/2·(R₂²-R₁²)·L to 1e-12 relative; closure ≤ 1e-12; no face with \|Sf\| < 1e-14 (`blockgen::tests::wedge_annulus_volume`, `wedge_pipe_axis_has_no_degenerate_faces`) | — | M | S · W1 |
| **BC-03** | `PatchKind::Wedge` is its own kind (`mesh.rs:84` no longer maps it to Symmetry). The wedge pair, axis, angle and mid-plane symmetry are derived and checked, the mesh report prints them, and non-planar, unpaired or asymmetric wedges are refused by name. | `src/mesh.rs`, `src/mesh/wedge.rs` (new), `src/io/polymesh.rs`, `cuda/momentum.cu`, `cuda/fv.cu` (patch-kind macros), §119.1 | Failing first: `mesh::tests::wedge_is_not_symmetry`. Axis direction error ≤ 1e-12 at θ = 1°, 5°, 10°. The three refusal tests. All f64 validate rows are unchanged (no shipped mesh has a wedge). | BC-02 | M | S · W1 |
| **BC-04** | `BcKind::Wedge`: the rotational self-couple of §2.3. A kernel rewrites refGrad from `R(±θ)U_P`, the wedge-face flux is prescribed as `U_b·Sf`, and scalars are zero-gradient. §119 carries the derivation, including the hoop-term factor and the centrifugal and Coriolis consistency. | `src/field.rs`, `cuda/field.cu` (no new evaluation branch; new `fldWedgeRefGrad`), `cuda/momentum.cu`, `src/momentum.rs`, §119.2 | Failing first: `wedge::tests::taylor_couette_swirl_survives` (with wedge = symmetry, u_θ error ~100 %). Taylor-Couette: u_θ L∞ relative error ≤ 2e-3 on the finest of three meshes, observed order in [1.8, 2.2] (§94). p(R₂)-p(R₁) within 1 % at θ = 5°, with the error ratio between θ = 5° and 2.5° in [3.5, 4.5] (O(θ²)). Per-cell mass imbalance from swirl ≤ 1e-14. | BC-03 | L | S · W2 |
| **BC-05** | The wedge joins `ofgpu-validate` as a new section: Hagen-Poiseuille through the axis prisms and radial conduction. Case JSON accepts a `wedge` patch preset. | `src/bin/validate.rs`, `src/io/case_json.rs`, `docs/schema/case-1.json` | u_max/Ū = 2 within 0.5 %; f·Re = 64 within 1 % at Re = 100; ln-profile T within 1e-3; new rows pass in both builds (P class) | BC-04, F32-01 | M | S · W2 |
| **BC-06** | Function1 on the device: constant, table (linear/step; outOfBounds clamp/error/warn/repeat), tableFile/csvFile, polynomial, sine, square, linear/quadratic/halfCosine/quarterSine/quarterCosine ramps, and scale. It feeds uniformFixedValue, uniformFixedGradient, uniformInletOutlet, uniformTotalPressure, the flowRateInletVelocity rate and fixedProfile (a Function1 of a coordinate). The `uniformFixedValue` refusal (`field.rs:723`) is removed. | `src/function1.rs` (new), `cuda/function1.cu` (new), `src/field.rs`, `src/field_setup.rs`, §120.1 | Device equals the host evaluation bitwise at 1000 times per type; the §81 capture gate passes with a ramped inlet captured; ∫Q(t)dt through a ramped inlet equals the closed form to 1e-12 (f64) | BC-01 | M | S · W1 |
| **BC-07** | Wall and slip family: rotatingWallVelocity, translatingWallVelocity, partialSlip (new vector branch: `U_b = (1-f)(I-nn)U_P + f·U_w`), fixedNormalSlip, fixedShearStress, directionMixed (tensor value fraction), extrapolatedCalculated, plus the `nutUSpaldingWallFunction` alias. | `src/field.rs`, `cuda/field.cu`, `src/field_setup.rs` | Couette flow with partialSlip at slip fraction f: the closed-form linear profile with slip length to 1e-10; a rotating wall's tangential velocity is exact (Ω×r) to 1e-14; the fixedShearStress wall gradient equals τ/μ to 1e-12 | BC-01 | M | S · W1 |
| **BC-08** | Velocity-inlet family: surfaceNormalFixedValue, cylindricalInletVelocity, swirlFlowRateInletVelocity, pressureInletVelocity, pressureInletUniformVelocity, pressureDirectedInletVelocity, pressureDirectedInletOutletVelocity, pressureInletOutletParSlipVelocity. | `src/field.rs`, `src/field_setup.rs`, `cuda/field.cu` (flux rule) | The prescribed flux equals the patch sum to 1e-12; the swirl inlet's rate and swirl ratio are exact; a pressure-driven channel with pressureInletVelocity reaches the Poiseuille rate within 1 % | BC-06 | M | S · W2 |
| **BC-09** | Outlets I: outletInlet (fr = 1 on outflow, a new kind outside the flux-switched block) and advective (the Orlanski convective outlet with euler/backward and lInf/fieldInf relaxation). Ref: Orlanski 1976, J. Comput. Phys. 21:251, doi:10.1016/0021-9991(76)90023-1. | `src/field.rs`, `cuda/field.cu`, `src/timescheme.rs` | A 1-D Gaussian pulse convected out through advective: reflected peak ≤ 1 % of incident (zeroGradient baseline measured and printed); outletInlet switch rows on a reversing-flow test | BC-01 | M | S · W1 |
| **BC-10** | Outlets II: fixedMean, fixedMeanOutletInlet, fixedFluxExtrapolatedPressure, outletMappedUniformInlet (outlet mean to inlet, with fraction and offset), matchedFlowRateOutletVelocity. | `src/field.rs`, `src/field_setup.rs` | Patch mean equals the target to 1e-12; outletMappedUniformInlet closes a loop duct with the inlet mean equal to the outlet mean ×fraction + offset to 1e-12 | BC-09 | M | S · W2 |
| **BC-11** | Buoyant pressure family: prghPressure, prghTotalPressure, prghTotalHydrostaticPressure, uniformDensityHydrostaticPressure. | `src/field.rs`, `src/field_setup.rs`, `src/bin/buoyant.rs`, `src/vof.rs` | A hydrostatic column open at the top with prghPressure: \|U\| ≤ 1e-10 and p = p₀ + ρg(h_ref - z) to 1e-12 (f64) | BC-06 | M | S · W2 |
| **BC-12** | Thermal family: externalWallHeatFluxTemperature (power/flux/coefficient, thicknessLayers/kappaLayers, emissivity) on the §98 ambient machinery for the fluid side, turbulentHeatFluxTemperature, lumpedMassTemperature, totalFlowRateAdvectiveDiffusive, and the alphat aliases. The `field.rs:682` refusal is removed. | `src/field.rs`, `src/cht/ambient.rs`, `src/energy.rs` | Coefficient mode on a 1-D slab: flux equals the series-resistance closed form to 1e-10; lumped mass follows `T∞ + (T₀-T∞)e^{-t/τ}` within 1e-4 at second order in Δt | BC-06 | M | S · W2 |
| **BC-13** | Atmospheric family: atmBoundaryLayerInletVelocity/K/Epsilon/Omega and nutkAtmRoughWallFunction (refusal at `field.rs:461` removed), plus nutUBlendedWallFunction. Refs: Richards & Hoxey 1993, J. Wind Eng. Ind. Aerodyn. 46-47:145, doi:10.1016/0167-6105(93)90124-7; Hargreaves & Wright 2007, JWEIA 95:355, doi:10.1016/j.jweia.2006.08.002. | `src/field.rs`, `src/wallfunctions.rs`, `cuda/wallfunctions.cu`, `src/field_setup.rs` | A 2-D empty fetch of 5 km (coarse, < 10 min): outlet U within 3 % and k within 10 % of the inlet profiles (horizontal homogeneity); ground u* within 3 % | BC-06 | L | S · W2 |
| **BC-14** | Recycled inflow: mapped/mappedFixedValue (sampling an offset plane through the parcel walk's point location, with setAverage) and turbulentInlet (seeded, deterministic fluctuation). | `src/field/mapped.rs` (new), `src/field.rs` | A recycled-inlet laminar channel reaches the developed Poiseuille profile within 1 % of the cyclic channel; setAverage holds the flux to 1e-12; turbulentInlet's RMS equals the prescribed scale ±5 % over 10⁴ samples with a fixed seed | BC-06 | M | S · W2 |
| **BC-15** | timeVaryingMappedFixedValue: `constant/boundaryData/<patch>/points` and time folders, a planar Delaunay with barycentric interpolation, and linear in time. Refs: Bowyer 1981 doi:10.1093/comjnl/24.2.162; Watson 1981 doi:10.1093/comjnl/24.2.167. | `src/field/tvmapped.rs` (new), `src/io/` | A linear field sampled at 200 scattered points is reproduced on the patch faces to 1e-12 (barycentric interpolation is exact for linear data); time interpolation is exact at the midpoints | BC-06 | M | S · W2 |
| **BC-16** | Synthetic LES inflow: turbulentDigitalFilterInlet (Klein, Sadiki & Janicka 2003, J. Comput. Phys. 186:652, doi:10.1016/S0021-9991(03)00090-1) and turbulentDFSEMInlet (Poletto, Craft & Revell 2013, Flow Turb. Combust. 91:519, doi:10.1007/s10494-013-9488-2). | `src/field/synthetic.rs` (new), `cuda/` | The prescribed Reynolds stresses are reproduced within 5 %; the two-point correlation length within 10 % of the prescribed one; the DFSEM field is divergence-free to 1e-10 (its property) | BC-14 | L | S · W3 |
| **BC-17** | Cyclic jumps: fixedJump, uniformJump (Function1) and the cyclic-baffle meaning of fan and porousBafflePressure (a jump as a function of the face flux). The open-patch §52/§53 kinds keep their names under the §13.4 rule, with the two meanings distinguished by the patch kind. | `cuda/field.cu`, `cuda/ldu.cu` (jump source on both rows), `src/field.rs`, §120.2 | 1-D duct with a baffle pair: a piecewise-linear p with a jump of exactly Δp (1e-12) and flow rate from the closed form within 1e-10; matrix symmetry (§48) holds | BC-06 | M | S · W2 |
| **BC-18** | Static non-conformal interfaces by **imprint**: the two patches' faces are intersected (Sutherland-Hodgman clipping in f64, Sutherland & Hodgman 1974, CACM 17:32, doi:10.1145/360767.360802) and split, so each side's cells become polyhedra with conformal sub-faces. | `src/mesh/imprint.rs` (new) | Sub-face areas sum to each original face's area to 1e-12; the two sides' sub-faces pair one-to-one; mesh closure ≤ 1e-12 on 3:2 and rotated 7° non-matching blocks | — | L | M · W2 |
| **BC-19** | `cyclicAMI` (static) is accepted: imprinted patches become an ordinary conformal cyclic (refusal at `field.rs:734` removed for static meshes; refused by name when the mesh moves). | `src/field.rs`, `src/mesh.rs`, `src/io/polymesh.rs` | A linear field's gradient is exact across the interface to 1e-11; flux mismatch across the interface ≤ 1e-14; a laminar channel across a 3:2 interface matches the single-block solution within its discretisation error (≤ 0.5 %) | BC-18 | M | S · W3 |
| **BC-20** | Compressible family: waveTransmissive (Poinsot & Lele 1992, J. Comput. Phys. 101:104, doi:10.1016/0021-9991(92)90046-2), totalTemperature, inletOutletTotalTemperature, supersonicFreestream, freestreamPressure supersonic switch, fixedPressureCompressibleDensity. | `src/field.rs`, plus the compressible stream's energy/density files | A 1-D acoustic pulse leaving through waveTransmissive with reflection coefficient ≤ 5 % (fixed-p baseline ~100 % printed); totalTemperature isentropic relation to 1e-12 | BC-09, CMP-16, CMP-17, CMP-18 (BC-09 and **the compressible stream's solver unit**) | M | S · W4 |
| **BC-21** | VOF family: variableHeightFlowRate, variableHeightFlowRateInletVelocity, outletPhaseMeanVelocity, phaseHydrostaticPressure. | `src/field.rs`, `src/vof.rs` | Inlet liquid flux equals the prescribed rate to 1e-12; alpha bounded in [0, 1] (§20 gate) | BC-06, BC-11 | M | S · W3 |
| **BC-22** | The studio's BC picker lists every implemented type with its required entries, from a catalogue the solver generates (JSON from `IMPLEMENTED_BC_NAMES` plus BC-01's entry lists), and the case schema's patch `bc.type` takes the new names. | `gui/shared/src/protocol.ts`, `gui/server/src/*`, `docs/schema/case-1.json` | A sync test: the gui catalogue equals the solver-generated JSON (like the PIPELINES test); server 286 + shared 32 tests green | BC-01 (BC-01 (refreshed after each BC unit)) | S | G · W5 |

**Assembler corrections:**

1. **The tree base:** `feat/core-2` is an ancestor of `8ab1c4a`, 0 ahead and 189 behind. "55 commits ahead of main"
   was measured against the stale local `main` (§3). §114 and §115 are on `8ab1c4a`.
2. **SPEC-LIT:** §P1, §P2 and §P3 became **§118, §119 and §120.**
3. **BC-20 is reduced** to `inletOutletTotalTemperature`, `supersonicFreestream`, `freestreamPressure` (the supersonic
   switch) and `fixedPressureCompressibleDensity`, built on CMP's `compressible/bc.rs`.
   - It depends on CMP-16/17/18.
   - Its "1-D acoustic pulse ≤ 5 %" row is dropped as a duplicate of Gate CMP-NR.
   - It keeps a closed-form gate per remaining name.
   - `waveTransmissive` and `totalTemperature` are COMP's.
4. **BC-09 takes the wave speed as a face-field input**, so CMP-17 reuses the kernel (contract note).
5. **BC-22 consumes the catalogue JSON the solver emits**, from BC-01's list and entries. It does not edit
   `docs/schema/case-1.json`, which is generated in lane S.
6. **Decision 11** (no new single-ignore) is adopted for every stream (§5.1.3).
7. **F32-03 stores f64 points in `PolyMeshRaw`** (§5.1.8). TET-11 and VIO-01 gain the dependency.
8. **Placement:**
   - F32-13/14 run in S W5, after VAL's in-place `validate.rs` edits (M, W2-W4). Rows added in tranche 17 carry
     their class from birth, so these two units classify only the rows that existed at `8ab1c4a`.
   - F32-16/17 run in S W6, and own automesher's tests in that wave.
   - F32-07 covers the native GAMG through AMG-06a rather than "once it lands".
9. **F32-19 ships in W5** against F32-06 (preview). The studio's default precision stays `double` (§10.1 A18), and
   "supported" is stated only after F32-18.
10. **The BC tail** (BC-16, 18, 19, 20, 21) stays in this tranche, last in its family, as "모두구현" says. BC-18 runs
    in lane M (W2), because `mesh/imprint.rs` is a new file.

### 12.5 MESHIO — native tetrahedra, volume output, Isaac (TET-, VIO-)

| id | what becomes TRUE | files | gate (quantitative; reference; tests) | depends | eff | lane · wave |
|---|---|---|---|---|---|---|
| **TET-00** | SPEC-LIT §116 "The native tetrahedral mesher" exists with every subsection, equation and citation of §2.1. §92.4's "Generation is external" is replaced by a pointer to §116, with the repair list R1–R4 kept. The `tetmesh` module skeleton and `TetSpec` config types compile. | `rust/SPEC-LIT.md`, `rust/src/tetmesh/mod.rs`, `rust/src/lib.rs`, `rust/PROVENANCE.md` | The xref audit (§80) passes; every DOI in §116 resolves (checked by the supervisor); `tetmesh::tests::spec_round_trips_json`; both builds. | — | S | M · W1 |
| **TET-01** | `orient3d` and `insphere` return the exact sign for any f64 input, whatever `Scalar` is. | `tetmesh/predicates.rs` | Shewchuk 1997 adaptive stages. For 10⁵ near-degenerate integer-coordinate configurations (\|x\| < 2²⁰, so the i128 determinant is exact), the sign equals the i128 evaluation 100 %. Cospherical lattice points give exactly 0. `predicates::matches_exact_i128`, `predicates::cospherical_is_zero`, `predicates::f64_under_single`. | TET-00 | M | M · W1 |
| **TET-02** | The perturbed `insphere` (lifting-map symbolic perturbation by vertex index) never returns 0, and it is antisymmetric under vertex swaps. | `tetmesh/perturb.rs` | Devillers & Teillaud 2011 / Edelsbrunner–Mücke 1990. On a 10³ cubic lattice, 0 zero results over all 5-point queries tested (10⁶ sampled), and sign(swap) = −sign for 10⁵ random swaps. `perturb::never_zero_on_lattice`, `perturb::antisymmetric`. | TET-01 | S | M · W1 |
| **TET-03a** | A Bowyer–Watson Delaunay tetrahedralisation with full adjacency, an infinite vertex and visibility-walk point location exists for any finite point set. | `tetmesh/delaunay.rs` | Delaunay lemma (Shewchuk notes). For 10⁴ random, 20³ lattice and 2·10³ on-sphere points: every interior facet is locally Delaunay under the perturbed predicate; adjacency is symmetric; Euler V−E+F−T = 1; Σ signed volumes = hull volume (lattice: exactly the box, rel 1e-12); no tet has orient3d = 0. `delaunay::locally_delaunay_*`, `delaunay::euler_ball`, `delaunay::volume_closes`. | TET-02 | L | M · W2 |
| **TET-03b** | Insertion order is BRIO with a Hilbert curve, and the kernel triangulates 10⁶ random points fast. | `tetmesh/order.rs`, `tetmesh/delaunay.rs` | Amenta, Choi & Rote 2003. 10⁶ uniform points in ≤ 15 s release single-thread on this machine (measured and recorded in §116), with walk steps per insertion ≤ 20 mean. All TET-03a gates rerun at 10⁵. `order::hilbert_is_bijective`, `delaunay::million_points_timed` (`#[ignore]`, run by the supervisor). | TET-03a | M | M · W2 |
| **TET-04a** | A size field h(x) on a background octree answers queries from base, per-patch surface size, distance bands (automesher `refinement` semantics) and boxes. | `tetmesh/sizefield.rs` | A point source h₀ with base H: h(x) = min(H, h₀ + (g−1)\|x\|) holds within one octree leaf's size, max relative error ≤ 2 %. Box sources are exact inside. `sizefield::point_source_cone`, `sizefield::box_exact`. | TET-00 | M | M · W1 |
| **TET-04b** | Curvature and proximity sources exist, and the gradation limit \|∇h\| ≤ g−1 holds everywhere. | `tetmesh/sizefield.rs` | Persson 2006. Sampled finite differences ≤ (g−1)(1+1e-9) on 10⁵ probe pairs; for a sphere of radius R with n_curv = 16, h ≤ 2πR/16 (+2 %) on the surface; for a 1 mm gap with n_gap = 3, h ≤ gap/3 inside it. `sizefield::gradation_bounded`, `sizefield::curvature_sphere`, `sizefield::proximity_gap`. | TET-04a | M | M · W2 |
| **TET-05** | Input surfaces are validated before meshing: closed, consistently oriented and free of self-intersections, with features and patch seams extracted. Each defect is refused by name. | `tetmesh/surfprep.rs` | Möller 1997. Fixtures: a sphere and a box pass; an open box gives the `require_closed` refusal; a flipped triangle gives an orientation refusal naming the triangle; two crossing boxes give a self-intersection refusal naming the pair. Cube: 12 feature polylines, 8 corners. `surfprep::refuses_open`, `::refuses_flipped`, `::refuses_self_intersection`, `::cube_features`. | TET-00 | M | M · W1 |
| **TET-06a** | An adaptive isotropic surface remesh (split/collapse/flip/tangential relaxation, projected onto the input) reaches the target edge length h(x). | `tetmesh/remesh.rs` | Botsch & Kobbelt 2004. Unit sphere at h = 0.05: ≥ 95 % of edges have length/h ∈ [4/5, 4/3]; ≥ 99 % of triangles have min angle ≥ 25°; Hausdorff distance to the input ≤ 0.01 h; the surface stays closed and 2-manifold. `remesh::sphere_lengths`, `::sphere_angles`, `::stays_closed`. | TET-04a, TET-05 | L | M · W2 |
| **TET-06b** | Feature edges, corners and patch seams survive remeshing exactly, and each triangle keeps its patch. | `tetmesh/remesh.rs` | Cube at h = 0.1: all 12 edges are present as polylines, the 8 corners are bit-identical and patch areas match within 1e-12 relative. Two-patch cylinder: the seam is preserved. A 30° wedge: the edge is preserved. `remesh::cube_features_exact`, `::patch_area_kept`, `::wedge_edge`. | TET-06a | M | M · W2 |
| **TET-07a** | Every edge of the remeshed surface appears in the tetrahedralisation, after Steiner splits on the edge where needed and protecting balls at input angles < 90°. | `tetmesh/recover.rs` | Murphy et al. 2001; Cohen-Steiner et al. 2004. Cube, 30° wedge and 10° spike: 100 % of edges recovered, the Steiner count is reported and termination is proven by a test with an iteration cap that is never reached. `recover::edges_cube`, `::edges_spike`. | TET-03a, TET-06b | L | M · W3 |
| **TET-07b** | Every surface triangle, after splits, is a union of tet faces, with Steiner points only on the piecewise-linear surface. | `tetmesh/recover.rs` | Schönhardt's twisted prism (1928) recovers with ≥ 1 Steiner point; sphere and cube: the boundary face set equals the split surface exactly; each Steiner point lies on its facet within 1e-12·h. `recover::schonhardt`, `::faces_exact_sphere`, `::steiner_on_surface`. | TET-07a | L | M · W3 |
| **TET-08** | Tets are carved into fluid/solid by flood fill across non-surface faces (external box, seed point, or declared bodies), and the kept volume closes exactly. | `tetmesh/carve.rs` | Divergence theorem: kept volume = enclosed volume of the split surface, relative 1e-12. Sphere in a box: box − sphere; a seed on the wrong side is refused by name; patch face counts equal the surface's. `carve::volume_closes`, `::seed_refusal`. | TET-07b | M | M · W3 |
| **TET-09** | Delaunay refinement leaves no tet with radius-edge ratio > 2 or circumradius > c·h(x), except tets next to protected small angles (counted and reported). Encroached surface triangles are split first. | `tetmesh/refine.rs` | Shewchuk 1998 (B = 2). Sphere in a box: 0 violators outside the reported set; mean edge/h(x) ∈ [0.85, 1.15]; the tet count is within ±20 % of ∫dV/(h³/(6√2)) × 1.1. `refine::radius_edge_bound`, `::size_conformance`, `::count_predicted`. | TET-04b, TET-08 | L | M · W4 |
| **TET-10a** | 2-3, 3-2 and edge-removal flips improve the worst min-dihedral in each changed star, monotonically, and never touch surface faces. | `tetmesh/optimise.rs` | Shewchuk 2002 (edge.pdf). On TET-09's outputs the worst dihedral never decreases in any pass, the boundary face set is unchanged bit for bit, and the tets stay valid (orient3d > 0). `optimise::flips_monotone`, `::boundary_frozen`. | TET-09 | M | M · W4 |
| **TET-10b** | Smoothing (interior vertices free, surface vertices tangential, feature vertices along their edge) and sliver perturbation bring every fixture to a min dihedral ≥ 10° and max ≤ 165°. | `tetmesh/optimise.rs` | Freitag & Ollivier-Gooch 1997; Cheng et al. 2000. Benchmark bounds from Labelle & Shewchuk 2007 (10.7°/164.8°). Fixtures: sphere in a box, cube minus a cylinder, a 30° wedge. Surface vertices stay on the surface within 1e-12. `optimise::dihedral_bounds_*`, `::surface_stays`. | TET-10a | L | M · W4 |
| **TET-11** | Tets become a `PolyMeshRaw` with patches, owner < neighbour and upper-triangular order. §92.3's gate passes. Under `single`, rounding the points to f32 that would flip any cell is refused, naming the cells. | `tetmesh/emit.rs` | `build_host_mesh` accepts; `quality::check` passes on all TET-10b fixtures; the f32 rounding check is exercised by a 1e4-m-offset sliver fixture (refused under single, accepted under f64). `emit::gate_passes_*`, `::f32_flip_refused`. | F32-03, TET-10b | M | M · W5 |
| **TET-12** | `ofgpu-automesher` meshes `kind: "tet"` configs end to end (`-check` included) and writes a summary JSON (tet/prism counts, dihedral min/max, Steiner counts, timings). Existing hex configs are unchanged. Docs say the STL→tet path needs no gmsh. | `automesher/mod.rs`, `automesher/driver.rs`, `tools/automesher/examples/box_sphere_tet.json`, `tools/automesher/README.md`, `tools/mesh/README.md` | `box_sphere_tet.json` → polyMesh in ≤ 30 s release; every hex golden hash (L1a goldens) is unchanged; unknown-key and kind-mismatch refusals are tested. `driver::tet_end_to_end`, `driver::hex_goldens_unchanged`. | TET-11 | M | M · W5 |
| **TET-13a** | Wall point normals (Aubry–Löhner "most normal") and per-point layer thicknesses with limiters (proximity, curvature, neighbour ratio ≤ 1.5) exist. | `tetmesh/inflate.rs` | Sphere: the normal is the radial direction within 1e-6 rad. Cube corner: the normal makes equal angles with the 3 faces within 1e-9. A gap of 3·T total is limited to ≤ gap/2. `inflate::normals_sphere`, `::corner_normal`, `::thickness_limited`. | TET-12 | M | M · W5 |
| **TET-13b** | Post-inflation displaces the tet mesh by explicit IDW deformation, with inversion check and local thickness cuts, and inserts N prism layers (t1, ratio r). The prism–tet interface is conformal. | `tetmesh/inflate.rs`, `tetmesh/emit.rs` | Luke et al. 2012; Kallinderis et al. 1996. Sphere in a box, 5 layers, t1 = 1e-3·D, r = 1.2: achieved t1 within 1 % at ≥ 95 % of wall points; 0 negative volumes; §92.3 G2 closure passes; wall non-orthogonality ≤ 5°. `inflate::sphere_layers`, `::conformal`. | TET-13a | L | M · W6 |
| **TET-13c** | Layers at sharp convex and concave edges and corners collapse or taper instead of inverting, and the report names each tapered point. | `tetmesh/inflate.rs` | Cube minus cylinder and a 30° wedge with 5 layers: 0 inverted cells; § 92.3 passes; coverage (fraction of wall points with the full N layers) ≥ 90 %, reported. `inflate::sharp_edges_taper`. | TET-13b | L | M · W6 |
| **TET-14** | A laminar pipe meshed with tets plus prisms solves to Hagen–Poiseuille, in both f64 and f32. | `tools/automesher/examples/pipe_tet.json`, a case under `cases/pipe_tet_L1/`, a test or validate row | Δp within 2 % of 8μLQ/(πR⁴) at Re = 100 (f64); f32 within the §112.1 floor. Solve ≤ 3 min on this GPU. The driver is the incompressible steady binary the §110 gates use (the supervisor names it). `pipe_tet_poiseuille`, plus an `ofgpu-validate` row. | TET-13b | M | M · W6 |
| **TET-15** | The size field and remesh run on `std::thread::scope` (no new crate), and a 1 M-tet sphere in a box meshes end to end within budget. | `tetmesh/{sizefield,remesh}.rs` | ≤ 120 s release end to end, measured; the output is bit-identical across thread counts 1/4/8. `perf::million_tets` (`#[ignore]`, supervisor), `perf::thread_invariant`. | TET-12 | M | M · W5 |
| **TET-16** | PROVENANCE, NOTICE, `tools/deps_licences.py` and LICENSING.md show the tet path has no GPL, LGPL or AGPL source or dependency. `tools/mesh/step_mesh.py` is documented as the optional external STEP route only. | `rust/PROVENANCE.md`, `NOTICE`, `LICENSING.md`, `tools/mesh/README.md` | `deps_licences.py` shows no new dependency; a grep for `gmsh` in `rust/src/tetmesh` returns 0 hits. | TET-12 | S | M · W5 |
| **TET-17** | The studio reads `kind: tet` automesher summaries (counts, dihedral bounds) and offers `kind` on the automesh tool. | `gui/server/src/formats/meshSummary.ts` (+ test fixture), `gui/shared/src/registry.ts` | `meshSummary.test.ts` passes on a TET-12 summary fixture; server and shared suites stay green (memory `gui-server-test-baseline`). | TET-12 | S | G · W6 |
| **TET-18** | The real drone sim surface (x500, about 251k triangles, BSD-3, output outside the repo) meshes with tets and layers and passes §92.3. | supervisor run, results in the §116 table | §92.3 passes; dihedral min/max reported; wall time recorded (a deferred long run is allowed). | TET-13c, TET-15 | M | M · W6 |
| **TET-19** | Declared bodies keep their own tet regions with conformal shared interfaces, reusing §92.15's region writer. | `tetmesh/carve.rs`, `automesher/driver.rs` | Two-cube fixture: interface faces are internal and paired; per-region volumes close at 1e-12; `regions_check` passes. `carve::two_regions`. | TET-12 | M | M · W6 |
| **VIO-00** | SPEC-LIT §117 "Volume output from any mesh, and the Isaac Sim profile" exists: lattice, rasterisation, tie rule, sampling, activation, file version, extinction scaling, profiles and probe rule. §44.1's sentence "a mesh that is not ... Cartesian has no voxels" points at §117. | `rust/SPEC-LIT.md` | The xref audit passes; the DOIs resolve. | — | S | M · W1 |
| **VIO-01** | `VoxelMap::build(raw, hm, spec)` assigns every lattice voxel to the cell containing its centre (lowest index on a tie), or to `NONE`, and reports coverage. | `rust/src/io/voxelmap.rs`, `io/mod.rs` | On a Cartesian box with voxel = cell size, the identity map holds exactly; on a `mesh/refined.rs` 2:1 mesh, 100 % agreement with brute-force `parcels::locate_cell` on 10⁴ voxels; on sphere in a box, the fluid fraction = 1 − V_s/V_b within A_s·voxel/V_b; the `maxVoxels` cap coarsens with a note. Same results under single. `voxelmap::identity_cartesian`, `::matches_locate_cell`, `::fluid_fraction_sphere`, `::cap_coarsens`. | F32-03, VIO-00 | M | M · W2 |
| **VIO-02** | `cell` and `linear` sampling: `linear` reproduces linear fields and creates no new extrema. | `io/voxelmap.rs` | Barth & Jespersen 1989. φ = a·x + b is reproduced at interior voxels to 1e-12 (f64) or 1e-5 (f32, the §112 floor); max over voxels ≤ max over cells; `cell` mode has Σ voxel·dV within the coverage error of Σ φV. `voxelmap::linear_exact`, `::no_new_extrema`, `::cell_conserves`. | VIO-01 | M | M · W2 |
| **VIO-03** | `VdbWriter` and `NvdbWriter` write from `WriteCtx.vox` when `cart` is `None`, with `fluid` or `dense` activation and f32/f16. The Cartesian bytes are unchanged. | `io/vdb.rs`, `io/nvdb.rs`, `io/writer.rs` | Round-trips through `vdb.rs`'s test reader: active count = fluid voxels (fluid) or all voxels (dense); values = the sampled values bit for bit (f32); the Cartesian output SHA-256 is unchanged from 8ab1c4a. `vdb::voxmap_round_trip_*`, `nvdb::voxmap_round_trip`, `writer::cartesian_bytes_unchanged`. | VIO-02 | M | S · W3 |
| **VIO-04** | The case keys `output.visualisation.lattice {voxel, box, maxVoxels, sample, activation, background}` are parsed and schema-exported. The non-Cartesian refusal is replaced by a refusal only when no lattice can be built (empty box, voxel ≤ 0, or a budget that a coarsening factor > 16 would still not meet), and each refusal text is tested. | `io/output_plan.rs`, `io/case_json.rs`, `docs/GUIDEBOOK*.md` | The old test `nvdb_writer_refuses_a_non_cartesian_mesh_by_default` is rewritten to the new named refusal (no refusal disappears silently); `-permissive` behaviour is kept. `output_plan::lattice_parses`, `::lattice_refusals`. | VIO-03 | M | S · W3 |
| **VIO-05a** | One helper `output_plan::volume_target(raw, hm, cart, spec)` returns Cartesian or Lattice, and `ofgpu-lowmach` and `ofgpu-k-omega` write `.vdb` on an octree mesh. | `io/output_plan.rs`, `bin/lowmach.rs`, `bin/k_omega.rs` | Box-sphere automesher mesh, a 20-step run: `.vdb` written, voxel count as predicted, run ≤ 2 min. `bin` smoke test (`#[ignore]` GPU) plus a CPU unit test of the helper. | VIO-04 | M | S · W3 |
| **VIO-05b** | The same holds for `k_epsilon`, `sa`, `vof`, `buoyant` and `case_cht` (per region). The three refusal call sites are converted. | `bin/{k_epsilon,sa,vof,buoyant}.rs`, `io/case_cht.rs` | Each driver's existing tests stay green; one non-Cartesian smoke run per driver ≤ 1 min (supervisor). | VIO-05a | M | S · W3 |
| **VIO-06** | `ofgpu-sample volume <case> <time> [-vdb] [-nvdb] [-usda] [-lattice ...] [-isaac ...]` writes volume files from a finished time directory without solving. | `bin/sample.rs` | On the drone solve (t = 0.5 s), the voxel count and fluid fraction match `isaac_export.py`'s recorded 0.999906 within 1e-3, and the free-stream U.x relative error is ≤ 0.005 (the drone G-FREESTREAM). Runs ≤ 2 min. `sample_volume_cli` test on a box case. | VIO-03, VIO-04 | M | S · W3 |
| **VIO-07** | The `.usda` boundary surface is the true face polygons, fanned, when the raw mesh is available, which removes the "Known approximation". | `io/usda.rs` | The patch partition is exact; Σ triangle area per patch = Σ b_mag_sf, relative 1e-10, on planar-face meshes; the extent equals the points' bounding box exactly. `usda::true_loops_area`, `::true_loops_partition`. | VIO-00 | S | M · W2 |
| **VIO-08** | `.vdb` header version 224, with the 225 difference established from the Apache-2.0 OpenVDB source and recorded in §117. | `io/vdb.rs` | Bytes 8..11 = 224; our reader reads both; supervisor check: Isaac's omni.volume `openvdb.readAll` and Blender 5.1's openvdb both read the file (≤ 2 min, via `isaac_vdb.py`'s loader). `vdb::file_version_224`. | VIO-00 | S | S · W2 |
| **VIO-09** | The Isaac profile exists: a `density` (extinction) grid under `opticalDepth` or `physical` scaling, stage `upAxis`/`metersPerUnit`, and `index` (sparse UsdVol) or `pathtrace` (Mesh + `primvars:isVolume` + `OmniVolumeDensity`, dense) scenes. `.nvdb` is never referenced. | `io/usda.rs`, `io/output_plan.rs`, `io/voxelmap.rs` | Max 1995: the computed max column optical depth equals τ within 1 %. Mulholland & Croarkin 2000: K = 8.7 m²/g·ρ_soot exactly. The `.usda` text is checked by our own tests, and `pxr` (usd-core, Apache-2.0) `Usd.Stage.Open` + `UsdVol` validation runs in a tools selftest (≤ 1 min). `usda::isaac_index_scene`, `::isaac_pathtrace_scene`, `voxelmap::optical_depth_target`. | VIO-03, VIO-07, VIO-08 | M | S · W4 |
| **VIO-10** | `tools/isaac/probe.py`: a generic headless probe (any scene, cameras auto-framed from the bounding box, the IndeX or PT renderer, a watchdog), with the visibility rule changed_px > 2 × noise. `tools/drone/isaac_probe.py` becomes a thin caller. | `tools/isaac/probe.py`, `tools/isaac/README.md`, `tools/drone/isaac_probe.py` | Plain-Python `--selftest` P1–P10 pass (no Isaac): framing, verdict and watchdog. The drone probe's selftest still passes 8/8. | VIO-09 | M | M · W5 |
| **VIO-11** | The drone octree solve, exported by `ofgpu-sample volume -isaac`, **is visible** in Isaac Sim 6.0 under both profiles. `isaac_export.py` uses the native writer (no Python resampling) and keeps its 9 gates. | `tools/drone/isaac_export.py`, supervisor run | Probe: `volume_visible` true with changed_px > 2 × noise, for `index` and for `pathtrace`, each ≤ 6 min. A box-sphere case is visible too. Today's baseline (31997 vs 18753) is recorded as the failing "before". | VIO-06, VIO-10 | M | M · W5 |

**Assembler corrections:**

1. **Ids are two-digit.** §116 and §117 are **final**, no longer provisional.
2. **VIO is split by file:**
   - VIO-00, 01, 02, 07, 10 and 11 are in M.
   - VIO-03, 04, 05a, 05b, 06, 08 and 09 are in S. They touch `WriteCtx`'s 19 literal sites in 9 files, the six
     drivers, `case_json.rs`, `output_plan.rs` and `vdb.rs`.
   - VIO-10 runs after VIO-09, as the planner's dependency says, so it sits in M W5.
3. **VIO-01 and TET-11 depend on F32-03** (f64 `PolyMeshRaw` points).
4. **TET-17 runs in G W6.** TET-19 (stretch) runs in M W6 only after F32-16 has landed the automesher tests, or it is
   deferred.
5. **The quality bars** (min dihedral ≥ 10°, max ≤ 165°, coverage ≥ 90 %) are adopted as fixed before measurement
   (§10.1 A17). TET-10c (exudation) and plan B are budgeted in §9.
6. **TET-03b's and TET-15's `#[ignore]` performance tests and TET-18 are supervisor runs (R).** They are performance
   and long-run measurements, not f32 ignores, so §5.1.3 does not apply to them.
7. **TET-16 also corrects PROVENANCE row 455's wording** about gmsh, if the user agrees (§11 item 6).

### 12.6 STUDIO — the studio and the autonomy gates (STU-, AUT-)

| id | what becomes TRUE | files | gate (quantitative; reference; tests) | depends | eff | lane · wave |
|---|---|---|---|---|---|---|
| **STU-01** | One test enumerates every `UiCommandSchema` type and fails on each one answered "does not implement it" or by falling through to the default. It is RED today with the list in §1.1; every later STU unit turns its rows green | `web/src/ws/uiBridge.census.test.ts` (new); the shared export of the type list in `shared/src/protocol.ts` | the census lists exactly the §1.1 types on the current tree (RED by construction). Every command answers in < 4,000 ms (fake timers). Test `uiBridge.census › no command is unimplemented` | — | S | G · W1 |
| **STU-02** | Axes and colour bars toggle from `show_overlay`. Every tab has an addressable id, so `close_tab`, `open_tab` (viewer, residuals, metrics, logs, problems, geometry, campaign) and `select_tab` (by id or label) work. All of it is reported in `ui.state` | `uiBridge.ts`, `state/uiStore.ts`, `viewer/ui/Viewer3D.tsx` (triad and legend visibility), `components/center/TabBar.tsx`, `shared/src/protocol.ts` (UiState fields) | census rows green for these four types. Toggling the triad and legend changes the rendered DOM, and the visibility appears in `ui.state.overlays` (vitest with DOM). `close_tab` on an unknown id answers `NO_TAB` with the list. i18n parity test green | STU-01 | S | G · W1 |
| **STU-03** | `set_post` applies colormap, range, component, representation, opacity, patches and log in one call. `show_metric` picks a run metric (case- and punctuation-blind) and the axis slot. `set_log_filter` drives LogsPane through the store. `show_chart surface` opens the per-patch surface-integral series the run reports, or refuses `NO_SURFACE_METRICS` with the list | `uiBridge.ts`, `uiStore.ts`, `terminal/LogsPane.tsx`, `chart/ResidualsChart.tsx`, `chart/series.ts` | census rows green. A GLM-shaped `set_post` (range as a string "[-2, 1]", stringified numbers) parses, per the `RangeTupleSchema` precedent. "Tmax" finds `T[max]`. Log filter round-trip: text, streams and follow appear in `ui.state.logs`. Tests `uiBridge.post.test.ts`, `LogsPane.test.tsx` | STU-01 | S-M | G · W1 |
| **STU-04** | `add_layer` maps slice, plane, isoSurface, streamlines and glyphs to the viewer's `add*` commands with argument coercion. `remove_layer` removes by id, and an unknown kind or id is refused with the valid list | `uiBridge.ts`, `viewer/api/viewerApi.ts` | census rows green. Each of the five kinds on the synthetic Cartesian dataset adds exactly one layer whose id is returned and reported. Bad args give `INVALID (add_layer): …` naming the field. Test `uiBridge.layers.test.ts` | STU-01 | S | G · W1 |
| **STU-05** | `new_case`, `validate_case` and `save_case {force}` work from the screen. new_case and validate_case delegate on the server to `case_create` and `case_validate`. save_case writes the open editor buffer, refuses when validation found errors unless `force`, and shows the findings in the Problems pane | `server/src/tools/gui.ts`, `server/src/tools/guiCase.ts` (new), `web/src/state/editorStore.ts`, `uiBridge.ts` | census rows green. On a fixture case with one schema error: save_case refuses `CASE_INVALID` with the finding count, `force:true` writes, and the bytes equal the buffer. new_case from `channel` writes a case that `case_validate` passes. Approval: new_case asks exactly like case_create (policy test). Tests `guiCase.test.ts`, `editorStore.save.test.ts` | STU-01 | M | G · W3 |
| **STU-06a** | `open_mesh_dialog` edits a per-session **mesh draft** on the server (preset, cells, outputDir, automesher config, regions layoutDir, check, dryRun). `start_mesh` runs it through `mesh_generate` with that tool's approval, then follows the job on screen | `server/src/tools/guiMesh.ts` (new), `gui.ts`, `shared/src/protocol.ts` (draft in UiState) | the draft round-trips. `cells` given as a number fills three axes, and a 2-D preset given one number is refused `CELLS_2D` naming [nx, ny, nz]. start_mesh with no draft gives `NO_MESH_DRAFT`. start_mesh calls mesh_generate with exactly the draft's input (spy), and the approval card is drawn once. Test `guiMesh.test.ts` | STU-01 | M | G · W3 |
| **STU-06b** | The studio has a Mesh dialog: three forms bound to the draft, Start, and the job's log. `open_mesh_view` opens a mesh output in the viewer with `surfaceEdges` | `web/src/components/side/MeshDialog.tsx` (new), `uiBridge.ts`, `uiStore.ts`, i18n | census rows green. Component test: each form edits the draft through `ui.command`. open_mesh_view on the fixture polyMesh loads it with representation `surfaceEdges`. A screenshot is reviewed by the supervisor. Test `MeshDialog.test.tsx` | STU-06a | M | G · W4 |
| **STU-07a** | `set_patch` changes one patch's type from the editor presets (wall, inlet, outlet, symmetry, empty, patch, wedge where the case kind allows it), or one field's condition, or resets it. The change goes through `case_edit` with its approval, refused by name when the patch is unknown or the preset does not fit the field | `server/src/tools/guiBoundary.ts` (new), `gui.ts` | on a fixture case: set_patch wall → the case diff touches exactly that patch, and the approval card is drawn once. An unknown patch gives `NO_PATCH` with the list. An inlet preset on a field without a value gives `PATCH_FIELD`. Test `guiBoundary.test.ts` | STU-05 | M | G · W4 |
| **STU-07b** | The studio has a Boundary editor panel listing patches (from the case, or from `constant/polyMesh/boundary` when meshed), with their types and per-field conditions, editable through `set_patch` | `web/src/components/side/BoundaryEditor.tsx` (new), `uiBridge.ts`, i18n | census row `open_boundary_editor` green. The component edits a fixture; the diff equals STU-07a's. i18n parity. Screenshot reviewed | STU-07a | M | G · W4 |
| **STU-08a** | Right panels Post, Inspector and Properties exist and `open_panel` opens each one. Post hosts the field, representation and LayerPanel controls. Inspector shows the last probe or selection. Properties shows the selected tree node's keys | `components/shell/AppShell.tsx`, `components/right/{PostPanel,InspectorPanel,PropertiesPanel}.tsx` (new), `uiBridge.ts` | census rows green for `open_panel` × 4. A probe then `open_panel Inspector` shows that cell's id and value. Tests `rightPanels.test.tsx` | STU-02, STU-04 | M | G · W4 |
| **STU-08b** | A project tree (Geometry → Mesh → Physics → Boundaries → Solver → Run → Results) built from the open case. `select_step` selects a node by id or label and opens its panel | `components/side/ProjectTree.tsx` (new), `uiStore.ts`, `uiBridge.ts` | census row `select_step` green. The tree from a fixture case has the 7 nodes with their children. An unknown step gives `NO_STEP` with the ids. Test `ProjectTree.test.tsx` | STU-06b, STU-07b, STU-08a | M | G · W5 |
| **STU-09** | `set_centerline {quantity}` opens a centreline chart: `line_sample` along the longest axis through the domain centre, or along an explicit line. `run_custom_tool` delegates to `custom_tool_run` with its policy | `web/src/chart/CenterlineChart.tsx` (new), `uiBridge.ts`, `server/src/tools/gui.ts` | census rows green. On the analytic fixture (`datasets/fixtures/makeCase.ts`), the centreline of U equals `analyticFields` at the 121 sample cells to 1e-6 relative. run_custom_tool asks like custom_tool_run (policy test) | STU-02 | S-M | G · W4 |
| **STU-10** | **A 3-D polyMesh with wall patches opens showing its walls, not the far-field box.** `fit_view` frames the visible patches, the DomainCard names the hidden ones, and `set_post patches:'all'` restores them | `viewer/controller/ViewerController.ts` (`defaultPatches`, fit), `viewer/ui/DomainCard.tsx`, a fixture builder `server/src/datasets/fixtures/makeSphereInBox.ts` (new: a hex box with a cut sphere wall, types patch and wall) | RED first: on the sphere-in-box fixture the default set is all 7 patches today. After: the visible set is `['sphere']` and the fit bounds equal the sphere's bbox to 1e-6 relative. A Cartesian fixture is still `'all'`, and a 2-D one still its empties (byte-equal manifests). Tests `ViewerController.patches.test.ts` | — | S-M | G · W1 |
| **STU-11** | Large surfaces are exact and do not crash. The edge key is exact beyond 2^21 vertices. StreamlineLayer survives ≥ 300 k points. A 3 M-cell ASCII polyMesh load is measured, and the number is recorded as a reported benchmark, not a flaky CI bound | `viewer/layers/edges.ts`, `viewer/layers/StreamlineLayer.ts`, `server/scripts/bench-polymesh.ts` (new) | RED first: two distinct edges with indices > 2^21 collide under today's key. After, they do not, and the edge counts on a 4 M-vertex synthetic strip equal the analytic count. Streamlines with 300 k points: no RangeError. The bench prints load seconds and peak RSS for 1 M and 3 M cells; the supervisor records them in the gui README (the 3 M run may be an L-run). Tests `edges.test.ts`, `StreamlineLayer.test.ts` | STU-10 | M | G · W3 |
| **STU-12a** | `polycut.ts` cuts any polyhedral polyMesh by a plane exactly: per-cell section polygons, triangulated, with their cell ids. Degenerate planes (through a vertex, an edge or a face) give no duplicate and no gap | `server/src/formats/polycut.ts` (new), `polycut.test.ts`, SPEC-LIT §148 subsection (sign rule; Sutherland–Hodgman DOI 10.1145/360767.360802) | on a unit cube meshed as a mix (5 tets, 1 prism, 1 pyramid, 1 hex, 1 general polyhedron from a merged pair): the section area at z = 0.37 equals 1 to 1e-12 relative, and every triangle's centroid lies in its cell (half-space test). Planes through a vertex or a face: area 1 ± 1e-12, with no cell counted twice. 200 random planes through the cube: area equals the independently computed convex plane∩cube polygon area to 1e-10. 1 M-cell synthetic: measured and reported | — | M | G · W3 |
| **STU-12b** | Slices and planes work on a non-lattice polyMesh. The dataset service runs polycut in the worker pool, stores the triangles as blobs, and `addSlice` / `addPlane` draw them coloured by the cell value. `NO_STRUCTURED_GRID` is gone for these two | `datasets/service.ts`, `datasets/worker.ts`, `shared/src/viewerDataset.ts`, `viewer/controller/ViewerController.ts:696`, `viewer/layers/ImageLayer.ts` or a new `UnstructuredSliceLayer.ts` | sphere-in-box fixture: `add_layer slice {axis:'x', position:0}` succeeds, and its triangle cell ids form exactly the set of cells polycut reports. Every coloured value equals that cell's field value bitwise. The lattice path stays unchanged (existing slice tests green). Test `unstructuredSlice.test.ts` | STU-10, STU-12a | M | G · W4 |
| **STU-13** | `probe {point}` and the Inspector locate the containing cell on a polyMesh (bucket candidates plus a convex half-space test). A warped or non-convex cell answers labelled `nearest-centre`, never silently | `server/src/tools/sample.ts` (export the index), `datasets/service.ts`, `viewer/controller/ViewerController.ts:174-190`, `uiBridge.ts:287` | on the mixed fixture, 10,000 random interior points: the located cell equals the brute-force oracle for 100 % of the convex cells. Points outside the domain give `MISS`. Census row `probe` with point green on polyMesh. Test `pointLocate.test.ts` | STU-12b | S-M | G · W4 |
| **STU-14** | Iso-surfaces work on a polyMesh: least-squares point reconstruction (exact for linear fields), then marching tetrahedra on the centre-and-face-fan decomposition | `server/src/formats/polyiso.ts` (new), service and controller wiring (`ViewerController.ts:494-499`), SPEC-LIT §148 subsection (Mavriplis NTRS 20040010717; Treece et al. DOI 10.1016/S0097-8493(99)00076-X) | f = 2x − y + 0.5z on the mixed cube: reconstructed point values equal f to 1e-12. The iso at f = 0.3 has every vertex within 1e-12 of the plane, and its area equals the analytic plane∩cube area to 1e-9. Cut-cell and lattice paths unchanged. Test `polyiso.test.ts` | STU-12b | M-L | G · W5 |
| **STU-15** | Streamlines on a polyMesh: RK4 on the reconstructed velocity, with a face-neighbour cell walk that terminates at boundary faces (optional; cut last if the tranche runs long) | `server/src/formats/polystream.ts` (new), wiring; SPEC-LIT §148 (Runge 1895, DOI 10.1007/BF01446807) | uniform U on the mixed cube: the streamline is straight, endpoint error ≤ 1e-12. U = (−y, x, 0) on a 20³-cell tet mesh of [−1, 1]³: after one revolution at step h/10 the radius drift is ≤ 1e-3 relative. A seed outside the domain is refused | STU-14 | M-L | G · W5 |
| **STU-16** | The server does STL booleans fuse, cut and common on parts with Manifold (Apache-2.0), refuses an open operand as `STL-OPEN` naming stl_repair, and writes a new STL beside the input | `server/src/tools/geometryBoolean.ts` (new), the route in `server/src/http/*`, `package.json` (`manifold-3d`), `LICENSING.md` dependency table, `NOTICE`, PROVENANCE row | two unit cubes offset by (0.5, 0, 0): fuse volume 1.5, common 0.5, cut 0.5, to 1e-12 relative. 20 seeded random rigid-rotated cube pairs: V(A∪B) + V(A∩B) = V(A) + V(B) to 1e-9 relative. Every output is closed by `formats/stl.ts`'s rule (0 open, 0 non-manifold). An open operand gives `STL-OPEN` with both counts. A licence-scan test asserts that no dependency in the lock file is GPL, LGPL or AGPL. Test `geometryBoolean.test.ts` | — | M | G · W1 |
| **STU-17** | `geometry_boolean` works on an STL geometry tab, and the result opens as a new tab | `web/src/state/geometryStore.ts:308-330`, `uiBridge.ts:216-223`, `api/rest.ts` | census `geometry_boolean` green for STL. A STEP tab keeps its current route unchanged. Store test: a boolean on a two-part STL fixture returns the out path, and the tab opens it. Test `geometryStore.boolean.test.ts` | STU-16 | S | G · W4 |
| **STU-18** | G-LLM's grounding half is re-measured live under GLM-5.3-Flash after the repair round (cffe323, 960d49e). The lint is not loosened | recipe of memory `ag6-live-neutrality-recipe.md`; the result goes in the docs/15 ledger | 20 campaign explanations: 0 ungrounded **after** the repair round (the docs/15 §F G-LLM bar, unchanged), with marked `[?]` counted as ungrounded. Mock also 0. If > 0 it is reported as FAIL, and the open options go to the user (§5) | — | S + live (~20 min, so an L-run) | G · W5 |
| **STU-19** | A Playwright walk drives every gui_control type in demo mode on the sphere-in-box and Cartesian fixtures. No answer reads "does not implement it", and each answer is either applied or a named refusal from the census's allow-list | `e2e/studio-control.e2e.ts` (new) | 100 % of the types answered under 4 s. Screenshots of the walls-only F1-like view, a polyMesh slice, the Mesh dialog and the Boundary editor are reviewed by the supervisor | STU-02, STU-03, STU-04, STU-05, STU-06b, STU-07b, STU-08b, STU-09, STU-11, STU-12b, STU-13, STU-14, STU-15, STU-16, STU-17 | M | G · W6 |
| **AUT-01** | The seed-2 full scope is recorded as withdrawn before running (the binary changed after its lock), and its REDUCED page stays published. One RED fixture per G defect class that ends SURFACE-OPEN today (duplicate, flip, hole of 33-64 edges, T-junction) is built from **tuning** ids only, through `inject.py` | docs/15 ledger, `evaluate/seed2/WITHDRAWN.md` (new), `tools/geom/fixtures/g_defects/` (new), `tools/geom/selftest.py` | the four fixtures fail today with the counts recorded (for example "open 59 → 59, repaired false"). The seed-2 lock is byte-unchanged. `evaluate.py --check --scope reduced` still PASS | — | S | M · W1 |
| **AUT-02a** | stl_repair closes duplicate-triangle and flipped-patch surfaces. Exact and opposite-coincident duplicates are removed; a flipped patch is re-oriented across its boundary edges, which the `_flip_makes_same` guard (`stl_repair.py:264-285`) refuses today | `tools/geom/stl_repair.py`, selftest | on the 26 tuning G rows of these two classes: 26/26 closed after repair, with area and signed volume equal to the regenerated parent's to 1e-9 relative. All A-F tuning parents byte-identical after repair (no regression). G-CORPUS detection unchanged | AUT-01 | S-M | M · W2 |
| **AUT-02b** | stl_repair resolves T-junctions: a vertex lying on another triangle's edge, within the weld tolerance, splits that edge | `stl_repair.py`, selftest | 14 tuning t_junction rows: ≥ 13/14 closed, with each residual named in the report. Parent area to 1e-9. A-F byte-identical | AUT-02a | M | M · W3 |
| **AUT-02c** | stl_repair fills holes up to 64 edges by Liepa's minimum-weight triangulation plus refinement and fairing, and the default `--max-hole-edges` becomes 64 | `stl_repair.py`, SPEC-LIT §23.2 subsection (Liepa, DOI 10.2312/SGP/SGP03/200-206), PROVENANCE row update | 16 tuning hole rows: 16/16 closed. The filled-patch area equals the parent's removed area within 2 %. The max distance of a filled vertex from the parent surface is ≤ 0.5·h_f at the rules' max level. A ≤ 32-edge planar hole is still byte-identical to the old ear-clip result. A-F byte-identical | AUT-02a | M | M · W3 |
| **AUT-03** | Output only: `stages[snap]` reports the k = 16 worst-residual boundary points with position, residual/h_f and nearest feature class (edge, corner, smooth, open-tip), and SPEC-LIT §92.14 documents it. No mesh changes | `rust/src/automesher/snap.rs`, `driver.rs`, `rust/SPEC-LIT.md` §92.14 | `cargo test --release --lib -- automesher` green plus the new test. Every castellated and snapped golden (L1a's 7 hashes, commit 95bfea4) byte-identical. On `wing_a_L4` the worst point's residual equals `max_residual` to 1e-12. Both `--release` and `--features single` build | — | S | M · W1 |
| **AUT-04** | The A-family wings: a rule R-TE (whitelisted knobs only: trailing-edge feature level, band and attraction radius ≥ h_f/8), designed from AUT-03's locations on tuning wings | `tools/autonomy/rules.py`, `remedies.py`, README section, selftest | quick set of 12 stratified tuning A bodies (< 10 min at 6 streams): F3-clean count at K = 4 moves from the recorded rules number up to a target **fixed in the brief from AUT-03's evidence before the run**. Every R-PLANE and smooth-body config byte-identical. If the target is missed, the unit reports the residual gap as an automesher snap item for the automesh stream (contract in §2.4 step 2) and is closed as measured | AUT-03 | M | M · W4 |
| **AUT-05** | R-SNAP: on curved bodies, `/snap/iterations` and `/snap/tolerance` move so p99 leaves the (92.28) dead band | `rules.py`, README | quick set of 4 B, 4 E and 4 G tuning bodies: median p99/h_f ≤ the tuning B0-template median per family, compared against B0-template on the same bodies. Snap seconds are reported. R-PLANE configs byte-identical. The full tuning effect is measured in AUT-12's L-run | — | S-M | M · W3 |
| **AUT-06** | `costbound.py` computes, per geometry, the fewest cells an octree can have that holds the R-WIN wall level on the whole wall, with 2:1 shells to the domain and 8 layers. It reports bound / B0 cells on the seed-1 15 both-pass bodies (post-hoc on spent data, labelled as such) and on the tuning both-pass set | `tools/autonomy/costbound.py` (new), README | analytic: an on-plane commensurate cube's bound equals the hand count to the cell. Bound ≤ actual full-system cells on 100 % of tuning rows (a bound must bound). The report states whether 1.5× is attainable. **If it is not, the gate goes to the user (D-8)**; the constant is not touched | — | S | M · W1 |
| **AUT-07** | R-ECON: band distances and levels set at the minimum the (92.x) gates and R-WIN allow, so cells drop with no loss of delivery | `rules.py`, README | quick set of 12 both-pass tuning bodies: median cells drop by a measured factor, with 0 new failures, BLC_8 per body no lower, and G1-G7 unchanged (preflight and `-check`). The ratio to B0 and to AUT-06's bound is reported | AUT-06 | M | M · W4 |
| **AUT-08** | G-EXPL is checked post-hoc on the nine seed-1 bundles with today's `explain.audit` (labelled post-hoc, not a claim), and the 63 records become a regression fixture | `tools/autonomy/explain.py` (selftest fixture only), README | 9/9 campaigns audit ok. The selftest fixture with the 63 last-attempt OPT records passes, and fails if cb69bc9's terminal rule is reverted (mutation test) | — | S | M · W1 |
| **AUT-09** | G-OPT's surrogate becomes the pre-registered hurdle model with R-WIN margin features. CV runs once on the fixed folds, plus a 20 % in-tuning geometry hold-out scored once | `tools/autonomy/optimise.py`, `optimise/aml2/` (new outputs), README | CV half: fail AUC ≥ 0.75 and **BLC_8 RMSE ≤ 0.15** (unchanged). The hold-out RMSE is reported beside it. One run only; a miss is published and the optimiser stays off. Selftest: the hurdle prediction equals P×E on fixtures to 1e-12. `optimise.py --check` PASS | — | M | M · W1 |
| **AUT-10** | The prior acts only where the surrogate predicts p_fail(rules attempt 1) ≥ 0.5, and votes among same-edge-class neighbours with a non-empty winning path. It is re-gated leave-one-geometry-out with the 3-seed shuffled control under the non-vacuous criterion (D-6) | `tools/autonomy/prior.py`, `prior/aml2/`, README | selftest on the oracle fixture (2/5/4 3 3 reproduction kept). The gate code computes the D-6 criterion and writes FAIL when it is not met. The gate's meshing rounds are an L-run (AUT-12) | AUT-09 | M | M · W4 |
| **AUT-11** | Prior and optimiser enable only from a gate file whose verdict is PASS and non-vacuous, whose sha matches, and whose model sha matches. `USER_SHIP` becomes the recorded rule "gate-driven since 2026-10-06 (the user's tranche-17 order)" | `prior.py:110`, `optimise.py`, README, selftest | property tests: a FAIL, a vacuous PASS, a tampered gate file and a missing file each keep the layer disabled and record PR-DISABLED or OPT-DISABLED with the reason. Only PASS plus non-vacuous plus matching hashes enables. G-QUAL and G-DET replays unchanged on the committed rows | AUT-09, AUT-10 | S | M · W4 |
| **AUT-12** | `projection.py` is the pre-registered go/no-go for spending a fresh seed: from a tuning `rules`/`full` re-measure it projects every headline and guard as the test bar would read them | `tools/autonomy/projection.py` (new), README | selftest on the L5 bundle reproduces "no-go" with its numbers (177/420 > 23). **L-run (supervisor):** the tuning re-measure after AUT-02…AUT-11 (≈ 2 h at 6 streams, as in L5), then AUT-09's refine round and AUT-10's re-gate rounds; the results go into the ledger | AUT-02c, AUT-04, AUT-05, AUT-07, AUT-11 | S + L-run | M · W5 |
| **AUT-13** | `split.py --write-fresh N` and `evaluate.py --manifest testN` generalise from seed 2 to any N, and seed 3 is written and locked **only on a go** from AUT-12 (or on the user's explicit spend) | `corpus/split.py`, `evaluate.py`, README | `--check-fresh 3` PASS against an independent oracle. The seed-1 and seed-2 manifests and locks are byte-unchanged. The plan lock records the D-L9 G-BLC-1 targets (0.09 / 0.05) unchanged. Selftests | AUT-12 | S | M · W6 |
| **AUT-14** | The held-out evaluation on seed 3, full scope: every §F gate decided once (G-FAIL, G-BLC-0, G-BLC-1, G-QUAL, G-FID with F3e, G-COST, G-DET, G-EXPL, G-OPT decided when the optimiser ships, G-ABL). The results page is published as measured | `evaluate/seed3/full/*`, docs/15 ledger | the gates as locked; nothing is re-run. **L-run (supervisor), hours of CPU** | AUT-13 | S + L-run | M · W6 |
| **AUT-15** | Output only: lowmach and the RAS drivers write a per-patch y⁺ min/mean/max JSON beside the run whenever a k field exists (docs/15 AR-1) | `rust/src/bin/lowmach.rs` (the §29.3 block at about 2821-2825), SPEC-LIT §29.3 note | a named test on a shipped case: the JSON values equal the printed line. No field or residual changes bitwise. Both `--release` and `--features single` build. **Overlaps the validation stream**: the assembler de-duplicates | — (the solver workflow's tree being free) | S-M | S · W2 |
| **AUT-16** | G-YPLUS: the solved per-patch y⁺ against the a-priori table on 30 passing seed-3 meshes (docs/15 AR-2) | `tools/autonomy/yplus.py` (new) | solved y⁺ ≤ 1 on ≥ 90 % of the area BLC_8 counts. If it fails, BLC is renamed and down-weighted (docs/15 §F). **GPU L-run** | AUT-14, AUT-15 | M + GPU L-run | M · W6 |

**Assembler corrections:**

1. **Ids are two-digit.** **Count: 23 STU + 18 AUT = 41.** The plan's text said 22 + 17 = 39.
2. **SPEC-LIT §148** is the studio's post-processing section (STU-12a, 14, 15).
3. **AUT-15 runs in lane S (W2)** because it edits `lowmach.rs`. VAL has no duplicate y⁺ unit, so nothing needed
   de-duplicating.
4. **Supervisor runs:** STU-18 (live, about 20 min), AUT-12, AUT-14 and AUT-16 are L-runs (§7.6).
5. **STU-16's `manifold-3d`** enters `LICENSING.md` and `NOTICE` with Clipper2 (BSL-1.0). The licence-scan test is
   part of its gate.
6. **`tools/autonomy` has one source of truth, the mesh tree.** The gui tree's diverging copy is refreshed only through
   the syncs (§7.3), never edited in lane G.

### 12.7 VAL — every open or missed gate (VAL-)

| id | what becomes TRUE | files | gate (quantitative; reference; tests) | depends | eff | lane · wave |
|---|---|---|---|---|---|---|
| **VAL-01** | The five §110 answer keys can be fetched or written byte-exact, and the manifest and markers name them | `tools/keys/fetch_answer_keys.py` (new); `reference/PROVENANCE.md` (+5 rows); `src/bin/validate.rs` (5 `// answer-key:` markers at the existing `load_text` calls); `.gitignore` (the MKM paths) | `every_answer_key_is_named_in_the_manifest_and_every_row_has_its_marker` passes with count 14. The script's `--check` reproduces the pinned SHA-256s. With keys present, the three §110 gates print `answer key <id>: ... sha256` and McCaffrey's four continuity rows hold at their stated 1-3 %. On a clean tree they print `answer key <id> missing` 5 times. Sources: MKM DOI 10.1063/1.869966, TMR backstep_val.html, NBSIR 79-1910 | — (D-VAL-1) | S | M · W2 |
| **VAL-02** | McCaffrey's nine constants are confirmed against NBSIR 79-1910 Table 1, or corrected, before any 110-C run | `rust/SPEC-LIT.md` §110.4 (the disclosure sentence at :33437 replaced by the confirmation and the page cited) | Opus reading, no GLM. Every one of the nine constants is quoted with its page. A changed constant also changes VAL-01's fenced block and digest | — (none) | S | M · W1 |
| **VAL-03** | Martin & Moyce's surge-front series is a key in the tree | `reference/martin_moyce_1952/surge_front.csv` (new); `reference/PROVENANCE.md` row `martin-moyce-1952`; SPEC-LIT §145.1 (source, which series, digitised or tabulated, u_D) | The file's SHA-256 is in the manifest. The values are monotone in T per series (a transcription guard test, `martin_moyce_key_is_monotone_per_series`). Source DOI 10.1098/rsta.1952.0006 | — (D-VAL-3 (access)) | S | G · W1 |
| **VAL-04** | The T3A measured Cf(Re_x) and free-stream Tu(x) are a local key, fetched by recipe | `reference/PROVENANCE.md` row `ercoftac-t3a`; `tools/keys/fetch_answer_keys.py` (+ the ERCOFTAC entry, or manual-placement text); SPEC-LIT §144.1 | Missing-key behaviour holds, as VAL-01. With the key placed, the loader reads N rows and the Cf rows are positive and finite. Source: Roach & Brierley (1992) via the ERCOFTAC Classic Database, URL in the row | — (D-VAL-2) | S | G · W3 |
| **VAL-05** | A measured splash-threshold set is a key | `reference/riboux_gordillo_2014/critical_speed.csv`; manifest row; SPEC-LIT §146.1 | Manifest test passes. Each row's We, Re and Oh recomputed from its stated properties matches the paper's printed values to their printed digits. Source arXiv:1401.6943 / PRL 113 (2014) 024507 | — (none) | S | G · W1 |
| **VAL-06** | Gate 5's O3 search is done and recorded: either an open source that reproduces the conduction limit at low Ra becomes a key, or none is found and O1 stands | SPEC-LIT §115.6 (amended); possibly `reference/<new>/` and a manifest row | Opus reading. An accepted source must give Nu within 1 % of the exact conduction value 0.357143 at Kr = 0.1, low Ra, because that is the test §115.4 used. Candidate: Computational Continuum Mechanics 5(3) (2012), "Laminar and turbulent conjugate regimes of natural convection in a square enclosure" (journal.permsc.ru, open access) | — (D-VAL-4) | S | M · W1 |
| **VAL-10** | Gate 95-A's shortfall is attributed: the exact discrete fixed point of the affine block map is known for each (ratio, n_y ≤ 8, and 5:1 n_y = 16), with its own error and order | `src/solid/coupled.rs` (one ignored probe test plus a host helper that assembles B column by column); SPEC-LIT §142.1 (diagnosis table) | The probe `gate_95a_probe_the_exact_fixed_point_of_the_block_map` checks: on 2.5:1 the dense fixed point equals the converged loop's answer to 1e-9 relative on all 3 meshes; the dense residual is ≤ 1e-12 × scale (109.6); it prints the condition estimate, e_u and p_u per ratio, and whether the 5:1 / 10:1 n_y = 4 fixed points exist. Runtime under 5 min (LU of at most 3 840). No numerics change | — (none) | M | M · W1 |
| **VAL-11** | (if VAL-10 says the loop) A restarted-GMRES outer iteration solves the block map's fixed point as an opt-in, and Gate 95-A is measured on it | `src/solid/outer.rs` (+ `Relaxation::Gmres(m)`); `src/solid/coupled.rs`; case knob `mechanics.solver.outer` in the mechanics lowering (§96.1 table + refusal of unknown names); `src/bin/validate.rs` `check_cantilever` (second loop rows); SPEC-LIT §142.2 | Bitwise: `outer: "anderson"` (the default) reproduces every 95-A row digit for digit. On block(6) free expansion GMRES(50) reaches 8 decades in at most 20 outer iterations. Gate 95-A (unchanged bars: converge on every mesh, p_u ≥ 1.9, p_sigma ≥ 0.9, tip ≤ 5 %) is printed per loop. Refs: Saad & Schultz 1986, Walker & Ni 2011. Captured replay bitwise per §81 for a fixed-iteration solve | VAL-10 | M | M · W4 |
| **VAL-12** | (if VAL-10 says pre-asymptotic; user D-VAL-6) Gate 95-A carries a fourth mesh n_y = 32 per ratio, and the study takes the finest three | `src/bin/validate.rs` `check_cantilever`; host test pinning the four meshes `cantilever_gate_runs_four_meshes_per_ratio`; SPEC-LIT §109.6/109.7 amended (table gains rows) | Bars unchanged. The coarsest mesh is still printed, and if it fails it fails by name. Scope runtime under 10 min (to be measured, since 10:1 n_y = 16 used 43 k BiCGStab iterations); if over, this becomes an R | VAL-10 (VAL-10, D-VAL-6) | S | M · W3 |
| **VAL-13** | (only if VAL-10/12 show an asymptotic order near 1; user D-VAL-7) The tangential face-gradient rows are implicit through a vertex stencil, a third block format, as an opt-in | `src/solid/coupled_wide.rs` (new), `cuda/solid_wide.cu` (new), SPEC-LIT §142.3 | Patch test on graded and jittered meshes ≤ 1e-12 × scale. The 95-A order with it is printed. Default bitwise unchanged. Split into 3 units (format and host twin; kernels; loop and gate) if taken | VAL-12 (VAL-12, D-VAL-7) | L (3×M) | M · W6 |
| **VAL-14** | (user D-VAL-6) Gate 95-G's ring carries nr = 96, and its order row is read on the finest pair | `src/solid/restated.rs` (`lame_ring` levels); `src/bin/validate.rs` `check_boundary_point_fit`; SPEC-LIT §95.11 amended | Bars unchanged: point value within 1 % of 5p/3 (S95.28) on the finest; observed order on the finest pair ≥ 0.9; the §94 study over the finest three; nr = 12 printed. Test `ring_gate_runs_four_meshes`. Runtime under 10 min | — (D-VAL-6) | S | M · W3 |
| **VAL-20** | An in-tree host 1-D fully developed RANS oracle reproduces §114's SST pipe table, and Launder-Sharma, SA and k-omega 1988 run in it | `src/vv/rans1d.rs` (new, host only, f64); `src/vv.rs` (mod line); SPEC-LIT §143.1 | Tests: `rans1d_sst_pipe_reproduces_spec_lit_114_table_3` (f 0.027258 and defect 3.6129 at 640 cells, Re_tau 576.69, to 1e-4 relative; U_b 6.02290 on the L1_lo cell set to 1e-5); `rans1d_laminar_pipe_is_poiseuille` (round-off); `rans1d_sa_log_layer_identity` (§56.4). All attributed f32-ignored where they bound f64. Runtime under 1 min | — (none) | M | M · W1 |
| **VAL-21** | Wilcox 2006 k-omega runs in the oracle, and the five models' TG0 bands are a measured table | `src/vv/rans1d.rs` (+ the 2006 terms); SPEC-LIT §143.2 (table, model × Re_tau: f against Prandtl, f against Blasius, log-law deviation, defect against 4.07) | Coefficients pinned against the TMR page to the digit (`wilcox2006_coefficients_are_the_tmr_page`). Cross-diffusion is exactly 0 where ∇k·∇ω ≤ 0. f_β = 1 for simple shear, bitwise. The table is printed by an ignored measurement test. Source tmbwg.github.io/turbmodels/wilcox.html; DOI 10.2514/1.36541 cited, not read if paywalled | VAL-20 | S | M · W1 |
| **VAL-22a** | (if D-VAL-5 picks Wilcox 2006) The model exists as a host twin with its coefficients and SPEC-LIT section | `src/models/k_omega_2006.rs` (new; host twin + coeffs + source split); SPEC-LIT §143.3 | Host twin equals the oracle's source terms cell for cell to 1e-14 on a 1-D column embedded in a 3-D block. Patankar split Sp ≥ 0 at every state | VAL-21 (VAL-21, D-VAL-5) | M | M · W4 |
| **VAL-22b** | (same) The Wilcox 2006 sources run on the device and capture | `cuda/k_omega_2006.cu` (new); `src/models/k_omega_2006.rs`; `build.rs` (kernel list); capture registry row | Device against host twin ≤ 1e-13 relative on uniform, jittered and graded blocks. Two identical runs give identical bits. Capture and replay bitwise (§81). `cargo build --features single` compiles the kernel at `ofscalar = float` | VAL-22a | M | M · W4 |
| **VAL-22c** | (same) A case can name `kOmega2006`, and the coupled drivers run it | `src/models/registry.rs` (`RasModel::KOmega2006`, the menu at :416), `src/models/coupled.rs` (impl), refusal list; SPEC-LIT §143.4 (what a case says) and the §58-style pair tests | Pair tests: `kOmega2006` against `kOmega` differ only where σ_d or the limiter act. A case that does not name it is bitwise unchanged (every §6.2/§6.3 output). The menu test lists it | VAL-22b | M | M · W4 |
| **VAL-23** | The turbulent pipe is a gate of `ofgpu-validate` (Gate 131-P), replayed from recorded `ofgpu-lowmach` runs for SST (omega tolerance 1e-12, the case-side O1) and for the D-VAL-5 model | `src/bin/validate_gates/pipe.rs` (new); `cases/pipeTurb{576,2358}.jsonc` (new); SPEC-LIT §143.5 | TG0's four bands unchanged: Prandtl ±5 %, Blasius ±5 %, log law ±1.0 for 30 ≤ y+ ≤ 0.2 Re_tau, defect within 10 % of Schlichting's 4.07. Verdict word MISSES (Nikuradse's measured pipes, via Schlichting). The §94 study over 3 radial levels. R: 2 models × 3 levels × 2 Re_tau, 8 000 iterations each | BC-04, BC-05, VAL-21, VAL-22c (VAL-21; BC stream's true wedge (or a 3-D quarter pipe if absent); VAL-22c if chosen) | M + R | M · W5 |
| **VAL-30** | blockgen builds a flat-plate mesh: symmetry upstream of the LE, wall downstream, graded to the wall and to the LE | `src/blockgen.rs` (`CaseKind::FlatPlate` + spec); `src/bin/generate_mesh.rs` (`plate` subcommand); `cases/flatPlateLaminar.jsonc` (new) | Host tests: `flat_plate_floor_is_split_at_the_leading_edge` (patch face ranges exact); `flat_plate_le_lies_on_a_face_boundary`; grading ratios to 1e-12; first-cell y+ estimate printed. The `PatchWindow` constraint (blockgen.rs:773-804) is respected. f32 build | — (none) | M | M · W3 |
| **VAL-31** | The flat-plate harness reproduces Blasius (Gate 132-L, live) | `src/bin/validate_gates/blasius_plate.rs` (new); SPEC-LIT §144.2 | Cf(x) against 0.664/sqrt(Re_x) (Blasius 1908; the crate's §88.7 solution): within 3 % for every wall face with 2e4 ≤ Re_x ≤ Re_L, on the finest of 3 meshes, plus the §94 study of Cf at Re_x = 1e5. u/U_inf against η profile within 2 % at one station. Verdict word MISSES if it misses (an exact solution). Runtime under 10 min (laminar, about 20 k cells at most) | VAL-30 | M | M · W3 |
| **VAL-32** | The T3A case runs on the harness, and leg 1 is measured in 2-D (R) | `cases/t3aPlate{LM,Gamma}.jsonc` (new); SPEC-LIT §144.3 | The solver's own Tu at the LE against the TMR's 3.300 %, and against the 1-D closed form 3.3530 %, to 0.5 %. R: 2 models × 3 levels | VAL-30, VAL-31 | S + R | M · W4 |
| **VAL-33** | Gates 88-T and 90-T gain leg 4: the 2-D Cf(Re_x) against Roach & Brierley's measurement | `src/bin/validate_gates/t3a.rs` (new; replay slots + verdict); SPEC-LIT §88.10/§90.10 amended (leg 4) + §144.4 | D4's bands, fixed in the brief before any run: onset ±15 % in Re_x; turbulent Cf ±10 % (+ u_D); laminar part within 10 % of Blasius. Verdict word MISSES (measurement). Legs 2 stay OPEN as written. The replay shape matches §110.1 | VAL-04, VAL-32 | M | M · W5 |
| **VAL-34** | The TMR ZPG flat plate (Re_L 5e6) runs with SA and SST, and is held against Karman-Schoenherr and Coles (Gate 132-Z) | `cases/zpgPlate{SA,SST}.jsonc` (new); `src/bin/validate_gates/zpg.rs` (new); `reference/PROVENANCE.md` rows `tmr-ks-cf`, `tmr-coles-uplus` (the TMR data files, US-government work, tracked); SPEC-LIT §56.11 amended + §144.5 | Cf within ±3 % of K-S over 4000 ≤ Re_theta ≤ 13000. u+ within 1.0 of Coles at Re_theta = 1e4 for 30 ≤ y+ ≤ 300. Verdict word OPEN (correlation). The TMR's C_d is NOT used (§22). R: 2 models × 3 levels | VAL-30, VAL-31 | M + R | M · W4 |
| **VAL-35** | (optional) T3B and T3A- are claimed with a per-case inflow ω fitted to ERCOFTAC's measured Tu(x) decay | `cases/t3{b,am}Plate.jsonc`; `t3a.rs` | The fitted decay reproduces the measured Tu(x) within 3 %, as a stated input check. Onset ordering and the Re_x spread are held against the data, which retires §88.10's 51.9× construction artefact. The bands are the same as VAL-33 | VAL-33 | S + R | M · W5 |
| **VAL-40** | §32.4's three verdicts are re-recorded at the current binary, and sibling rows at `PrtModel KaysCrawford` are printed beside the default ones | `src/bin/validate.rs` (replay slots at :3059/:3101/:11370 refreshed; 3 sibling `enter_gate` rows); SPEC-LIT §32.4 table + §69.5 sentence | The default rows are recomputed from the new records and keep their definitions and bands (Gnielinski ±10 %, f named every time). The sibling rows are named "at PrtModel KaysCrawford (opt-in, §37)". A pass of a sibling never changes the default row's word. R: 4 runs × 40 000 iterations | — (D-VAL-8 (default question, asked, not assumed)) | S + R | M · W4 |
| **VAL-41** | Why the wall-function leg's realised f is −28 % of the pipe f is measured | ignored probe test in `src/bin/lowmach.rs`; SPEC-LIT §32.5.3 addendum | Diagnosis only, §114/§115 shape. Candidates ruled in or out by measurement: the u_tau form (ρu_τ² against viscous), first-cell y+ = 58 against the log-law constants, the bounded correction. No numerics change. Runtime under 10 min per probe | VAL-40 | M | S · W5 |
| **VAL-50** | The §110 recorded runs are priced: 200 iterations of each case at each level, timed | none (R only); timing table in SPEC-LIT §110.5 | Seconds per iteration per level printed. Projected wall time per run. Peak device memory against §111's 4.72 M-cell cliff (the 110-C fine level is 2.79 M) | VAL-01 | S (R) | M · W3 |
| **VAL-51** | Gate 110-A has a verdict at three Re_tau | `src/bin/validate.rs` replay records for :12426; SPEC-LIT §110.5 results rows | B1/B2/B3 exactly as §110.2. The study through `grid_study`. Verdict as printed. R: 9 runs × 40 000 iterations | VAL-01, VAL-50 | S + R | M · W4 |
| **VAL-52** | Gate 110-B has a verdict | replay records for :12447; §110.5 | x_r/H against 6.26 ± 0.10, the stated band, with the inflow δ99 as u_input. R: 3 runs × 6 000 iterations (fine ≈ 1 M cells) | VAL-01, VAL-50 | S + R | M · W4 |
| **VAL-53** | Gate 110-C has a verdict (hot gas, no radiation) | replay records for :12468; §110.5 | P1-P4 as §110.4. Expected OPEN on P1/P2 high by 16-41 % / 8-19 %, stated in advance. P3 is the closable leg. R: 3 runs × 4 000 iterations | VAL-01, VAL-02, VAL-50 | S + R | M · W4 |
| **VAL-54** | Gate 110-C gains leg 2: the same plume with participating-media radiation from the CHEM/RAD stream | `cases/plumeMcCaffreyRad.jsonc`; replay rows; §110.4 addendum | The same P1-P4 bands, unchanged. The radiant fraction the run computes is printed beside McCaffrey's 20-40 %. R: 3 runs | CHR-23, VAL-53 (VAL-53; CHEM/RAD stream (radiation in `ofgpu-lowmach`)) | S + R | M · W6 |
| **VAL-60** | Martin & Moyce is a live gate of `ofgpu-validate` (Gate 133-A) | `src/bin/validate_gates/martin_moyce.rs` (new; drives the `ofgpu::vof` library as `ofgpu-vof -surge` does); SPEC-LIT §145.2; vof.rs:48-52 doc amended (the table now exists, in reference/) | Z(T) against the key at every keyed T in the stated window, within the band fixed in the brief from VAL-03's u_D (for example max(u_D, 0.1 Z_exp)). The Ritter bound holds. Phase volume conserved to 1e-12. The §94 study on Z at T = 2 over 3 meshes. Verdict word MISSES. Runtime under 10 min | F32-10, VAL-03 | M | G · W4 |
| **VAL-70** | Gate 78-D gains a measured leg: each threshold criterion against Riboux & Gordillo's critical speeds | `src/bin/validate_gates/splash.rs` (new); SPEC-LIT §78.10 addendum + §146.2 | Per liquid, the criterion's predicted threshold speed against the measured one. The band is stated before the run as ±20 % per row (or the paper's own error bar where printed), with at least 6 of 8 rows inside. MISSES per criterion otherwise. The 4.78 row unchanged (OPEN). Host only, under 1 min | VAL-05 | S | G · W4 |
| **VAL-71** | An injector can draw droplet diameters from a Rosin-Rammler distribution, deterministically | `src/parcels/inject.rs` (or the file §66.8 names); `cuda/parcels.cu` if sampling is on the device; SPEC-LIT §147.1 | Sampled CDF against the closed form: KS distance ≤ 1.36/sqrt(N) at N = 1e5. Same seed, same bits. Mass median diameter within 0.5 %. Default (monodisperse) bitwise unchanged. f32 build | — (none) | M | S · W4 |
| **VAL-72a** | Two-way parcels run inside `ofgpu-lowmach`'s pressure-coupled loop | `src/bin/lowmach.rs` (+ parcel step at §68.7's order point); SPEC-LIT §147.2 | §68.9's conservation rows hold in the coupled driver: momentum given equals momentum gained to 1e-12 relative. No parcels means bitwise unchanged (§68.10 Gate 68-B re-run in the driver). Capture stance stated | VAL-71 (INPUT stream's parcel case block (or VAL builds the controls in code as §68 does); VAL-71) | M | merged into CHR-26 |
| **VAL-72b** | Gate 68-C gains its coupled leg: Theobald test 3 on a stream-resolving mesh, then all 90 (R) | `src/bin/validate_gates/theobald_coupled.rs` (new; replay); `cases/theobald*.jsonc` generator; SPEC-LIT §68.12 addendum + §147.3 | Bar unchanged: mean pred/meas within ±10 %, 2σ scatter ≤ 30 %, over all 90. Still-air leg kept as written (MISSES). Test 3's costing run first; the overnight batch only if its projected total fits a night | CHR-26, CHR-26 (VAL-72a; adapt (§75) or MESH stream's refinement box) | M + R | S · W5 |
| **VAL-80** | Gate 5's verdict is recorded per D-VAL-4 | `src/bin/validate.rs` (:6286 headline cites §115; or a new row against VAL-06's key); SPEC-LIT §60.5 | O1: the definition, bar and every number unchanged; the headline adds "reference-limited, SPEC-LIT §115", and the word stays MISSES. O3: a new row against the new key, live, with the same 3 % bar | VAL-06 (VAL-06, D-VAL-4) | S | M · W4 |
| **VAL-81** | Every new VAL test and gate has a stated f32 posture, and each new scope has run under `--features single` | SPEC-LIT §112.3/§112.4 (rows for each new scope); test attributes | `cargo test --features single` on each new scope. §112.4 gains one row per new gate (holds / reason). No f32 tolerance invented (§112.2) | — (all code units) | S | S · W6 |
| **VAL-82** | The miss/open lists, §110.5's table and the guidebook state what the full run printed | `README.en.md` :172-181, `README.md` (ko), `docs/GUIDEBOOK*.md` §11, SPEC-LIT §110.5 | Generated from one full `ofgpu-validate` run (R, deferred batch). The README counts equal the registry's printed counts. Citation audit (§80) and provenance audit pass | CMP-28, CHR-27, VAL-81, F32-18 (all) | S + R | S · W6 |

**Assembler corrections:**

1. **Count: 30 firm + 8 conditional rows.** VAL-13 is 3 units if it is taken. The plan's text said 31 firm.
   **SPEC-LIT:** §130-§135 became **§142-§147.**
2. **VAL-72a is merged into CHR-26** (the same scope). VAL-72b depends on CHR-26 and VAL-71.
3. **The test-attribute convention is replaced** by §5.1.3: no new single-ignore. The 1-D oracle stays host f64 and
   says so.
4. **D10 (`validate_gates/`) applies to every stream.** VAL-54's leg-2 replay rows also go into a `validate_gates/`
   file, because `validate.rs` is S-owned in W6.
5. **Placement:**
   - VAL-11 runs in M W4, when `io/case_cht.rs`, which holds `lower_mechanics`, is lent to M.
   - The other in-place `validate.rs` edits (VAL-01, 12, 14, 40, 51-53, 80) finish by W4.
   - VAL-03/05/60/70 run in lane G.
6. **VAL-23 depends on BC-04/05 explicitly.** The 3-D quarter pipe is used only if the wedge fails, and that is the
   user's call. VAL-60 depends on F32-10 (local origin).
7. **VAL-50's pricing (W3) gates every R batch** (§7.6).

## Appendix A — id renames

Every other id is unchanged. CMP, F32, BC and VAL ids were already two-digit; CHR keeps `NN[a-b]`.

| old | new |
|---|---|
| CHR-S1 | CHR-00a |
| CHR-S2 | CHR-00b |
| CHR-A1 | CHR-00c |
| CHR-A2 | CHR-00d |
| PAR-1 | PAR-01 |
| PAR-2 | PAR-02 |
| PAR-3 | PAR-03 |
| PAR-4a | PAR-04a |
| PAR-4b | PAR-04b |
| PAR-5 | PAR-05 |
| PAR-6 | PAR-06 |
| PAR-7 | PAR-07 |
| PAR-8 | PAR-08 |
| PAR-9 | PAR-09 |
| PAR-G1 | AMG-01 |
| PAR-G2 | AMG-02 |
| PAR-G3a | AMG-03a |
| PAR-G3b | AMG-03b |
| PAR-G4a | AMG-04a |
| PAR-G4b | AMG-04b |
| PAR-G5 | AMG-05 |
| PAR-G6a | AMG-06a |
| PAR-G4c | AMG-04c |
| PAR-G6b | AMG-06b |
| PAR-G7a | AMG-07a |
| PAR-G7b | AMG-07b |
| PAR-X1 | AMX-01 |
| PAR-X2 | AMX-02 |
| PAR-U1 | PAR-17 |
| TET-0 | TET-00 |
| TET-1 | TET-01 |
| TET-2 | TET-02 |
| TET-3a | TET-03a |
| TET-3b | TET-03b |
| TET-4a | TET-04a |
| TET-4b | TET-04b |
| TET-5 | TET-05 |
| TET-6a | TET-06a |
| TET-6b | TET-06b |
| TET-7a | TET-07a |
| TET-7b | TET-07b |
| TET-8 | TET-08 |
| TET-9 | TET-09 |
| VIO-0 | VIO-00 |
| VIO-1 | VIO-01 |
| VIO-2 | VIO-02 |
| VIO-3 | VIO-03 |
| VIO-4 | VIO-04 |
| VIO-5a | VIO-05a |
| VIO-5b | VIO-05b |
| VIO-6 | VIO-06 |
| VIO-7 | VIO-07 |
| VIO-8 | VIO-08 |
| VIO-9 | VIO-09 |
| STU-1 | STU-01 |
| STU-2 | STU-02 |
| STU-3 | STU-03 |
| STU-4 | STU-04 |
| STU-5 | STU-05 |
| STU-6a | STU-06a |
| STU-6b | STU-06b |
| STU-7a | STU-07a |
| STU-7b | STU-07b |
| STU-8a | STU-08a |
| STU-8b | STU-08b |
| STU-9 | STU-09 |
| AUT-1 | AUT-01 |
| AUT-2a | AUT-02a |
| AUT-2b | AUT-02b |
| AUT-2c | AUT-02c |
| AUT-3 | AUT-03 |
| AUT-4 | AUT-04 |
| AUT-5 | AUT-05 |
| AUT-6 | AUT-06 |
| AUT-7 | AUT-07 |
| AUT-8 | AUT-08 |
| AUT-9 | AUT-09 |

## Appendix B — files

- The stream plans: `plan17/{COMP,CHEMRAD,PAR,PREC,MESHIO,STUDIO,VAL}.md`.
- The scheduler and its checks, which can be re-run after any change to the plan: `plan17/asm/assemble.py` (dependency
  check and lane hours), `filecheck.py` (same-wave file ownership), `cp.py` (critical paths) and `waves.py` (the §7.5
  table).
