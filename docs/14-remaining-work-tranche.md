# 14 — The remaining backend work, cut into units a 256k coder can carry

**Status:** adopted 2026-09-18. This document does not restate the plans it draws from; it names, orders and
scopes the units that remain from `docs/10` (Part II: ALE, FSI, §98, §100), `docs/11` (the GPU core), `docs/13`
(the data-centre axis) and `docs/2026-09-18-f1-aero-issues.md` (what the F1 session left), and says which of
them are NOT cut and why. Coded by GLM-5.3-Flash one unit at a time from a brief; every unit reviewed,
verified and committed by an Opus 5 supervisor; no GPL/LGPL/AGPL source consulted; existing solver numerics
never changed to make a gate pass.

## A. What is not cut, and why

| item | why it waits |
|---|---|
| D3 (`docs/13`): the data-centre convergence / divergence stop rule | changes what a run reports — **the user's decision**, not yet given |
| non-orthogonal correctors on the pressure equation (F1 §3) | a change to the solver's numerics — **the user's decision**; the F1 session's 79° cut cells are a mesh problem first |
| §106 FSI (S15–S18) and everything of WF-C (§101–§104, §107 overset, §108 VOF mesh) | §106 waits on §109's cantilever (`docs/10` §J); WF-C waits on WF-B's review. Their briefs are written when §109 lands |
| multi-GPU, a real algebraic multigrid, f16/bf16 | `docs/11` stage 4: deferred by design and named so nobody looks for them |
| merging the F1 session's two commits (`origin/feat/ontology` 2f9e60d, e59502c) to main | they carry two 11.8 MB STLs of a downloaded car model; redistribution terms unverified — the user decides |

## B. Three chains, three trees

One tree per chain so units never collide; a chain is sequential; the Rust chain owns the one GPU.

| chain | tree | branch (from) | what runs there |
|---|---|---|---|
| **R** (Rust core, solver, Part II) | `Iteration-CFD-solver` | `feat/core-2` (origin/main 573fa75) | every unit that builds `rust/`; GPU tests one process at a time |
| **M** (mesh tools, Python + automesher) | `Iteration-CFD-mesh` | `feat/mesh-2` (origin/main 573fa75) | `tools/mesh`, `tools/geom`, `rust/src/automesher` (CPU tests only) |
| **G** (TypeScript: data-centre stage 2, studio screen) | `Iteration-CFD-gui` | `feat/gui-2` (origin/feat/ontology e59502c — the F1 session's `uiBridge.ts` is there and not on main) | `gui/` only; test servers on `CFD_PORT=8799`, vite 5183+; never 8787 / 5180 |

`docs/11` S1's uncommitted work (the `ofgpu-regions` binary and `ChtCase.mesh: {regions}` lowering, left in
the solver worktree on 2026-09-14) is `stash@{0}` of that worktree; K1 adopts it.

## C. Chain R — in this order

| id | title (what is true when it is done) | from | gate / proof |
|---|---|---|---|
| **K2** | `reference/` exists with a `PROVENANCE.md` manifest (file, source, DOI/URL, licence, sha256); `validate.rs` reads a key from a file where one exists; the four source files that say "vendored" say the truth; a gate whose key is missing is `OPEN` by name, never passed | docs/11 S2 | a test that lists every literal answer key and its file; `ofgpu-validate` prints the key's sha256 beside the verdict |
| **K3** | the lid-driven cavity runs un-ignored against Ghia's table in `reference/`, through §94's uncertainty | docs/11 S3 | the Ghia section of `ofgpu-validate`, `PASS` with an uncertainty |
| **K1** | the region layout reaches the solver: `ofgpu-regions split/check/list`, `ChtCase.mesh: {regions: path}`, R1–R8 refusals, Gate 97-B bit-for-bit against the concatenated block run | docs/11 S1 (adopt `stash@{0}`) | Gate 97-B registered and green |
| **R1** | a run never overwrites its initial fields: the final state goes to a time directory (or `-restartWrite`'s), `0/` stays what the case shipped, and a restart from a written snapshot reproduces the continued run bitwise | F1 §4 | pair test: run 200, restart from 100 + run 100, bitwise equal |
| **R2** | a run that stops before its iteration count says why: exit code, the last line names the reason (NaN, refusal, signal, budget); `-restartFrom` and `-writeInterval` are exercised by a test that observes the snapshot directories | F1 §4 | test over a 30-iteration case with `-writeInterval 10`: three directories, named |
| **K9** | `precon.cu` has its host twin: scatter-shaped DIC/DILU factorisation and sweep in `reference.rs`, device vs host to 1e-12 on three meshes | docs/11 S9 | the three-mesh test |
| **K10** | `pressure/fft.rs`, `pressure/mod.rs`, `simple.rs` each carry a capture gate or a stated refusal; `UNGATED_CEILING` has headroom again | docs/11 S10 | `capture::registry` tests; the ceiling test |
| **K4** | every enum a case can name is audited: a value silently substituted today is either implemented or refused by name, with a §13.4.1 pair test each (`GAMG` is already refused — the audit covers the rest) | docs/11 S4 | one pair test per substituted value; the audit table in SPEC-LIT |
| **R3** | `-output nvdb` on a non-Cartesian mesh is a visible warning naming the reason and the alternatives; `ofgpu-generate-mesh`'s usage documents per-axis `n` (`big dir 128 52 41`) | F1 §5, §3 | usage text test; the warning in the log with a named test |
| **R4** | the viscous drag is integrated directly on wall faces (τ_w = μ_eff ∂u/∂n on the face, summed over cut faces) beside the pressure drag, and the report prints both and the flat-plate estimate for comparison | F1 task 4 | Poiseuille/Couette channel: wall shear to 1 % of the closed form |
| **K5** | the 4.72 M-cell memory cliff is diagnosed (mem_info after every allocation at 4.72 M and 4.80 M, diffed), fixed, and gated by a memory-model test that asserts bytes per cell stay linear; a "cells that fit in N GB" table the test keeps honest | docs/11 S5 | the memory-model test; the table |
| **K6** | f32 builds, is tested and is honest: the dead `1e-300` floors are fixed, `--features single` is a second test invocation of the house command, and what changes at f32 is published (holds / loosens / fails) | docs/11 S6 | both invocations green; the table |
| **K7** | the hot path stops draining: `check_interval` gets a measured default, the four one-thread kernels fold into neighbours, the per-iteration host flag read leaves the graph-captured path; same case, same iterations, bitwise same answer, wall-clock difference published | docs/11 S7 | bitwise test + the number |
| **K8** | §81.12's graph-capture table is re-taken on a named machine with contention stated | docs/11 S8 | the section says which machine |
| **K11** | the published fluid gates §10 promises: Moser/Kim/Mansour channel DNS, Driver & Seegmiller, the McCaffrey plume, keys in `reference/`, verdicts in §32.4's band discipline | docs/11 S11 | three `ofgpu-validate` sections |
| **B1** | §109a: a second matrix format beside the one-entry-per-face LDU — a 3×3 coefficient per face for the three displacement components in one matrix (Cardiff, Tuković, Jasak & Ivanković 2016, DOI 10.1016/j.compstruc.2016.07.004): storage, assembly of the block-coupled solid operator, host reference, device kernels diffed to 1e-12 | docs/10 §J | assembly test on the patch test and the cube (matches the segregated operator's residual to round-off) |
| **B2** | §109b: the block-coupled solve (block-Jacobi / block-ILU preconditioned Krylov), the segregated loop replaced by it when `mechanics.coupled: true`; **Gate 95-A** (the end-loaded cantilever at observed order through §94) earned; the slenderness refusal lifted only where the gate holds | docs/10 §J, docs/09 §95 | Gate 95-A at 2.5:1, 5:1, 10:1 with uncertainty; 95-B/C/F unchanged bitwise in segregated mode |
| **A1** | §105a ALE: resident geometry recompute in place (`GpuGeometry::recompute_in_place`), swept-volume kernel per face, volume history `v0/v00`, ALE ddt kernels; Gate 105-A space conservation to 1e-12, uniform flow uniform to round-off over 100 steps | docs/10 S12 | Gate 105-A |
| **A2** | §105b: `phi_rel` buffer routed through every convective consumer, moving-wall `ref_value = u_mesh,f`, IDW smoother (Luke, Collins & Blades 2012), the `motion` case block; Gate 105-B piston with §94 time order; the static suite bitwise | docs/10 S13 | Gate 105-B; static suite bitwise |
| **A3** | §105c: Turek–Hron CFD3 on the M5 mesh with prescribed interior wobble within 0.5 % of static; CFD1/CFD2 drag/lift within 2 % with GCI | docs/10 S14 | Gate 105-C |
| **H1–H2** | §98: a face that exchanges heat with something not meshed (the two units of Part I, numbers shifted) | docs/09 §98 | its gates as written there |
| **P1–P2** | §100: `k(T)`, `cp(T)`, `mu(T)`, `alpha(T)`, `E(T)`, `q'''(x,t,T)` (the two units of Part I) | docs/09 §100 | its gates as written there |
| **D4** | the free-cooling sweep: `supplyTemperatureSweep`, N solves, the highest `T_supply` with `RCI_HI = 100` reported as `free_cooling_ceiling` | docs/13 D4 | the sweep on the shipped case, monotone, the ceiling printed |
| **D5** | per-patch y+ and Gr/Re² for the data-centre driver, as `ModelCaveat`s a report carries | docs/13 D5 | caveats on the shipped case with numbers |

## D. Chain M

| id | title | from |
|---|---|---|
| **T1** | the mesh tools stop hiding things: `variant_from_checkpoint.py` finds the checkpoint under the case's real `out_dir`; `run_step_mesh.cmd` keeps stderr in `run.log`; `step_mesh.py` refuses a STEP whose largest solid is not a fluid volume (no terrain floor) by name instead of meshing a box | 2026-09-17 site work |
| **T2** | an STL is repaired to watertight where it can be (weld, small-hole fill, orientation), with a report of what could not be; `-permissive` parity voting becomes the exception | F1 task 2 |
| **T3** | the F1 pipeline scripts that exist locally (`cases/f1-c42/drag_post.py`, `cases/f1-fast/fast_patch.py`) move under `tools/aero/` with the house header, a usage line and a self-test on a synthetic case | F1 task 5 |
| **M6** | automesher regions per `facts-automesher` §8.1 A–K: castellate-and-snap one mesh, split into region meshes after snap, layers per region, the region layout written | docs/10 M6 |

## E. Chain G

| id | title | from |
|---|---|---|
| **C7** | the logical-form solver over L1+L2 (`Retrieval/Sort/Math/Deduce/Output`; `Math` binds `rci_hi/rti/shi_rhi`) | docs/13 C7 |
| **E1** | the studio edits geometry on screen: `geometry_part`, `transform`, `boolean`, `save` are real `ui.command`s applied through the geometry server, no longer UNSUPPORTED | F1 task 1 |
| **E2** | `set_tool`, `set_projection`, `probe`, `post_screenshot` answered on screen | F1 §1 |
| **E3** | `split_view` and `compare_run` exist | F1 §1 |
| **E4** | multi-session honesty: the GPU badge shows shared compute processes; the server warns when a code edit will restart it under a running turn | F1 task 7 |

## F. The rules every unit runs under

The brief is binding and is the coder's whole world (`units/*/_TEMPLATE.md` shape: reading list with line
ranges, requirements table, owned files, verbatim contracts, Verify block, report). The supervisor reads
the brief, its `<id>-findings.md` and the chain's `CONTRACT.md`, corrects the brief against the tree,
dispatches the wrapper ONCE in the FOREGROUND, reads every changed file, runs Verify plus the house checks
(`xref`, `provenance_audit`, `capture::registry` for Rust; `selftest.py` for mesh; typecheck + tests for
TypeScript), stages only owned files, and commits in the house style. A unit whose GLM run finished but
whose supervisor died is adopted, never re-dispatched. Never push; never create or remove a worktree with
`gui/node_modules` junctions; kill by PID only; ports 8787 and 5180 are the user's.
