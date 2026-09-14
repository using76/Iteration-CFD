# 11 — The GPU core: what it does today, what it is missing, and how it will be proved

**Status:** adopted 2026-09-14. Part III of the solver programme (Part I `docs/09`, Part II `docs/10`).
Planned and reviewed by Opus 5; coded by GLM-5.3-Flash one unit at a time; every unit verified and committed
by an Opus 5 supervisor. No GPL/LGPL/AGPL source is consulted at any point.

Everything below was measured on this machine on 2026-09-14 (one RTX 5070 Ti, 16 GB, driver 596.49,
sm_120, nvcc 13.3, f64) and the raw logs are kept beside the fact sheets
(`facts-core-inventory.md`, `facts-does-it-run.md`, `facts-answer-keys.md`, `facts-gpu-performance.md`).

## A. What actually runs today

The core was exercised end to end, not just unit-tested. The headline is that it works.

| what | evidence | number |
|---|---|---|
| device and kernels | `ofgpu-probe` | `max |gpu − cpu| = 0.000e0`, bitwise identical |
| the whole gate suite | `ofgpu-validate` | **880/880 checks passed**, 835 live + 45 replayed, 35.7 min, exit 0 |
| conjugate + thermo-elastic | `ofgpu-cht dieStack.cht.jsonc` | 4 regions, 300 interface pairs, converged, 1.82 s; peak von Mises 2.68e6 / 2.36e8 / 6.18e8 Pa |
| the new gates | same run | 95-D (3 meshes), 95-E (bimetal, both bond treatments), 97-A (imported region, bit-for-bit) all `ok` |
| an independent check nobody had done | bimetal tip deflection parsed out of the VTU | −2.1291e-5 m against Timoshenko's −2.0728e-5, **2.71 %** |
| the output a user opens | VTU parser written for this audit | shared points (3298 for 1536 cells), σ symmetric to 0.0, von Mises consistent to 2.3e-16, no non-finite value |
| the fluid path after the merge | `ofgpu-k-epsilon cases/channel` | converged in 25 iterations, 0.172 s, 3.48 Mcell-iter/s |
| the Python toolchain | four selftests | all `SELFTEST PASS` (16.2 s, 13.4 s, 13.0 s, 74.4 s) |
| determinism | every run repeated | peak von Mises, iteration counts and residual lines bitwise identical |

Two things are wired but unproven, and both are the same unfinished unit: **`ofgpu-regions` does not
exist** (18 `[[bin]]` entries, none named `regions`) and **`ofgpu-cht` cannot read a `regions.json`** —
`ChtPolyMeshRef` takes one `polyMesh` path per region. The library half landed (commit `1c6f8bb`); the
binary and the case-format half sit uncommitted and unverified in the solver worktree. That is item S1
below and it comes first.

## B. What the measurement found that nobody knew

1. **A memory cliff at 4.72 M cells.** Between 4,720,000 and 4,800,000 cells the k-ε model's device
   footprint steps by ~6.4 GB (498 → 1870 B/cell), deterministically, four times in a row. It cuts the
   real ceiling on this card from ~14 M cells to ~6.0–6.5 M, and 8 M cells thrashes (111.6 → 480.1 ms per
   iteration for 1.33× the cells). Nothing in the source asks for 6 GB; the async memory pool is the only
   mechanism in the stack that can do that without a line of code.
2. **There is no answer-key store.** `reference/` does not exist in any tree, is gitignored on purpose,
   and is absent from this machine — yet ten source files and two documents call it "vendored in this
   repository". Every published number a gate uses is a hand-typed Rust literal; the gate binary carries
   3 DOIs against SPEC-LIT's 112. The one published fluid benchmark that is implemented (Ghia) is
   `#[ignore]`d, and three more that §10's own table promises (Moser/Kim/Mansour, Driver & Seegmiller,
   McCaffrey) appear nowhere in the code.
3. **`solver GAMG;` silently runs PBiCGStab.** There is no multigrid in the default build. A case that
   asks for it gets a different solver and no refusal — the exact defect class this project keeps removing.
4. **Launch submission dominates at small sizes.** 91–93 % of an iteration at 8–24 k cells is submission,
   1.4 % at 2.4 M. Four of the ~22 launches per PBiCGStab iteration are one-thread kernels, and
   `check_interval` defaults to 1, so the pipeline drains every Krylov iteration (1666 drains in one
   potential-flow solve). A CUDA graph is worth 11.8× here and, measured against a second process on the
   card, is a contention shield as much as a launch shield.
5. **Verification is deep in three places and shallow in 34.** Only `fv.cu`, `ldu.cu` and `solver.cu`
   have the independent scatter-shaped host twin the project's own argument rests on; `precon.cu`'s
   DIC/DILU device sweep has no numeric host check at all. The capture registry sits at
   `UNGATED_CEILING = 3` with zero headroom (`pressure/fft.rs`, `pressure/mod.rs`, `simple.rs`).
6. **A latent f32 defect nothing can catch.** `(ofscalar)1e-300` silently becomes `0.0f` in four files
   (`les.cu:73`, `sst.cu:80`, `wallfunctions.cu:530`, `fan.cu:290`), turning floors into `log(0)` and
   divisions by zero. Two other files switch the constant correctly, so the pattern is known and applied
   in a third of the places. Nothing ever builds `--features single`, and there is no CI.
7. **A published table no longer reproduces.** SPEC-LIT §81.12 prints a 2.96× graph speed-up where this
   card measures 15.2× at the same cell count — the per-launch column is 6–19× slower than the document.
   Either the document names a machine it no longer describes, or the sweep was taken under different
   contention; either way it must be re-measured and re-stated.

## C. The work, in order

Each unit is one GLM brief in the house shape, verified and committed by an Opus supervisor. The house
rules are unchanged: no GPL source, headers and PROVENANCE rows and counts, SPEC-LIT text in the same
style, existing numerics never changed to make a gate pass, and the coder never commits.

### Stage 1 — finish what is half-built, and make the suite honest (highest value per hour)

* **S1 — the region layout reaches the solver.** Adopt the uncommitted work in the solver worktree:
  the `ofgpu-regions` binary (`split`, `check`, `list`) and `ChtCase.mesh: {regions: path}` lowering to
  the same `Vec<RegionInput>`, with §C's R1–R8 refusals. Gate 97-B (split a two-zone block, load through
  the manifest, bit-for-bit against the concatenated block run) is written and registered. Without this,
  M4/M5's Turek–Hron and cantilever layouts are files nothing reads.
* **S2 — the answer-key store exists, or the claim is withdrawn.** Create `reference/` with a manifest
  (`reference/PROVENANCE.md`: file, source, DOI/URL, licence, sha256), teach `validate.rs` to read a
  reference file instead of a hand-typed literal where one exists, and fix the ten source files and two
  documents that call data "vendored" that is not. A gate whose key is missing says so by name and is
  reported `OPEN`, never silently passed. This is the precondition for every published gate below.
* **S3 — Ghia runs.** Un-`#[ignore]` the lid-driven cavity, put its table in `reference/`, and report it
  through §94's uncertainty. It is the only published fluid answer key already implemented.
* **S4 — a solver a case asks for is the solver it gets.** `solver GAMG;` is refused by name (naming the
  AMGX feature and this plan's §D as the routes) instead of silently becoming PBiCGStab; the same audit
  is run over every enum a case can name, and a §13.4.1 pair test is added for each one that is silently
  substituted today.

### Stage 2 — the capacity and speed of the card

* **S5 — the 4.72 M-cell cliff.** Diagnose (print `mem_info` after each allocation at 4.72 M and 4.80 M
  and diff the two traces; check the async pool's release threshold), fix, and gate: a memory-model test
  that asserts the measured bytes per cell stay linear across the cliff, and a documented
  "cells that fit in N GB" table that the test keeps honest. Payoff is measured: ×2.3 on the ceiling.
* **S6 — f32 is built, tested and honest.** Fix the four dead floors, add `--features single` to a CI-less
  world as a second `cargo test` invocation the house command runs, and publish what changes: which gates
  hold at f32, which loosen, which fail. Then the mixed-precision question can be asked with evidence.
* **S7 — the hot path stops draining.** `check_interval` gets a measured default, the four one-thread
  kernels are folded into their neighbours, and the per-iteration host flag read is removed from the
  graph-captured path. Gate: the same case, same iterations, bitwise identical answer, with the measured
  wall-clock difference published.
* **S8 — §81.12 is re-measured.** The graph-capture table is re-taken on a named machine with the
  contention stated, and the section says which machine it describes.

### Stage 3 — the verification the core does not have

* **S9 — `precon.cu` gets its host twin.** A scatter-shaped DIC/DILU factorisation and sweep in
  `reference.rs`, and a device-vs-host test to 1e-12 on three meshes. It is the one hot kernel with no
  numeric check.
* **S10 — the ungated three.** `pressure/fft.rs`, `pressure/mod.rs` and `simple.rs` get capture gates or
  a stated refusal, and `UNGATED_CEILING` goes back to having headroom.
* **S11 — the published fluid gates §10 promises.** Moser/Kim/Mansour channel DNS, Driver & Seegmiller,
  and the McCaffrey plume, each with its key in `reference/` and its verdict in §32.4's band discipline.

### Stage 4 — deferred, and named so nobody looks for them

Multi-GPU (the decompose path is proved bitwise at P = 2, 4 but every part runs on one device in
sequence, 10.5× slower at P = 16 — a real multi-GPU run needs a device per part and a communicator);
a real algebraic multigrid; an f16/bf16 path; anything that changes a scheme.

## D. Answer keys — what to acquire, cheapest first

The full ranking is in `facts-answer-keys.md`. The three that matter now:

1. **NAFEMS P18/TNSB Rev. 3, £45.** Not institutionally paywalled. The free ESRD restatement already
   gives LE1/LE10/LE11's material, boundary conditions, loading and targets verbatim (including LE11's
   α = 2.3e-4 /°C); the £45 buys the geometry, which lives in raster figures. `docs/09`'s "not written
   until the primary is in hand" is a one-invoice blocker, and it unblocks Gate 95-G.
2. **Ghia et al. (1982)** — already implemented, ignored, and free.
3. **Turek–Hron** — already downloaded, with the `.point` series, for Part II's FSI gates.

Gate 95-D has no SPEC-LIT section at all (`grep -c "95-D" SPEC-LIT.md` → 0) while `validate.rs` prints
its banner and three checks: S2 writes that section as part of making the suite honest.
