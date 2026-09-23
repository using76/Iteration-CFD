<!-- meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). Source-available, not Open Source. No GPL-licensed source was consulted. -->

# tools/autonomy — the autonomous mesh-setup engine: schemas, locked gate constants and the definitions everything is scored against

## What is here

- `schema.py` — the stdlib JSON-Schema draft-2020-12 subset validator (written from the keyword
  list, no third-party import on the validation path), the canonical-sha256 gate lock, the knob
  whitelist and legal-range table checks, the twelve semantic checks, and the a priori y⁺ helpers
  (`--selftest | --print-lock | --write-lock | --validate KIND FILE`).
- `selftest.py` — the package gate: runs `schema.py --selftest`, then the README and licence checks.
- `schema/*.schema.json` — the seven schemas: FlowSpec, ManifestRow, Fingerprint, DecisionRecord,
  AttemptRow (`autonomy-attempt/1`), GateConstants, Knobs.
- `schema/knobs.json` — the 21-knob whitelist, the forbidden pointers (`/quality` prefix,
  `/layers/cell_frac`, `/layers/medial_frac`) and the forbidden flag `-permissive`.
- `gates.json` — the §I-3 gate constants, adopted 2026-09-23. `gates.lock` — their sha256 lock,
  written once before any tuning; relocking is the user's decision (docs/15 §F).
- `fixtures/good.json` — one good instance per kind: flow, manifest, fingerprint, decision, attempt.

## The binding definitions (docs/15 §D, verbatim)

<!-- BEGIN VERBATIM docs/15-autonomous-setup-plan.md lines 136-203 at f636fa6 -->
## D. What we measure, exactly

These definitions are written into `tools/autonomy/README.md` by AM-1 and are binding. Every signal named
here either exists today (file and line) or is added by the named unit.

### D.1 Mesh failure

A geometry's **final** mesh, after at most K attempts (K locked in §I-3), is a **failure** if any one holds:

| flag | condition | signal | exists? |
|---|---|---|---|
| **F1** no mesh | exit ≠ 0, timeout or crash | exit code (`bin/automesher.rs` main), campaign timeout | yes |
| **F2** gate | `ofgpu-automesher <cfg> -check <case>` exits 1 | §92.14.5 `-check` | yes |
| **F3** snap fidelity | (a) pinned fraction `n_pinned / n_boundary_points` > 5 %, or (b) `p99_residual` > 0.1·h_f, or (c) `max_residual` > 0.5·h_f, where h_f = `base_size / 2^max_level`; (d) once AM-R2 lands, a per-patch mesh/STL area ratio outside [0.98, 1.02] | summary `stages[snap]` (driver.rs:597), config; (d) AM-R2 | a–c yes, d AM-R2 |
| **F4** topology | a requested wall patch absent (castellate `wall_patches` row 0) or `n_regions ≠ 1` | `stages[castellate]` (driver.rs:556), `quality` | yes |
| **F5** cost | cells over the budget | `mesh.n_cells` | yes |

**MFR** (mesh failure rate) = failures / geometries, per family and stratum, with Clopper-Pearson 95 %
intervals. When there are 0 failures in n, the rule of three gives the bound 3/n (1.7 % at n = 180). If F3(d)
is not in place before AM-12 measures the baseline, it is left out of the baseline and of the final result
alike, so both sides use one definition.

A requested layer patch that was dropped is **not** a failure here. It scores 0 in BLC below. Every report
also prints **strict failure** = failure OR any requested layer patch dropped, so the two can never be traded
against each other out of sight.

### D.2 Boundary-layer capture

For each wall patch p of the STL (all of them, not only the ones the config requested), A_p is its area from
the STL (features.py). Its layer row comes from `stages[layers].patches` or the per-region rows
(driver.rs:677-687, 428-438).

- delivered_p = (`dropped == null` and `n_layers ≥ 8`).
- y⁺ₚ = `t1_requested`·u_τ/ν, with u_τ from §D.3. Retreats only make the first layer thinner, so this is an
  upper bound on the a priori y⁺ of what was built.
- **BLC_8** = Σ A_p · [delivered_p ∧ y⁺ₚ ≤ 1] / Σ A_p. This is the analogue of their BL-CP.
- **BLC_full** = Σ A_p · `full_area_frac_p` · [y⁺ₚ ≤ 1] / Σ A_p. It is stricter: only the area that got the
  whole stack counts.
- **BLC_β** (reported; exact per face only after AM-R1): the area share with achieved/requested thickness ≥ β.
  Before AM-R1 it is bounded per patch, from `t1_min` and `full_area_frac`.

A missing, dropped or unrequested patch counts as 0, never as not-applicable. Every BLC name carries
"a priori" in reports until AR-2 has compared it with a solved y⁺.

### D.3 The y⁺ source and the feasible window (decision §I-4)

R-YP computes Re_L = U·L/ν from the manifest row and takes C_F from the correlations the project already
transcribes in the solver-tree `SPEC-LIT.md` §32.5.6 (Schlichting & Gersten, *Boundary-Layer Theory*, 8th ed.,
2000): 1.328/√Re_L below 5e5 and 0.455/(log10 Re_L)^2.58 above. Then u_τ = U·√(C_F/2) and t1 = y⁺·ν/u_τ with
y⁺ = 1. Worked example: L = 1 m, ν = 1.5e-5 m²/s, Re_L = 2e4 gives t1 = 7.30e-4 m. The mean C_F is not
conservative near a leading edge, which is why the metric is named "a priori".

R-WIN combines two constraints. The G5 early check (layers.rs:1256-1285) needs 3·t1/h_min ≥ 0.05, i.e.
h_min ≤ 60·t1. The stack limiter, read from (92.45) by the design survey, needs
T = t1·(gⁿ−1)/(g−1) ≤ `cell_frac`·h with `cell_frac` fixed at its default 0.5. At n = 8 the window h/t1 is:

- [16, 60] as g → 1;
- [33.0, 60] at g = 1.2;
- [47.7, 60] at g = 1.3;
- empty above g ≈ 1.36.

[16, 60] spans more than a factor of 2, so some octree level always lands inside it, and R-WIN picks that level
and then the largest growth that fits. Two cautions. First, `h_min` in the G5 check is the shortest wall-face
edge after snap, not h_f. It equals h_f only on a castellated wall; on box_sphere it was 0.0376 m where h_f was
0.125 m. So on snapped walls L0's check is a prediction, and G-PREFLIGHT measures how often it is wrong.
Second, the limiter inequality is verified by a run in AM-9's gate before anything depends on it. With these
rules the corpus flow window at L4–L6 is Re_L ≈ 1e4–1e5 (L4 at 1 m covers about 1.4e4–3.1e4; L5 about
3.6e4–8.1e4).
<!-- END VERBATIM -->

## score.py

`score.py` (AM-2) turns one automesher run into §D's Outcome. It reads only what `ofgpu-automesher`
itself prints and writes: the `=== stage i/n <name> ===` banners and `--- <name>: x.x s` elapsed lines
of `driver.rs`, the `error:` refusal of `bin/automesher.rs`'s main in SPEC-LIT §92.3's fixed grammar, and
`<name>_summary.json` (§92.14.3). On `exit != 0` the summary is IGNORED even if given — a refusal writes
nothing (§92.14.4), so any summary there is stale — and the refusal line is classed, first rule that fires:
`surface/closed:` → `surface_closed`; `quality gate G<k> (` → `gate_G<k>@<last stage>`; the (92.51)
thin-first-layer text → `layer_t1_G5`; `io error on ` → `io`; anything else → `config`. `exit 0` with no
summary is `io`; a timeout or a missing process is `timeout`/`crash`. An unknown layer drop text raises
`ScoreParseError` — the scorer never guesses past what the reports say.

The F-flags are stored in the RATIO form `check_attempt` re-derives (S13–S16): `pinned_frac > 0.05`,
`p99/h_f > 0.1`, `max/h_f > 0.5` with `h_f = base_size / 2**max_level`, `F4` a requested wall patch with a
zero `wall_patches` row or `n_regions != 1`, `F5` cells over the 2 M budget. **BLC_8** and **BLC_full** use
every STL wall patch as denominator; a patch counts only when delivered (dropped is null, n_layers ≥ 8) and
its a priori y⁺ (`schema.yplus_a_priori`, named `_a_priori` until AR-2) is ≤ 1. **BLC_beta** is the area
share with achieved/requested ≥ beta; until AM-R1 reports per-face tau it is BOUNDED per patch from
`t1_min`, the area-weighted `mean_frac` and `full_area_frac` (two Markov-type inequalities, `exact: false`).
A requested patch that was dropped is not a failure — it scores 0 — but it makes `strict_failure` true, so
the two can never be traded out of sight. `missing_signals` names what today's reports lack instead of
guessing: the `-check` that was not run, AM-R2's `area_ratio` and octree gate fields, AM-R1's per-face tau.

`content_sha256` hashes `constant/polyMesh`'s points, faces, owner, neighbour and boundary, name- and
length-prefixed — equal across two runs of one config, different after one byte. `run_check` is §92.14.5's
`-check`: it returns the exit code, the refusal line and the gate name, and `check_exit != 0` is flag F2.
`score.py --hash-gate CONFIG` runs one config three ways (A, B, then C with one whitelisted knob moved by
`--pointer`/`--to`) and demands A == B and C != A by content sha256 — the content-hash half of G-DET. The 31
frozen probes under `fixtures/probes/` are the docs/15 surveys' own automesher outputs, re-scored on every
`--selftest`: the exit codes, patch areas, hand labels and expected numbers in `labels.json` are the
SUPERVISOR'S, computed from the summaries independently of the scorer, and the scorer is wrong whenever
they disagree.

## The locked constants

| field | value |
|---|---|
| pinned_frac_max | 0.05 |
| p99_residual_over_hf_max | 0.1 |
| max_residual_over_hf_max | 0.5 |
| area_ratio_min / max | 0.98 / 1.02 |
| attempts_k | 4 |
| cell_budget | 2000000 |
| delivered_min_layers | 8 |
| yplus_max_a_priori | 1.0 |
| adopted | 2026-09-23 |

Whole-gates-file sha256 over canonical JSON: `93cc1648563ec0ac96b81d837e8625c9b98140114c094e63ee20b793a1aef1aa`. The lock lives in `gates.lock` — the hash of
the parsed file, one hash per field, and the hash of `schema/knobs.json`. **Relocking is the user's
decision (docs/15 §F)**: a constant that proves infeasible goes back to the user; it is never relaxed
quietly, and this file is never rewritten to make a number pass.

## y⁺ is a priori

R-YP takes C_F from the correlations the solver tree already transcribes in SPEC-LIT §32.5.6
(`feat/core-2` at `d272391`, `rust/SPEC-LIT.md:3898-3899`):

```
C_F = 1.328 / sqrt(Re_L)                Re_L < 5e5   (Blasius, laminar)
C_F = 0.455 / (log10 Re_L)^2.58         Re_L >= 5e5  (Prandtl-Schlichting, turbulent, to ~1e9)
```

with `Re_L <= 0` or non-finite giving no estimate (None). Then `u_tau = U·sqrt(C_F/2)` and
`t1 = y⁺·nu/u_tau` with `y⁺ = 1`. Worked example: L = 1 m, ν = 1.5e-5 m²/s, Re_L = 2e4 gives
`t1 = 7.30e-4 m` (exactly 7.29699e-4). The mean C_F is not conservative near a leading edge, which is
why the metric is named "a priori" (docs/15 §D.3, decision §I-4). Until AR-2 has compared it with a
solved y⁺, every report name that carries a y⁺-derived number carries `_a_priori`
(`flat_plate_cf` cites: SPEC-LIT §32.5.6; Schlichting & Gersten, Boundary-Layer Theory, 8th ed.,
Springer (2000)), and `schema.py`'s `a_priori_name_violations` refuses any y⁺-bearing field name
that does not (docs/15 §I-4).

## The action space

The loop may edit these 21 knobs, and nothing else (docs/15 §C L0 (d)); every row cites the Rust
field it points at (`rust/src/automesher/mod.rs`):

| pointer | type | stage | min | max | min_excl | validator/declared |
|---|---|---|---|---|---|---|
| /domain/base_size | float | octree | 0 (excl) | — | yes | mod.rs:546 |
| /domain/extent | extent | octree | — | — | — | mod.rs:527 |
| /refinement/max_level | int | octree | 0 | 6 | no | mod.rs:559 |
| /refinement/levels/*/patch | str | octree | — | — | — | preflight checks the name |
| /refinement/levels/*/bands/*/distance | float | octree | 0 (excl) | — | yes | a distance band is positive |
| /refinement/levels/*/bands/*/level | int | octree | 0 | 6 | no | no level past the octree cap |
| /refinement/levels/*/feature_level | int | octree | 0 | 6 | no | no level past the octree cap |
| /snap/iterations | int | snap | 0 | 200 | no | default 30; 60 measured as enough for the 1e-6 dead band |
| /snap/tolerance | float | snap | 1e-9 | 0.01 | no | fraction of base_size; default 1e-3 |
| /snap/smoothing_passes | int | snap | 0 | 10 | no | default 3; the L4 box uses 0..3 |
| /snap/smoothing | float | snap | 0 | 1 | no | mod.rs:571 |
| /snap/undo_limit | int | snap | 0 | 10 | no | default 4 |
| /snap/feature_tolerance | float | snap | 0 | 1 | no | mod.rs:577 (no upper bound in the validator) |
| /layers/patches | str_list | layers | — | — | — | preflight checks the names |
| /layers/n | int | layers | 0 | 16 | no | docs/15 §D.2 counts n >= 8 |
| /layers/first_thickness | float | layers | 0 (excl) | — | yes | set by R-YP from §D.3 |
| /layers/growth | float | layers | 1.0 | 2.0 | no | mod.rs:603; the §D.3 window is empty above ~1.36 |
| /layers/normal_passes | int | layers | 0 | 10 | no | default 3 |
| /layers/smoothing | float | layers | 0 | 1 | no | a relaxation weight |
| /layers/smoothing_passes | int | layers | 0 | 10 | no | default 4 |
| /layers/retreat_limit | int | layers | 0 | 8 | no | default 4 |

Forbidden, never in the whitelist: `/quality` (prefix), `/layers/cell_frac`, `/layers/medial_frac`,
and the flag `-permissive`. Per docs/15 §I-5 (D-D), `cell_frac` and `medial_frac` are **limiters**,
not knobs: the ambitious design put `cell_frac` in its whitelist, and that is exactly how a loop
would buy "capture" with thinner stacks. The loop improves meshes; it does not loosen the gate that
judges them (docs/15 §C, §H). Anything unlisted (`/layers/min_thickness`, `/snap/max_area_ratio`,
`/castellation/*`, `/input/*`, `/output/*`, `/refinement/feature_angle_deg`, ...) is refused as
`WL-UNLISTED`; refusals come back as DecisionRecords with rule ids `WL-POINTER`, `WL-FORBIDDEN`,
`WL-UNLISTED`, `WL-TYPE`, `WL-RANGE` and `WL-FLAG`.

## Licences

```text
PACKAGE        VERSION   INSTALLED DECLARED                                 DETECTED         VERDICT
-------------- --------- --------- ---------------------------------------- ---------------- -------
numpy          2.2.6     yes       Copyright (c) 2005-2024, NumPy Developer BSD-3-Clause     ok
  bundled: lapack-lite                                BSD-3-Clause
  bundled: dragon4                                    MIT
  bundled: libdivide                                  Zlib
  bundled: Meson                                      Apache 2.0
  bundled: spin                                       BSD-3
  bundled: tempita                                    MIT
  bundled: OpenBLAS                                   BSD-3-Clause
  bundled: LAPACK                                     BSD-3-Clause-Attribution
  bundled: GCC runtime library                        GPL-3.0-with-GCC-exception
  NOTE: GCC runtime library carries GPL-3.0-with-GCC-exception - the GCC runtime library exception, which is what numpy's and scipy's OpenBLAS DLL carry
scipy          1.15.2    yes       Copyright (c) 2001-2002 Enthought, Inc.  BSD-3-Clause     ok
  bundled: OpenBLAS                                   BSD-3-Clause-Attribution
  bundled: LAPACK                                     BSD-3-Clause-Attribution
  bundled: GCC runtime library                        GPL-3.0-with-GCC-exception
  NOTE: GCC runtime library carries GPL-3.0-with-GCC-exception - the GCC runtime library exception, which is what numpy's and scipy's OpenBLAS DLL carry
scikit-learn   1.6.1     yes       BSD 3-Clause License                     BSD-3-Clause     ok
  bundled: Microsoft Visual C++ Runtime Files         (not stated)
jsonschema     4.24.0    yes       MIT                                      MIT              ok
referencing    0.36.2    yes       MIT                                      MIT              ok
```

## corpus/ — families A and B

- `corpus/stl_io.py` — the pure-numpy ASCII STL writer every corpus family shares
  (`%.9e` numbers, `\n` ends, shared vertices printed from one point array so they
  are bit-identical, `+0.0` canonical so no `-0.` is written) and the in-memory
  oracles: per-edge open/non-manifold/same-direction counts, Euler characteristic,
  signed volume, shortest edge, and `check_closed`, which raises instead of repairing.
- `corpus/gen_wing.py` — family A, NACA 4-digit wings. The section is NACA Report 460
  verbatim (open trailing edge, −0.1015; the −0.1036 closed-TE variant is NOT used
  because Report 460 does not contain it): Jacobs, E. N., Ward, K. E. & Pinkerton,
  R. M. (1933), *The characteristics of 78 related airfoil sections from tests in the
  variable-density wind tunnel*, NACA Report No. 460 (NACA-TR-460), NTRS 19930091108,
  https://ntrs.nasa.gov/citations/19930091108 (no DOI; US government work). Full-span
  planform: quarter-chord sweep, linear taper, linear twist, blunt-TE base strip,
  planar tip caps. Camber is capped at m ≤ 4 % so the section area stays within 1 %
  of 0.685·t·c² (measured 1.03 % at 6 %, 0.43 % at 4 %); the closed form is
  0.685·t·span·(c_root² + c_root·c_tip + c_tip²)/3.
- `corpus/gen_lathe.py` — family B, bodies of revolution about x: prolate
  ellipsoids (4/3·π·a·b²), tangent-ogive + cylinder and tangent-ogive + cylinder +
  conical boat-tail bodies with flat bases (nose volume
  π(ρ²L_n − L_n³/3 − (ρ−R)ρ² asin(L_n/ρ)), cylinder πR²L_c, frustum
  πL_b(R² + R·r_b + r_b²)/3). Every pole is ONE vertex; a flat base's outer ring is
  the last side ring.
- `corpus/gate.py` — §F G-CORPUS: closed, dryrun, no_negzero, sha, regen, row,
  volume, min_edge — closure judged on the re-read file by `stl_repair` and by
  `ofgpu-automesher -dryRun`, byte-identical regeneration in a child process, and
  four negative controls proving the gate is not vacuous.

Both generators expose the same API (AM-6/AM-7 depend on it): `sample_params`,
`validate_params`, `build`, `closed_form_volume`, `stratum`, `l_ref`, `flow`,
`geometry_id`, `make_row`, `write_row`, `generate`. No global RNG: parameters come
from `numpy.random.default_rng([SALT, seed, index])` in a fixed draw order, so a
geometry is a pure function of (family, seed, index). Rows are `autonomy-manifest/1`
ManifestRows without `split` (AM-7's split.py adds it); STLs and manifests are never
committed (docs/15 §E) — only the generators are.

## Running

    python tools/autonomy/schema.py --selftest            # the 8 schema/lock/knob checks
    python tools/autonomy/selftest.py                     # the package gate (adds README + licences)
    python tools/autonomy/schema.py --print-lock          # the lock lines, no comments
    python tools/autonomy/schema.py --validate KIND FILE  # valid / one error per line
    python tools/deps_licences.py --python                # the Python-side licences above

    python tools/autonomy/corpus/gen_wing.py --seed 1 --n 120 --out DIR   # family A STLs + manifest_A.jsonl
    python tools/autonomy/corpus/gen_lathe.py --seed 1 --n 120 --out DIR  # family B
    python tools/autonomy/corpus/gate.py --family A --family B --n 120 --seed 1   # G-CORPUS

The mesh tree's own gate (`python tools/mesh/selftest.py`) runs `tools/autonomy/selftest.py` last.
