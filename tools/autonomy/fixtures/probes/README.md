<!-- meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). Source-available, not Open Source. No GPL-licensed source was consulted. -->

# The G-SCORER probe fixtures (docs/15 §B, §F)

Each directory is one run of `ofgpu-automesher` made by the docs/15 design surveys on 2026-09-23, frozen here so
`score.py` is judged on what the mesher really printed. On 2026-09-24 the 27 runs that wrote a summary were re-run,
in a mirror of the survey tree, with the mesh tree's release binary at `77a33b5` (sha256 `054bba67c8650082...`), so
their summaries carry the fields AM-R2 (`n_pinned_boundary`, `area_ratio`, octree `gate_passed`) and AM-R1
(`area`, `area_frac_tau_ge`) added and the user's decisions of 2026-09-24 read. Against the 2026-09-23 freeze every
mesh number is equal: the only differences are the time fields, `identity.written_at`, and `identity.mesh_id` of
the 18 box_sphere and cube probes (it hashes the case path, and the mirror lives at another path). The four refused
runs keep their 2026-09-23 logs byte for byte; each was re-run and refused with the same `error:` line.

- `log.txt`: the run's stdout and stderr as one stream (the surveys ran `2>&1`); the refusal of a refused
  run is its last `error:` line.
- `config.json`: the config the run was given (the shipped `box_sphere.json` example with its `//`
  comments stripped).
- `summary.json`: `<name>_summary.json` as the run wrote it; absent for the four refused runs
  (NO15, NO25, thin_t1, rev_L4), which write nothing (SPEC-LIT §92.14.4).

The survey's scratchpad directory is replaced by `<survey>` in every path; nothing else is changed, and
none of these three files carries a header, because they are the mesher's own bytes. The STLs are not
committed except `../stl/cube.stl` and `../stl/cubep.stl` (1.5 kB each, the live selftest's input).

`labels.json` is the supervisor's: the exit code (0 when the run wrote its summary; 1 for the four refused
runs, re-confirmed on 2026-09-24), the STL patch area (float64 sum over the STL's
triangles, checked against the log's 3-decimal surface line), the hand label read off each log, and the
expected numbers computed from the summary JSON independently of `score.py` - since 2026-09-24 under the
user's three decisions: F3a is `n_pinned_boundary / n_boundary_points`, F3d is every non-null
`area_ratio[].ratio` inside the locked [0.98, 1.02], and BLC_beta is exact from `area_frac_tau_ge`.
`score.py` never writes it.
The docs/15 text counts "26 probes"; the surveys' saved outputs hold 31 distinct runs (21 box_sphere and
cube probes, 10 corpus probes; the 12 timing replicas of wing_b L4 are not frozen), and all 31 are here.

**The capture signal (2026-09-27).** The 13 probes with a sharp edge whose attraction reached it (`box_L4`,
`cube_cf`, `cube_n5`, `cube_ok`, `cubep_cf`, `cubep_defaults`, `cubep_ok`, `wb_L4`, `wing_a_L3`, `wing_a_L4`,
`wing_a_L5`, `wing_b_L4`, `wing_c_L4`) were re-run, in a fresh mirror of the survey tree, with the mesh tree's
release binary (sha256 `7ff16117...`; output-only against `054bba67...` by lreplay's 97-row replay). Every
mesh number of every re-run summary equals the frozen one; what differs is the time fields, `identity`, the
mirror's paths, and the keys the binary adds (`stages[snap].feature_capture`, (92.62), and the layer stage's
`ladder` and `drop_cause`). **Only `stages[snap].feature_capture` was spliced into those 13 `summary.json`
files, at its sorted key position; every other byte is the 2026-09-24 freeze.** Their `labels.json` expect
carries `feature_capture = captured_length_m / sharp_length_m` and `F3e: false`; every share is above 0.
The other 18 probes are unchanged: their F3e does not depend on the signal (no sharp edge, the attraction
off, or a refused run).

**The chain form and D-L5 (2026-10-02).** The 13 `feature_capture` signals above were measured per feature
segment (binary `7ff16117`), and the user's D-L5 threshold applies to the chain form of (92.62) (the L0c build,
binary `72f7851e`, and later). So the 13 probes were re-run once more, in a fresh mirror of the survey tree,
with `72f7851e`. Every mesh number of every re-run summary equals the frozen one again (the differences are
the time fields, `identity`, the mirror's paths, and the layer stage's `ladder` and `drop_cause` keys). Only
the value of `stages[snap].feature_capture` was replaced; its keys, their order and every other byte are as
before. `sharp_length_m` and `tol_m` are bit-equal; `captured_length_m` changed on the 7 corpus probes and on
none of the 6 cubes (their edges are single segments). The chain-form shares are 0.4849 (`wing_a_L3`), 0.5410
(`wing_a_L4`), 0.5582 (`wing_a_L5`), 0.6167 (`wb_L4`), 0.7036 (`wing_c_L4`), 0.7956 (`wing_b_L4`), 0.9997
(`box_L4`) and 0.9994 / 0.9997 on the cubes, where the per-segment form read 0.0217 to 0.9722 on the corpus
probes. Their `labels.json` expect carries the chain-form share and `F3e = share < 0.5` (gates.json
`feature_capture_min`), so `wing_a_L3` fails F3e and the other 12 pass it.
