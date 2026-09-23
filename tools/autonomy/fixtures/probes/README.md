<!-- meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). Source-available, not Open Source. No GPL-licensed source was consulted. -->

# The G-SCORER probe fixtures (docs/15 §B, §F)

Each directory is one run of `ofgpu-automesher` (the mesh tree's release binary as it stood at `c93e4c1`,
sha256 `f59f66ce3c3c4476...`) made by the docs/15 design surveys on 2026-09-23, frozen here so
`score.py` is judged on what the mesher really printed:

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
runs, re-confirmed on 2026-09-24 with the same binary), the STL patch area (float64 sum over the STL's
triangles, checked against the log's 3-decimal surface line), the hand label read off each log, and the
expected numbers computed from the summary JSON independently of `score.py`. `score.py` never writes it.
The docs/15 text counts "26 probes"; the surveys' saved outputs hold 31 distinct runs (21 box_sphere and
cube probes, 10 corpus probes; the 12 timing replicas of wing_b L4 are not frozen), and all 31 are here.
