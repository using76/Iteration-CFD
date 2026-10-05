# rescore.py - the tuning campaigns under the 2026-09-26 rule

`rescore.py --run` re-scores the committed TUNING bundles under the feature-edge rule of section D, with no re-meshing: every attempt's config is rebuilt from its row (`optimise.rows_from`, sha-checked), F3e is decided by `score.feature_capture` from the fingerprint, `preflight.plane_path` and, where a campaign directory is given with `--cases`, the attempt's own `stages[snap]` report, and a geometry passes when one of its attempts that ran still passes.

It is the MFR of the attempts actually run: a geometry whose passing mesh now fails ended there, so the attempts a re-run would have made instead never ran, and a re-run can only rescue.

The seal holds (a bundle that is not a tuning campaign is refused); the test split is spent and is not read.

| system (tuning, 420 geometries) | failures before | after | MFR before -> after | after, plane path forbidden too |
|---|---|---|---|---|
| B0-template | 384 | 384 | 0.914 -> 0.914 | 384 (0.914) |
| B0-LHS | 240 | 348 | 0.571 -> 0.829 | 348 (0.829) |
| rules | 193 | 340 | 0.460 -> 0.810 | 376 (0.895) |
| rules+opt | 178 | 340 | 0.424 -> 0.810 | 376 (0.895) |

Family by family (failures), before -> after:

| system | A | B | D | E | F | G |
|---|---|---|---|---|---|---|
| B0-template | 84 -> 84 | 63 -> 63 | 84 -> 84 | 29 -> 29 | 42 -> 42 | 82 -> 82 |
| B0-LHS | 84 -> 84 | 13 -> 51 | 24 -> 63 | 18 -> 27 | 29 -> 42 | 72 -> 81 |
| rules | 78 -> 84 | 5 -> 55 | 13 -> 64 | 18 -> 29 | 17 -> 29 | 62 -> 79 |
| rules+opt | 76 -> 84 | 5 -> 55 | 6 -> 64 | 13 -> 29 | 16 -> 29 | 62 -> 79 |
