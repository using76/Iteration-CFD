<!-- meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). Source-available, not Open Source. No GPL-licensed source was consulted. -->

# The explain.py fixtures (docs/15 §F G-EXPL)

Written by the supervisor on 2026-09-24 from live runs of the mesh tree's release binary (sha256 `054bba67…`)
at `57a2b4d`, not by explain.py.

- `rows.jsonl`: six autonomy-attempt/1 rows, campaign `am15-fixture`. box_sphere and wing_a_L3 are survey probe
  configs run as given (split `reported`); D-1-002 is a tuning box built by rules.setup and passed by preflight
  with a live octree probe. Each row is valid under `schema.errors` and `schema.check_attempt`.
- `records.json`: every DecisionRecord of those runs, tagged with the attempt it decided (a remedy's record is
  tagged with the attempt its edit produced; a terminal record with the attempt it ended on).
- `records_by_id.json`: one real record per (rule_id, verdict) emitted by schema.py, preflight.py, rules.py and
  remedies.py, 59 records of 41 ids. A scratch path is replaced by `<work>`.
- `meta.json`: the family and stratum of the three geometries (the probes are family X, stratum probe).
- `expect.json`: the supervisor's labels for the golden texts and the summary, and nine Clopper-Pearson
  references computed with scipy's beta quantile.
- `golden/`: written by `explain.py --write-golden` and reviewed by the supervisor; the selftest compares them
  byte for byte.

JSON Lines rows carry no header line, because each line must validate as an AttemptRow.
