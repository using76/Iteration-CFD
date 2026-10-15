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
  remedies.py, 65 records of 45 ids (WL-SHARP-FT0 added 2026-09-26; RM-SNAP-TAU added 2026-10-02; R-SNAP's
  apply, abstain and pass added 2026-10-14 by AUT-05; RM-TE's apply added 2026-10-15 by AUT-04, proposed on
  A-1-012's live attempt-1 outcome from the AUT-04 quick set). A scratch
  path is replaced by `<work>`.
- `meta.json`: the family and stratum of the three geometries (the probes are family X, stratum probe).
- `expect.json`: the supervisor's labels for the golden texts and the summary, and nine Clopper-Pearson
  references computed with scipy's beta quantile.
- `golden/`: written by `explain.py --write-golden` and reviewed by the supervisor; the selftest compares them
  byte for byte.
- `seed1_opt_end.json.gz`: written by the supervisor on 2026-10-11 (AUT-08) from the nine committed seed-1
  bundles, not by explain.py; POST-HOC, not a held-out claim. The 63 geometries (one ending optimiser record
  each) of 5 of the 9 campaigns, with those campaigns' attempts and record lines copied unchanged in bundle
  order; gzip with mtime 0. explain.py's selftest group 15 reads it.

JSON Lines rows carry no header line, because each line must validate as an AttemptRow.
