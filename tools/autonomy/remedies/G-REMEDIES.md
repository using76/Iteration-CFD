<!-- meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). Source-available, not Open Source. No GPL-licensed source was consulted. -->

# G-REMEDIES (AM-10), 2026-09-24: PASS

Run at tree `3afd66e`, binary sha256 `054bba67...a90b`. The report JSON is `remedies/G-REMEDIES.json`.

- Part 1 PASS: 31 probes, 32 sequences, box_sphere CAPABILITY-LIMITED after 1 attempt(s); 0 mismatches.
  Probe/sequence rule ids: RM-CAPABILITY-LIMITED 10, RM-EXHAUSTED 10, RM-SNAP-WALL 10, RM-NO-REMEDY 9, RM-PLANE 7, RM-PASS 4, RM-SNAP-FT 3, RM-BUDGET-FAR 2, RM-BUDGET-FEAT 1, RM-BUDGET-WALL 1, RM-LAYER-FIT 1, RM-PLANE-FINER 1, RM-SNAP-REFINE 1, RM-T1-RAISE 1, RM-T1-REFINE 1, RM-TOPO-REFINE 1.
- Part 2 PASS: the static scan, 35 _set calls in 12 _rm_* functions, 0 violations, 2004 string constants scanned.
- Part 3 PASS: 600 single steps (188 apply, 412 terminal, 0 violations), 300 loops (CAPABILITY-LIMITED 61, EXHAUSTED 87, NO-REMEDY 107, PASS 45; attempts 1:172, 2:90, 3:29, 4:9), 0 violations; exhaustive sweep 2853 cases, 0 violations; 3928 records.
- Part 4 PASS: the six cube loops live (attempt 1 frozen, attempt 2 run by the binary):
  - cube_cf: EXHAUSTED after 2 attempt(s), attempt 2 layer_dropped:min_thickness, n_layers 0, full 0, BLC_8 0, 0.24 s.
  - cube_n5: PASS after 2 attempt(s), attempt 2 no drop, n_layers 8, full 1, BLC_8 1, 0.16 s.
  - cube_ok: PASS after 2 attempt(s), attempt 2 no drop, n_layers 8, full 1, BLC_8 1, 0.17 s.
  - cubep_cf: EXHAUSTED after 2 attempt(s), attempt 2 layer_dropped:retreat_snapped, n_layers 0, full 0, BLC_8 0, 0.14 s.
  - cubep_defaults: PASS after 2 attempt(s), attempt 2 no drop, n_layers 8, full 1, BLC_8 1, 0.07 s.
  - cubep_ok: PASS after 2 attempt(s), attempt 2 no drop, n_layers 8, full 1, BLC_8 1, 0.07 s.
