# Autonomous mesh setup: the held-out evaluation (docs/15 §F)

- date: 2026-09-26
- split: test, sha256 69a6c9399b5ace39696ebbc2b6123214f0b5134f7cfb8b8709e392a871fb010a, 180 geometries
- binary sha256: 054bba67c8650082431ac01325e497208a9bc663c4760a36f7472ff62814a90b
- git sha: 5a8d1cf08dc8213140b9fac5f0f456c69d658e96
- plan sha256: 1ceecf603ac7964695555acdf73d3f4794effc78774470f674a0b790af0470eb
- opened at: 2026-09-25T17:16:32Z

The evaluation is in the style of the ablation tables of DOI 10.3389/frobt.2025.1566623; it reproduces nothing of that paper, and its numbers are not comparable with ours.

Every BLC number is a priori (docs/15 §D.3) until the solved y+ check (G-YPLUS) exists.

Every rule on this page was fixed before the test split was opened. A missed gate is reported as missed and is not re-run with other settings; the test split is now spent, and a second claim needs a fresh test seed and a new lock (docs/15 §F).

## Headlines

G-FAIL FAIL: MFR full 0.405556 [0.333146, 0.481129] vs B0-template 0.911111 (4 x 73 <= 164 False); 95 % upper 0.481129 (<= 0.10 False); McNemar b 92 c 1 p 1.89831e-26 (< 0.01 True); families worse none

G-BLC-0 PASS: tier 0 (15 D/F commensurate) BLC_8 0.933333 (>= 0.90 True), BLC_full 0.933333 (>= 0.80 True), a priori; B0-template 0 / 0

## Gates

| gate | verdict | numbers |
|---|---|---|
| G-FAIL | FAIL | MFR full 0.405556 [0.333146, 0.481129] vs B0-template 0.911111 (4 x 73 <= 164 False); 95 % upper 0.481129 (<= 0.10 False); McNemar b 92 c 1 p 1.89831e-26 (< 0.01 True); families worse none |
| G-BLC-0 | PASS | tier 0 (15 D/F commensurate) BLC_8 0.933333 (>= 0.90 True), BLC_full 0.933333 (>= 0.80 True), a priori; B0-template 0 / 0 |
| G-QUAL | PASS | 1791 configs, 0 off the reference quality block, 0 of 3312 edits outside the whitelist, 0 config sha mismatches, 0 forbidden-flag literals, remedies scan ok True |
| G-FID | FAIL | 6 family metrics compared, worse B:p99_over_hf,E:p99_over_hf,G:p99_over_hf, re-measured B0-template equal to the seal True |
| G-COST | FAIL | cells 43.4769 (FAIL), over budget 0 (PASS), wall 7615.19 s at 6 streams (PASS), peak 0.592408 of RAM (PASS), orphans 0 (PASS) |
| G-DET | PASS | 21/21 rows, 0 row diffs, content 20/20, replays True True |
| G-EXPL | FAIL | 9 campaigns audited, 5 not ok |
| G-OPT | FAIL | full vs rules MFR 0.45 -> 0.405556 (gain 0.0444444), BLC_8 0.155556 -> 0.177778 (gain 0.0222222), regressed B; optimiser marginal MFR 0.438889 -> 0.405556 |
| G-ABL | REPORTED | full MFR 0.405556 BLC_8 0.177778, -preflight MFR 0.394444 BLC_8 0.188889, -remedies MFR 0.511111 BLC_8 0.144444, -prior MFR 0.416667 BLC_8 0.172222, -optimiser MFR 0.438889 BLC_8 0.161111, rules + remedies only MFR 0.45 BLC_8 0.155556, B0-template (sealed) MFR 0.911111 BLC_8 0, B0-LHS best of 4 (sealed) MFR 0.577778 BLC_8 0.0333333 |

## Mesh failure per family (G-FAIL)

| family | n | B0-template | B0-LHS best | full | full MFR [95 % CI] | full strict |
|---|---|---|---|---|---|---|
| A | 36 | 36 | 35 | 29 | 0.806 [0.640, 0.918] | 36 |
| B | 36 | 29 | 12 | 1 | 0.028 [0.001, 0.145] | 36 |
| D | 36 | 36 | 9 | 5 | 0.139 [0.047, 0.295] | 16 |
| E | 18 | 11 | 7 | 4 | 0.222 [0.064, 0.476] | 17 |
| F | 18 | 18 | 12 | 7 | 0.389 [0.173, 0.643] | 11 |
| G | 36 | 34 | 29 | 27 | 0.750 [0.578, 0.879] | 33 |
| all | 180 | 164 | 104 | 73 | 0.406 [0.333, 0.481] | 149 |

## Boundary-layer capture, a priori (G-BLC-0)

Tier 0 (15 D/F commensurate geometries): full BLC_8 0.933, BLC_full 0.933; B0-template 0.000 / 0.000; a priori.

| family | n | BLC_8 | BLC_full | with CAPABILITY-LIMITED patches |
|---|---|---|---|---|
| A | 36 | 0.000 | 0.000 | 7 |
| B | 36 | 0.028 | 0.025 | 35 |
| D | 27 | 0.444 | 0.402 | 11 |
| E | 18 | 0.056 | 0.052 | 13 |
| F | 12 | 0.083 | 0.080 | 4 |
| G | 36 | 0.083 | 0.054 | 6 |

CAPABILITY-LIMITED patches lose their layers on a snapped curved wall, which this mesher cannot grow (docs/15 §I-1); they are reported, never gated.

## Fidelity (G-FID)

| family | full meshes | B0 meshes | feature_tolerance 0 on sharp bodies | metric | full median | B0 median | no worse |
|---|---|---|---|---|---|---|---|
| A | 7 | 0 | 7 | p99_over_hf | 0.038 | absent | - |
| A | 7 | 0 | 7 | pinned_frac | 0.000 | absent | - |
| A | 7 | 0 | 7 | feature_share | 0.000 | absent | - |
| B | 35 | 7 | 24 | p99_over_hf | 0.035 | 0.016 | False |
| B | 35 | 7 | 24 | pinned_frac | 0.000 | 0.000 | True |
| B | 35 | 7 | 24 | feature_share | 0.000 | absent | - |
| D | 31 | 0 | 31 | p99_over_hf | 0.032 | absent | - |
| D | 31 | 0 | 31 | pinned_frac | 0.000 | absent | - |
| D | 31 | 0 | 31 | feature_share | 0.000 | absent | - |
| E | 14 | 7 | 8 | p99_over_hf | 0.032 | 0.016 | False |
| E | 14 | 7 | 8 | pinned_frac | 0.000 | 0.000 | True |
| E | 14 | 7 | 8 | feature_share | 0.000 | absent | - |
| F | 11 | 0 | 11 | p99_over_hf | 0.000 | absent | - |
| F | 11 | 0 | 11 | pinned_frac | 0.000 | absent | - |
| F | 11 | 0 | 11 | feature_share | 0.000 | absent | - |
| G | 9 | 2 | 7 | p99_over_hf | 0.032 | 0.016 | False |
| G | 9 | 2 | 7 | pinned_frac | 0.000 | 0.000 | True |
| G | 9 | 2 | 7 | feature_share | 0.000 | absent | - |

A mesh snapped with feature_tolerance 0 captures no feature edge and scores a share of 0.

## Cost (G-COST)

- cells: median full 469898 / median B0-template 10808 on 15 both-pass geometries, ratio 43.477 (FAIL, <= 1.50)
- over budget: 0 geometries over the cell budget (PASS)
- wall: 7615.2 s at 6 streams, the best a 12-stream run could do is 4091.6 s (PASS, <= 10800 s)
- RAM: peak 0.592 of total (PASS, <= 0.60)
- orphans: 0 (PASS)

## Determinism (G-DET)

The 20 G-DET geometries, fixed before any outcome: A-1-043 A-1-045 A-1-059 A-1-084 B-1-101 B-1-103 B-1-115 B-1-117 D-1-009 D-1-021 D-1-028 E-1-034 E-1-044 E-1-047 F-1-013 F-1-052 F-1-058 G-1-006 G-1-062 G-1-080.

gdet-1 against gdet-2: 21/21 rows, 0 row diffs, content 20/20 equal; verdict PASS.

Replay through rules.py and remedies.py: gdet-1 ok True (38 decisions), gdet-2 ok True (38 decisions).

The full campaign against gdet-1 on these ids: 21 rows, 0 row diffs, content 20/20 equal; ok True.

## Guards (G-QUAL, G-EXPL)

1791 configs, 0 off the reference quality block, 0 of 3312 edits outside the whitelist, 0 config sha mismatches, 0 forbidden-flag literals, remedies scan ok True; 9 campaigns audited, 5 not ok.

- b0-template: audit ok True, 162 rows, 0 untemplated.
- full: audit ok False, 234 rows, 0 untemplated.
- gdet-1: audit ok False, 21 rows, 0 untemplated.
- gdet-2: audit ok False, 21 rows, 0 untemplated.
- no-optimiser: audit ok True, 223 rows, 0 untemplated.
- no-preflight: audit ok False, 249 rows, 0 untemplated.
- no-prior: audit ok False, 371 rows, 0 untemplated.
- no-remedies: audit ok True, 148 rows, 0 untemplated.
- rules: audit ok True, 362 rows, 0 untemplated.

## Ablation (G-ABL)

| system | MFR [95 % CI] | strict | BLC_8 | BLC_full | cells median |
|---|---|---|---|---|---|
| full | 0.406 [0.333, 0.481] | 149 | 0.178 | 0.164 | 177480 |
| -preflight | 0.394 [0.323, 0.470] | 147 | 0.189 | 0.175 | 205124 |
| -remedies | 0.511 [0.436, 0.586] | 154 | 0.144 | 0.134 | 226444.0 |
| -prior | 0.417 [0.344, 0.492] | 150 | 0.172 | 0.150 | 165997 |
| -optimiser | 0.439 [0.365, 0.515] | 152 | 0.161 | 0.150 | 168996 |
| rules + remedies only | 0.450 [0.376, 0.526] | 153 | 0.156 | 0.141 | 160904 |
| B0-template (sealed) | 0.911 [0.860, 0.948] | 180 | 0.000 | 0.000 | 10778.0 |
| B0-LHS best of 4 (sealed) | 0.578 [0.502, 0.651] | 174 | 0.033 | 0.013 | 19256.0 |

| system | A | B | D | E | F | G |
|---|---|---|---|---|---|---|
| full | 0.806 | 0.028 | 0.139 | 0.222 | 0.389 | 0.750 |
| -preflight | 0.806 | 0.028 | 0.056 | 0.278 | 0.389 | 0.750 |
| -remedies | 0.972 | 0.028 | 0.222 | 0.500 | 0.500 | 0.833 |
| -prior | 0.833 | 0.028 | 0.167 | 0.222 | 0.389 | 0.750 |
| -optimiser | 0.833 | 0.028 | 0.167 | 0.389 | 0.444 | 0.750 |
| rules + remedies only | 0.861 | 0.028 | 0.194 | 0.389 | 0.444 | 0.750 |
| B0-template (sealed) | 1.000 | 0.806 | 1.000 | 0.611 | 1.000 | 0.944 |
| B0-LHS best of 4 (sealed) | 0.972 | 0.333 | 0.250 | 0.389 | 0.667 | 0.806 |

### G-OPT

- as written (full vs rules + remedies only): MFR 0.450 -> 0.406 (gain 0.044), BLC_8 0.156 -> 0.178 (gain 0.022), regressed B.
- optimiser marginal (full vs -optimiser): MFR 0.439 -> 0.406 (gain 0.033), BLC_8 0.161 -> 0.178 (gain 0.017), regressed none.
- rules+opt vs rules: MFR 0.450 -> 0.417 (gain 0.033), BLC_8 0.156 -> 0.172 (gain 0.017), regressed none.

### The per-round tuning curve (tuning split, not the result)

| system | failures | MFR | BLC_8 |
|---|---|---|---|
| rules | 193 | 0.460 | 0.140 |
| round-1 | 179 | 0.426 | 0.171 |
| round-2 | 180 | 0.429 | 0.167 |
| round-3 | 180 | 0.429 | 0.167 |
| round-4 | 179 | 0.426 | 0.169 |
| round-5 | 178 | 0.424 | 0.169 |

## What the prior and the optimiser decided on the test split

- full: OPT-NOFEAS 26, OPT-PICK 14, PR-FAR 37, PR-KEEP 33, PR-KNN 78
- no-optimiser: PR-FAR 37, PR-KEEP 33, PR-KNN 78
- no-preflight: OPT-NOFEAS 28, OPT-PICK 14, PR-FAR 39, PR-KEEP 34, PR-KNN 79
- no-prior: OPT-NOFEAS 25, OPT-PICK 13
- no-remedies: PR-FAR 37, PR-KEEP 33, PR-KNN 78

## The campaigns

| campaign | system | ablate | geometries | rows | meshed | reused | RAM-guarded | wall s | peak RSS MiB | harness errors | replay | audit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | full | none | 180 | 234 | 234 | 0 | 0 | 7615.2 | 16744.7 | 0 | True | False |
| b0-template | b0-template | none | 180 | 162 | 162 | 0 | 0 | 200.3 | 359.4 | 0 | True | True |
| gdet-1 | full | none | 20 | 21 | 21 | 0 | 0 | 499.9 | 8342.5 | 0 | True | False |
| gdet-2 | full | none | 20 | 21 | 21 | 0 | 0 | 522.1 | 9191.1 | 0 | True | False |
| no-remedies | full | remedies | 180 | 148 | 0 | 148 | 0 | 52.9 | 291.6 | 0 | True | True |
| no-optimiser | rules+prior | none | 180 | 223 | 0 | 223 | 0 | 53.6 | 390.0 | 0 | True | True |
| rules | rules | none | 180 | 362 | 166 | 196 | 0 | 5724.6 | 9651.6 | 0 | True | True |
| no-prior | rules+opt | none | 180 | 371 | 0 | 371 | 0 | 60.5 | 358.0 | 0 | True | False |
| no-preflight | full | preflight | 180 | 249 | 9 | 233 | 7 | 711.7 | 7108.8 | 0 | True | False |

## Tuning context (not the result)

On the tuning split: rules-only 80 of 420 tuning geometries pass at attempt 1, the prior 214, the shuffled control 169.667 on average: of the 134-geometry gain, 89.6667 is reached with shuffled fingerprints and 44.3333 is the fingerprint's own.

The optimiser's tuning verdict: PASS (enabled True); final MFR gain 0.036, BLC_8 gain 0.029, regressed none.

## Departures

- docs/15 §F measures G-COST's campaign wall time at 12 streams; the house caps campaigns at 6 while the solver workflow runs, so the wall item passes when the 6-stream time is at most 3 h, fails when even perfect 12-stream scaling (half the 6-stream time) or the longest single geometry exceeds 3 h, and is undecided in between.
- The ablations reuse what earlier campaigns of this evaluation measured: an attempt whose (geometry, config sha) was meshed returns that outcome (the mesher is deterministic, §F G-DET). The full system, the B0-template re-measure and both G-DET runs are meshed fresh.
- The sealed B0-template rows carry no feature-edge counts, so B0-template is re-meshed once on the test split for G-FID; its rows must equal the sealed rows apart from time fields and git_sha, and every other B0 number comes from the seal.
- G-FID's feature-edge share is n_snapped_to_edge / n_feature_edges; a mesh snapped with feature_tolerance 0 extracts no edges and reports 0 of them, so on a body whose fingerprint has sharp edges its share is 0.0, not undefined. A body without sharp edges is left out.
- The -preflight ablation meshes configs L0 would refuse. On this shared machine an attempt whose octree probe predicts more than 8192 MiB of peak memory is not run and is scored as an F1 no-mesh failure (class crash); the count is reported, and it is 0 wherever L0 runs, whose budget check keeps a probe under 2 M leaves (4.9 GiB).
- G-OPT as written compares the full system with rules+remedies only (mode rules); the optimiser's own marginal (full against -optimiser) and rules+opt against rules are reported beside it.
- -remedies runs K = 1, so the optimiser, which acts only after the remedies are spent with attempts left, never acts in it either.
- The test manifest is opened once by this evaluation (split.load('test', 'evaluate')) after a write-once lock records the plan; each campaign re-reads the same lock-checked rows by name. A second evaluation with any other plan is refused: the split is spent (§F).
- The 20 G-DET geometries are fixed from the lock's test ids before any outcome exists: within each family by sha256('gdet:' + id), drawn round-robin A, B, D, E, F, G.
- The tier-0 stratum of G-BLC-0 is the D and F rows whose manifest field commensurate is true.
- A geometry that ends SURFACE-OPEN, SURFACE-REFUSED, REFUSED or HARNESS-ERROR is a failure in every system (§D.1 F1), the sealed baselines included (read through campaign.terminal_of).
