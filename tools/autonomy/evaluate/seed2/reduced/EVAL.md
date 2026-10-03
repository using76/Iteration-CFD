# Fresh-seed evaluation (corpus seed 2, scope reduced)

REDUCED: a pre-registered subset of 6 of the 180 fresh test geometries; these verdicts are not a headline.

- date: 2026-10-04
- split: test2, sha256 3bd59a3dde90f1929ec61888b40b20cc40a598f6543805584e9ca7a88b675bb9, 6 geometries
- binary sha256: 3d90ce9156119b16ee6fdd1359c5913152ebc29be3e218ebb903fe70900d923b
- git sha: 1b628045cae6e30315cd7a69592cbc5073fc4ab2
- plan sha256: 60a05a97624b0875fbaa536804748d063bfea28a7a83ad90f63702256bdd1063
- opened at: 2026-10-03T15:23:52Z

The evaluation is in the style of the ablation tables of DOI 10.3389/frobt.2025.1566623; it reproduces nothing of that paper, and its numbers are not comparable with ours.

Every BLC number is a priori (docs/15 §D.3) until the solved y+ check (G-YPLUS) exists.

Every rule on this page was fixed before the fresh test split was opened. A missed gate is reported as missed and is not re-run with other settings; the full scope runs under the same locked plan, and a run with any other plan is refused (docs/15 §F).

## Headlines

REDUCED G-FAIL FAIL: MFR full 0.5 [0.118117, 0.881883] vs B0-template 0.666667 (4 x 3 <= 4 False); 95 % upper 0.881883 (<= 0.10 False); McNemar b 1 c 0 p 1 (< 0.01 False); families worse none

REDUCED G-BLC-0 PASS: tier 0 (1 D/F commensurate) BLC_8 1 (>= 0.90 True), BLC_full 1 (>= 0.80 True), a priori; B0-template 0 / 0

REDUCED G-BLC-1 FAIL: tier 1 (4 A/B/E/F) BLC_8 0 (>= 0.09 False), BLC_full 0 (>= 0.05 False), a priori; B0-template 0 / 0

## Gates

| gate | verdict | numbers |
|---|---|---|
| G-FAIL | G-FAIL FAIL | MFR full 0.5 [0.118117, 0.881883] vs B0-template 0.666667 (4 x 3 <= 4 False); 95 % upper 0.881883 (<= 0.10 False); McNemar b 1 c 0 p 1 (< 0.01 False); families worse none |
| G-BLC-0 | G-BLC-0 PASS | tier 0 (1 D/F commensurate) BLC_8 1 (>= 0.90 True), BLC_full 1 (>= 0.80 True), a priori; B0-template 0 / 0 |
| G-BLC-1 | G-BLC-1 FAIL | tier 1 (4 A/B/E/F) BLC_8 0 (>= 0.09 False), BLC_full 0 (>= 0.05 False), a priori; B0-template 0 / 0 |
| G-QUAL | G-QUAL PASS | 63 configs, 0 off the reference quality block, 0 of 44 edits outside the whitelist, 0 config sha mismatches, 0 forbidden-flag literals, remedies scan ok True |
| G-FID | G-FID FAIL | 4 family metrics compared, worse B:p99_over_hf,E:p99_over_hf, B0-template measured fresh |
| G-COST | G-COST FAIL | cells 43.6016 (FAIL), over budget 0 (PASS), wall 332.284 s at 6 streams (PASS), peak 0.0966416 of RAM (PASS), orphans 0 (PASS) |
| G-DET | G-DET PASS | 4/4 rows, 0 row diffs, content 4/4, replays True True |
| G-EXPL | G-EXPL PASS | 9 campaigns audited, 0 not ok |
| G-OPT | G-OPT REPORTED | full vs rules MFR 0.5 -> 0.5 (gain 0), BLC_8 0.166667 -> 0.166667 (gain 0), regressed none; optimiser marginal MFR 0.5 -> 0.5 |
| G-ABL | G-ABL REPORTED | full MFR 0.5 BLC_8 0.166667, -preflight MFR 0.5 BLC_8 0.166667, -remedies MFR 0.5 BLC_8 0.166667, -prior MFR 0.5 BLC_8 0.166667, -optimiser MFR 0.5 BLC_8 0.166667, rules + remedies only MFR 0.5 BLC_8 0.166667, B0-template (measured fresh) MFR 0.666667 BLC_8 0 |

## Mesh failure per family (G-FAIL)

| family | n | B0-template | full | full MFR [95 % CI] | full strict |
|---|---|---|---|---|---|
| A | 1 | 1 | 1 | 1.000 [0.025, 1.000] | 1 |
| B | 1 | 0 | 0 | 0.000 [0.000, 0.975] | 1 |
| D | 1 | 1 | 0 | 0.000 [0.000, 0.975] | 0 |
| E | 1 | 0 | 0 | 0.000 [0.000, 0.975] | 1 |
| F | 1 | 1 | 1 | 1.000 [0.025, 1.000] | 1 |
| G | 1 | 1 | 1 | 1.000 [0.025, 1.000] | 1 |
| all | 6 | 4 | 3 | 0.500 [0.118, 0.882] | 5 |

## Boundary-layer capture, a priori (G-BLC-0)

Tier 0 (1 D/F commensurate geometries): full BLC_8 1.000, BLC_full 1.000; B0-template 0.000 / 0.000; a priori.

| family | n | BLC_8 | BLC_full | with CAPABILITY-LIMITED patches |
|---|---|---|---|---|
| A | 1 | 0.000 | 0.000 | 0 |
| B | 1 | 0.000 | 0.000 | 1 |
| E | 1 | 0.000 | 0.000 | 1 |
| F | 1 | 0.000 | 0.000 | 0 |
| G | 1 | 0.000 | 0.000 | 0 |

CAPABILITY-LIMITED patches lose their layers on a snapped curved wall, which this mesher cannot grow (docs/15 §I-1); they are reported, never gated.

## G-BLC-1 (headline 2, tier 1)

Tier 1 (4 A/B/E/F geometries): full BLC_8 0.000 (target >= 0.09), BLC_full 0.000 (target >= 0.05); B0-template 0.000 / 0.000; a priori.

| family | n | BLC_8 | BLC_full |
|---|---|---|---|
| A | 1 | 0.000 | 0.000 |
| B | 1 | 0.000 | 0.000 |
| E | 1 | 0.000 | 0.000 |
| F | 1 | 0.000 | 0.000 |

## Fidelity (G-FID)

| family | full meshes | B0 meshes | feature_tolerance 0 on sharp bodies | metric | full median | B0 median | no worse |
|---|---|---|---|---|---|---|---|
| B | 1 | 1 | 0 | p99_over_hf | 0.062 | 0.016 | False |
| B | 1 | 1 | 0 | pinned_frac | 0.000 | 0.000 | True |
| B | 1 | 1 | 0 | feature_share | absent | absent | - |
| B | 1 | 1 | 0 | feature_capture | absent | absent | - |
| D | 1 | 0 | 1 | p99_over_hf | 0.000 | absent | - |
| D | 1 | 0 | 1 | pinned_frac | 0.000 | absent | - |
| D | 1 | 0 | 1 | feature_share | 0.000 | absent | - |
| D | 1 | 0 | 1 | feature_capture | 1.000 | absent | - |
| E | 1 | 1 | 0 | p99_over_hf | 0.032 | 0.016 | False |
| E | 1 | 1 | 0 | pinned_frac | 0.000 | 0.000 | True |
| E | 1 | 1 | 0 | feature_share | absent | absent | - |
| E | 1 | 1 | 0 | feature_capture | absent | absent | - |

A mesh snapped with feature_tolerance 0 captures no feature edge and scores a share of 0.

## Cost (G-COST)

- cells: median full 421911.0 / median B0-template 9676.5 on 2 both-pass geometries, ratio 43.602 (FAIL, <= 1.50)
- over budget: 0 geometries over the cell budget (PASS)
- wall: 332.3 s at 6 streams, the best a 12-stream run could do is 332.3 s (PASS, <= 10800 s)
- RAM: peak 0.097 of total (PASS, <= 0.60)
- orphans: 0 (PASS)

## Determinism (G-DET)

The 20 G-DET geometries, fixed before any outcome: D-2-106 F-2-026.

gdet-1 against gdet-2: 4/4 rows, 0 row diffs, content 4/4 equal; verdict PASS.

Replay through rules.py and remedies.py: gdet-1 ok True (6 decisions), gdet-2 ok True (6 decisions).

The full campaign against gdet-1 on these ids: 4 rows, 0 row diffs, content 4/4 equal; ok True.

## Guards (G-QUAL, G-EXPL)

63 configs, 0 off the reference quality block, 0 of 44 edits outside the whitelist, 0 config sha mismatches, 0 forbidden-flag literals, remedies scan ok True; 9 campaigns audited, 0 not ok.

- b0-template: audit ok True, 5 rows, 0 untemplated.
- full: audit ok True, 9 rows, 0 untemplated.
- gdet-1: audit ok True, 4 rows, 0 untemplated.
- gdet-2: audit ok True, 4 rows, 0 untemplated.
- no-optimiser: audit ok True, 9 rows, 0 untemplated.
- no-preflight: audit ok True, 9 rows, 0 untemplated.
- no-prior: audit ok True, 9 rows, 0 untemplated.
- no-remedies: audit ok True, 5 rows, 0 untemplated.
- rules: audit ok True, 9 rows, 0 untemplated.

## Ablation (G-ABL)

| system | MFR [95 % CI] | strict | BLC_8 | BLC_full | cells median |
|---|---|---|---|---|---|
| full | 0.500 [0.118, 0.882] | 5 | 0.167 | 0.167 | 445124 |
| -preflight | 0.500 [0.118, 0.882] | 5 | 0.167 | 0.167 | 445124 |
| -remedies | 0.500 [0.118, 0.882] | 5 | 0.167 | 0.167 | 445124 |
| -prior | 0.500 [0.118, 0.882] | 5 | 0.167 | 0.167 | 445124 |
| -optimiser | 0.500 [0.118, 0.882] | 5 | 0.167 | 0.167 | 445124 |
| rules + remedies only | 0.500 [0.118, 0.882] | 5 | 0.167 | 0.167 | 445124 |
| B0-template (measured fresh) | 0.667 [0.223, 0.957] | 6 | 0.000 | 0.000 | 9676.5 |

| system | A | B | D | E | F | G |
|---|---|---|---|---|---|---|
| full | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| -preflight | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| -remedies | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| -prior | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| -optimiser | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| rules + remedies only | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| B0-template (measured fresh) | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 |

### G-OPT

- as written (full vs rules + remedies only): MFR 0.500 -> 0.500 (gain 0.000), BLC_8 0.167 -> 0.167 (gain 0.000), regressed none.
- optimiser marginal (full vs -optimiser): MFR 0.500 -> 0.500 (gain 0.000), BLC_8 0.167 -> 0.167 (gain 0.000), regressed none.
- rules+opt vs rules: MFR 0.500 -> 0.500 (gain 0.000), BLC_8 0.167 -> 0.167 (gain 0.000), regressed none.

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

- full: OPT-DISABLED 2, PR-DISABLED 5
- no-optimiser: PR-DISABLED 5
- no-preflight: OPT-DISABLED 2, PR-DISABLED 5
- no-prior: OPT-DISABLED 2
- no-remedies: PR-DISABLED 5

## The campaigns

| campaign | system | ablate | geometries | rows | meshed | reused | RAM-guarded | wall s | peak RSS MiB | harness errors | replay | audit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | full | none | 6 | 9 | 9 | 0 | 0 | 332.3 | 2731.6 | 0 | True | True |
| b0-template | b0-template | none | 6 | 5 | 5 | 0 | 0 | 9.1 | 249.9 | 0 | True | True |
| gdet-1 | full | none | 2 | 4 | 4 | 0 | 0 | 326.1 | 1867.9 | 0 | True | True |
| gdet-2 | full | none | 2 | 4 | 4 | 0 | 0 | 343.0 | 2488.2 | 0 | True | True |
| no-remedies | full | remedies | 6 | 5 | 0 | 5 | 0 | 3.9 | 179.5 | 0 | True | True |
| no-optimiser | rules+prior | none | 6 | 9 | 0 | 9 | 0 | 4.0 | 180.8 | 0 | True | True |
| rules | rules | none | 6 | 9 | 0 | 9 | 0 | 4.0 | 181.6 | 0 | True | True |
| no-prior | rules+opt | none | 6 | 9 | 0 | 9 | 0 | 3.9 | 182.3 | 0 | True | True |
| no-preflight | full | preflight | 6 | 9 | 0 | 9 | 0 | 3.9 | 183.0 | 0 | True | True |

## Tuning context (not the result)

On the tuning split: rules-only 241 of 420 tuning geometries pass at attempt 1, the prior 241, the shuffled control 240.667 on average: of the 0-geometry gain, -0.333333 is reached with shuffled fingerprints and 0.333333 is the fingerprint's own.

The optimiser's tuning verdict: PASS (enabled True); final MFR gain 0.036, BLC_8 gain 0.029, regressed none.

## Departures

- docs/15 §F measures G-COST's campaign wall time at 12 streams; the house caps campaigns at 6 while the solver workflow runs, so the wall item passes when the 6-stream time is at most 3 h, fails when even perfect 12-stream scaling (half the 6-stream time) or the longest single geometry exceeds 3 h, and is undecided in between.
- The ablations reuse what earlier campaigns of this evaluation measured: an attempt whose (geometry, config sha) was meshed returns that outcome (the mesher is deterministic, §F G-DET). The full system, the B0-template re-measure and both G-DET runs are meshed fresh.
- G-FID's feature-edge share is n_snapped_to_edge / n_feature_edges; a mesh snapped with feature_tolerance 0 extracts no edges and reports 0 of them, so on a body whose fingerprint has sharp edges its share is 0.0, not undefined. A body without sharp edges is left out.
- The -preflight ablation meshes configs L0 would refuse. On this shared machine an attempt whose octree probe predicts more than 8192 MiB of peak memory is not run and is scored as an F1 no-mesh failure (class crash); the count is reported, and it is 0 wherever L0 runs, whose budget check keeps a probe under 2 M leaves (4.9 GiB).
- G-OPT as written compares the full system with rules+remedies only (mode rules); the optimiser's own marginal (full against -optimiser) and rules+opt against rules are reported beside it.
- -remedies runs K = 1, so the optimiser, which acts only after the remedies are spent with attempts left, never acts in it either.
- The 20 G-DET geometries are fixed from the lock's test ids before any outcome exists: within each family by sha256('gdet:' + id), drawn round-robin A, B, D, E, F, G.
- The tier-0 stratum of G-BLC-0 is the D and F rows whose manifest field commensurate is true.
- A geometry that ends SURFACE-OPEN, SURFACE-REFUSED, REFUSED or HARNESS-ERROR is a failure in every system (§D.1 F1), the sealed baselines included (read through campaign.terminal_of).
- The fresh test seed is corpus seed 2: one pool per family at seed 2 with the seed-1 sizes, the same stratified largest-remainder draw (salt 17, no spent ids), and only its 180 test rows are written, to corpus/manifests/seed2/ under a new write-once lock; the seed-1 split and its lock are untouched and spent.
- No sealed baseline exists for the fresh seed: B0-template is meshed fresh in this evaluation with the same binary and is the baseline of G-FAIL, G-BLC-0, G-FID and G-COST; B0-LHS is not measured and is left out of G-ABL.
- G-BLC-1 (tier 1: A, B, E, F) passes when the full system's mean a-priori BLC_8 over those geometries is at least 0.09 and its mean BLC_full at least 0.05, the targets the user fixed (D-L9) from the tuning re-measure before this seed was opened.
- G-FID gains the F3e row: among passing meshes the median chain-form feature-capture share is no lower than B0-template's, per family.
- A reduced scope runs a pre-registered subset under the same lock: per family the first id by sha256('reduced:' + id), family D drawn among its commensurate rows, G-DET on the subset's D and F ids. Its verdicts are REDUCED and never a headline; the full scope runs all 180 geometries under the same plan.
- Both learned layers ship disabled (2026-10-03), so full, -prior, -optimiser and rules + remedies mesh the same configs and G-OPT is reported, not decided.
- The fresh test manifest is opened by each scope (split.load('test2', 'evaluate')) only after the write-once lock records the plan; the reduced and full scopes share that one plan, and a run with any other plan is refused: the seed is spent (§F).
