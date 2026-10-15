<!-- meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). Source-available, not Open Source. No GPL-licensed source was consulted. -->
# R-TE - the quick set of tuning wings (AUT-04)

- date 2026-10-15; R-TE campaign r-te-k4 (git db2542c15d82, binary a6e1583510bb, 3 streams); the rules before it rules-k4 (git db2542c15d82, binary a6e1583510bb)
- the rule: after an F3a-F3c attempt on a sharp body off R-PLANE with a feature bump, RM-TE sets feature_level = wall level, max_level = the largest level left, tau = 1.5 h_f and R-SNAP's band h_f/256 at the new h_f
- bodies (pre-registered, te_quick_ids): A-1-012 A-1-024 A-1-047 A-1-037 A-1-013 A-1-017 A-1-049 A-1-118 A-1-098 A-1-034 A-1-099 A-1-109
- identity over the tuning rows with an L5 fingerprint: 37 R-PLANE and 44 smooth rows propose the same config for a synthetic F3c with and without RM-TE, 0 changed (ok); 169 of 276 sharp rows take RM-TE; 15 refused
- **F3-clean at K = 4 (final attempt): R-TE 8 of 12** against the target >= 6; the rules before it 0; L5 recorded 0. Any attempt clean: 8 / 0 / 0. Verdict: **PASS**; harness errors 0

| body | stratum | L5 recorded | rules before R-TE | R-TE | R-TE chain | max/h_f | p99/h_f | pinned | capture | cells | s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A-1-012 | easy | F3b F3c F3d F3e (EXHAUSTED) | F3c F3d F3e (EXHAUSTED) | F3-clean (CAPABILITY-LIMITED) | R-SNAP RM-TE | 0.2553 | 0.0677 | 0.0241 | 0.5035 | 370374 | 891 |
| A-1-024 | medium | F3b F3c F3d F3e (EXHAUSTED) | F3b F3c F3d F3e (EXHAUSTED) | F3e (NO-REMEDY) | R-SNAP RM-TE | 0.3854 | 0.0825 | 0.0301 | 0.3569 | 231654 | 749 |
| A-1-047 | hard | F3b F3c F3e (EXHAUSTED) | F3b F3c F3e (EXHAUSTED) | F3-clean (CAPABILITY-LIMITED) | R-SNAP RM-TE | 0.1891 | 0.0039 | 0.0114 | 0.8458 | 440411 | 1020 |
| A-1-037 | medium | F3d F3e (EXHAUSTED) | F3d F3e (EXHAUSTED) | F3-clean (CAPABILITY-LIMITED) | R-SNAP RM-TE | 0.2169 | 0.0385 | 0.0165 | 0.5683 | 386420 | 763 |
| A-1-013 | hard | F3b F3c F3e (EXHAUSTED) | F3b F3c F3e (EXHAUSTED) | F3-clean (CAPABILITY-LIMITED) | R-SNAP RM-TE | 0.1881 | 0.0039 | 0.0098 | 0.7628 | 610486 | 1608 |
| A-1-017 | medium | F3c (EXHAUSTED) | F3c F3e (EXHAUSTED) | F3-clean (CAPABILITY-LIMITED) | R-SNAP RM-TE | 0.1582 | 0.0111 | 0.0146 | 0.9139 | 384649 | 1177 |
| A-1-049 | hard | F3e (NO-REMEDY) | F3e (NO-REMEDY) | F3e (NO-REMEDY) | R-SNAP | 0.0039 | 0.0038 | 0.0000 | 0.2837 | 292075 | 68 |
| A-1-118 | medium | F3c (EXHAUSTED) | F3c F3e (EXHAUSTED) | F3-clean (CAPABILITY-LIMITED) | R-SNAP RM-TE | 0.3307 | 0.0039 | 0.0049 | 0.8557 | 252391 | 762 |
| A-1-098 | hard | F3b F3c F3d F3e (EXHAUSTED) | F3a F3b F3c F3d F3e (EXHAUSTED) | F3d F3e (EXHAUSTED) | R-SNAP RM-TE RM-SNAP-WALL | 0.0025 | 0.0023 | 0.0000 | 0.0309 | 17208 | 245 |
| A-1-034 | medium | F3c (EXHAUSTED) | F3c F3e (EXHAUSTED) | F3-clean (CAPABILITY-LIMITED) | R-SNAP RM-TE | 0.2574 | 0.0039 | 0.0072 | 0.8950 | 273685 | 870 |
| A-1-099 | hard | F3b F3c F3d F3e (EXHAUSTED) | F3b F3c F3d F3e (EXHAUSTED) | F3d F3e (EXHAUSTED) | R-SNAP RM-TE RM-SNAP-TAU | 0.0044 | 0.0038 | 0.0000 | 0.1308 | 125290 | 299 |
| A-1-109 | medium | F3c F3e (EXHAUSTED) | F3c F3e (EXHAUSTED) | F3-clean (CAPABILITY-LIMITED) | R-SNAP RM-TE | 0.3042 | 0.0038 | 0.0043 | 0.6983 | 248705 | 722 |

A quick set decides the rule; the full tuning effect is AUT-12's L-run.
