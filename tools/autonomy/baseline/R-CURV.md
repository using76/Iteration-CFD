<!-- meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). Source-available, not Open Source. No GPL-licensed source was consulted. -->
# R-CURV - its F3 cost on the tuning split

- date: 2026-09-25
- binary: 054bba67c8650082431ac01325e497208a9bc663c4760a36f7472ff62814a90b
- pairs run: 133 of 420 rows; skipped: family 84, rcurv_abstain 34, rcurv_pass 147, refused 1, same config 21
- harness errors 0
- peak rss 9579 MiB, max live mesher 6, orphans 0, wall 12170.1 s

| family | n | level with | level without | F3 with | F3 without | F3 with-only | F3 without-only | failure with | failure without | strict with | strict without | refusals with | refusals without | cells median with | cells median without | s median with | s median without | capability-limited area with | capability-limited area without |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 33 | 5 | 4 | 33 | 33 | 0 | 0 | 33 | 33 | 33 | 33 | 0 | 4 | 395113 | 124525 | 255.2 | 85.8 | 1.000 | 1.000 |
| B | 70 | 6 | 4 | 45 | 52 | 0 | 7 | 45 | 52 | 70 | 70 | 0 | 5 | 1239907 | 25382 | 625.6 | 14.4 | 1.000 | 1.000 |
| D | 27 | 6 | 4 | 27 | 27 | 0 | 0 | 27 | 27 | 27 | 27 | 0 | 0 | 647158 | 45584 | 304.6 | 22.3 | 1.000 | 1.000 |
| E | 1 | 5 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 398438 | 53828 | 97.9 | 12.8 | 1.000 | 1.000 |
| F | 2 | 5 | 4 | 1 | 1 | 0 | 0 | 2 | 2 | 2 | 2 | 0 | 0 | 635835 | 41666 | 143.7 | 11.2 | 0.500 | 0.500 |
| all | 133 | 6 | 4 | 106 | 113 | 0 | 7 | 107 | 114 | 133 | 133 | 0 | 9 | 695183 | 46262 | 283.1 | 23.5 | 0.992 | 0.992 |

Reported, not gated: the user decides keep / retune / drop (DECISIONS 2026-09-24).
