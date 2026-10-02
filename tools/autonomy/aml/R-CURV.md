<!-- meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). Source-available, not Open Source. No GPL-licensed source was consulted. -->
# R-CURV - its F3 cost on the tuning split

- date: 2026-10-02
- binary: 3d90ce9156119b16ee6fdd1359c5913152ebc29be3e218ebb903fe70900d923b
- pairs run: 133 of 420 rows; skipped: family 84, rcurv_abstain 34, rcurv_pass 147, refused 1, same config 21
- harness errors 0
- peak rss 6846 MiB, max live mesher 6, orphans 0, wall 5675.4 s

| family | n | level with | level without | F3 with | F3 without | F3 with-only | F3 without-only | failure with | failure without | strict with | strict without | refusals with | refusals without | cells median with | cells median without | s median with | s median without | capability-limited area with | capability-limited area without |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 33 | 5 | 4 | 29 | 32 | 0 | 3 | 30 | 33 | 33 | 33 | 0 | 0 | 395113 | 124525 | 133.9 | 46.5 | 1.000 | 1.000 |
| B | 70 | 6 | 4 | 0 | 14 | 0 | 14 | 0 | 14 | 58 | 67 | 0 | 1 | 1254895 | 25928 | 181.7 | 6.6 | 0.829 | 0.957 |
| D | 27 | 6 | 4 | 2 | 6 | 2 | 6 | 2 | 6 | 15 | 26 | 0 | 0 | 722270 | 45584 | 45.5 | 7.8 | 0.556 | 0.963 |
| E | 1 | 5 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 398438 | 53828 | 101.3 | 12.0 | 1.000 | 1.000 |
| F | 2 | 5 | 4 | 0 | 1 | 0 | 1 | 1 | 2 | 2 | 2 | 0 | 0 | 635835 | 41666 | 25.9 | 4.4 | 0.500 | 0.500 |
| all | 133 | 6 | 4 | 31 | 53 | 2 | 24 | 33 | 55 | 109 | 129 | 0 | 1 | 762285 | 50048 | 156.6 | 10.8 | 0.812 | 0.962 |

Reported, not gated: the user decides keep / retune / drop (DECISIONS 2026-09-24).
