<!-- meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). Source-available, not Open Source. No GPL-licensed source was consulted. -->

# G-DET - the campaign runner gate (AM-11, docs/15 §F)

date: 2026-09-24
binary sha256: 054bba67c8650082431ac01325e497208a9bc663c4760a36f7472ff62814a90b
git head: b007ef2eb4cf069ba61564c6025ed8b7c0b0e654
verdict: PASS

## part 1 - PASS

rows 29, terminals {"CAPABILITY-LIMITED": 9, "EXHAUSTED": 3, "NO-REMEDY": 1, "PASS": 4, "REFUSED": 1, "SURFACE-OPEN": 2}, decisions 94, audited 6, peak rss 3087.0/3183.4 MiB (10.9%/11.3% of RAM), max mesher 6, l5 peak 708.1 MiB, l5 cap 12, wall 646.1 s

- 3: {"n": 6, "peak_mib_max": 36.9375, "peak_mib_median": 36.18359375, "seconds_median": 1.50891114998376}
- 4: {"n": 36, "peak_mib_max": 439.97265625, "peak_mib_median": 123.771484375, "seconds_median": 15.535848399973474}
- 5: {"n": 10, "peak_mib_max": 708.05859375, "peak_mib_median": 585.51171875, "seconds_median": 30.582631699973717}
- 6: {"n": 6, "peak_mib_max": 1323.83203125, "peak_mib_median": 1288.880859375, "seconds_median": 81.42263520002598}

## part 2 - PASS

seal a ok, b ok, c ok (first test id A-1-008)

- a: true
- b: true
- c: true

## part smoke - PASS

rows 4, terminals {"CAPABILITY-LIMITED": 1, "PASS": 2, "SURFACE-OPEN": 1}, decisions 14, audited 8, peak rss 147.9 MiB, wall 37.6 s
