<!-- meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). Source-available, not Open Source. No GPL-licensed source was consulted. -->

# G-PREFLIGHT (docs/15 §F) - preflight.py's gate

date 2026-09-24T04:47:32Z; binary sha256 054bba67c8650082431ac01325e497208a9bc663c4760a36f7472ff62814a90b; git HEAD ead44b0505713d0284a5ed8800d3489a28ac99ff; seed 1; n 10000; streams 6

verdict: PASS

## part 1 - the mirror against -dryRun (10000 configs) - PASS

10000 rows, 0 mirror-refused/dryRun-passed, 0 mirror-passed/dryRun-refused, 0 field mismatches, 0 bad exits, 0 undetected, 3740 refused / 6260 passed by -dryRun, min kind 122 of 60.

per STL: box_sphere: 1899 refused / 3101 passed, wing_b: 1841 refused / 3159 passed.

preflight-only refusals (dryRun passed, preflight refused): PF-DOMAIN 66, PF-PATCH 128, PF-QUALITY 68, PF-THIN 2090, WL-FORBIDDEN 119, WL-RANGE 841, WL-UNLISTED 319.

## part 2 - C-THIN to the digit - PASS

h = 0.03757424300735249; preflight refuses 3 * 0.0001 / 0.03757424300735249 = 0.007984192787098767 < 0.05; the live refusal line is bit-equal (True); "%.6f" % ratio = True; the castellated prediction refuses 3 * 0.0001 / 0.125 = 0.0024 (True).

mesher: layers: the first layer is thinner than the quality gate allows - 3 * 0.0001 / 0.03757424300735249 = 0.007984192787098767 < min_thickness_ratio = 0.05 (92.51); every layer cell would fail G5, so none is inserted

## part 3 - the castellated h prediction against 48 real stage-5 outcomes - PASS

castellated: n 12, refused 6, not_refused 6, not_reached 0, crashes 0, false_pass 0 (rate 0.0), false_refuse 0 (rate 0.0), h_pred/h_mesher {'min': 1.0, 'median': 1.0, 'max': 1.0}.

snapped: n 36, refused 2, not_refused 34, not_reached 0, crashes 0, false_pass 0 (rate 0.0), false_refuse 11 (rate 0.3055555555555556), h_pred/h_mesher {'min': 2.4893993071395766, 'median': 3.3267469946244863, 'max': 3.3267469946244863}.

on snapped walls the castellated h is a prediction; these rates are reported, not gated (docs/15 §F).
