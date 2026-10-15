# The go / no-go projection (AUT-12): NO-GO

System `aut12-rules` (rules, binary `e956347fed45`) against B0-template `b0-template-tuning` on 420 tuning geometries, projected to a 180-body test split (projection.py, PJ-1..PJ-8).

The AUT-12 tuning re-measure (campaign aut12-rules, 2026-10-15: rules after AUT-02..AUT-11 with R-TE and R-ECON on, binary e956347f; its bundle is the one prior/aml2 was gated on) against B0-template re-meshed with the same binary the same evening (projection/tuning_b0-template_AUT12.json.gz); 6 streams each, while lanes S and M shared the 4 CPUs.

| gate | projected | numbers |
|---|---|---|
| G-FAIL | FAIL | failures 71/420 (bar 23 = floor(420 x 10 / 180)); B0-template 380; McNemar b 310 c 1 -> 133/0, p 1.83671e-40; families worse: none |
| G-BLC-0 | PASS | n 35, BLC_8 0.971429, BLC_full 0.961719 (B0 0, 0) |
| G-BLC-1 | PASS | n 252, BLC_8 0.150794, BLC_full 0.091762 (B0 0, 0) |
| G-QUAL | PASS | 493 edits, 0 off the whitelist (not projected: the configs' quality block (a bundle carries no configs)) |
| G-FID | PASS | compared 6, worse none |
| G-COST | FAIL | cells FAIL (median 52632 vs 9008, ratio 5.84281 on 39 both-pass); budget PASS (0 over); wall PASS (2139.96 s scaled from 4993.25 s); RAM PASS; orphans PASS |
| G-EXPL | PASS | the bundle's summary audit |
| G-DET | NOT-PROJECTED | needs two runs of the same ids |
| G-OPT | NOT-PROJECTED | decided on the test ablation |

**Per family** (failures system / B0-template, BLC_8 system):

- A: 39 / 84 of 84, BLC_8 0
- B: 0 / 63 of 84, BLC_8 0.142857
- D: 1 / 84 of 84, BLC_8 0.511905
- E: 1 / 29 of 42, BLC_8 0.119048
- F: 15 / 42 of 42, BLC_8 0.5
- G: 15 / 78 of 84, BLC_8 0.285714

**Verdict: NO-GO.**
- G-FAIL FAIL: 71/420 > 23 (the 95 % upper bound on 180 test bodies holds at <= 10 failures)
- G-COST FAIL: cells FAIL; cells median 52632 vs B0 9008 (ratio 5.84281 > 1.5)
