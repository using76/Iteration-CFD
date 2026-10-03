# TG4 turbulent nozzle study / TG4 난류 노즐 연구

The turbulent twin of `../g4_nozzle` (docs/16 §H.5 TG4): the locked bilingual brief's
axisymmetric air contraction (inlet 300 mm, exit 212.1 mm, Q 1.590 m3/s, no wall separation,
exit non-uniformity at most 1 % over r <= 0.8 R_e, the a priori acceleration parameter
K_max <= 3e-6 at Re_De 6.30e5 as a hard row decided at CAD time and re-checked from the CFD
edge velocity, wall at least 3 mm, Mach at most 0.3, shortest total length) evaluated by
`tg4.py` through the loop with the turbulent evaluator `cfd_turb`
(`evaluate_cfd.evaluate_turb`: `wedge_mesh.run_turb`, `case_writer.write_turb_case`,
`post.post_turb`). The start (poly7, L/D_i 0.5) fails the a priori K row on purpose
(K_max 4.97e-6), which is the G4-style precondition.

`../g4_nozzle`의 난류 쌍둥이(docs/16 §H.5 TG4)이다. 잠긴 이중 언어 브리프의 축대칭 공기 수축
노즐(입구 300 mm, 출구 212.1 mm, Q 1.590 m3/s, 벽 박리 없음, 출구 불균일도 r <= 0.8 R_e에서
1 % 이하, CAD 시점에 1-D 면적 규칙으로 결정하고 CFD 에지 속도로 재검사하는 하드 행
K_max <= 3e-6 at Re_De 6.30e5, 벽 3 mm 이상, 마하수 0.3 이하, 전체 길이 최소)을 난류 평가기
`cfd_turb`(`evaluate_cfd.evaluate_turb`: `wedge_mesh.run_turb`, `case_writer.write_turb_case`,
`post.post_turb`)로 루프를 통해 `tg4.py`가 평가한다. 시작점(poly7, L/D_i 0.5)은 선험적 K 행을
일부러 실패한다(K_max 4.97e-6) - G4식 전제조건이다.

## Files / 파일

- `brief.json` - the locked bilingual brief, NFC, two paragraphs. / 잠긴 이중 언어 브리프(NFC, 두 문단).
- `proposal.json` - the seven rows compiled to REQ-001..007 plus SYS-*. / 일곱 행이 REQ-001..007과 SYS-*로 컴파일된다.
- `report.json` - `reqs.py check` output, status ok, no refusals. / `reqs.py check` 결과(status ok, refusal 없음).
- `requirements.json`, `requirements.lock` - `reqs.py lock report.json docs16-H5`, write-once. / write-once 잠금 문서.
- `start.json` - the pinned start (poly7 L/D_i 0.5, evaluator cfd_turb, repeat_band 0.0). / 고정 시작점.
- `tg4.py` - init, evaluate, judge_tg4, record, report and the selftest. / init·evaluate·judge_tg4·record·report와 셀프테스트.

## Commands / 명령

```
python tools/cad/studies/tg4_nozzle/tg4.py --selftest
python tools/cad/studies/tg4_nozzle/tg4.py init STUDY_DIR [--registry R]
python tools/cad/studies/tg4_nozzle/tg4.py evaluate STUDY_DIR NAME LEVEL [PARAMS_JSON] [--registry R]   # supervisor only
python tools/cad/studies/tg4_nozzle/tg4.py record STUDY_DIR OUT_JSON [--registry R]
python tools/cad/studies/tg4_nozzle/tg4.py report STUDY_DIR OUT_MD
```

## Status / 상태

TG4 is OPEN while TG0 is OPEN (docs/16 §H.5: no turbulent nozzle number counts before TG0
passes), so the full walk is deferred - the CAD prefilter of 4 x 256 candidates, the at most
24 L1 evaluations, the L2 confirmation at 6000 iterations and the deep replay are the
supervisor's live steps.

TG0이 OPEN인 동안 TG4도 OPEN이다(docs/16 §H.5: TG0이 통과하기 전에는 어떤 난류 노즐 수치도
인정하지 않는다). 따라서 전체 워크는 보류된다 - 4 x 256 후보의 CAD 프리필터, 최대 24회 L1
평가, 6000 반복의 L2 확인, 깊은 리플레이는 감독자의 라이브 단계다.

REDUCED live run (2026-10-04, `tg4_record.json`, binary 50471caa, GPU not shared): the start stops at
checks on REQ-005 (a priori K_max 4.967e-6 > 3e-6), so the precondition is met; poly3 at L/D_i 0.6
passes the a priori row (2.85e-6) but its L1 solve is unsteady on U_decades, so its CFD rows are
not evaluable; its reported values are exit non-uniformity 6.7 % and a CFD K_max of 4.04e-6, over
3e-6. `report.md` comes from iterations.jsonl alone, which holds only the genesis row, because the
walk has not run.

REDUCED 실측(2026-10-04, `tg4_record.json`, 바이너리 50471caa, GPU 비공유): 시작점은 REQ-005
(선험 K_max 4.967e-6 > 3e-6)에서 checks 단계에 멈추므로 전제조건이 충족된다. L/D_i 0.6의 poly3는
선험 행(2.85e-6)을 통과하지만 L1 해석이 U_decades 기준 unsteady여서 CFD 행은 판정되지 않는다.
보고값은 출구 불균일도 6.7 %, CFD K_max 4.04e-6으로 3e-6을 넘는다. 워크가 돌지 않았으므로
iterations.jsonl에는 시작 행만 있고, `report.md`는 그 파일만 읽어 만든다.
