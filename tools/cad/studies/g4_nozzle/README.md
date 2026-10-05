# g4_nozzle / G4 노즐 연구

The G4 nozzle study of docs/16 §H.4: an axisymmetric air contraction, inlet 60 mm, exit 20 mm,
7.07 L/s, no wall separation, exit non-uniformity at most 1 % over r <= 0.8 R_e, Cd at least
0.96, wall at least 3 mm, Mach at or below 0.3, total length minimised. The locked requirement
set is the truth; the loop's evaluator is `cfd` (tools/cad/evaluate_cfd.py), the real S3-S10
path, on the REDUCED route: each evaluation runs outside the walk and lands in the study's
cache, which a later walk reuses with zero work.

docs/16 §H.4의 G4 노즐 연구: 축대칭 공기 수축 노즐, 입구 60 mm, 출구 20 mm, 유량 7.07 L/s,
벽 박리 없음, 출구 불균일도 r <= 0.8 R_e에서 1 % 이하, Cd 0.96 이상, 벽 두께 3 mm 이상,
마하수 0.3 이하, 전체 길이 최소화. 잠긴 요구사항 집합이 진실의 원천이고, 평가기는 루프의
`cfd`(tools/cad/evaluate_cfd.py, 실제 S3-S10 경로)이며 REDUCED 경로로 돈다. 각 평가는 워크
밖에서 실행되어 연구의 캐시에 들어가고, 나중 워크가 0 작업으로 재사용한다.

## Files / 파일
- `brief.json` - the fixed brief, English and Korean / 고정 브리프(영·한)
- `proposal.json` - the flat requirement rows / 요구 행
- `report.json` - reqs.py check of the proposal against the brief / 심사 보고서
- `requirements.json`, `requirements.lock` - the locked set, write-once / 잠긴 집합(일회 기록)
- `start.json` - the study start: evaluator `cfd`, repeat_band 0.0, poly7 L/D 0.5 t_wall 0.004
- `g4.py` - init, evaluate, judge_g4, prefilter, report, record / 초기화·평가·판정·프리필터·보고·기록
- `prefilter_subset.jsonl` - written into the study directory at run time, not committed (it can
  name temp paths) / 실행 때 연구 디렉터리에 기록되며 커밋하지 않는다(임시 경로를 실을 수 있다)

## Commands / 명령
```
python tools/cad/studies/g4_nozzle/g4.py init STUDY_DIR [--registry R]
python tools/cad/studies/g4_nozzle/g4.py evaluate STUDY_DIR NAME LEVEL [PARAMS_JSON] [--registry R]
python tools/cad/studies/g4_nozzle/g4.py prefilter STUDY_DIR N_PER_LAW [--registry R]
python tools/cad/studies/g4_nozzle/g4.py record STUDY_DIR OUT_JSON [--registry R]
python tools/cad/studies/g4_nozzle/g4.py report STUDY_DIR OUT_MD
```

## Status / 상태
Under the cad-wedge/2 recipe the study start meshes and passes GC-6 at L0, L1 and L2, so the
start can be evaluated through the mesh gate. `prefilter` runs the walk's own CAD prefilter on
the first N Sobol points of each law (the REDUCED subset, at most `sobol_pool` per law), keys the
rows by (law, sobol_index) in `prefilter_subset.jsonl` and reuses an existing row without
recomputing it. The live numbers are in `g4_record.json`, written by `record`.

cad-wedge/2 레시피에서 이 연구의 시작 설계는 L0, L1, L2 전부에서 메시가 만들어지고 GC-6를 통과하므로
시작 설계를 메시 게이트 너머까지 평가할 수 있다. `prefilter`는 각 법칙의 첫 N개 Sobol 점(REDUCED
부분집합, 법칙당 최대 `sobol_pool`)에 워크의 CAD 프리필터를 돌리고, 행을 (law, sobol_index)로
`prefilter_subset.jsonl`에 기록하며 이미 있는 행은 다시 계산하지 않고 재사용한다. 살아있는 수치는
`record`가 쓰는 `g4_record.json`에 있다.

G4 stays OPEN until the full run - the walk's CAD prefilter of 4 x 256 candidates, at most 24
L1 evaluations, the L2 confirmation at 6000 iterations and the deep replay - which is deferred
to the supervisor. `cfd_u` is OPEN: the nominal record knows no gci_fine yet, so the loop must
not lock on this u (docs/16 §E.4).

전체 실행(워크의 4 x 256 CAD 프리필터, 최대 24회 L1 평가, 6000 반복 L2 확인, 깊은 리플레이)은
감독자에게 남겨 두었으므로 G4는 OPEN이다. `cfd_u`도 OPEN이다. 명목 기록이 아직 gci_fine을
모르므로 루프는 이 u에 못을 박아서는 안 된다(docs/16 §E.4).
