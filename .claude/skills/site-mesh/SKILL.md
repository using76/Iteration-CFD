---
name: site-mesh
description: STEP 부지 형상에서 수렴하는 사면체 격자를 만들 때(암모니아 누출 부지 같은 대형 부지, 풀·지붕 원천, Fluent 납품) 따르는 절차 — 설정 레시피, 검토·수정 게이트, 실패 진단 스크립트, 함정. "부지 격자", "step_mesh", "풀 inlet", "격자 발산", "PLC Error", "음수 부피" 같은 말이 나오면 이 스킬을 먼저 읽는다.
---

# 부지 사면체 격자 만들기 (tools/mesh)

배경과 근거는 `docs/08-site-mesh-playbook.md`에 있다. 여기는 순서와 명령만 적는다.

## 0. 원칙

- 솔버 수학은 손대지 않는다. 수렴은 형상·크기·후처리·입력으로 만든다.
- 원천(inlet) 주변 격자는 정확하게, 먼 곳은 과감히 뭉갠다(건물 볼록껍질 치환).
- 로그의 잔차만 믿지 않는다. 필드의 극값 위치(`diag/field_extremes.py`)까지 봐야 국소 폭발이 보인다.
- 긴 작업은 보이는 콘솔에서. 콘솔은 **Bash**에서 `cmd //c start "" cmd //k <cmd>`로 띄운다(PowerShell 도구의 `PYTHONIOENCODING=utf-8`이 한글 경로를 깨뜨린다). Fluent는 영문 경로만 읽으므로 납품 폴더는 ASCII 경로에 둔다.

## 1. 설정 (레시피)

`tools/mesh/examples/pool_ring_case.json`을 복사해 `step`, `out_dir`, `points`만 바꾼다. 핵심 값:

| 키 | 값 | 이유 |
|---|---|---|
| `points.<name>` | `{x, y, r, r_inner, h: 0.5}` | 지면에 원을 그려 0.5 m 끌어올린 원기둥. 윗면이 inlet `pool_<name>`, 옆면 `wall_pool_<name>`; `r_inner`로 원판과 링 분리 |
| `sizes` | `pool 0.35, box 2.5, min 0.3, near_struct 6, far_struct 20, max 60` | 700만 셀 이상; 풀 상자 ±40 m는 0.35 m |
| `sizes.gap_ratio` | `2.5` | 틈 규칙 h ≤ g/2.5 (표면 삼각형이 마주보는 면까지의 거리 g). 해결 불가한 틈은 요약 `gap_unresolvable`에, 얕은 이면각(<30°) 쐐기는 `wedges`에 위치가 나온다 |
| `solids.hull_beyond_m`, `hull_pad_m`, `fuse` | `150, 1.0, true` | 원천에서 150 m 밖 건물은 볼록껍질 프리즘(+1 m)으로 치환하고 합친 뒤 그룹별로 다시 볼록껍질 — 벽 사이 슬롯·2~5 cm 모서리 틈이 사라진다 |
| `solids.hull_snap_m` | `0.05` | 프리즘 꼭짓점 스냅 |
| `solids.touch_warn_m` | `0.05` | 절단 뒤 1 mm~5 cm 근접 정점 쌍을 기록(3-D 실패 지점 후보) |
| `post` | `sliver_rel 0.02, sliver_edge_rel 0.3, thin_push_m 0.3, min_thickness 0.05, repair_rounds 3` | 셀 크기에 비례하는 sliver 병합·밀어내기, 두께 게이트 τ=3V/A_max^1.5 ≥ 0.05 반복 수정. 절대 임계(`sliver_edge_m 0.6`, `sliver_vol_m3 0.2`)는 미세 격자를 망가뜨리므로 0 |
| `repairs` | `[]` | 선박(tag 33)도 원천에서 277 m라 다른 먼 건물처럼 볼록껍질 프리즘이 된다(재표본 1.5 m/6000면은 상부구조에 1 cm 틈, 3 m는 수밀 실패) |
| `trim.shrink_xy_m` | `600` (도메인 ±650 m) | 원 STEP ±1,250 m의 먼 곳은 지형 계단·기저 경사·벽-출구 접합 쐐기가 무한히 나와 스테디 해가 매번 다른 곳에서 터졌다; ±650 m에서 50회 무발산 |
| `solids.hull_box_snap_m`, `hull_box_inset_m` | `20, 5` | 경계 20 m 안의 프리즘 꼭짓점을 경계 안쪽 5 m로 당겨 어떤 벽도 출구에 닿지 않게; 띠 안에서 윤곽이 퇴화한 건물은 제거 |
| `sizes.gap_min_m` | `1.0` | 틈 규칙 세분의 먼 곳 바닥 — 없으면 2,400만 셀로 폭발 |
| `mesh` | `algo2d 6, algo3d 1, threads 32` | HXT(10)는 좌표를 찍어 주는 진단용으로만 |
| `classification.pool_prefix` | `""` (기본 `pool_`) | 점 이름이 그대로 inlet 패치 이름이 된다 — 점 `inlet1`·`inlet2`·`inlet3` → 패치 `inlet1`…, 옆면 `wall_inlet1`…; 변환기는 `inlet`으로 시작하는 이름을 velocity-inlet으로 기본 배정한다. 한 형상에 원천 여러 개(원판+링+다른 원판)를 넣을 때 |

설정에 모르는 키(`_note` 등)를 넣으면 도구가 거부한다. 메모는 README.txt에.

## 2. 실행

```
tools\mesh\run_step_mesh.cmd <case>.json                # 처음부터 (STEP → 절단 → 풀 → 체크포인트 → 격자)
tools\mesh\run_step_mesh.cmd <case>.json --from-checkpoint   # 크기·후처리만 바꿀 때 (절단 생략)
ofgpu-convert-mesh <case>.msh <caseDir> -fluent <case>_fluent.msh -fluentType pool_<name>=velocity-inlet
```

여러 케이스는 납품 폴더의 `run_all.cmd [케이스…]`(체크포인트가 있으면 자동 재개)로. 절단 뒤 로그에서 확인할 것:

- `solids beyond … replaced by padded convex-hull prisms: N` / `fused groups replaced …: M`
- `near-touching solid vertices: K pairs` — 20개 이하면 정상, 100개 이상이면 프리즘 교차 정점이 남은 것
- `gap pass … N need refinement … U unresolvable spots` — U > 0이면 위치를 보고 형상을 뭉갠다
- `thickness gate: worst tau …` — 남은 셀이 있으면 `thickness_gate` 목록의 위치를 본다
- `triangles reconciled … dropped a orphan, b interior, c duplicate` — 남은 미대조 면이 있으면 변환하지 않는다

## 3. 검토 (격자를 솔버에 넣기 전)

```
python tools/mesh/diag/nonorth_faces.py <caseDir> <pool_x> <pool_y>    # 85°/87°/89° 초과 면 수와 군집 위치
python tools/mesh/diag/badface_radial.py <caseDir> <pool_x> <pool_y>   # 그 면들의 원천 거리 분포
ofgpu-automesher -check <caseDir>                                        # 게이트 G1~G7 (부피·폐합·영역·비직교·두께·조건수·주소)
```

기준: 87° 초과 면 0에 가깝게(원본 부지는 1,442 → 레시피 후 0), 두께 게이트 잔여 0, 영역 1, 변환기가 "regions: … dropped" 몇 개 이내.

## 4. 솔버 시험 (6회면 충분)

```
python tools/mesh/solver_test/make_case_air.py <caseDir> 6     # 공기 1 m/s, 부력 없음, 바람 10 m/s, upwind·uncorrected·U0.3/p0.1
tools/mesh/solver_test/run_air.sh 6 3                            # 3회·6회에 필드 저장
python tools/mesh/diag/field_extremes.py <caseDir> 6 <pool_x> <pool_y>
```

판정: 6회째 |U| 최대가 수십 m/s 이내, |p| > 1000인 셀이 수십 개 이내, 극값 위치가 원천 주변이 아니어야 한다. 로그의 잔차는 국소 폭발을 숨긴다(v8 격자: 잔차 0.17이었지만 1.2 km 밖 셀에서 |U| 1e7).

## 5. 실패 진단

| 증상 | 스크립트 | 그 다음 |
|---|---|---|
| `PLC Error: A segment and a facet intersect` (3-D 실패) | `variant_from_checkpoint.py <json> hxt mesh.algo3d=10` → HXT가 `Vertex: (x,y,z)` 좌표를 찍는다; `inspect_point.py <pools.brep> x y z 3` | 좌표가 근접 정점 쌍이면 형상 치환(hull)이나 `exclude_tags`; 운(절점 번호) 문제면 `sizes.pool`을 0.01 바꿔 재시도 |
| 근접 정점 쌍이 많음 | `scan_near_vertices.py <pools.brep> 0.05` | 솔리드 쌍을 보고 뭉갠다 |
| 솔버가 셀 하나를 거부(음수 부피·미폐합) | `cell_faces.py <caseDir> <cell>` | 어느 면이 잘못 붙었는지 → 후처리·변환기 문제 |
| 국소 폭발 | `field_extremes.py` → `inspect_cells.py <caseDir> --near x y z 25` | 얇은 셀(3V/A_max)·비직교 면·슬롯 확인 → 형상 뭉개기 또는 `gap_ratio` |
| Fluent가 음수 부피 경고 | 2026-09-10 이후 변환기 출력이면 납작 셀(일부만 음수) → 배정도(3ddp)로 읽기; 그 전 파일이면 방향 규약(전부 음수) → `fluent_invert.py in.msh out.msh` | Fluent가 원하는 규약은 매뉴얼 문구의 반대(foamMeshToFluent와 같음) |
| 유체 솔리드가 필요(다른 격자기·CAD) | `export_fluid.py <trimmed.brep> <out> [--trim 3.05]` | STEP(mm) 1솔리드 |

## 6. 함정 (전부 겪은 것)

- 절대 sliver 임계는 0.35 m 격자를 52만 셀이나 지우고 품질을 떨어뜨린다 → 상대 임계.
- 볼록껍질만 치환하면 프리즘 교차 정점(1~9 mm)이 100개 넘게 생겨 3-D가 실패한다 → 합친 뒤 그룹별 재치환. OCC 퍼지 fuse(5 cm)는 실패하고 `healShapes`는 격자기가 거부하는 형상을 남긴다.
- 후처리 후 고아·내부·중복 삼각형을 걸러야 변환기가 셀을 뒤집지 않는다.
- 밀봉 포켓(건물 밑 셀 섬)은 압력 방정식을 특이하게 만든다 → 변환기가 자동 제거(`regions:` 줄 확인).
- `chcp 949` 콘솔 + `PYTHONIOENCODING=utf-8` = 한글 경로 즉시 실패. 배치 파일엔 한글 금지.
- 프로세스는 이름·PID로만 죽인다(명령줄 부분 일치는 내 셸까지 죽였다).
