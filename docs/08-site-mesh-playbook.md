# 08 — 부지 사면체 격자 플레이북 (암모니아 누출 부지, 2026-09-09/10)

`tools/mesh/step_mesh.py`로 2.5 km 부지 STEP에서 700만 셀급 사면체 격자를 만들어 자체 GPU
솔버와 Fluent에 넘기기까지, 하루 동안 부딪힌 문제와 그 해법을 규칙으로 적는다. 명령만 필요하면
`.claude/skills/site-mesh/SKILL.md`, 설정 키의 뜻은 `tools/mesh/README.md`를 본다.

## 1. 무엇이 격자를 망가뜨렸나

| 현상 | 원인 | 위치 |
|---|---|---|
| 스테디 해가 10회 안에 1e150 유량 불균형으로 폭발 | 벽 사이 0.2~0.4 m 슬롯을 20 m 셀이 채우며 생긴 슬리버 | 원천에서 1.24 km, 건물 블록 3개가 겹치는 곳 |
| 잔차는 0.17로 멀쩡한데 필드에 \|U\| 1e7·p −4e10 | 같은 곳의 셀 수백 개만 터짐 — 잔차는 전체 평균이라 안 보임 | 위와 같음 |
| 비직교 85° 초과 면 2,395개(87° 초과 1,442, 89° 초과 362) | 건물 발치 지면에 붙은 납작한 쐐기 셀(두께 1 cm~0.7 m, 변 4~22 m) | 전부 원천에서 300 m 밖, 94%가 800 m 밖 |
| 3-D 경계 복원 실패 `PLC Error: A segment and a facet intersect` | 이웃 건물 모서리 1.9~4.9 cm 틈(22쌍)의 절점을 격자기가 중복점으로 처리; 통과 여부는 절점 번호 순서 운 | 1.57 km 밖 등 |
| 후처리 뒤 셀 52만 개 소실·품질 저하 | 절대 임계 sliver 병합(V<0.2 m³, 변<0.6 m)이 0.35~0.55 m 정상 셀을 병합 | 풀 세분 영역 |
| 솔버가 셀 하나를 부피 −30 m³·미폐합으로 거부 | 후처리가 남긴 고아·내부·중복 삼각형을 변환기가 경계면으로 씀(또는 남은 원인 조사 중) | 임의 |
| 밀봉 포켓(셀 섬) | 건물 밑 절단 잔여물; 압력 방정식이 특이해져 압력 해가 죽음 | 건물 밑 |

교훈 하나로 줄이면: **셀은 자기가 놓인 틈보다 두꺼울 수 없다.** 틈이 셀보다 좁으면 슬리버가 생기고,
슬리버 하나가 압력 방정식을 망친다. 그러니 (a) 틈을 없애거나(형상 뭉개기), (b) 틈에 맞춰 셀을
줄이거나(틈 규칙), (c) 그래도 남은 얇은 셀을 고쳐야(두께 게이트) 한다.

## 2. 레시피 (설정 키와 값)

1. **원천은 정확하게**: `points.<name> = {x, y, r, r_inner, h: 0.5}` — 지면에 원을 그려 0.5 m 원기둥으로
   끌어올린다. 윗면 `pool_<name>`이 inlet(수천 면), 옆면 `wall_pool_<name>`. 풀 상자(±max(40, r+15) m,
   지면 −1~+10 m)는 0.35 m, 그 밖 ±150 m 상자는 2.5 m, 최소 0.3 m. 원천 200 m 안에는 건물이 없다.
2. **먼 곳은 뭉갠다**: `solids.hull_beyond_m 150, hull_pad_m 1.0, hull_snap_m 0.05, fuse true`.
   각 건물의 발자국 볼록껍질을 1 m 밀어낸 프리즘으로 바꾸고(거의 일직선 꼭짓점·5 cm 안 꼭짓점 정리),
   겹치는 것끼리 합친 뒤(464 → 264 그룹), **합쳐진 그룹마다 다시 볼록껍질 프리즘**으로 바꾼다.
   합친 형상 그대로 두면 거의 일직선인 벽 사이에 수 cm 계단면(면적 0.01 m², 변 2 mm)이 남아
   3-D가 실패한다. OCC 퍼지 불리언(5 cm)은 `Union failed`, `healShapes`는 격자기가 거부하는 형상을
   남기므로 쓰지 않는다. 결과: 근접 정점 쌍 22 → 13, 85° 초과 면 2,395 → 6, 87° 초과 0, SICN 0.1 미만
   8,519 → 130, 최소 SICN 0.000 → 0.020.
3. **틈 규칙** `sizes.gap_ratio 2.5`: 표면 격자 삼각형마다 안쪽 법선 방향으로 마주보는(법선 내적 < −0.5)
   가장 가까운 삼각형까지 거리 g를 재고, g/2.5가 현재 크기보다 작으면 그 자리를 세분한다(10×10×5 m
   격자 상자로 반영, 표면 재격자). g/2.5 < `sizes.min`인 틈(0.75 m 미만)은 격자로 해결할 수 없으니
   요약 `gap_unresolvable`에 위치를 남긴다 — 부지에서는 선박 상부구조의 1 cm 틈(재표본 1.5 m/6000면의
   잔재) 444곳이 나왔고(3 m 재표본은 수밀이 깨진다), 답은 선박도 볼록껍질 프리즘으로 바꾸는 것이었다.
   먼 곳 세분의 바닥 `sizes.gap_min_m 1.0`이 없으면 슬롯마다 g/2.5로 세분해 2,400만 셀이 된다. 마주보지
   않는 두 면이 30° 미만으로 접히는 쐐기(기저 경사와 수면, 벽과 도메인 옆면)는 틈 규칙이 못 보므로
   `wedges`로 따로 검출해 보고하고 양쪽을 세분한다.
4. **후처리는 크기에 비례**: `post.sliver_rel 0.02, sliver_edge_rel 0.3, thin_push_m 0.3`(|V| < 0.02·e³인
   셀의 0.3·e보다 짧은 변 병합; 밀어내기), `min_thickness 0.05, repair_rounds 3`(τ = 3V/A_max^1.5 게이트,
   최대면 반대 절점을 필요한 높이만큼 법선 방향으로 밀기, 주변 셀 반전 금지, 풀 상자 안 절점 고정),
   그리고 삼각형-사면체 대조(고아·내부·중복 제거). 절대 임계는 0.
5. **도메인**: `trim.shrink_xy_m 600`(±650 m). ±1,220 m 도메인은 40회까지 잔차가 내려가다 매번 다른
   먼 곳(지형 가장자리 계단, 기저 경사 쐐기, 벽-출구 접합, 경계 띠 안의 작은 건물)에서 터졌고, ±650 m는
   50회 무발산(Ux 0.011·p 0.026). 경계 20 m 안의 프리즘 꼭짓점은 안쪽 5 m로 당기고(`hull_box_snap_m 20,
   hull_box_inset_m 5`), 그 띠에서 윤곽이 퇴화한 건물은 제거한다.
6. **변환**: `ofgpu-convert-mesh … -fluent … -fluentType pool_<name>=velocity-inlet`. 밀봉 포켓은 자동
   제거되고 `regions:` 줄에 개수가 남는다. Fluent 존: pool_* velocity-inlet, wall_* wall, top/west/south/
   north pressure-outlet, east velocity-inlet.

## 3. 검토 게이트

| 검사 | 도구 | 통과 기준 |
|---|---|---|
| 부피·폐합·영역·비직교 | 솔버 로더(`ofgpu-buoyant <case>` 첫 줄), `ofgpu-automesher -check` | 음수 0, 폐합 오차 1e-12, 영역 1, 비직교 최대 < 87° |
| 두께 τ = 3V/A_max^1.5 | 후처리 요약 `thickness_gate`, automesher G5 | 0.05 미만 0 |
| 비직교 면 분포 | `tools/mesh/diag/nonorth_faces.py`, `badface_radial.py` | 87° 초과 0에 가깝게; 원천 45 m 안 85° 초과 0 |
| 근접 정점 쌍 | 절단 로그 `near-touching solid vertices` | 20개 이하 |
| 틈 | 격자 로그 `gap pass … unresolvable` | 0, 아니면 형상 수정 |
| 삼각형 대조 | 후처리 로그 `triangles reconciled` | 미대조 경계면 0 |
| 6회 스테디 시험 | `tools/mesh/solver_test/` + `diag/field_extremes.py` | 6회째 \|U\| 최대 수십 m/s, \|p\|>1000 셀 수십 개 이내, 극값이 원천 밖 |

## 4. 진단 도구 (`tools/mesh/diag/`)

- `field_extremes.py <case> <time> [px py]` — U·p 극값과 그 셀의 중심(풀 거리), 분위수, 상위 10셀.
- `nonorth_faces.py <case> [px py]` — 면 비직교 분포와 85° 초과 군집; `badface_radial.py`가 그 면들의 원천 거리 분포와 영역 축소 시나리오를 센다.
- `inspect_cells.py <case> --cells a,b --near x y z r` — 셀 부피·두께·최대 면 각·변 길이·패치.
- `cell_faces.py <case> <cell>` — 한 셀의 면·owner/neighbour·면적 벡터·면 기반 부피(뒤집힌 셀 진단).
- `inspect_point.py <pools.brep> x y z pad` — 체크포인트 형상에서 그 점 주변 솔리드·면·곡선·정점.
- `scan_near_vertices.py <pools.brep> tol` — 1 mm~tol 근접 정점 쌍과 소속 솔리드.
- `inspect_ring_curves.py <pools.brep> cx cy r` — 풀 원기둥 주변 곡선·중복 검사.
- `variant_from_checkpoint.py <json> <name> key=value…` — 체크포인트 사본으로 변형 실행(HXT 진단 `mesh.algo3d=10`, `sizes.pool=0.34`, `sizes.gap_ratio=2.5`).
- `fluent_invert.py in out` — Fluent 격자의 면 절점 순서 반전(방향 규약 시험). `polymesh_to_fluent.py`는 foamMeshToFluent 규약의 파이썬 작성기.
- `export_fluid.py <brep> <out> [--trim z]` — 유체 영역을 STEP(mm) 1솔리드로.
- `desliver.py <src case> <dst case> <thr>` — polyMesh 절점 이동으로 3V/A ≥ thr 확보(옛 방식, 40패스).

## 5. 운영 함정

- Fluent는 한글 경로를 읽지 못한다 → 납품은 `nh3_fluent/pool_cases` 같은 ASCII 경로.
- PowerShell 도구 프로세스의 `PYTHONIOENCODING=utf-8`이 cp949 콘솔에서 한글 경로를 깨뜨려 배치가 즉시
  실패한다 → 콘솔은 Bash에서 `cmd //c start "" cmd //k <cmd>`로. 배치 파일에 한글 금지.
- 설정에 모르는 키를 넣으면 도구가 거부한다(메모용 `_note`도).
- GPU 16 GB에 850만 셀은 15.9 GB — 한계. 반복당 23 s(정상)와 245 s(메모리 스래싱)의 차이.
- 프로세스는 이름·PID로만 죽인다.
- Fluent 면 방향: ANSYS 매뉴얼 B.3.7의 문구("엄지가 c1을 향한다")대로 쓰면 Fluent 2022 R2가 모든 면을
  왼손 방향, 모든 셀을 음수 부피로 판정한다. 맞는 규약은 그 반대(foamMeshToFluent와 같음: 절점 순서를
  뒤집어 엄지가 c0을 향하고 경계면 법선이 셀 안쪽)이며 변환기는 2026-09-10부터 그렇게 쓴다. 옛 파일은
  `tools/mesh/diag/fluent_invert.py`로 뒤집으면 된다.

## 6. 남은 일

- (해결) 면 기반 부피 −30 m³ 셀 = 거의 일직선 지붕 삼각형 위 바늘 tet → 표면 슬리버 규칙·HULL_FLAT_M 1 m.
- (해결) 세 케이스 최종 레시피(±650 m) 재생성 완료; 케이스 2 50회 무발산.
- 전체 도메인(±1,220 m)을 쓰려면 먼 곳 쐐기(기저 경사·지형 계단)를 메우는 형상 규칙이 더 필요하다.
- (해결) Fluent 음수 부피 경고 = 방향 규약; 변환기가 뒤집힌 순서로 쓰도록 고침.
- 자체 격자기 automesher(SPEC-LIT §92)에 같은 게이트·수정 규칙을 넣는 것.
