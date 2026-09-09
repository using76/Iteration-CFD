# tools/mesh/diag — 격자 진단 스크립트

`step_mesh.py`가 만든 격자(Gmsh `.msh`, 체크포인트 `.brep`, 변환된 `polyMesh`)와 솔버가 쓴
필드를 들여다보는 도구들. 전부 numpy(+scipy)와 gmsh만 쓰고, 사면체 격자(모든 면이 삼각형)를
가정한다. 배경은 `docs/08-site-mesh-playbook.md` §4, 언제 무엇을 쓰는지는
`.claude/skills/site-mesh/SKILL.md` §5.

| 스크립트 | 입력 | 무엇을 보여 주나 |
|---|---|---|
| `field_extremes.py <case> <time> [px py]` | polyMesh + 시간 폴더 | U·p 극값과 그 셀의 중심(원천 거리), 분위수, 상위 10셀 — 로그 잔차가 숨기는 국소 폭발 |
| `nonorth_faces.py <case> [px py]` | polyMesh | 면 비직교 분포(70/80/85/87/89° 초과 수), 85° 초과 면의 20 m 군집, 최악 10면; `nonorth_deg.npy` 저장 |
| `badface_radial.py <case> px py` | polyMesh + `nonorth_deg.npy` | 85° 초과 면의 원천 거리 분포와 영역 ±500~800 m 축소 시 잔존 수 |
| `inspect_cells.py <case> --cells a,b [--near x y z r]` | polyMesh | 셀 부피·두께 3V/A_max·최대 면 각·변 길이·패치·중심 |
| `cell_faces.py <case> <cell>` | polyMesh | 한 셀의 면마다 절점·owner/neighbour·패치·면적 벡터, 면 기반 부피와 폐합 — 뒤집힌 셀 |
| `inspect_point.py <pools.brep> x y z [pad]` | 체크포인트 | 그 점 주변의 솔리드(bbox)·유체 경계면·곡선·정점 |
| `scan_near_vertices.py <pools.brep> [tol]` | 체크포인트 | 1 mm~tol 근접 정점 쌍과 소속 솔리드, 거리 히스토그램 |
| `inspect_ring_curves.py <pools.brep> cx cy r` | 체크포인트 | 풀 원기둥 주변 면·곡선·중복 곡선·정점 |
| `variant_from_checkpoint.py <case.json> <name> key=value…` | 설정 + 체크포인트 | 체크포인트 사본으로 변형 설정 작성(`mesh.algo3d=10`이면 HXT가 실패 정점 좌표를 찍는다) |
| `fluent_invert.py in.msh out.msh` | Fluent ASCII 격자 | 모든 면의 절점 순서 반전(c0/c1 유지) — 방향 규약 시험 |
| `polymesh_to_fluent.py` | polyMesh | foamMeshToFluent 규약의 파이썬 Fluent 작성기 |
| `export_fluid.py <brep> <out> [--trim z] [--stl-size m] [--stl-only]` | 체크포인트 | 유체 영역을 STEP(mm, 1솔리드)·STL(m)로 |
| `desliver.py <src case> <dst case> <thr>` | polyMesh | 절점 이동으로 3V/A ≥ thr 확보(40패스; 옛 방식, 지금은 후처리 게이트가 대신) |

큰 polyMesh(1,600만 면)에서 `nonorth_faces.py`·`inspect_cells.py`는 5~10분, 메모리 4~8 GB를 쓴다.
