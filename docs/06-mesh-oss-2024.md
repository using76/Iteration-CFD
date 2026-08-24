# 유체영역 메쉬 생성 오픈소스 — 2024년 이후 동향

주식회사 메테오시뮬레이션 · 조사일 2026-08-24

관점을 둘로 나눕니다. **외부 도구**(실행해서 메쉬 파일만 받음 — GPL이어도
제품에 무관)와 **내장·이식 가능**(GPL·AGPL 배제, LGPL은 동적 링크만).

## 결론 먼저

2024년 이후 "혁신적 신형 메쉬 생성기"는 나오지 않았습니다. 대신 실질적 변화
네 가지가 있었습니다.

1. **snappyHexMesh에 수년 만의 신규 모드** — v2412(2024-12)의
   `castellatedBufferLayer`: 스냅 전에 버퍼 층을 넣어 고질적인 경계층 커버리지
   문제를 정면으로 공략. v2506에서 버그 정리, 현재 v2606.
2. **Netgen이 가장 활발한 오픈 경계층 코드** — `boundarylayer.cpp`에
   2024–2026년 26커밋(검증됨), 2026-05 곡면 경계층 지원. Salome 9.14/9.15의
   ViscousLayers 수정으로 사용자에게 도달.
3. **허용 라이선스 기하 커널의 약진** — 우리 같은 상용 제품에 결정적:
   OpenVDB가 v12(2024-10)에서 Apache-2.0으로 재라이선스 + GPU 동적 토폴로지,
   Manifold(Apache-2.0)가 Blender 4.5의 불리언 엔진으로 채택될 만큼 성숙,
   Geogram v1.10(2026-05)의 불리언 4배 가속.
4. **강건 사면체 연구의 정리** — Attene의 CDT(GPL/LGPL 이중)가 TetGen의
   후계로 자리잡음. TetGen(AGPL, 2020년 이후 정지)은 사실상 은퇴.

## 외부 도구 (라이선스 무관 — 실행해서 파일만 받음)

| 도구 | 최신 | 2024+ 개선 | 유체영역 적합성 |
|---|---|---|---|
| **Gmsh** (GPL) | 4.15.2, 2026-03 | OCC 디피처링(4.13), 이방성 표면 개선(4.15.x) | STEP 입력 최강, physical group→패치 보존, `.msh 4.1` = 우리 계획 입력. 3-D 경계층은 여전히 취약 |
| **snappyHexMesh** (GPL) | v2606, 2026-06 | **castellatedBufferLayer** (v2412) | 오픈 진영 최강 경계층, polyMesh 직접 출력 = 우리가 바로 읽음 |
| **Netgen** (LGPL) | 6.2.2606, 2026-06 | 경계층 코드 집중 개선, 곡면 BL (2026-05) | STEP→사면체+프리즘 BL, 면 이름 보존. polyMesh 변환 필요 |
| **Salome/SMESH** (LGPL) | 9.15, 2025-09 | ViscousLayers 수정(9.14) | STEP→BL 가능한 최고 오픈 GUI. 출력이 MED/UNV라 변환 고리 필요 |
| **CGAL Alpha Wrapping** (GPL 패키지) | 6.2, 2026-06 | 6.0(2024-09)에서 2배 가속 + 체적 랩 | **더러운 CAD→방수 표면의 표준.** 우리 castellation 전처리로 최적. 패치명은 소실되어 재투영 필요 |
| **FreeCAD + CfdOF** (LGPL) | 1.1.3, 2026-07 | 1.0(2024-11) 토포네이밍 해결 | STEP 열고 GUI로 패치 지정 → cfMesh/snappy → **polyMesh** — 비전문가용 최단 경로 |
| **IfcOpenShell/Bonsai** (LGPL/GPL) | 0.8.x, 주간 릴리스 | BlenderBIM→Bonsai 개편(2024 말) | **화재·AEC의 정문**: IFC→OBJ/STL. 요소 정체성은 스크립트로 클래스별 분리 |
| cfMesh (GPL) | v2406, 유지보수만 | 호환성 수정뿐 | polyMesh 직출력이나 2024+ 신기능 없음 |
| Mmg/ParMmg (LGPL) | 5.8.0, 2024-10 | 2년 만의 재개 릴리스 | 초기 생성이 아니라 **적응 재메쉬** — 컷셀 시대에 유용 |
| MeshLib | 주간 릴리스 | 매우 활발 | **비상용 전용 라이선스 — 상용 벤더는 무료 외부 사용도 불가.** 배제 |

## 내장·이식 가능 (우리 제품에 넣거나 읽고 배울 수 있음)

| 라이브러리 | 라이선스 | 최신 | 우리 계획과의 접점 |
|---|---|---|---|
| **OpenVDB/NanoVDB** | Apache-2.0 (v12+) | 13.0, 2025-11 | mesh→level-set 복셀화, GPU 동적 토폴로지 — **castellation·컷셀 분율 계산의 정석 도구** |
| **Manifold** | Apache-2.0 | 3.5.2, 2026-06 | 보장된 manifold 불리언 — 건물 부재 합집합→방수 고체. Blender 4.5 채택으로 실전 검증 |
| **Geogram** | BSD-3 | 1.10.0, 2026-05 | 정확 술어, 병렬 3-D Delaunay, 강건 STL 불리언 — **컷셀 단계의 최우선 내장/이식 후보** |
| **libigl** (core) | MPL-2.0 | 2.6.0, 2025-05 | fast winding number 참조 구현 (`igl/copyleft/`는 GPL — 그 서브트리만 회피) |
| **wildmeshing-toolkit** | **MIT** | 무버전, 매우 활발 | TetWild 계보의 2024+ 신작 — 선언적 메쉬 편집 프레임워크. 읽을 가치 최고 |
| fTetWild | MPL-2.0 | 수정만 | 깨진 STL도 무조건 사면체화 — 비상용 전처리기 |
| VTK SurfaceNets | BSD-3 | 9.5, 2025-06 | 복셀→매끈한 방수 표면 (역방향 — 격자 해상도 디피처링) |
| NVIDIA Warp | Apache-2.0 (2025-05 재라이선스) | 1.16, 2026-08 | GPU winding-number·BVH 커널의 Apache 참조 — 프로토타이핑용 |
| AMReX EB | BSD-3 | 월간 | 컷셀 자료구조·작은셀 처리의 최고 공개 참조 |
| EBGeometry | **GPL-3** | 활발 | 구조적으로 우리 컷셀 요구와 가장 근접하나 **읽기 전용** — 설계만 배움 |

## 참조 전용 (코드 사용 불가·불필요, 동향 파악)

- **FDS GEOM/컷셀** (퍼블릭 도메인, 6.11.1 2026-07): 우리와 가장 가까운 제품의
  경고 사례 — **10년 개발한 컷셀 경로가 아직도 Beta**. 2024–26 버그 수정이
  몰린 지점(컷면 연결, 메쉬 경계 처리)이 곧 어려운 지점의 지도.
- **CDT (Diazzi/Attene)** (GPL, `-DLGPL=ON` 빌드 가능): TetGen이 실패하는
  입력에서 100% 성공하는 강건 CDT의 기준 구현.
- **MFC** (MIT, 2025 Gordon Bell 파이널리스트): GPU 침지경계 구현이 자유롭게
  읽고 이식 가능.

## Rust 생태계의 정직한 상태

**2026-08 현재 순수 Rust 생산급 3-D 사면체 메쉬 생성기는 없습니다.**
- `spade`(2-D CDT, MIT/Apache) — 성숙하나 2-D 전용
- `baby_shark`(MIT) — 복셀 리메쉬·불리언, 단독 저자 sub-1.0
- `vtkio`(MIT/Apache) — VTU 출력용으로 유용
- `tritet` — **TetGen(AGPL)을 번들** — 제품에 링크 금지
- `mshio`(MIT, 2020 정지) — 죽었지만 MSH 4.1 문법의 정확한 지도로 읽을 가치
- Barill 2018 fast winding number의 성숙한 Rust 크레이트 없음 — **직접 구현**
  (이미 SPEC-LIT §23.3에 반영)

## 화재·AEC 사용자를 위한 구체 체인 (전부 2024+ 버전으로 성립)

```
IFC 건물 모델
  → IfcOpenShell 0.8 / Bonsai (요소 클래스별 분리 스크립트)
  → OBJ/STL
  → Manifold 합집합 (또는 절망적 모델은 CGAL Alpha Wrap 외부 실행)
  → 방수 STL
  → meteor-cfd castellation          ← 지금 구현 중인 경로
```

공학 유동(체적 정합 필요)은:

```
STEP → FreeCAD+CfdOF GUI (패치 지정) → snappy/cfMesh → polyMesh → meteor-cfd
STEP → Gmsh 4.15 (physical groups)  → .msh 4.1      → (계획된 리더)
```

## meteor-cfd 계획에의 반영

- 우리의 **castellation(§23) + 향후 컷셀** 노선은 2024+ 동향과 정합 — 혁신은
  생성기가 아니라 **허용 라이선스 기하 커널**(OpenVDB·Manifold·Geogram)에서
  일어났고, 그것이 정확히 우리가 내장할 수 있는 층입니다.
- 컷셀 설계 문서를 쓸 때 읽을 순서: AMReX EB(BSD, 이식 가능) → EBGeometry
  (GPL, 설계만) → FDS 6.10/6.11 컷셀 수정 이력(퍼블릭 도메인, 함정 지도).
- polyMesh 유지 결정(§4.3.1)이 재확인됨: FreeCAD+CfdOF·snappy 체인의 종착이
  polyMesh라 **외부 체적 정합 메쉬가 공짜로 들어옵니다.**
