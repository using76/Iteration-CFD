# Iterations-CFD 테크니컬 가이드북

GPU에 상주하는 유한체적 CFD 솔버를 **실제로 돌리는 법**을 다루는 문서입니다.
설치부터 격자, 솔버 선택, 수렴 판독, 결과 보기, 그리고 자주 만나는 오류의
원인까지 순서대로 갑니다.

이 문서가 다루지 않는 것: 이산화 식의 유도와 원논문 대조는
[`rust/SPEC-LIT.md`](../rust/SPEC-LIT.md)에, 파일별 출처는
[`rust/PROVENANCE.md`](../rust/PROVENANCE.md)에 있습니다. 이 문서는 그 위에서
**쓰는 법**만 말합니다.

English: [`GUIDEBOOK.en.md`](GUIDEBOOK.en.md)

---

## 목차

1. [무엇인가, 그리고 무엇이 아닌가](#1-무엇인가-그리고-무엇이-아닌가)
2. [설치](#2-설치)
3. [5분: 첫 해석](#3-5분-첫-해석)
4. [케이스의 구조](#4-케이스의-구조)
5. [격자 만들기](#5-격자-만들기)
6. [솔버 고르기 — 가장 많이 틀리는 곳](#6-솔버-고르기--가장-많이-틀리는-곳)
7. [실행하고 수렴을 읽기](#7-실행하고-수렴을-읽기)
8. [결과 보기](#8-결과-보기)
9. [Studio — AI가 붙은 작업대](#9-studio--ai가-붙은-작업대)
10. [MCP — 쓰던 AI에게 솔버 넘기기](#10-mcp--쓰던-ai에게-솔버-넘기기)
11. [검증과 근거](#11-검증과-근거)
12. [할 수 없는 것](#12-할-수-없는-것)
13. [문제 해결](#13-문제-해결)
14. [라이선스](#14-라이선스)

---

## 1. 무엇인가, 그리고 무엇이 아닌가

**시간 적분 루프 전체가 GPU에 머무릅니다.** 격자와 필드를 한 번 올린 뒤에는
반복마다 장치 메모리를 새로 잡지도, 필드를 호스트로 내리지도 않습니다. 이것이
속도의 유일한 이유이고, 동시에 제약의 이유이기도 합니다 — **GPU 한 장, 프로세스
하나**입니다. MPI도 다중 GPU도 없습니다.

호스트는 Rust 1.85, 커널은 CUDA C++, 기본 정밀도는 배정밀도(`single` 기능으로
단정밀도), 의존성은 `cudarc`와 `thiserror` 둘뿐입니다(AMGX는 선택 기능).

**쓰임새**는 비압축성·저마하 유동입니다. RANS/LES/하이브리드 난류, 부력 플룸,
가변밀도 저마하, 2상 VOF, 켤레 열전달, 면대면 복사, 라그랑주 분무, 팬과 다공성
점프를 갖춘 환기·데이터센터 기류.

**아닌 것**: 압축성·천음속 해석기가 아닙니다. 유한율 화학이 없습니다. 다중 GPU
확장성 수치를 발표하지 않습니다. 자세한 목록은 [§12](#12-할-수-없는-것).

그리고 하나 더 — **비교가 불충분합니다. 많은 도움 부탁드립니다.** 검증은
인위해법(MMS), 해석해, 공개 벤치마크를 씁니다. 무엇이 아직 닫히지 않았는지는
숨기지 않고 `ofgpu-validate`가 매 실행 이름으로 부릅니다. 더 견줄 만한 측정과
케이스를 알고 계시면 알려 주십시오([§11](#11-검증과-근거)).

---

## 2. 설치

### 요구사항

| | |
|---|---|
| GPU | CUDA 13을 지원하는 NVIDIA GPU 한 장 |
| OS | Windows 10/11 x64 |
| 런타임 | Visual C++ 2015–2022 재배포 패키지 |
| (소스 빌드 시) | Rust 1.85+, Visual Studio 2022 C++ 워크로드, CUDA Toolkit 13.x |

CUDA 13은 **정적으로 링크**되어 있으므로, 설치 프로그램으로 설치하는 경우
CUDA Toolkit을 따로 깔 필요가 없습니다. 드라이버만 있으면 됩니다.

### 설치 프로그램

릴리스에서 `iterations-*.exe`를 받아 실행합니다. `bin`을 PATH에 넣는 옵션을
켜면 어느 프롬프트에서든 `ofgpu-*`가 돕니다.

설치되는 것: 솔버 바이너리 16개, 케이스 정의(`cases/*.jsonc`), 경주차 샘플,
`docs/`(이 문서와 SPEC-LIT, PROVENANCE, 스키마), 라이선스 문서.

### 소스에서 빌드

```powershell
cd rust
cargo build --release
```

`build.rs`가 `vcvars64.bat`으로 MSVC 환경을 구성하고 모든 `.cu`를 PTX가 아닌
**CUBIN**으로 컴파일합니다. 즉 빌드한 기계의 아키텍처에 맞춰 미리 컴파일되며,
런타임 JIT이 없습니다.

### 설치 확인

```powershell
ofgpu-probe
```

장치 이름, 스칼라 정밀도, 그리고 **장치 결과가 호스트 결과와 비트 단위로
같은지**를 출력합니다. `max |gpu - cpu| = 0.000e0`과 `PASS`가 나와야 합니다.
여기서 실패하면 그 아래 어떤 결과도 신뢰할 수 없습니다.

---

## 3. 5분: 첫 해석

동봉된 경주차 샘플이 가장 빠른 길입니다. 형상 하나와 명령 두 줄입니다.

```powershell
cd cases
.\racecar.cmd
```

이 스크립트가 하는 일:

```powershell
# 1. 단위 풍동(1 m 정육면체)에 차체를 컷셀로 새깁니다.  몇 분 (모든 코어 병렬)
ofgpu-generate-mesh big racecar_case 128 -stl car=racecar.stl -cutcell

# 2. 운동량까지 풉니다.                                   1-2분 (GPU)
ofgpu-lowmach racecar_case -iters 3000 -check 250 -output foam
```

1단계의 형상 분류 — 210만 개 셀 각각을 STL 표면에 대해 안/밖으로 가르고, 표면에
걸친 셀을 잘라내는 작업 — 은 모든 코어에서 병렬로 돕니다. 32코어에서 측정한
경주차 96³ 조각이 559초에서 48초로 줄었습니다(약 11.6배); 이 샘플 크기인
128³은 수 분입니다. 2단계는 GPU에 상주하는 루프라 몇 분이면 끝납니다.

`p`와 `T`도 생성기가 `0/`에 직접 쓰므로 예전 레시피에 있던 복사 단계는
사라졌습니다. 어느 드라이버가 무엇을 푸는지는
[§6](#6-솔버-고르기--가장-많이-틀리는-곳) — 요약하면 난류 전용 드라이버는
얼린 `U` 위에서 난류 두 방정식만 풉니다.

끝나면 `racecar_case/0/`에 `U`, `p`, `T`, `k`, `epsilon`, `omega`, `nut`이
OpenFOAM ASCII로 들어 있습니다. Studio의 3D 뷰어로 열거나 ParaView로 바로
읽힙니다.

자세한 것은 [`cases/racecar.md`](../cases/racecar.md).

---

## 4. 케이스의 구조

케이스를 주는 방법은 두 가지입니다.

### (a) JSONC 케이스 — 파일 하나

주석과 trailing comma를 허용하는 JSON 한 장에 격자·물성·경계조건·수치를 모두
적습니다. 스키마는 [`docs/schema/case-1.json`](schema/)에 자동 생성되어 있고,
예시는 [`docs/case-example.json`](case-example.json)과 `cases/*.jsonc`입니다.

읽는 드라이버: `ofgpu-lowmach`, `ofgpu-k-epsilon`, `ofgpu-cht`,
`ofgpu-datacentre`, `ofgpu-decompose`.

결과는 `<stem>_jsonc/<time>/`에 쓰입니다.

### (b) OpenFOAM 케이스 디렉터리

```
racecar_case/
  0/                       초기·경계 필드 (U, p, T, k, epsilon, omega, nut, ...)
  constant/
    polyMesh/              points, faces, owner, neighbour, boundary
    physicalProperties     viscosityModel, nu
    momentumTransport      simulationType, RAS { model kEpsilon | kOmega | ... }
    g                      (있으면) 중력 벡터
  system/
    controlDict            startTime, endTime, deltaT, writeControl
    fvSchemes              ddt, div, grad, laplacian, snGrad
    fvSolution             solvers, relaxation, correctors
```

OpenFOAM ASCII 형식을 읽고 씁니다. **이것은 상호운용을 위한 것이지 파생물이
아닙니다** — ofgpu는 OpenFOAM의 어떤 부분과도 링크하지 않고 그 소스를 포함하지
않습니다. 파일 형식은 저작물이 아닙니다.

### 알아 두면 좋은 것

- `momentumTransport`의 `model`이 실제 난류 모델을 정합니다. 메쉬 생성기는
  `kEpsilon`을 씁니다. k-omega 계열로 풀려면 여기를 고쳐야 합니다.
- 인식되지 않는 설정은 **조용히 대체되지 않고 거부**됩니다(SPEC-LIT §13.4).
  기본값으로 넘어가고 싶으면 `-permissive`를 주십시오 — 무엇을 무엇으로
  대체했는지 출력합니다.
- `constant/g`가 있으면 중력이 실제로 방정식에 들어갑니다. 뷰어의 위쪽 축을
  바꾸려는 목적으로 넣지 마십시오. 없으면 뷰어는 바닥 패치(`bottomWall`,
  `floor`, `ground`)의 법선에서 위쪽을 읽습니다.

---

## 5. 격자 만들기

```powershell
ofgpu-generate-mesh <preset> <outputDir> [nx ny nz] [-stl [name=]path]...
                    [-cutcell [-s N] [-thetaMin X]]
                    [-extent xlo xhi ylo yhi zlo zhi] [-grading x|y|z=r]...
                    [-wallModel standard|spalding|rough|lowRe [-Ks x [-Cs y]]]
                    [-cyclic x|y|z] [-permissive]
```

프리셋: `channel`, `cavity`, `step`, `big`, `plume`, `room`, `damBreak`.
`big`은 한 변 1 m의 정육면체 풍동이고 셀 수를 **하나만** 받습니다(`n³`).

생성되는 것은 바로 돌릴 수 있는 완전한 케이스입니다 — `constant/polyMesh`,
`constant/physicalProperties`, `constant/momentumTransport`,
`system/{controlDict,fvSchemes,fvSolution}`, 그리고 `0/`에 `U`, `k`, `epsilon`,
`omega`, `nut` — 프리셋이 `channel`·`cavity`·`step`·`big`이거나 부력 쌍
(`plume`/`room`)이면 여기에 `p`와 `T`까지 들어갑니다(블록·컷셀 양쪽 경로 모두).
`damBreak`만 예외로, 2상 경로라 `0/`이 `alpha.water`와 `p_rgh`를 담습니다.

### 부지에 맞춘 블록 — `-extent`, `-grading`

```powershell
ofgpu-generate-mesh big site 320 260 40 -extent -1240 400 -800 500 3.5 200 -grading z=6 -stl site=site.stl -cutcell
```

프리셋의 고정 크기(`big`은 단위 정육면체)로는 실제 부지를 담을 수 없을 때,
`-extent xlo xhi ylo yhi zlo zhi`(미터)가 블록의 여섯 면을 직접 정하고
`-grading x|y|z=r`은 그 축의 셀 성장비(마지막 셀 / 첫 셀)를 정합니다 — `r > 1`이면
가장 작은 셀이 축의 `lo` 끝(z축이면 지면)에 놓입니다. 두 플래그는 일반 경로는
물론 `-stl` 캐스텔레이션과 `-cutcell`에도 함께 적용됩니다. 블록의 여섯 패치는
프리셋의 이름과 타입을 그대로 유지합니다(`big`이면 xMin `inlet`, xMax `outlet`,
`bottomWall`, `topWall`, yMin `backWall`, yMax `frontWall`) — 필요하면 나중에
`constant/polyMesh/boundary`와 `0/`에서 이름이나 타입을 고치십시오. `plume`,
`room`, `damBreak`는 개구부와 물기둥을 프리셋 자체 크기에서 놓으므로 `-extent`를
거부합니다.

### STL을 넣기

```powershell
ofgpu-generate-mesh big case 128 -stl car=racecar.stl -cutcell
```

`-stl [이름=]경로`로 형상을 넣고, `-cutcell`을 주면 계단식이 아니라 **컷셀**로
잘라냅니다. 형상 표면은 주어진 이름의 wall 패치가 됩니다.

컷셀 분류(SPEC-LIT §23.2)는 **닫힌 다양체**를 요구합니다. 맞닿은 부품이 면을
공유하면 `non-manifold edge(s)`로 거절되므로, 서로 **겹치게** 만드십시오.

`-s N`은 셀당 세부 분할 수, `-thetaMin X`는 병합 임계 부피비입니다. 기본값은
`s = 16`, `theta_min = 0.2`입니다. 출력은 이렇게 나옵니다:

```
[cutcell] 128 x 128 x 128 block, s = 16, theta_min = 0.2:
          3183 solid, 2087745 fluid, 6224 cut (917 merged) -> 2093052 cells
[cutcell] new wall patch car: 5850 face(s)
```

`solid`는 형상 안쪽이라 셀이 사라진 자리, `cut`은 표면에 잘린 셀,
`merged`는 너무 작아 이웃에 합쳐진 셀입니다.

### 벽 처리 프리셋

`-wallModel`은 한 케이스의 모든 벽에 `nut`/`k`/`epsilon`/`omega`(에너지
방정식을 풀면 `T`까지) 경계 타입을 **일관된 한 행**으로 채웁니다. 필드마다 따로
고르다 서로 모순되는 조합을 만드는 실수를 막기 위한 것입니다.

| 프리셋 | `nut` | `k` | `epsilon`/`omega` |
|---|---|---|---|
| `standard`(기본) | `nutkWallFunction` | `kqRWallFunction` | `epsilonWallFunction`/`omegaWallFunction` |
| `spalding` | `nutUWallFunction` | `kqRWallFunction` | 〃 |
| `rough` | `nutkRoughWallFunction`(`-Ks` 필수) | `kqRWallFunction` | 〃 |
| `lowRe` | `nutLowReWallFunction` | `kLowReWallFunction` | `epsilon`: `fixedValue`(값 없음 → 0) / `omega`: `zeroGradient` |

자세한 표(LES일 때의 축약 포함)는 [`cases/README.md`](../cases/README.md).

### STEP 형상에서 격자로, 그리고 Fluent로

CAD가 주는 것은 STEP 파일이고 솔버가 먹는 것은 격자입니다. 그 사이는 두 단계입니다 —
Gmsh가 격자를 만들고, 이 저장소의 변환기가 그것을 가져갑니다. 이 절은 그 두 단계의
사용설명서 전체입니다. 요약(스키마, 10단계 파이프라인 표, 검사기 사용법)만 필요하면
[`tools/mesh/README.md`](../tools/mesh/README.md)가 대신하고, 여기서는 서술을
담습니다. 실례는 암모니아 누출 부지 하나입니다 — STEP 한 장에서 사면체 360만 개까지,
아래의 시간은 실제로 돌려 측정한 값입니다.

#### 필요한 것

- **STEP 파일 하나.** 단위는 설정의 `scale`이 정합니다. 도구는 파일의 단위 선언을
  읽지 않고 곱하기만 합니다 — mm STEP이면 `0.001`, 이미 미터면 `1.0`.
- **Python 3.13**에 `gmsh`, `numpy`, `scipy`, `pymeshlab`. `step_mesh.py`가
  직접 import하는 것은 gmsh와 numpy이고, 표면 수리 경로([`repairs`](#설정-쓰기))가
  pymeshlab을 더 요구합니다. 이 문서의 수치는 gmsh 4.14.1 환경에서 얻은 것입니다.
- **빌드된 변환기** `rust\target\release\ofgpu-convert-mesh.exe` —
  `cd rust; cargo build --release`로 만듭니다([§2](#2-설치)). Studio 쪽 도구는
  `OFGPU_BIN_DIR` 환경 변수를 먼저 보고, 없으면 같은 빌드 트리를 찾습니다.

둘 다 갖춰졌는지 확인하는 가장 빠른 길입니다. 작은 STEP을 스스로 만들어 네 가지
경로로 메시하고(초 단위), 변환기가 빌드되어 있으면 polyMesh와 Fluent 메쉬까지
검사합니다:

```powershell
python tools\mesh\selftest.py
```

#### 설정 쓰기

설정은 JSON 파일 하나입니다. 스키마 전체는
[`tools/mesh/README.md`](../tools/mesh/README.md)에 있고, 모르는 키는 이름과
함께 거절되며 빠진 키는 기본값을 받습니다. `step`, `out_dir`, `domain_box` 셋만
필수입니다. 암모니아 부지 전체가 이 한 장입니다
([`tools/mesh/examples/nh3_site.json`](../tools/mesh/examples/nh3_site.json)):

```json
{
  "step": "C:/Users/sdd32/Desktop/암모니아누출/cfd_optimized_defeatured_recommended.step",
  "scale": 0.001,
  "out_dir": "C:/Users/sdd32/Desktop/암모니아누출/mesh",
  "name": "nh3_site",
  "fluid": {"tag": 1},
  "domain_box": [-1250.0, -1250.0, -7.5, 1250.0, 1250.0, 200.0],
  "outer_tol": 0.05,
  "solids": {"sink_m": 1.5, "fuse": false, "exclude_tags": []},
  "repairs": [
    {"tag": 33, "method": "resample", "cell_m": 1.5, "target_faces": 6000, "lift_z": 3.05}
  ],
  "trim": {"below_z": 3.05},
  "sea_z": 3.05,
  "points": {
    "tank_shell":    [-916.9, 349.8],
    "ESDV1":         [-910.2, 329.3],
    "skid":          [-869.5, 233.8],
    "pipe_mid":      [-434.8, 116.9],
    "ESDV2":         [-23.2,    9.4],
    "ship_manifold": [  18.5,  -7.5]
  },
  "pool_radius_m": 26.0,
  "roof_patches": {"nh3_source": 306},
  "sizes": {
    "min": 1.5, "max": 40.0, "pool": 2.0, "box": 4.0, "growth_from": 2.5,
    "near_struct": 4.0, "far_struct": 12.0, "near_radius": 400.0,
    "size_mult": 1.0, "roof_boxes": [2.5, 5.0, 10.0]
  },
  "mesh": {"algo2d": 6, "algo3d": 1, "optimize_passes": 5, "threads": 32},
  "post": {
    "flat_tets": true, "flat_threshold": 1e-7, "seam_merge_m": 0.02,
    "sliver_edge_m": 0.6, "sliver_vol_m3": 0.2, "thin_push_m": 0.0
  },
  "classification": {"wall_prefix": "wall_", "big_roof_is_ground_m2": 2000.0}
}
```

키 하나를 바꾸면 무엇이 달라지는가:

| 키 | 바꾸면 |
|---|---|
| `step` | 읽을 STEP. 다른 부지는 여기만 바꾸고 시작합니다 |
| `scale` | STEP 좌표에 곱해지는 값 (`Geometry.OCCScaling`, import 전에 적용). 틀리면 형상이 천 배 커지거나 작아지고, 절단이 예상 질량 검사에서 거절합니다 |
| `out_dir` | `.msh`/`.vtk`/`_summary.json`과 `work/` 체크포인트가 놓이는 곳 |
| `name` | 모든 출력과 체크포인트 파일의 접두어 |
| `fluid` | 어느 imported 고체가 유동 영역인가. 태그를 모를 때는 `{"largest": true}` |
| `domain_box` | 계산 영역 상자. 여섯 면 중 다섯이 `top`/`west`/`east`/`south`/`north` 패치가 됩니다. 키우면 원역장 셀(`sizes.max`)이 그 부피만큼 더 늘어납니다 |
| `outer_tol` | 바깥 면 판정의 허용 오차. 바깥 패치가 비어 거절되면 이것부터 올려 봅니다 |
| `solids.sink_m` | 모든 건물 베이스를 그 지붕을 중심으로 이만큼 아래로 신장합니다 (지형 위에 뜬 베이스가 만드는 머리카락 공기층을 제거). 단 베이스가 trim 평면에 닿지 않게 — 2.0이면 베이스가 정확히 3.0이 되어 1,814면이 trim 밑으로 내려가고, 1.5면 0면입니다 |
| `solids.fuse` | `true`면 절단 전에 고체들을 fuse합니다. 맞닿거나 겹치는 이웃의 동면과 머리카락 틈을 합쳐 주지만 불리언은 더 무겁습니다 |
| `solids.exclude_tags` | 절단에서 아예 빠지는 고체 (모델에서 제거) |
| `repairs[].tag` | 수리로 대체할 고체. 자기교차 고체 — 이 부지에서는 선박 선체 — 전용입니다 |
| `repairs[].method` | `"resample"` 하나뿐입니다. 3 m로 2-D 메시 → pymeshlab 균일 재샘플링(`cell_m` 격자) → quadric 감쇄(`target_faces`/10000/16000 사다리 중 닫히고 다양체이고 자기교차 없는 가장 거친 것) → 평면 삼각형의 OCC 고체. 결과는 `work/repaired_<tag>.brep`에 캐시되고 `"brep"`로 재사용할 수 있습니다 |
| `repairs[].cell_m` | 재샘플링 격자. 굵게 하면 선체가 둔해지고 면도 적어집니다 |
| `repairs[].target_faces` | 감쇄 사다리의 첫 목표. 낮추면 프록시가 거칠어집니다 |
| `repairs[].lift_z` | 수리 후 올릴 높이. 이 STEP은 선박을 STEP의 z=0에 띄워 두는데 해수면은 +3이라 3.05입니다. 원래 높이를 유지하려면 0 |
| `trim.below_z` | 이 평면 아래를 상자로 잘라 냅니다. `null`이면 trim 없음. 이 부지가 3.05인 이유: 진짜 해수면은 z=+3인데 절단 평면을 기존 면 바로 위 5 cm에 두면 불리언이 기존 면을 따라 자르지 않고, 잘라 새로 만들어진 평면이 곧 `wall_sea_surface`가 됩니다 |
| `sea_z` | 이 높이(±0.06)의 평평한 면을 `wall_sea_surface`로 분류합니다. trim과 같은 값으로 둡니다 |
| `points` | 이름 → (x, y). 점마다 지면 높이를 z=−1부터 0.25 m 간격 스캔으로 찾고, 풀 디스크·±40 m 상자·±150 m 상자·성장 필드가 하나씩 붙습니다. 이름을 바꾸면 체크포인트가 거절하므로 처음부터 다시 돌려야 합니다 |
| `pool_radius_m` | 각 점에 새겨지는 풀 디스크의 반경. 풀은 그 점의 지면이 평평하다고 가정합니다 |
| `roof_patches` | 이름 → 고체 태그. 그 고체의 평평한 지붕이 별도 패치가 되고 주위로 세 개의 세분화 상자(지붕 ±60 m, ±250 m, 그리고 −700 m 하풍 팔)가 깔립니다 |
| `sizes.min` | `Mesh.MeshSizeMin` — 그 어디서도 이보다 짧은 변이 생기지 않는 하한 |
| `sizes.max` | `Mesh.MeshSizeMax`이자 모든 필드의 바깥 크기 — 원역장 셀 크기 |
| `sizes.pool` | 각 점의 ±40 m, 지면 +10 m 상자(풀과 그 바로 위) 안의 크기 |
| `sizes.box` | 각 점의 ±150 m, +30 m 상자(지시서의 세분화 상자) 안의 크기 |
| `sizes.growth_from` | 점에서 이 크기로 시작해(40 m까지 유지) 700 m에 걸쳐 `max`까지 자랍니다 |
| `sizes.near_struct` | 점에서 `near_radius` 안의 구조물 가까이 크기 (80 m에 걸쳐 `max`로) |
| `sizes.far_struct` | 나머지 구조물 가까이 크기 (120 m에 걸쳐 `max`로) |
| `sizes.near_radius` | near/far 구분 반경 — 점의 (x, y)에서 잰 거리 |
| `sizes.size_mult` | 1보다 크게 하면 모든 크기(min, max, 상자 VIn, threshold, Thickness)에 곱해집니다. 굵은 격자로 빨리 돌리는 솔버 강건성 재현용 |
| `sizes.roof_boxes` | 지붕 패치 주변 세 상자의 크기 [가까움, 중간, 하풍] |
| `mesh.algo2d` | 2-D 알고리즘 (6 = Frontal-Delaunay). 겹치는 삼각형이 남은 면은 5(Delaunay), 그다음 1(MeshAdapt)로 자동 재메시됩니다 |
| `mesh.algo3d` | **1(Delaunay)로 두십시오.** HXT(10)는 이 부류의 형상에서 경계 복구 중 미완성 Steiner 경로에 걸려 프로세스째 죽습니다. 0개 사면체로 끝나면 도구가 스스로 Delaunay로 재시도합니다 |
| `mesh.optimize_passes` | gmsh 자체 최적화(변·면 교환 + 스무딩)의 패스 수, 임계 0.5 |
| `mesh.threads` | 2-D/3-D 스레드 수 |
| `post.flat_tets` | 3-D 패스 뒤의 평면 사면체 제거 단계. Delaunay는 평면 경계 위에 부피 0 사면체를 남기므로 기본 켭니다 |
| `post.flat_threshold` | \|V\|가 이 값 × (노드 0에서 가장 긴 변)³보다 작으면 평면 사면체 |
| `post.seam_merge_m` | 평면 사면체 안의 이보다 가까운 노드쌍을 메쉬 전체에서 병합 (이음새 처리) |
| `post.sliver_edge_m`, `post.sliver_vol_m3` | 부피가 `sliver_vol_m3`보다 작은 사면체 안의 `sliver_edge_m`보다 짧은 변을 접어 슬리버를 제거 |
| `post.thin_push_m` | 0보다 크면 gamma < 0.02의 얇은 사면체 노드를 최대 이만큼 밀어냅니다. 선박과 해수면 사이 얇은 쐐기가 남을 때 (예 0.2) |
| `classification.wall_prefix` | 네 `wall_*` 그룹의 접두어 |
| `classification.big_roof_is_ground_m2` | 100 × 100 m보다 큰 슬랩의 평면 지붕 중 면적이 이보다 큰 것은 건물이 아니라 지형(`wall_ground_land`)으로 분류합니다 |

설정에 이름이 없는 숫자들 — 점 상자의 ±40/±150 m, 지붕 상자의 ±60/±250 m과 −700 m
하풍 팔, 성장 거리 40/700 m, 구조물 필드의 0–80/120 m, 지면 스캔의 −1..60 m at
0.25 m, 선체 판정의 0.5 m 여유, 대형 지붕의 100 × 100 m 형태 검사 — 는 참조
스크립트가 하드코드했던 값을 그대로 옮긴 것입니다. 그 격자 배치가 이 부지에서
튜닝된 값입니다.

#### 돌리기

먼저 축소판으로:

```powershell
python tools\mesh\step_mesh.py step.json --dry-run
```

import와 절단까지만 돌고 부피·질량·경계 면 수·점별 지면 높이를 인쇄합니다
(`work/dry_run_summary.json`에도 씁니다). 부지에서 몇 분입니다. 형상이 거절되는
대부분의 일은 여기서 끝납니다.

진짜 실행은 콘솔에서 보이게:

```powershell
.\tools\mesh\run_step_mesh.cmd step.json
```

이 .cmd는 step_mesh.py를 현재 콘솔에서 돌립니다 — 단계 배너가 생기는 대로
인쇄됩니다 — 그리고 모든 줄을 `<out_dir>\work\run.log`에도 복사하고, 종료 코드는
step_mesh.py의 것을 그대로 돌려줍니다.

사이즈를 만지는 반복에는 체크포인트가 있습니다:

```powershell
python tools\mesh\step_mesh.py step.json --stop-after-checkpoint   # 절단+풀까지
python tools\mesh\step_mesh.py step.json --from-checkpoint --tag coarse
```

`--stop-after-checkpoint`는 체크포인트(`work/<name>_pools.brep` + `.json`)를 쓰고
멈춥니다. `--from-checkpoint`는 그 결과부터 시작합니다 — 미터 단위 체크포인트를
읽고, 느슨한 면을 걷어내고, 지면 높이와 풀 상태를 `.json`에서 가져옵니다 — 그러므로
`sizes`나 `post`만 고친 반복은 절단과 풀(부지에서 약 15분)을 다시 하지 않고 trim부터
몇 분 만에 끝납니다. 설정의 `points`가 체크포인트의 것과 다르면 거절합니다 — 지면
높이와 풀은 체크포인트에 속합니다. `--tag NAME`은 출력을 `<name>_<NAME>.msh` 등으로
접미사 붙여 따로 둡니다.

실행은 10단계로 이루어지고, 각 단계가 시작될 때
`========== [ 7/10] trim  (elapsed ...) ==========` 모양의 배너와 경과 시간을
찍으며 각 줄 앞에도 경과 초가 붙습니다:

1. **import** — STEP을 `scale`로 읽고(`Geometry.OCCScaling`), 유동 고체를 태그나
   최대 부피로 찾아 질량을 보고합니다
2. **cut** — 고체별 수리, 침하 신장, 선택적 fuse, 불리언 절단 한 번, 밀폐 포켓은
   목록과 함께 제거
3. **ground heights** — 점마다 z=−1부터 0.25 m 간격의 `isInside` 스캔
4. **classification** — 모든 경계 면을 정확히 하나의 패치로 (풀 전에 한 번, trim 뒤에
   다시 보고)
5. **pool discs** — 점마다 지면 높이에 디스크를 하나씩 새깁니다; 하나가 실패하면
   절단 직후 체크포인트로 되돌리고 그 풀 없이 계속합니다
6. **checkpoint** — `work/<name>_pools.brep` + `.json` 기록 (`--stop-after-checkpoint`는
   여기서 끝)
7. **trim** — 설정이 null이 아니면 `below_z` 아래를 상자로 절단하고, 새 평면 면 수와
   면적, trim 평면 밑까지 내려가는 면 수를 보고합니다
8. **groups + fields** — 물리 그룹과 크기 필드(점 상자, 성장 threshold, near/far
   구조물 필드, 지붕 상자)를 만들고 Min으로 하나로 합칩니다
9. **mesh** — `generate(1)`, 코인시던트 곡선 계열 통일, `generate(2)`, 겹치는
   삼각형이 남은 면 재메시(5, 그다음 1), `generate(3)`, gmsh 최적화
10. **post + write** — 평면 사면체 제거와 품질 전후 비교, 세 파일 기록

부지에서의 실측:

| 단계 | 걸린 시간 |
|---|---|
| import | 12 s |
| 선박 수리 (repairs) | 60 s |
| cut | 3 min |
| pool discs (여섯 점) | 9–11 min |
| trim | 1.5 min |
| 2-D 메시 | 20 s |
| 3-D 메시 | 3 min |
| post + write | 2 min |
| **합계 (사면체 약 3.6 M)** | **약 23 min** |

출력은 `out_dir`에 모입니다:

```
mesh/
  nh3_site.msh              Gmsh 4.1 ASCII, 물리 그룹 = 패치
  nh3_site.vtk              바이너리, 보기 전용
  nh3_site_summary.json     개수·그룹·품질·시간·설정 전체
  work/
    nh3_site_cut.brep       절단 직후 (풀 실패 시 복원점)
    nh3_site_pools.brep     체크포인트 (+ .json) — --from-checkpoint의 시작점
    nh3_site_trimmed.brep   trim 직후
    repaired_33.brep        선박 수리 캐시
    run.log                 run_step_mesh.cmd로 돌렸을 때 모든 줄
```

`_summary.json`에는 gmsh 버전, 단계별 `timings_s`, 유동 태그와 질량, 고체 bbox,
수리 치환 기록, 제거된 포켓, 점별 지면 높이, 풀 상태(`imprinted` / `not imprinted` /
`failed`), 패치별 면 수와 면적(`groups`), trim이 없앤 질량, 겹침으로 재메시한 면,
삼각형·사면체 개수, 품질(제거 전후), 평면 사면체 단계의 노트, 파일 바이트 수,
그리고 설정 전체가 들어갑니다.

종료 코드는 셋입니다: **0** 성공(`--dry-run`, `--stop-after-checkpoint` 포함),
**1** 거절 — 설정의 알 수 없는 키, 유동 태그 없음, 절단 질량 불일치, 빈 바깥
패치, 분류가 경계 면을 다 덮지 못함 — 메시지가 무엇이 문제인지 이름을 댑니다,
**3** 3-D 패스가 빈 격자 — 이때 표면 메시가 `work/surface_only.msh`로,
요약이 `work/failed_summary.json`으로 저장되므로 위의 PLC 오류 지점을 볼 수
있습니다.

#### 결과 읽기

패치 이름은 분류가 정합니다: `top`/`west`/`east`/`south`/`north`(domain_box의 다섯
면), `wall_sea_surface`(sea_z의 평면), `wall_ship_hull`(수리된 고체 bbox 주변 0.5 m),
`wall_buildings`(고체 bbox 안쪽 면), `wall_ground_land`(나머지 전부 + 대형 슬랩의
평면 지붕), `pool_<점 이름>`(점 지면 높이의 디스크 조각), 그리고 `roof_patches`에
쓴 이름 하나씩(이 부지에서는 `nh3_source`). `_summary.json`의 `groups`에 패치별 면
수와 면적이 있으니 어느 패치가 비어 있거나 비정상적으로 넓은지 여기서 봅니다.

품질은 두 측도로 인쇄됩니다 — 3-D 직후와 평면 사면체 제거 뒤 각각:

```
  minSICN min 0.2999  p1 0.xxx  p5 0.xxx  p50 0.xxx  <0.1: 0  <0: 0
```

`minSICN`은 부호 있는 최소 역조건수로, 1이 정사면체, 0이 붕괴, 음수가 뒤집힌
셀입니다. `p1`/`p5`/`p50`은 백분위, `<0.1`은 그 미만의 사면체 수, `<0`은 음의
부피 수입니다. `<0`이 0이 아니면 그 격자는 솔버로 가면 안 됩니다. (selftest의
작은 상자는 min 0.2999 근처로 나옵니다.) `gamma`도 같은 모양의 줄로 함께
인쇄됩니다.

post 단계의 노트는 평면 사면체 몇 개를 발견해 경계를 다시 삼각분할해 몇 개를
지웠는지, 이음새 노드쌍을 몇 쌍 병합해 사면체·삼각형 몇 개가 접혔는지, 슬리버 변을
몇 개 접고 몇 번 되돌렸는지(사면체가 뒤집히는 접기는 거부), 얇은 사면체를 몇 개
밀었는지를 `_summary.json`의 `flat_tets_notes`에 기록합니다.

#### 솔버로

```powershell
ofgpu-convert-mesh mesh\nh3_site.msh nh3_case -type pool_tank_shell=wall
```

하나의 .msh로 케이스 디렉터리의 `constant/polyMesh`를 씁니다(위의 (b) 형식). 패치
타입의 규칙은 이름의 접두어입니다 — `wall`로 시작하면 `wall`, `empty`면 `empty`,
`symmetry`면 `symmetry`(전부 대소문자 무시), 그 외는 `patch`. 이름은 Gmsh가 쓴 그대로
유지되고 타입만 정해집니다. `pool_*` 패치는 지면 위의 풀이므로 솔버 입장에서는
벽입니다 — 벽 처리는 타입에서 고르므로 `patch`로 남겨 두면 벽함수가 전혀 작동하지
않습니다. `-type 이름=타입`은 반복해 패치별로 정하고 접두어 규칙을 이깁니다. 알 수
없는 타입 값은 허용 목록(`patch`, `wall`, `mappedWall`, `empty`, `symmetry`,
`symmetryPlane`, `wedge`, `cyclic`, `cyclicAMI`, `cyclicSlip`, `processor`,
`processorCyclic`)과 함께 거절됩니다.

변환기가 거절하는 것도 알아 두십시오: 대상 디렉터리에 이미 `constant/polyMesh`가
있으면 덮어쓰지 않고 거절합니다(어떤 전처리 사슬이 남긴 유일한 사본일 수
있습니다). 그리고 `-fluent` 출력은 사면체 전용이라, 사각형 면이나 네 면이 아닌 셀은
이름을 댄 채 거절됩니다.

**한 반복으로 격자부터 점검하십시오.**

```powershell
ofgpu-buoyant nh3_case -iters 1
```

필드를 읽기 전에 로더가 격자 통계를 인쇄합니다 — `mesh: <cells> cells, ...` 줄,
패치 표(이름·타입·면 수), `volume: total ..., min ..., max ...`,
`non-orthogonality: max ... deg, mean ... deg`, `face closure`,
`lduAddressing: upper-triangular`. 이 줄들이 부지 격자의 성적표입니다. 변환만 한
케이스에는 `0/`이 없으므로 그다음은 이렇게 끝납니다:

```
error: no time directory with initial fields found in nh3_case
```

**이 거절이 정상입니다** — 격자가 로드되고 구조가 검증되었다는 뜻입니다. 음의
부피나 비직교각이 마음에 걸리면 이 점검표의 `volume`과 `non-orthogonality` 줄이
그 증거이고(아래의 문제 해결 표 참고), 이대로 좋으면
`0/`, `constant/`, `system/`을 채웁니다. 케이스의 구조는
[§4](#4-케이스의-구조), 필드와 사전의 상세는
[`cases/README.md`](../cases/README.md), 어느 드라이버가 무엇을 푸는지는
[§6](#6-솔버-고르기--가장-많이-틀리는-곳) — 부력 케이스인 이 부지는
`ofgpu-buoyant`로 갑니다.

#### Fluent로

```powershell
ofgpu-convert-mesh mesh\nh3_site.msh nh3_case -fluent nh3_site_fluent.msh
```

`-fluent`는 polyMesh에 적용한 것과 **같은** 타입 정리를 ANSYS Fluent가 읽는 ASCII
메쉬로 씁니다. 형식은 ANSYS FLUENT 12.0 User's Guide 부록 B의 B.3.7 "Faces"입니다
(공개 미러:
[afs.enea.it/project/neptunius/docs/fluent/html/ug/node1471.htm](http://afs.enea.it/project/neptunius/docs/fluent/html/ug/node1471.htm)).
방향 규칙도 그 문서의 것 — "노드 순서대로 오른손 손가락을 말면 엄지가 c1을
가리킨다" — 을 따르며, 모든 지수는 16진수 1-based입니다.

존(zone) 이름은 패치 이름을 따릅니다. 기본값은 polyMesh 타입을 먼저 따르고(`wall`은
`wall`, `symmetry`는 `symmetry`), 그다음 이 프로젝트가 입구에 붙이는 이름을
따릅니다 — `east`, `inlet`으로 시작, `_source`로 끝나는 이름은 `velocity-inlet`,
나머지는 전부 `pressure-outlet`입니다. Fluent는 읽은 뒤 스스로 재존하므로 기본값은
출발점일 뿐입니다. `-fluentType 이름=존`으로 하나를 정하면 이 기본값을 이기며,
모르는 존 이름(`wall`, `velocity-inlet`, `pressure-inlet`, `pressure-outlet`,
`outflow`, `symmetry`, `interior` 외)은 거절됩니다. 존 번호는 1이 유동 셀 존,
2가 내부 면 존이고 패치는 10부터 순서대로입니다.

Fluent에서는 File > Read > Mesh로 읽습니다. 좌표는 미터로 취급하고, 읽은 뒤에는
Mesh > Check를 돌리십시오.

**경로 주의.** Fluent의 Cortex가 한글이 섞인 폴더 경로에서 실패합니다 — 이 부지의
작업 폴더(`암모니아누출`)가 그랬습니다. Fluent에 읽힐 메쉬 파일은 ASCII로만 된
경로에 두십시오.

상대가 CGNS를 원한다면 Gmsh 단계에서 `gmsh.write('x.cgns')`로 직접 쓰고 File >
Import > CGNS로 여는 것이 대안입니다.

**독립 검사.** 변환기가 쓴 파일을 도구 없이 검사할 수 있습니다:

```powershell
python tools\mesh\tester_mesh.py 0.5
ofgpu-convert-mesh tester_tet.msh case -fluent tester_ours.msh
python tools\mesh\fluent_check.py tester_ours.msh tester_tet_geometry.json
```

`tester_mesh.py`는 알려진 작은 케이스(육면체 상자에서 사면체를 파낸 것)를 만들고,
`fluent_check.py`는 Fluent ASCII 메쉬를 밑바닥부터 다시 파싱해 헤더 개수와 실체의
일치, 모든 셀이 닫히고 양의 부피를 갖는 것, 총 부피가 상자−사면체와 같은 것,
경계 면이 셀 하나를 갖고 법선이 도메인 밖을 가리키는 것, c0/c1 방향 관례(가정하지
않고 측정합니다), 모든 존이 이름에 맞는 평면·고체 위에 있는 것을 검사합니다.
2026-09-09에는 이 테스터를 OpenFOAM의 foamMeshToFluent와도 대조했습니다 — 두
파일의 셀·노드·토폴로지·부피가 같고, 다른 것은 방향 관례뿐이었습니다(상대는
매뉴얼의 반대를 씁니다).

#### Studio 안에서

이 두 단계는 Studio의 기본 도구로도 등록되어 있습니다. 서버가 시작할 때
`gui/server/tools.defaults.json`에서 읽으므로 사용자가 만들 필요가 없습니다.

- **`mesh_from_step`** — 입력은 `config`(설정 JSON 경로, 필수)와 `extra`(선택적
  추가 토큰, 예: `--from-checkpoint`). 실행하는 명령은 콘솔과 같습니다 —
  `python <repo>\tools\mesh\step_mesh.py <config> <extra>`.
- **`mesh_to_fluent`** — 입력은 `msh`, `case`, `fluent`(필수)와 `types`(선택적
  `-type 이름=타입` 토큰). 역시 같은 명령 — `<OFGPU_BIN_DIR>\ofgpu-convert-mesh
  <msh> <case> -fluent <fluent> <types>`.

선택 필드는 비어 있으면 토큰째 사라지고, 공백을 포함한 값은 단어별로 나뉘어
인자가 됩니다. `<repo>`는 저장소 루트로 풀리고 `<OFGPU_BIN_DIR>`는 솔버와 같은
순서(`OFGPU_BIN_DIR` → `rust/target/release` → `rust/target/debug`)로 바이너리를
찾습니다.

실행은 `custom_tool_run`을 통하고 다른 도구와 마찬가지로 승인을 받습니다. 같은
이름의 사용자 도구를 `custom_tool_create`로 등록하면 **그쪽이 이깁니다** — 병합은
사용자 도구를 먼저 두고 기본값은 이름이 겹치지 않을 때만 채웁니다. 기본값이
사용자의 파일(`gui/config/custom-tools.json`)에 기록되지는 않습니다.

하나만 알아 두십시오: command 도구에는 60초의 시간 상한이 있습니다. 부지 전체
실행(약 23분)은 이 상한 안에 끝나지 않으므로 콘솔의 `run_step_mesh.cmd`로
돌리는 것이 맞고, Studio 도구는 `--dry-run`이나 변환 같은 짧은 단계에 씁니다.

#### 문제 해결 — 부지에서 실제로 만난 것

| 증상 | 원인 | 처방 |
|---|---|---|
| `algo3d: 10`(HXT)을 고르면 예고 없이 프로세스째 죽는다 | gmsh의 미완성 Steiner 경로가 경계 복구 중 걸린다. 예외가 아니라 죽음이라 잡을 방법이 없다 | `algo3d: 1`(Delaunay)로 두십시오. 다른 알고리즘이 0개 사면체로 끝내면 도구가 스스로 Delaunay로 재시도합니다 |
| Netgen 최적화가 access violation으로 죽는다 | 이 gmsh 빌드(4.14.1)의 결함 | Netgen은 아예 옵션에 없습니다. gmsh 자체 최적화(`optimize_passes`)를 쓰십시오 |
| `No elements in volume` — 3-D 패스가 모든 사면체를 버린다 | 3-D 패스 **전에** 중복 노드를 병합했다 | 3-D 전에는 절대 병합하지 마십시오. 이음새 병합은 post 단계의 `seam_merge_m`이 3-D 뒤에 합니다 (2026-09-08에 찾은 작업 경로) |
| 자기교차 고체(선박 선체)가 볼륨이 되지 않는다 | STEP 자체가 서로를 꿰뚫는 면을 담고 있다 | `repairs`의 `resample` — `cell_m` 격자로 표면을 재샘플링한 닫힌 프록시로 바꿔치기하십시오. 캐시(`work/repaired_<tag>.brep`)를 `brep`로 재사용하면 재수리는 건너뜁니다 |
| 건물이 지형 위에 떠 있다 | STEP의 베이스가 그 아래 지형보다 위에 있다 | `solids.sink_m`으로 베이스를 묻으십시오. 단 베이스가 trim 평면에 닿지 않게 — 2.0은 베이스를 정확히 3.0에 두어 1,814면이 trim 밑으로 내려가고, 1.5는 0면입니다 |
| 건물 사이에 머리카락 틈이 있다 | STEP에서 이웃 고체가 몇 센티미터 어긋나 있다 | 고체를 뚱뚱하게 부풀리지 마십시오. 맞닿거나 겹치는 이웃이면 `solids.fuse: true`로 합칩니다 |
| 3-D 패스가 겹치는 면으로 실패한다 (exit 3) | 틈을 메우려고 고체를 부풀린 대가 — 두 벽이 같은 자리를 차지한다 | 부풀림을 거두십시오. 뜬 베이스는 `sink_m`으로 묻고 맞닿는 고체는 fuse로 합칩니다 |
| 해수면이 브리프가 말하는 높이에 없다 | 브리프가 틀렸다 | 브리프 대신 기하를 보십시오 — 면적이 맞는 평평한 면을 찾습니다. 이 부지의 해수면은 z=+3이고(z=0의 겉보기 시트는 중복 껍질이다), trim/sea는 그 위 5 cm인 3.05에 둡니다 |
| 풀이 새겨지지 않았다 (`not imprinted`) | 그 점의 지면이 평평하지 않다 — 디스크가 경사에 파묻힌다 | 풀 디스크는 평평한 지면을 가정합니다. 실행은 그 풀 없이 계속되고 summary에 기록되므로, 필요하면 점을 평평한 자리로 옮기십시오 |
| GPU 솔버가 얇은 셀에 민감하다 | 선박–해수면, 지붕–슬랩 사이의 얇은 쐐기 | 한 반복 점검의 로더 줄 — `volume: ... min ...`과 `non-orthogonality: max ... deg` — 이 그 증거입니다. `post.thin_push_m`(예 0.2)으로 노드를 밀어 보십시오. 솔버 쪽 대응은 별도 트랜치입니다 |

---

## 6. 솔버 고르기 — 가장 많이 틀리는 곳

**이 절이 이 문서에서 가장 중요합니다.**

드라이버 이름이 난류 모델처럼 생겼다고 해서 그것이 유동을 푸는 것은 아닙니다.
`ofgpu-k-epsilon`, `ofgpu-k-omega`, `ofgpu-sa`는 **난류 방정식만** 풉니다.
속도장은 **얼린 채로** 둡니다.

즉 이 셋으로 외부 공력 케이스를 돌리면:

- 유선은 직선으로 나옵니다
- 컨투어는 균일한 한 색으로 나옵니다
- `1/`에 `U`도 `p`도 쓰이지 않습니다

차체 주위 유동을 보고 있는 것이 아니라 **초기장**을 보고 있는 것입니다.

### 무엇이 무엇을 푸는가

| 드라이버 | 푸는 것 | 케이스 형식 |
|---|---|---|
| `ofgpu-k-epsilon` | k, ε — **U는 얼림** | 디렉터리 / JSONC |
| `ofgpu-k-omega` | k, ω — **U는 얼림** | 디렉터리 |
| `ofgpu-sa` | ν̃ — **U는 얼림** | 디렉터리 |
| `ofgpu-lowmach` | **U, p**, T, 난류 | 디렉터리 / JSONC |
| `ofgpu-buoyant` | **U, p**, T, 부력 | 디렉터리 |
| `ofgpu-plume` | **U, p**, T, 부력 플룸 | 디렉터리 |
| `ofgpu-vof` | **U, p**, α — 2상 자유표면 | 디렉터리 |
| `ofgpu-cht` | 다영역 전도 + 켤레 자연대류 | JSONC |
| `ofgpu-datacentre` | 데이터센터 룸(팬·타일·습공기·지표) | JSONC |

속도장이 필요하면 **굵게 표시된 것 중에서 고르십시오.**

### 어떤 모델이 어느 드라이버로 가는가

모델 이름이 어느 바이너리에서 돌아가는지는 코드가 정합니다.
`rust/src/bin/common/mod.rs`의 `driver_for`는 정확히 이렇게 갈립니다:

| 모델 | 드라이버 |
|---|---|
| `kEpsilon`, `RealizableKE`, `RNGkEpsilon` | `ofgpu-k-epsilon` |
| `LaunderSharmaKE` | `ofgpu-buoyant` 또는 `ofgpu-lowmach` |
| `kOmega` | `ofgpu-k-omega` |
| `kOmegaSST` | `ofgpu-buoyant` 또는 `ofgpu-lowmach` |
| `kOmegaSSTLM`, `kOmegaSSTGamma` | `ofgpu-buoyant` 또는 `ofgpu-lowmach` |

마지막 행이 `ofgpu-k-omega`가 아닌 이유: 이 모델들은 SST에 방정식을 더 얹은
전이 모델이고 `build_coupled`를 통해서만 도달합니다. `ofgpu-k-omega`에서
돌리면 전이 케이스를 전연에서부터 완전 난류로 풀어 버립니다 — 그럴듯하게
수렴하는 그릇된 답으로, SPEC-LIT §13.4가 막으려는 바로 그것입니다.

`kOmegaSST`는 마지막 행과 같은 드라이버로 가지만 이유는 다릅니다: SST는 벽
거리가 필요하고(SPEC-LIT §6.6) `KOmegaSst::new`는 그것을 필수 인자로
받습니다. `build_coupled`는 모델을 만들기 전에 벽 거리를 계산하지만
`ofgpu-k-omega`는 전혀 계산하지 않으므로, SST는 `ofgpu-buoyant`와
`ofgpu-lowmach`를 통해서만 도달합니다.

### 난류 전용 드라이버는 언제 쓰는가

정해진 속도장 위에서 난류량만 보고 싶을 때입니다 — 모델 자체의 검증, 벽함수
비교, 특정 필드에 대한 난류 응답 조사. `plume.jsonc`처럼 얼린 `U` 위의 두
방정식 수렴이 목적인 케이스가 그렇습니다.

### 운동량 드라이버에 필요한 것

`ofgpu-lowmach`는 `U`와 `p`를 함께 풀고, 저마하 루프가 `T`를 요구합니다. 메쉬
생성기는 이 셋을 `0/`에 직접 쓰므로 — `channel`, `cavity`, `step`, `big`과
부력 쌍 `plume`/`room`이 대상이고 컷셀 경로도 마찬가지 — 갓 생성된 케이스를
그대로 받아 돌립니다. RANS 모델이 `k`와 `epsilon`/`omega`를 읽는 것도 마찬가지로
생성기가 채워 둡니다.

없이 돌리면 — 직접 만든 케이스에 필드를 빠뜨렸거나, `0/`이 `alpha.water`와
`p_rgh`뿐인 생성된 `damBreak`에 들이댄 경우 — 이렇게 거절합니다:

```
error: cases/racecar_case\0 has no p field;
       ofgpu-lowmach with kOmega solves U, p, T, k, omega
```

---

## 7. 실행하고 수렴을 읽기

### 공통 옵션

```
-iters N          최대 반복 (수렴하면 그전에 멈춤)
-fixedIters N     수렴 판정 없이 정확히 N회
-check N          N회마다 잔차 출력
-write NAME       결과 디렉터리 이름
-noWrite          결과를 쓰지 않음
-output LIST      출력 형식: foam, vtu, nvdb, vdb, usda
-permissive       인식되지 않는 설정을 경고로 낮추고 문서화된 기본값으로 대체
```

`-output`은 **형식** 목록이지 필드 목록이 아닙니다. `-output "U,p"`는 거절됩니다.

### 잔차 줄 읽기

`ofgpu-lowmach` 기준:

```
iter    500  |U| res 0.0928755  |p| res 1.351e-01  contErr 3.39691e-05
             T [293.15, 293.15] K  rho [1.2041, 1.2041] kg/m3  p0 101325 Pa
```

- `|U| res`, `|p| res` — 정규화 잔차. 정상해석에서 이것이 내려가면 수렴 중입니다.
- `contErr` — 연속방정식 오차. **이것이 내려가지 않으면 압력 해가 실패한 것**이고,
  잔차가 예뻐도 결과를 믿을 수 없습니다.
- `T`, `rho` 범위 — 물리적으로 말이 되는지 한눈에 보는 용도입니다. `T [inf, -inf]`가
  나오면 필드가 비었거나 깨진 것입니다.

난류 전용 드라이버는 대신 이렇게 나옵니다:

```
   250  omega res 7.457e-09 (0)  k res 1.799e-03 (1)  max dk/k 9.999e-04
```

괄호 안은 그 반복에서 상한/하한에 걸린 셀 수, `max dk/k`는 최대 상대 변화입니다.

### 발산하면

**멈춥니다.** 조용히 계속 돌지 않습니다. 발산한 반복 번호를 찍고, 무엇이
비유한이 되었는지 말합니다:

```
[ofgpu-lowmach] a field went non-finite at step 0 - stopping
iter 0 ... T [inf, -inf] K ... *** NaN/Inf ***
error: solution diverged (NaN/Inf)
```

### 격자 경고

```
[ofgpu] mesh warning: maximum non-orthogonality 79.7 deg;
        the explicit correction will dominate and the solution may need
        extra non-orthogonal correctors
```

컷셀 격자에서는 정상입니다 — 잘린 셀이 큰 비직교각을 만듭니다. 수렴이 나쁘면
`fvSolution`에서 비직교 보정 횟수를 늘리십시오.

---

## 8. 결과 보기

### 출력 형식

| `-output` | 무엇 |
|---|---|
| `foam` | OpenFOAM ASCII — ParaView·`foamToVTK`가 바로 읽음 |
| `vtu` | VTK 비정렬 격자 |
| `nvdb` / `vdb` | NanoVDB / OpenVDB 볼륨 |
| `usda` | USD ASCII |

### Studio 3D 뷰어

절단면, 임의 평면, 등가면, 유선, 벡터 글리프를 그 자리에서 그립니다. WebGPU로
그리고, 브라우저가 지원하지 않으면 조용히 WebGL2로 내려갑니다.

**컷셀 격자에서도 됩니다.** 외부 공력 케이스는 블록에서 형상을 파낸 것이라 정확
격자 검출기는 이것을 비정렬로 읽습니다. 뷰어는 셀 중심의 축별 히스토그램에서
격자점을 스파이크로, 잘린 중심을 잡음으로 구분해 **원래 블록을 복원**하고,
형상 안쪽에 해당하는 자리를 구멍으로 표시합니다:

```
structured block recovered from the cut-cell mesh:
128 x 128 x 128, 4,100 site(s) inside the body
```

구멍인 격자점은 보간에서 제외됩니다. 차체 안쪽을 찍으면 0이 아니라 **데이터
없음**으로 나오고, 유선은 표면에서 멈춥니다.

---

## 9. Studio — AI가 붙은 작업대

파일 탐색기, 편집기, 터미널, 실시간 잔차 그래프, 3D 뷰어, 그리고 그 전부를
대신 조작하는 AI 패널로 이루어진 데스크톱 작업대입니다.

말로 시키면 AI가 케이스 파일을 읽고, 바꿔야 할 부분을 찾아 **변경 내역으로
보여준 뒤, 확인을 받고 나서야 저장**합니다. 솔버를 실행하면 GPU를 잡는 순간부터
잔차가 실시간으로 그려집니다.

현재 **공개 베타**이며 이메일 주소만으로 이용할 수 있습니다. 언어는 영어와
한국어를 지원하고 설정에서 바꿉니다(기본 영어).

---

## 10. MCP — 쓰던 AI에게 솔버 넘기기

Studio는 클라이언트 하나일 뿐입니다. 솔버는 Model Context Protocol도 말하므로,
Claude Code든 Claude Desktop이든 MCP를 말하는 도구라면 무엇이든 여러분의 GPU에서
직접 격자를 만들고 케이스를 풀 수 있습니다.

```powershell
claude mcp add ofgpu -- node mcp\server.mjs
```

도구 6개(`ofgpu_probe`, `ofgpu_list_cases`, `ofgpu_generate_mesh`,
`ofgpu_solve`, `ofgpu_validate`, `ofgpu_read_case_file`), 그중 4개는 읽기
전용입니다. 셸을 쓰지 않고, 모든 경로를 작업 폴더 안으로 제한하며, 삭제 도구는
아예 없습니다.

전체 사용법: [`mcp/README.md`](../mcp/README.md).

---

## 11. 검증과 근거

세 개의 문서가 서로를 받칩니다.

- **[`rust/SPEC-LIT.md`](../rust/SPEC-LIT.md)** — 모든 이산화 식을 원논문 인용과
  함께 명세합니다. 코드가 아니라 이 문서가 기준입니다.
- **[`rust/PROVENANCE.md`](../rust/PROVENANCE.md)** — 파일별로 무엇을 읽고
  구현했는지 기록합니다.
- **`ofgpu-validate`** — 인위해법(MMS), 해석해, 공개 벤치마크를 실행합니다.

```powershell
ofgpu-validate
```

실행할 때마다 통과 수를 출력하고, **이어서 빗나가는 게이트와 열린 게이트를 이름
그대로 부릅니다.** 그 목록은 손으로 유지되지 않고, 게이트가 자신의 판정을
보고하는 바로 그 지점에서 들어가는 레지스트리로부터 생성됩니다(SPEC-LIT §69).
판정을 출력하는 것과 등록하는 것이 같은 호출이므로, 게이트가 늘어나도 목록에서
빠질 수 없습니다.

현재의 통과 수와 MISSES/OPEN 목록은 [`README.md`](../README.md)의 「현황」에
있으며, 그 숫자는 그 작업 트리에서 실제로 돌려 얻은 것입니다.

**중요한 구분**: "`ofgpu-validate`가 실행하는 모든 항목이 통과한다"는 "이
프로젝트가 비교하는 모든 발표된 벤치마크를 재현한다"와 다른 진술입니다. 둘을
혼동해서는 안 됩니다. 빗나가는 게이트는 숨기지 않고 이름과 수치와 함께
출력합니다.

그리고 **GPL 소스는 참조하지 않았습니다.** 파일마다 헤더로 선언하고, 문장이
아니라 시험으로 강제합니다(`provenance_audit`).

---

## 12. 할 수 없는 것

- **MPI·다중 GPU 없음.** 분할·헤일로·분산 PCG/PBiCGStab이 구현되고 게이트되어
  있으나 한 장의 카드 위 한 프로세스에서 돕니다. 통신 라이브러리를 링크하지
  않으며 강한 확장성 수치도 발표하지 않습니다.
- **압축성·천음속 없음.** 밀도 가중 시간미분은 VOF에서 쓰이지만 압력 방정식은
  비압축성입니다.
- **유한율(아레니우스) 화학 없음** — 경직 ODE 적분기도, 반응 메커니즘도 없습니다.
- **복사는 면대면만.** 투명 매질을 가로지르는 회색 확산 교환이며, 체적 안에서
  흡수·방출·산란하는 것은 없습니다.
- **면대면 복사·화학종 수송·라그랑주 스프레이는 케이스 형식이 없습니다.** 셋 다
  라이브러리 API로 명세되고 게이트되어 있으나 어느 드라이버도 케이스 파일에서
  읽지 않습니다.
- **적응 세분화는 어떤 솔버에도 연결되어 있지 않습니다.**
- **사면체 격자의 다면체 쌍대 변환은 없습니다.** 사면체 격자를 읽고 그 위에서
  푸는 것은 됩니다(Gmsh MSH 4.1, 면 기반 모델이라 읽힌 사면체는 일반 다면체
  메쉬로 취급). 없는 것은 절점 둘레의 사면체를 하나의 다면체로 합치는
  `polyDualMesh`식 쌍대 연산이며, 모든 면이 평면이고 모든 셀이 볼록하다는
  SPEC-LIT의 가정을 다시 검토하는 일과 경계 절점의 쌍대 셀이 열려 있다는
  경계 처리의 결정에서 작업이 시작되어야 변환기가 됩니다.
- **AMGX는 기본 비활성**이며, 비활성 상태에서도 선택기가 명시적으로
  "unavailable"이라고 보고합니다.
- **DES 계열이 발표된 박리 유동 통계를 재현한다고 주장하지 않습니다.**

빗나가는 게이트 두 개와 열린 게이트들의 이름·수치는 [`README.md`](../README.md)에
있습니다.

---

## 13. 문제 해결

실제로 자주 만나는 순서대로 놓았습니다.

### `non-manifold edge(s)` — 컷셀이 STL을 거절함

맞닿은 부품이 **면을 공유**하고 있습니다. 컷셀 분류는 닫힌 다양체를 요구하므로
안/밖을 가를 수 없습니다. 부품을 서로 **겹치게** 만드십시오. 살짝 파고들게 하는
편이 정확히 맞대는 것보다 언제나 낫습니다.

### `has no p field` — 운동량 드라이버가 거절함

생성된 케이스에서는 더 이상 만나지 않습니다 — 생성기가 `0/p`와 `0/T`를 직접
쓰기 때문입니다(`channel`, `cavity`, `step`, `big`, `plume`, `room`). 만나는
길은 따로 있습니다: 직접 만든 케이스에 필드를 넣지 않았거나, `0/`이
`alpha.water`와 `p_rgh`뿐인 생성된 `damBreak`에 운동량 드라이버를 들이댄
경우, 또는 필드 파일을 지우거나 옮긴 경우입니다. 빠진 필드를 `0/`에 넣으면
됩니다. [§6](#6-솔버-고르기--가장-많이-틀리는-곳) 참고.

### 유선이 직선이고 컨투어가 한 색

난류 전용 드라이버로 돌렸습니다. 속도장이 얼려 있어 초기장을 보고 있는
것입니다. `ofgpu-lowmach` 같은 운동량 드라이버로 다시 돌리십시오.

### `NO_STRUCTURED_GRID` — 뷰어가 절단면을 거절함

컷셀 격자에서 블록이 복원되지 않았습니다. 최신 버전에서는 히스토그램 기반
복원이 들어가 있으므로, 이 오류를 만나면 (a) 격자가 실제로 블록 기반이 아니거나
(b) 구멍이 너무 많아(사이트의 15 % 초과) 복원을 거부한 경우입니다. 후자는
"블록처럼 보이는 다른 격자"를 블록으로 읽지 않기 위한 의도된 거절입니다.

### 뷰어가 케이스를 옆으로 눕혀 그림

케이스에 `constant/g`가 없고, 바닥 패치 이름이 `bottomWall`/`floor`/`ground`가
아닙니다. 패치 이름을 맞추거나, 중력이 실제로 필요한 케이스라면 `constant/g`를
넣으십시오. **뷰어를 돌리려는 목적만으로 `g`를 넣지 마십시오** — 중력은
방정식에 실제로 들어가고 해가 달라집니다.

### `-output "U,p"` 가 거절됨

`-output`은 형식 목록입니다. `foam`, `vtu`, `nvdb`, `vdb`, `usda` 중에서
고르십시오.

### 결과가 `1/`이 아니라 `0/`에 쓰임

`system/controlDict`의 `startTime`/`endTime`/`writeControl`이 정합니다. 정상해석
드라이버는 최종 상태만 쓰며, 어느 시간 디렉터리에 쓸지는 이 설정을 따릅니다.
초기장을 보존하려면 원본을 따로 두십시오.

### 인식되지 않는 설정으로 거절됨

의도된 동작입니다(SPEC-LIT §13.4) — 조용한 대체는 없습니다. 기본값으로
넘어가려면 `-permissive`를 주십시오. 무엇을 무엇으로 대체했는지 출력합니다.

### 비직교 경고

컷셀 격자에서는 정상입니다. 수렴이 나쁘면 `fvSolution`의 비직교 보정 횟수를
늘리십시오.

---

## 14. 라이선스

**Prosperity Public License 3.0.0 + 라이선서 해석 조항.**

무료: 개인 학습·취미, 교육기관, 대학과 그 소속 연구소, 정부기관, 공공
안전·보건·환경 단체, 자선단체.

그 밖의 상업적 이용은 **30일 시험 후 유상 라이선스**입니다 — 사람이 아니라 회사
단위입니다.

**연구는 기관의 소유 주체가 아니라 목적으로 판단합니다.** 소방·의료·보건·안전·
재난·환경 보호를 목적으로 하는 연구는 정부출연연구기관에서 수행하더라도
무료입니다. 특정 제품이나 산업 이전을 목적으로 하는 기술개발 — 전기차,
철도·추진, 항공기 엔진 등 **기술료 징수 대상** — 은 정출연이라도 상업적
이용입니다.

전문: [`LICENSE`](../LICENSE), 해설: [`LICENSING.md`](../LICENSING.md).

---

## 더 읽을 것

| 문서 | 무엇 |
|---|---|
| [`README.md`](../README.md) | 개요, 현황 수치, 할 수 있는 것과 없는 것, 참고문헌 |
| [`rust/SPEC-LIT.md`](../rust/SPEC-LIT.md) | 모든 이산화 식의 명세와 원논문 인용 |
| [`rust/PROVENANCE.md`](../rust/PROVENANCE.md) | 파일별 출처 |
| [`cases/README.md`](../cases/README.md) | 케이스 형식, 벽 모델 프리셋, 케이스별 기록 |
| [`cases/racecar.md`](../cases/racecar.md) | 경주차 샘플 상세 |
| [`mcp/README.md`](../mcp/README.md) | MCP 서버 사용법 |
| [`docs/07-lowmach-solver.md`](07-lowmach-solver.md) | 저마하 솔버의 설계와 진단 |
| [`docs/01-model-catalog.md`](01-model-catalog.md) | 모델 카탈로그 |

---

소유: 주식회사 이터레이션즈 · 협력 및 기여: 주식회사 메테오시뮬레이션
· 문의: simul@msimul.com
