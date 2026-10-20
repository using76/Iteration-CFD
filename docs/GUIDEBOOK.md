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

**쓰임새**는 비압축성·저마하·압축성 유동입니다. RANS/LES/하이브리드 난류, 부력 플룸,
가변밀도 저마하, 2상 VOF, 켤레 열전달, 면대면 복사, 라그랑주 분무, 팬과 다공성
점프를 갖춘 환기·데이터센터 기류.

**아닌 것**: 검증된 천음속 해석기가 아닙니다. `ofgpu-compressible`는 새로 들어왔고 천음속 게이트가 미달입니다(§12). 유한율 화학이 없습니다. 다중 GPU
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

설치되는 것: `ofgpu-*` 바이너리 22개, 케이스 정의(`cases/*.jsonc`), 경주차 샘플,
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

끝나면 `racecar_case/3000/`(반복 횟수가 디렉터리 이름입니다)에 `U`, `p`, `T`,
`rho`, `k`, `epsilon`, `nut`이 OpenFOAM ASCII로 들어 있고, `0/`은 초기장 그대로입니다.
Studio의 3D 뷰어로 열거나 ParaView로 바로 읽힙니다.

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

> 큰 부지(수 km, 건물 수백 개)를 격자로 만들 때 무엇이 깨지고 어떤 레시피로 고쳤는지는
> `docs/08-site-mesh-playbook.md`에, 절차와 명령은 `.claude/skills/site-mesh/SKILL.md`에 있다.

```powershell
ofgpu-generate-mesh <preset> <outputDir> [nx ny nz] [-stl [name=]path]...
                    [-cutcell [-s N] [-thetaMin X]]
                    [-extent xlo xhi ylo yhi zlo zhi] [-grading x|y|z=r]...
                    [-wallModel standard|spalding|rough|lowRe [-Ks x [-Cs y]]]
                    [-cyclic x|y|z] [-permissive]
```

프리셋: `channel`, `cavity`, `step`, `big`, `plume`, `room`, `damBreak`.
`big`은 한 변 1 m의 정육면체 풍동이고 셀 수를 하나(`n`, `n³` 정육면체)로 받거나 축별로 셋(`nx ny nz`, 예: `big dir 128 52 41`)으로 받습니다. 둘만 주면 이름을 대고 거부합니다.

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

격자기가 건물 아래에 봉인된 셀 포켓을 남기기도 합니다: 모든 면이 포켓 안의
내부면이거나 둘러싼 벽 패치(`wall_ground_land`)의 경계면뿐인 셀 무리입니다.
포켓을 본격과 잇는 면이 하나도 없으므로 그 벽에 `p`가 zeroGradient이면 포켓의
압력 블록은 특이행렬이 되고, 압력은 1e16으로 흘러가 선형솔버가 잔차를 정규화하는
전체 셀 평균을 오염시킵니다 — 4회 반복부터 압력 솔버는 반복 0회를 보고하고,
나머지 계산은 죽은 압력식을 향해 걸어갑니다. 변환기는 셀 영역을 세고
기본값으로 가장 큰 영역만 남겨 나머지를 버립니다 — 셀과 내부면과 경계면을
지우고 셀 번호를 다시 매기되 패치 순서와 상삼각 면 순서는 그대로 두고 — 이렇게
말합니다:

```
regions: 9; dropped 8 sealed region(s), 24 cell(s), 77 face(s) (19 internal, 58 on wall_ground_land)
kept region: 343466 cells
```

버림이 포켓 크기를 넘으면 거절합니다: 전체 셀의 1 %가 넘거나 가장 큰 영역이
두 번째의 10배가 못 되면 그것은 포켓이 아니라 두 번째 도메인이므로 결정은
기본값이 아니라 사용자의 몫입니다. `-keepRegions`이 그 결정을 기록하는
방법입니다 — 모든 영역을 그대로 변환하고 `regions:` 줄만 인쇄합니다. 그래도
여러 영역이 남은 케이스라면 로더가 알려 줍니다: 압력 솔버가 반복 0회를
보고하기 시작하면 가장 먼저 읽을 것이 로더의 `regions:` 줄입니다.

**한 반복으로 격자부터 점검하십시오.**

```powershell
ofgpu-buoyant nh3_case -iters 1
```

필드를 읽기 전에 로더가 격자 통계를 인쇄합니다 — `mesh: <cells> cells, ...` 줄,
패치 표(이름·타입·면 수), `volume: total ..., min ..., max ...`,
`non-orthogonality: max ... deg, mean ... deg`, `face closure`,
`lduAddressing: upper-triangular`, `regions: 1`. 이 줄들이 부지 격자의 성적표입니다. 변환만 한
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
| `ofgpu-cht` | 다영역 전도 + 켤레 자연대류 + 열응력 | JSONC |
| `ofgpu-datacentre` | 데이터센터 룸(팬·타일·습공기·지표) | JSONC |
| `ofgpu-compressible` | **U, p, T** — 압축성, 전속도(SPEC-LIT §122-§127) | 디렉터리 / JSONC |

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

### 볼륨 파일의 복셀 격자 — `output.visualisation.lattice`

`.vdb`/`.nvdb` 파일은 복셀 격자입니다. 균일 카테시안 상자에서는 셀이 곧 복셀입니다.
그 밖의 격자는 정육면체 복셀 격자에 샘플링되며, JSONC 케이스는 그 격자를
`output.visualisation` 안에 적습니다.

```jsonc
"output": { "visualisation": { "format": "vdb",
  "lattice": { "voxel": 0.01, "box": [0, 0, 0, 2, 1, 1], "maxVoxels": 8000000,
               "sample": "cell", "activation": "dense", "background": 0.0 } } }
```

| 키 | 뜻 | 기본값 |
|---|---|---|
| `voxel` | 복셀 한 변, m | 가장 작은 셀 부피의 세제곱근 |
| `box` | `[xlo, ylo, zlo, xhi, yhi, zhi]`, m | 격자 점들의 경계 상자 |
| `maxVoxels` | 복셀 예산; 이보다 큰 격자는 맞을 때까지 성기게 함 | 8 000 000 |
| `sample` | `cell`(셀 값) 또는 `linear`(잘라낸 기울기를 더함) | `cell` |
| `activation` | `dense`(모든 복셀) 또는 `fluid`(셀 안의 복셀만) | `dense` |
| `background` | 어느 셀에도 들지 않는 복셀의 값 | `0.0` |

키는 모두 생략할 수 있습니다. 이 블록으로 격자를 만들 수 없으면 솔버는 루프 전에 키 이름을
대고 거절합니다: 어느 축에서 `lo >= hi`인 `box`, 0 이하의 `voxel`, `maxVoxels: 0`, 예산을
맞추려면 16배를 넘게 성기게 해야 하는 명시적 `voxel`, 모르는 `sample`·`activation` 값.
`-permissive`를 주면 대신 시각화 출력을 쓰지 않습니다.

모든 드라이버(`ofgpu-lowmach`, `ofgpu-k-omega`, `ofgpu-k-epsilon`, `ofgpu-sa`, `ofgpu-vof`,
`ofgpu-buoyant`, `ofgpu-cht`)가 어떤 격자에서도 이 격자를 깝니다. 케이스 블록을 읽는 드라이버는
그 블록을 쓰고, 격자를 말하지 않는 명령줄(`-output vdb`)에서는 위의 기본값을 씁니다. 격자는
루프 전에 한 번 만들고 크기, 복셀, 유체 비율을 출력합니다. 얇거나 경사가 심한 격자에서는 기본
복셀(가장 작은 셀 부피의 세제곱근)이 영역의 두께보다 성길 수 있으니 `voxel`을 적으십시오.
`ofgpu-cht`는 영역마다 따로 격자를 깔고(기본 상자는 그 영역의 상자), 영역마다
`VDB/<영역>_000000.vdb`를 씁니다. 여러 영역을 담는 `.usda` 장면은 만들지 않으므로 이 형식에서는
`usdScene`을 거절합니다.

### Isaac Sim 장면 — `output.visualisation.isaac`

`isaac` 블록은 볼륨 출력을 Isaac Sim이 그리는 형태로 바꿉니다. `.vdb` 위에 `density`
격자 하나, 그것을 가리키는 `.usda` 장면 하나입니다. 이 블록은 장면을 함축합니다 — `usda`를
적거나 붙게 두십시오. 다만 `vdb`는 요구합니다. Isaac 장면은 절대 `.nvdb`를 가리키지
않기 때문입니다.

```jsonc
"output": { "visualisation": { "format": "vdb",
  "isaac": { "profile": "index", "field": "smoke", "scaling": "opticalDepth", "tau": 2.0 } } }
```

| 키 | 뜻 | 기본값 |
|---|---|---|
| `profile` | `index`(NVIDIA IndeX, `def Volume`) 또는 `pathtrace`(RTX 경로 추적기, 볼륨 `Mesh`) | `index` |
| `field` | 밀도를 만들 셀 필드 | 필수 |
| `scaling` | `opticalDepth` 또는 `physical`(`K_m rho_s`) | `opticalDepth` |
| `tau` | 가장 짙은 열의 광학 두께 | `2.0` |
| `range` | 정규화의 `[f_lo, f_hi]` | 유체 위에서 필드의 최소·최대 |
| `Km` | 비소광계수, m^2/kg(`physical`) | `8700`(Mulholland & Croarkin 2000) |
| `kind` | `density` 또는 `massFraction`(이때 `rho_v f_v`) | `density` |
| `upAxis`, `metersPerUnit` | 스테이지 메타데이터 | `"Z"`, `1.0` |

`opticalDepth`는 정규화된 필드에 배율을 맞춰 가장 짙은 복셀 열의 광학 두께가 정확히
`tau`가 되게 합니다. 범위와 배율을 첫 쓰기에서 잡아 두므로 자라는 플룸은 자라는 것처럼
보입니다.
`physical`은 대신 실제 그을음 밀도의 소광을 씁니다. 격자 `activation`은 케이스가 적지
않았다면 프로파일이 정합니다(`index` -> `fluid`, `pathtrace` -> `dense`), 어긋나는 값은
이름을 대고 거절합니다. `-permissive`가 누그러뜨리지 않는 두 거절이 있습니다: 실행에 없는
`isaac.field`, 그리고 `rho`가 없는 실행의 `massFraction` — 대신할 것이 없기 때문입니다.
`ofgpu-sample volume`은 같은 블록을 `-isaac JSON`으로 받고, 같은 거절을 그대로 적용합니다.

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

현재의 통과 수와 MISSES/OPEN 목록은 전체 실행 한 번의 것이며, SPEC-LIT §110.5가 기록하고 이 절의 끝에 표로 실었습니다. [`README.md`](../README.md)의 「현황」도 같은 생성 블록을 싣습니다.

### 검증 항목 하나하나 — 어떤 검증이며, 무엇을 근거로 견주었는가

아래 표는 `ofgpu-validate`가 출력하는 절(`=== … ===` 제목) 하나하나에 대해 세 가지를
적습니다: 그 절이 **어떤 종류의 검증**인지, 그 판정이 **무엇에 견준 것**인지(논문의
해석해·벤치마크 수치·상관식·공개 데이터·이 저장소가 기록한 측정), 그리고 그 자료의
**접속 주소**(DOI 또는 URL — 서지에 DOI가 없는 오래된 문헌은 서지사항만 적고 "DOI
없음"이라 표시)입니다. 서지사항 전체와 "읽지 않은 출처" 표시는
[`README.md`](../README.md)의 「참고문헌」에 있고, 각 식의 출처 인용은
[`rust/SPEC-LIT.md`](../rust/SPEC-LIT.md)에 있습니다.

검증의 종류는 다섯입니다. **항등식**(이산화가 정확히 만족해야 하는 닫힌 식 — 외부
자료 없이 SPEC-LIT 자체가 근거), **참조 구현 대조**(SPEC-LIT §3을 호스트에서 scatter
루프로 독립 전사한 `ofgpu::reference`와 장치 gather의 일치), **제조해/해석해**(MMS의
관측 수렴 차수, 논문의 닫힌 해와의 오차), **공개 벤치마크/상관식/데이터**(발표된
수치·상관식·공개 데이터셋과의 비교 — 이 프로젝트가 "맞다"고 주장하는 유일한 외부
근거), **기록 재생**(이 실행이 아니라 이전에 측정해 `docs/07-lowmach-solver.md`에
적어 둔 숫자를 다시 판정 — 833개 중 45개). 다른 CFD 코드의 출력과는 어느 항목도
비교하지 않습니다(SPEC-LIT §0 규칙 4).

| `ofgpu-validate` 절 | 어떤 검증인가 | 무엇에 견주었나 (쓰인 검증 자료) | 접속 주소 |
|---|---|---|---|
| `3-D graded block`, `3-D sheared block (non-orthogonal)`, `2-D block with empty front and back` — 격자 항등식 | 항등식: 셀 닫힘 \|ΣS_f\|/V^(2/3), 부피 = 해석 부피, 모든 면이 두 셀 중심을 분리, 보간 가중치 ∈ [0,1], lduAddressing 상삼각 (SPEC-LIT §10) | SPEC-LIT §2–§3의 정의 자체(외부 자료 없음). 그 정의의 출처: Jasak (1996) 박사논문 3장; Moukalled, Mangani & Darwish (2016) | Jasak: <http://hdl.handle.net/10044/1/8335>; Moukalled 등: Springer, DOI 10.1007/978-3-319-16874-6 |
| 같은 절 — 명시 연산자 | 항등식+참조 구현: 선형장의 `fvc::grad` = 해석값, 균일 유량의 발산 = 0, 라플라시안 차수 (SPEC-LIT §10 「Gradient」「Divergence」「Laplacian order」) | 해석 항등식(선형장·균일 유량)과 `ofgpu::reference`(SPEC-LIT §3의 호스트 전사); Gauss 정리의 이산형은 Jasak (1996) 3장 | 위와 같음 |
| 같은 절 — 스큐 보정 | 참조 구현 대조: 스큐 면에서 §74.4 보정을 장치 gather vs 호스트 scatter로 대조, 보정 유무가 답을 바꾸는지 | `ofgpu::reference`; 보정식은 Jasak (1996) §3.3, Ferziger & Perić (2002) | 위와 같음; Ferziger & Perić: Springer, DOI 10.1007/978-3-642-56026-2 |
| 같은 절 — 묵시 조립 | 항등식: 대각·비대각·완화·경계 접기·Amul을 참조 구현과 대조, 세 대류 스킴 (SPEC-LIT §10) | `ofgpu::reference`; 완화·경계 접기 규칙은 Patankar (1980) 4–6장; Rhie & Chow (1983) 면 보간 | Patankar: Hemisphere, ISBN 0-89116-522-3 (DOI 없음); Rhie & Chow: *AIAA J.* 21, 1525, DOI 10.2514/3.8284 |
| `3-D block with 2:1 refinement interfaces`, `the adapt: refine, coarsen, and what a rebuild costs` | 항등식: 2:1 계면 격자의 닫힘·조립, 세분·병합 후 적분 보존, 재구성 비용 측정 (SPEC-LIT §82–84) | SPEC-LIT §82–84 자체(적분 보존은 정의상 정확); 계면 처리의 출처는 Jasak (1996), Moukalled 등 (2016) | 위와 같음 |
| `linear solvers` | 해석해 대조: Krylov 해 vs 밀집 직접해(가우스 소거), cuFFT 직접 Poisson 해 vs 같은 행렬의 반복해 (SPEC-LIT §10 「Solver」「FFT Poisson」) | 직접해 자체가 기준(외부 자료 없음). 해법의 출처: Hestenes & Stiefel (1952) CG, van der Vorst (1992) BiCGStab, Saad (2003), Swarztrauber (1977) FFT Poisson | Hestenes & Stiefel: DOI 10.6028/jres.049.044; van der Vorst: DOI 10.1137/0913035; Saad: DOI 10.1137/1.9780898718003; Swarztrauber: *SIAM Rev.* 19, 490, DOI 10.1137/1019071 |
| `method of manufactured solutions, -lap(psi) = f` | 제조해: 3-D graded / 3-D sheared / 2-D empty에서 한 번 세분화해 관측 차수 log₂(e_coarse/e_fine) 보고 | 제조해 방법론: Roache (1998) *Verification and Validation in Computational Science and Engineering*, Hermosa; Roache (2002) *J. Fluids Eng.* 124, 4 | Roache 1998: ISBN 0-913478-08-3 (DOI 없음); Roache 2002: DOI 10.1115/1.1436090 |
| `buoyancy` | 항등식: b = g(T_ref/T − 1)의 산술, 정수압 균형, 부력 부호 (SPEC-LIT §9, §10 「Hydrostatic」「Buoyancy sign」) | SPEC-LIT §9 자체. 정식의 출처: Rehm & Baum (1978) *J. Res. NBS* 83, 297; Spiegel & Veronis (1960) ΔT/T ≪ 1 조건 | Rehm & Baum: NIST J. Res. 아카이브 <https://nvlpubs.nist.gov/nistpubs/jres/>; Spiegel & Veronis: *Astrophys. J.* 131, 442 (DOI 없음) |
| `buoyancy production, sources, species, phi I/O` | 항등식: 안정/불안정 성층에서 G_b의 부호, 체적 열원이 엔탈피를 정확히 P만큼, 화학종 합 = 1 및 [0,1], 기록·재읽기한 `phi`의 첫 압력 잔차 비트 동일 (SPEC-LIT §17, §22) | G_b의 정의: Rodi (1987) *J. Geophys. Res.* 92, 5305; Henkes, van der Vlugt & Hoogendoorn (1991) *IJHMT* 34, 377. 나머지는 SPEC-LIT §22의 항등식 자체 | Rodi: DOI 10.1029/JC092iC05p05305; Henkes 등: DOI 10.1016/0017-9310(91)90258-G |
| `volume of fluid (SPEC-LIT 20, the 22 rows)` | 해석해+보존: 20.1 이동 계면의 비확산, 20.2 Zalesak 회전 슬롯 원판, 20.3 질량 플럭스·밀도 일관, 20.4 원의 곡률과 Laplace 압력 점프, 20.5 정지 성층 유체(p_rgh) | Zalesak (1979) *J. Comput. Phys.* 31, 335(원판 문제); Brackbill, Kothe & Zemach (1992) *J. Comput. Phys.* 100, 335(CSF 곡률·Laplace 점프); Hirt & Nichols (1981) *J. Comput. Phys.* 39, 201(VOF); Ubbink (1997)·Rusche (2002) 박사논문(계면 압축) | Zalesak: DOI 10.1016/0021-9991(79)90051-2; Brackbill 등: DOI 10.1016/0021-9991(92)90240-Y; Hirt & Nichols: DOI 10.1016/0021-9991(81)90145-5; Ubbink·Rusche: Imperial College 박사논문(DOI 없음) |
| `msh hex closure, cut-cell closure (SPEC-LIT 23, 24)` | 항등식: 실제 `parse_msh`를 통과한 Gmsh 4.1 육면체의 닫힘, STL 컷셀의 면적 벡터 닫힘 | 공개 파일 형식 명세: Gmsh MSH 4.1 형식 문서, STL 형식. 닫힘 자체는 SPEC-LIT §23–24의 정의 | Gmsh MSH: <https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format> |
| `the low-Mach reference pressure (SPEC-LIT 25)` | 해석해: 밀폐 상자의 히터 P가 dp₀/dt = (γ−1)P/V로 정확히, 열린 영역은 p₀ 불변 | 저마하 정식: Rehm & Baum (1978); Majda & Sethian (1985) *Combust. Sci. Technol.* 42, 185; NIST *FDS Technical Reference Guide* (SP 1018-1, 퍼블릭 도메인) | Rehm & Baum: 위와 같음; FDS 기술 참조서: <https://pages.nist.gov/fds-smv/manuals.html> |
| `wall treatment: Ks -> 0, the thermal wall function (SPEC-LIT 29)` | 항등식: 거친벽 법칙이 K_s = 0에서 매끈벽을 반올림 오차로(두 벽함수 계열), Jayatilleke P(Pr/Pr_t = 1) = 0, 한 셀 전도도 = 해석 열유속 | Jayatilleke (1969) *Prog. Heat Mass Transfer* 1, 193(열 벽함수 P 함수); Cebeci & Bradshaw (1977)(거친벽 상수, Nikuradse 모래알 데이터); Spalding (1961) *J. Appl. Mech.* 28, 455 | Jayatilleke: Pergamon (DOI 없음); Cebeci & Bradshaw: Hemisphere (DOI 없음); Spalding: DOI 10.1115/1.3641728 |
| `Werner-Wengle, coupled-solver turbulence selection (SPEC-LIT 30)` | 항등식: WW 벽모델 두 분기의 연속성, 제조한 τ_w의 역산 복원; 부력 드라이버의 `kOmegaSST` 선택 확인 | Werner & Wengle (1991) *8th Symp. Turbulent Shear Flows*(멱법칙 벽모델); Menter (1994) *AIAA J.* 32, 1598, Menter, Kuntz & Langtry (2003)(SST) | Werner & Wengle: Springer *Turbulent Shear Flows 8*, DOI 10.1007/978-3-642-77674-8_12; Menter 1994: DOI 10.2514/3.12149 |
| `periodic domains: cyclic-pair invariants (SPEC-LIT 31.1)` | 항등식: 주기 면 짝의 기하 매칭 | SPEC-LIT §31.1의 정의 자체(외부 자료 없음) | — |
| `the thermal wall-function gate, redesigned (SPEC-LIT 32)` | 항등식+상관식+LIVE+재생: `flux_to_grad` 항등식; Dittus–Boelter와 Gnielinski의 상호 ±20–25 %(Re 1.6e4, Pr 0.71); 실현 마찰계수를 힘 균형과 층류 Poiseuille f·Re에(LIVE); 벽함수 판정 재생 — §32.4의 **OPEN 3건**이 여기 | Dittus & Boelter (1930/1985 재수록) Nu 상관식; Gnielinski (1976) *Int. Chem. Eng.* 16, 359(±10 % — 셋 다 이 상관식에 견줌); Petukhov (1970) 매끈관 f; 기록: `docs/07-lowmach-solver.md` §1.1, 케이스 `cases/channelPeriodicFluxWF.jsonc` | Dittus–Boelter 재수록: *Int. Commun. Heat Mass Transfer* 12 (1985) 3, DOI 10.1016/0735-1933(85)90003-X; Gnielinski: DOI 없음; Petukhov: *Adv. Heat Transfer* 6, 503, DOI 10.1016/S0065-2717(08)70153-9 |
| `Launder-Sharma low-Re k-epsilon: damping functions (SPEC-LIT 33.3)` | 항등식: f_μ, f₂의 Re_t → ∞ / 0 극한, 단조성, 표준 모델로의 환원 | Launder & Sharma (1974) *Lett. Heat Mass Transfer* 1, 131; 표준 계수: Launder & Spalding (1974) *CMAME* 3, 269; 저Re 계열 리뷰: Patel, Rodi & Scheuerer (1985) *AIAA J.* 23, 1308 | Launder & Sharma: DOI 10.1016/0094-4548(74)90150-7; Launder & Spalding: DOI 10.1016/0045-7825(74)90029-2; Patel 등: DOI 10.2514/3.9086 |
| `resolved leg mesh resolution, replayed (SPEC-LIT 33.2/34)` | 기록 재생: 해상 격자 첫 셀 y⁺ = 0.00174, y⁺ < 20인 셀 192/400 | 이 저장소의 기록: `docs/07-lowmach-solver.md` §1.1, 케이스 `cases/channelPeriodicFluxLowRe.jsonc`, 판정 `ofgpu::models::mesh_resolution_report` | 저장소 내부 |
| `the bulk-temperature thermostat (SPEC-LIT 35)` | 항등식(LIVE): 비례 제어기의 부호와 정상 오프셋 | SPEC-LIT §35.1의 비례 법칙 자체(외부 자료 없음) | — |
| `thermostat weighting: the decisive experiment, replayed (SPEC-LIT 35.3.2)` | 기록 재생: 질량유속 가중이 (T_w − T_b)를 넓히고 Nu를 낮추며 해상 격자에서 더 크게 — 예측한 세 진술을 네 측정값에 | 이 저장소의 기록: `docs/07-lowmach-solver.md` §1.1(두 격자 × 두 가중의 네 실행) | 저장소 내부 |
| `bounded convection on momentum: the isolation, replayed (SPEC-LIT 3.1/32.5.5)` | 기록 재생: `bounded` 접두어와 대류 스킴 차수를 분리한 조합 — `bounded`를 빼면 항력 균형이 닫힘 | 이 저장소의 기록: `docs/07-lowmach-solver.md` §1.1(해상 다리 4조합, 벽함수 다리 3조합) | 저장소 내부 |
| `Kays-Crawford turbulent Prandtl number (SPEC-LIT 37.1/37.2)` + 실험 재생 | 상관식 산술: Pe_t → 0에서 2·Pr_t∞ = 1.70(Kays의 공기 1.5–1.9 안), Pe_t → ∞에서 0.85; 재생: Nu가 두 다리에서 낮아지고 해상 격자에서 더 크게 | Kays (1994) *ASME J. Heat Transfer* 116, 284(Pr_t 상관식과 벽 근처 상승); 기록: `docs/07-lowmach-solver.md` §1.1 | Kays: DOI 10.1115/1.2911398 |
| `realizable and RNG k-epsilon (SPEC-LIT 40, 41)` | 항등식+LIVE: 실현가능성 ⟨u_a u_a⟩ ≥ 0(GPU의 C_μ, Shih의 임계 λ_max k/ε = 3.7037 자체 계산), 계수의 닫힌 형식, 균질 전단 ODE의 점근 Sk/ε, 강한 변형에서 상수 C_μ의 실패 | Shih, Liou, Shabbir, Yang & Zhu (1995) *Comput. Fluids* 24, 227 — NASA TM-106721로 읽음; Yakhot 등 (1992) *Phys. Fluids A* 4, 1510 — ICASE 91-65로 읽음; Reynolds AGARD-755 (1987)·Lumley (1978) 실현가능성 조건 | Shih 등: <https://ntrs.nasa.gov/citations/19950005029>; Yakhot 등: <https://ntrs.nasa.gov/citations/19910021152>; Lumley: *Adv. Appl. Mech.* 18, 123, DOI 10.1016/S0065-2156(08)70266-7 |
| `the output block, and fp16 voxels (SPEC-LIT 44, 45)` | 실행 검사: `output` 블록 해석·거절, 작성기가 실제로 파일을 쓰고 측정·삭제 | SPEC-LIT §44–45 자체; 형식 명세: OpenFOAM ASCII 필드 형식, VTK, OpenVDB/NanoVDB(공개 형식) | — |
| `conjugate heat transfer (SPEC-LIT 46, 47, 48)` | 해석해+보존+벤치마크: 접촉저항 2층 슬래브, 두 자유 극한, 과도 계면 온도, 보존; Gate 5 Kaminski & Prakash (1986) **MISSES**(−7.11 %), Gate 6 Qu & Mudawar (2002), Gate 7 Flageul 등 (2015) 미실행 | 해석해: Carslaw & Jaeger (1959) 1장; 접촉 전도: Cooper, Mikic & Yovanovich (1969); 분할 안정성: Meng 등 (2017), Henshaw & Chand (2009), Verstraete & Scholl (2016), Giles (1997); Gate 5 비교값은 Kaminski & Prakash(유료, 미열람)이 아니라 **2차 출처 Belazizia 등 (2012)** | Kaminski & Prakash: DOI 10.1016/0017-9310(86)90017-7; Cooper 등: DOI 10.1016/0017-9310(69)90011-8; Meng 등: DOI 10.1016/j.jcp.2017.04.052; Henshaw & Chand: DOI 10.1016/j.jcp.2009.02.007; Verstraete & Scholl: DOI 10.1016/j.ijheatmasstransfer.2016.05.041; Giles: DOI 10.1002/(SICI)1097-0363(19970830)25:4<421::AID-FLD557>3.0.CO;2-J; Belazizia 등: *Adv. Theor. Appl. Mech.* 5 (2012), DOI 없음 |
| `the conjugate fluid/solid interface (SPEC-LIT 59, 60)` | 공개 벤치마크: Gate 59-A de Vahl Davis (1983) 정사각 공동 자연대류, Ra = 10⁴에서 Nu = 2.243 (0.6 %); 59-B 정확 항등식; Gate 5 켤레 자연대류(위) | de Vahl Davis (1983) *Int. J. Numer. Methods Fluids* 3, 249 — 표 I의 벤치마크 Nu | de Vahl Davis: DOI 10.1002/fld.1650030305 |
| 같은 절 안 — §79 강제대류 Gate 6 | 공개 벤치마크: Qu & Mudawar (2002) 마이크로채널 히트싱크의 열저항 — 저자의 Fig. 4(b)/4(c)를 **디지타이즈**해 견줌(공개 1); Kawano 등 (1998)의 입·출구 열저항 측정 | Qu & Mudawar (2002) *IJHMT* 45, 3973(저자 공개본으로 완독); Kawano, Minakami, Iwasaki & Ishizuka (1998) *ASME HTD-361-3* 173 | Qu & Mudawar: DOI 10.1016/S0017-9310(02)00101-1; Kawano 등: DOI 없음 |
| `surface-to-surface radiation (SPEC-LIT 49, 50, 51)` | 해석해: 시야계수 C-11·C-14(무장애), Shapiro FACET의 장애 구성 F₁₂ = 0.115621, 대규모 닫힘; 회색 무한 평행판, 동심체, 재복사 벽이 있는 3면 밀폐, 복사 평형; 결정성 | Howell 시야계수 카탈로그 C-11, C-14; Shapiro (1983) FACET UCID-19887; Walton (2002) NISTIR 6925(장애 적분법); 방사도 해석해: Modest (2013) 5장, Hottel & Sarofim (1967) 3·5장 | Howell: <https://www.thermalradiation.net/>; Shapiro: DOI 10.2172/5607653; Walton: <https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nistir6925.pdf>; Modest: Academic Press, ISBN 978-0-12-386944-9 (DOI 없음) |
| `fan curves, porous jumps, psychrometrics, metrics (SPEC-LIT 52, 53, 54, 55)` | 닫힌 형식+공개 데이터: 팬 작동점(정확), **Gate 52-B는 NIST FDS의 `fan_test`/`qfan_test` 입력과 발표 CSV(퍼블릭 도메인 데이터)에 직접 대조**; 다공 점프 직렬 저항·타일 유량 분배; ASHRAE 습공기 13계수·IAPWS 비등점; RCI/RTI/SHI 항등식과 도달 가능했던 외부 수치 하나 | FDS 검증 세트 `Verification/HVAC/fan_test.fds`, `qfan_test.fds` + CSV; AMCA 210 / ASHRAE 51(팬 곡선의 정의); ASHRAE Handbook—Fundamentals (2021) 1장(Hyland–Wexler 계수), Gatley 등 (2008) 건공기 몰질량; IAPWS; Ward (1964) Darcy–Forchheimer; Karki & Patankar (2006) 타일; Herrlin (2005, 2008) RCI/RTI; Sharma, Bash & Patel (2002) SHI/RHI | FDS HVAC 케이스: <https://github.com/firemodels/fds/tree/master/Verification/HVAC>; Gatley 등: DOI 10.1080/10789669.2008.10391032; IAPWS: <http://www.iapws.org/>; Ward: DOI 10.1061/JYCEAJ.0001096; Karki & Patankar: DOI 10.1016/j.buildenv.2005.03.005; Herrlin 2005: <https://www.semanticscholar.org/paper/99b942df4aa448a1e06f77d36b48d5d52a40c6e0>; Sharma 등: DOI 10.2514/6.2002-3091 |
| `Spalart-Allmaras, DES97/DDES/IDDES (SPEC-LIT 56, 57, 58)` | 닫힌 형식+공개 수치: 원역 ν_t/ν의 TMR 표(6자리), DES 길이척도 항등식, 실험 LIVE; TMR 평판(§56.11)·주기 언덕 Fröhlich 등 (2005)(§57.12)은 **실행하지 않음** | NASA/TMBWG Turbulence Modeling Resource SA 페이지(정부 문서, 자리수 그대로); Allmaras, Johnson & Spalart (2012) ICCFD7-1902; Spalart & Allmaras (1994); DES97 Spalart 등 (1997); DDES Spalart 등 (2006) *TCFD* 20, 181; IDDES는 Shur 등 (2008)(유료, 미열람) 대신 공개 재서술 Herr 등 (2023) arXiv:2301.07223, Savino 등 (2026) arXiv:2603.08875 | TMR SA: <https://tmbwg.github.io/turbmodels/spalart.html>; Allmaras 등: <https://www.iccfd.org/iccfd7/assets/pdf/papers/ICCFD7-1902_paper.pdf>; Spalart 등 2006: DOI 10.1007/s00162-006-0015-0; Herr 등: <https://arxiv.org/abs/2301.07223>; Savino 등: <https://arxiv.org/abs/2603.08875> |
| `gamma-Re_theta transition (SPEC-LIT 88, 89)`, `the 2015 gamma transition model (SPEC-LIT 90)` | 닫힌 형식+해석해: 상관식의 TMR 전개형 vs 논문 중첩형(1e-12), Blasius 운동량 두께 ∫f'(1−f') = 0.664, max(Re_V)/Re_θ = 2.193, λ_B 기준값 −0.003434; Gate 88-T/90-T는 T3A 자유류 감쇠(Tu 3.3 %)를 재현하고 천이 위치는 **OPEN** | Langtry & Menter (2009) *AIAA J.* 47, 2894; Menter, Smirnov, Liu & Avancha (2015) *FTC* 95, 583(유료, 미열람 — 모든 숫자는 TMR 페이지에서); NASA/TMBWG TMR의 Menter-γ-2015 페이지; Blasius 해는 이 저장소가 직접 적분 | Langtry & Menter: DOI 10.2514/1.42362; Menter 등 2015: DOI 10.1007/s10494-015-9622-4; TMR γ-2015: <https://tmbwg.github.io/turbmodels/menter_gamma_3eqn.html>; TMR 목록: <https://tmbwg.github.io/turbmodels/> |
| `Lagrangian parcels (SPEC-LIT 66)` | 해석해+결정성: 66-A 종단속도를 해석 힘 균형에 네 시간간격으로, 66-B 탄도 파셀의 셀·직선 위치, 66-C 두 실행과 CUDA 그래프 재생의 비트 동일 | 운동 방정식: Maxey & Riley (1983), Crowe, Sommerfeld & Tsuji (1998); 항력: Schiller & Naumann (1933, Clift·Grace·Weber 1978 편찬); 파셀 모델: Dukowicz (1980) | Maxey & Riley: DOI 10.1063/1.864230; Dukowicz: DOI 10.1016/0021-9991(80)90087-X; Clift 등: Academic Press, ISBN 0-12-176950-X (DOI 없음) |
| `the parcel sort and gather-shaped deposition (SPEC-LIT 67)` | 결정성: 정렬·셀별 CSR 정준화가 결과를 바꾸지 않고 재생이 비트 동일 | PSI-CELL: Crowe, Sharma & Stock (1977); 정렬·스캔 알고리즘: Satish, Harris & Garland (2009), Merrill & Grimshaw (2009), Blelloch (1990), Hillis & Steele (1986) — 논문만 읽고 구현체는 열지 않음 | Crowe 등: DOI 10.1115/1.3448756; Satish 등: DOI 10.1109/IPDPS.2009.5161005; Hillis & Steele: DOI 10.1145/7902.7903; Elghobashi (1994) 결합 지도: DOI 10.1007/BF00936835 |
| `two-way coupling of the dispersed phase (SPEC-LIT 68)` | 항등식+실험: 68-A 파셀이 가져간 운동량·에너지 = 기체가 받은 것(반올림), 68-B 파셀 없음 = 비트 불변, 68-C Theobald (1981) 소방 수류 약 90회 **MISSES**(정지 기체에서 측정 거리의 61.29 %, 무항력 괄호 198.65 %) | Theobald (1981) *Fire Safety J.* 4, 1–13(노즐 설계와 수류 도달거리 실험); Ranz & Marshall (1952) Nu 상관식의 현열 반쪽 | Theobald: *Fire Safety J.* 4 (1981) 1 (README 서지에 DOI 없음); Ranz & Marshall: *Chem. Eng. Prog.* 48, 141·173 (DOI 없음) |
| `droplet heating and evaporation (SPEC-LIT 76)` | 해석해+상관식: d² 법칙(닫힌 형식), 정착 온도를 자체 균형(반올림)과 ASHRAE 습구에, 파셀 질량 보존 | Spalding (1953, 1963) B_M과 Stefan 유동; Godsave (1953); Abramzon & Sirignano (1989) B_T; Ranz & Marshall (1952); 물성: Watson (1943), Marrero & Mason (1972), NIST Chemistry WebBook SRD 69; 리뷰 Sazhin (2006) | Abramzon & Sirignano: DOI 10.1016/0017-9310(89)90043-4; Marrero & Mason: DOI 10.1063/1.3253094; NIST WebBook: <https://webbook.nist.gov/chemistry/>; Sazhin: DOI 10.1016/j.pecs.2005.11.001 |
| `the vapour into the gas (SPEC-LIT 77)` | 항등식+상관식: 77-A 잃은 질량 = 받은 질량, 77-B 상변화 에너지 원장, 77-C 발산의 두 반쪽, 77-D ASHRAE 단열포화온도(밀폐 단열 상자에 살수) | ASHRAE Handbook—Fundamentals (2021) 1장의 습공기 관계식(단열포화·습구); Lewis (1922) 관계; Rehm & Baum (1978) 발산 항 | ASHRAE Handbook: <https://www.ashrae.org/technical-resources/ashrae-handbook> (유료); Lewis: *Trans. ASME* 44, 325 (DOI 없음) |
| `droplet-wall impact (SPEC-LIT 78)` | 항등식+공개 기준: 78-A 영역 경계를 발표 기준의 닫힌 역함수와 10⁻¹² 양쪽에서, 78-B 벽에 닿은 질량의 비트 원장, 78-C 벽을 안 만나면 비트 불변, 78-D 두 발표 스플래시 기준의 상호 불일치(We 4.78배) — **OPEN** | Mundo, Sommerfeld & Tropea (1995) *IJMF* 21, 151(K = Oh·Re^1.25, K_crit = 57.7, 기본값 — 실험 데이터 자체는 전사하지 않음); Bai & Gosman (1995) SAE 950283(대안 문턱); Yarin (2006) 리뷰; IAPWS R1-76 표면장력 | Mundo 등: DOI 10.1016/0301-9322(94)00069-V; Bai & Gosman: DOI 10.4271/950283; Yarin: DOI 10.1146/annurev.fluid.38.050304.092144; IAPWS R1-76: <http://www.iapws.org/> |
| §38.9 비뉴턴 채널 (Gate 1 Herschel–Bulkley 평면 Poiseuille LIVE, Gate 2 Buckingham–Reiner) | 해석해: 두 격자·네 멱지수에서 §38.9가 유도한 닫힌 프로파일; Buckingham–Reiner = Bingham 프로파일의 적분(계수 1, −4/3, +1/3을 수치 적분으로 확인); Papanastasiou m ↑ 에서 단조 접근 | Herschel & Bulkley (1926) *Kolloid-Z.* 39, 291; Papanastasiou (1987) *J. Rheol.* 31, 385(정규화); Buckingham–Reiner: Chhabra & Richardson (2008) *Non-Newtonian Flow and Applied Rheology* 2판 | Herschel & Bulkley: DOI 10.1007/BF01432034; Papanastasiou: DOI 10.1122/1.549926; Chhabra & Richardson: Butterworth-Heinemann, ISBN 978-0-7506-8532-0 (DOI 없음) |
| §39.7 접촉각 (Jurin, Hoffman, Tanner, Šikalo) | 해석해: Jurin 높이 h = 2σcosθ/(ρgR)의 부호(θ > 90°는 하강, 90°는 0), Hoffman 곡선을 Jiang–Oh–Slattery 상관식으로, Tanner R ∼ t^(1/10), Šikalo 등의 동접촉각 | Jurin (1718) 모세관 상승 / Washburn (1921) *Phys. Rev.* 17, 273; Hoffman (1975) *JCIS* 50, 228; Jiang, Oh & Slattery (1979) *JCIS* 69, 74; Tanner (1979) *J. Phys. D* 12, 1473; Šikalo 등 (2005) *Phys. Fluids* 17, 062103 | Washburn: DOI 10.1103/PhysRev.17.273; Hoffman: DOI 10.1016/0021-9797(75)90225-8; Jiang 등: DOI 10.1016/0021-9797(79)90081-X; Tanner: DOI 10.1088/0022-3727/12/9/009; Šikalo 등: DOI 10.1063/1.1928828 |
| `cargo test --release --bin ofgpu-validate -- --ignored` — Ghia 뚜껑 구동 공동 Re 100/400 | 공개 벤치마크(분 단위라 `#[ignore]`): 공동 중심선의 u, v 17점을 Ghia, Ghia & Shin (1982) 표 I·II에 2 % 안(Re 400은 논문의 정오표 반영) | Ghia, Ghia & Shin (1982) *J. Comput. Phys.* 48, 387 — 표 I(u), II(v) | Ghia 등: DOI 10.1016/0021-9991(82)90058-4 |

**처음 빗나간 게이트와 열린 게이트의 근거.** 2026-09-09 실행이 마지막에 이름으로 부른 여덟 개가 무엇에 견준 것인지입니다. 2026-10-20 전체 실행이 출력한 목록 전부는 이 절의 끝에, 그 기록에서 생성해 실었습니다.

| 게이트 | 판정 | 견준 자료 | 접속 주소 |
|---|---|---|---|
| §60.5 Gate 5 — Kaminski & Prakash (1986) 켤레 자연대류 | MISSES (Kr = 0.1에서 −7.11 %) | 1차 문헌은 유료라 읽지 못했고 비교는 2차 출처 Belazizia 등 (2012)의 표 | Kaminski & Prakash: DOI 10.1016/0017-9310(86)90017-7; Belazizia 등: DOI 없음 |
| §68.12 Gate 68-C — Theobald (1981) 소방 수류 | MISSES (61.29 %) | Theobald (1981) *Fire Safety J.* 4, 1–13의 약 90회 수류 실험 | DOI 없음(서지사항) |
| §32.4 verdict 1·2 — 평판 채널 Nu, 세 다리 | OPEN | Gnielinski (1976) 상관식 ±10 %(측정값이 아니라 상관식에 견줌), Petukhov (1970) f; 기록 `docs/07-lowmach-solver.md` §1.1 | Petukhov: DOI 10.1016/S0065-2717(08)70153-9; Gnielinski: DOI 없음 |
| §88 Gate 88-T, §90 Gate 90-T — 천이 위치 | OPEN | T3A 평판의 측정된 천이 개시 Re_x를 구하지 못해 비교를 닫지 못함; 자유류 감쇠는 TMR의 Tu 3.3 %에 맞춤 | <https://tmbwg.github.io/turbmodels/> |
| 78-D — 두 스플래시 기준의 불일치 | OPEN | Mundo 등 (1995)와 Bai & Gosman (1995)의 문턱이 같은 액적에서 We 4.78배 서로 어긋남 | DOI 10.1016/0301-9322(94)00069-V; DOI 10.4271/950283 |

이 표는 2026-09-09의 `ofgpu-validate` 출력(833/833, 37개 절; 2026-10-20 전체 실행은 95개 절이며 아래 표에 있습니다)과
[`README.md`](../README.md) 「참고문헌」·[`rust/SPEC-LIT.md`](../rust/SPEC-LIT.md)의
인용으로 만들었습니다. 유료라 읽지 못한 출처(Kaminski & Prakash, Menter 등 2015, Shur 등
2008)와 2차 출처로 대신한 곳은 표에 그대로 적었습니다.

### 전체 실행, 절마다

전체 `ofgpu-validate` 실행 한 번이 빌드마다 출력한 것이며, `python tools/validate_record.py`가 `docs/validate-run/`의 기록에서 생성합니다(실행 방법은 SPEC-LIT §110.5). 굵은 칸은 실패하는 행이 있는 절, NOT RUN은 실행이 예산 안에 끝내지 못한 절, 대시는 행이 없는 절입니다(이 트리는 Gate 105-C의 격자를 만들지 않으며, `reference/mkm99/`가 없는 체크아웃에서 Gate 110-A는 답안 키가 없습니다).

<!-- validate-run:status:ko begin -->
```
ofgpu-validate (f64)   1634 / 1708 checks passed in 95 sections (2026-10-20, 필터 없는 실행 1회, 3,533 s; SPEC-LIT §110.5)
                       section 95개 중 77개 통과, 16개 미달, 행 없음 2개, NOT RUN 0개
                       이어서 판정이 MISSES로 등록된 게이트 35개와 OPEN 37개 (아래 표)
ofgpu-validate (f32)   1431 / 1655 checks passed in 95 sections (2026-10-20, 실행 5회, 5,087 s; SPEC-LIT §110.5)
                       section 95개 중 49개 통과, 43개 미달, 행 없음 2개, NOT RUN 1개
                       이어서 판정이 MISSES로 등록된 게이트 42개와 OPEN 37개 (아래 표)
```
<!-- validate-run:status:ko end -->

<!-- validate-run:sections:ko begin -->
| # | 절 | f64 행 | f64 s | f32 행 | f32 s |
|---|---|---|---|---|---|
| 1 | 3-D graded block | 71/71 | 0.0 | 71/71 | 0.0 |
| 2 | 3-D sheared block (non-orthogonal) | 36/36 | 0.0 | 36/36 | 0.0 |
| 3 | 3-D block with 2:1 refinement interfaces | 35/35 | 0.0 | 35/35 | 0.0 |
| 4 | the adapt: refine, coarsen, and what a rebuild costs | 34/34 | 1.1 | 34/34 | 1.1 |
| 5 | 2-D block with empty front and back | 37/37 | 0.0 | 37/37 | 0.0 |
| 6 | linear solvers | 6/6 | 0.2 | **10/12** | 1.8 |
| 7 | method of manufactured solutions, -lap(psi) = f | 6/6 | 0.1 | 6/6 | 0.1 |
| 8 | observed order and reported uncertainty (SPEC-LIT 94) | 10/10 | 0.3 | 10/10 | 0.4 |
| 9 | buoyancy | 12/12 | 0.0 | **8/12** | 0.0 |
| 10 | buoyancy production, sources, species, phi I/O | 14/14 | 0.1 | **13/14** | 0.1 |
| 11 | volume of fluid (SPEC-LIT 20, the 22 rows) | 16/16 | 2.5 | **13/16** | 13.6 |
| 12 | msh hex closure, cut-cell closure (SPEC-LIT 23, 24) | 10/10 | 0.0 | 10/10 | 0.0 |
| 13 | the low-Mach reference pressure (SPEC-LIT 25) | 3/3 | 0.0 | 3/3 | 0.0 |
| 14 | wall treatment: Ks -> 0, the thermal wall function (SPEC-LIT 29) | 5/5 | 0.0 | 5/5 | 0.0 |
| 15 | Werner-Wengle, coupled-solver turbulence selection (SPEC-LIT 30) | 9/9 | 0.1 | 9/9 | 0.2 |
| 16 | periodic domains: cyclic-pair invariants (SPEC-LIT 31.1) | 4/4 | 0.0 | 4/4 | 0.0 |
| 17 | the thermal wall-function gate, redesigned (SPEC-LIT 32) | 27/27 | 0.0 | **25/27** | 0.0 |
| 18 | Launder-Sharma low-Re k-epsilon: damping functions (SPEC-LIT 33.3) | 10/10 | 0.0 | 10/10 | 0.0 |
| 19 | resolved leg mesh resolution, replayed (SPEC-LIT 33.2/34) | 3/3 | 0.0 | 3/3 | 0.0 |
| 20 | the bulk-temperature thermostat (SPEC-LIT 35) | 17/17 | 0.0 | 17/17 | 0.0 |
| 21 | thermostat weighting: the decisive experiment, replayed (SPEC-LIT 35.3.2) | 5/5 | 0.0 | 5/5 | 0.0 |
| 22 | bounded convection on momentum: the isolation, replayed (SPEC-LIT 3.1/32.5.5) | 10/10 | 0.0 | 10/10 | 0.0 |
| 23 | Kays-Crawford turbulent Prandtl number (SPEC-LIT 37.1/37.2) | 10/10 | 0.0 | **6/10** | 0.0 |
| 24 | realizable and RNG k-epsilon (SPEC-LIT 40, 41) | 39/39 | 6.0 | **37/39** | 19.7 |
| 25 | the output block, and fp16 voxels (SPEC-LIT 44, 45) | 22/22 | 0.0 | 22/22 | 0.0 |
| 26 | conjugate heat transfer (SPEC-LIT 46, 47, 48) | 16/16 | 0.5 | **15/16** | 0.5 |
| 27 | the per-region residual - Gate 93-B (SPEC-LIT 8.4 on each region's rows) | 6/6 | 0.0 | **3/6** | 0.1 |
| 28 | an equation that lives on a region - Gate 93-A (SPEC-LIT 93) | 8/8 | 0.0 | 8/8 | 0.0 |
| 29 | the conjugate fluid/solid interface (SPEC-LIT 59, 60) | 48/48 | 182.9 | **21/48** | 786.7 |
| 30 | surface-to-surface radiation (SPEC-LIT 49, 50, 51) | 47/47 | 2.3 | **45/47** | 0.7 |
| 31 | fan curves, porous jumps, psychrometrics, metrics (SPEC-LIT 52, 53, 54, 55) | 83/83 | 0.2 | **81/83** | 0.3 |
| 32 | Spalart-Allmaras, DES97/DDES/IDDES (SPEC-LIT 56, 57, 58) | 43/43 | 0.2 | **39/43** | 0.3 |
| 33 | gamma-Re_theta transition (SPEC-LIT 88, 89) | 20/20 | 0.0 | **19/20** | 0.0 |
| 34 | the 2015 gamma transition model (SPEC-LIT 90) | 24/24 | 0.1 | **21/24** | 0.1 |
| 35 | Lagrangian parcels (SPEC-LIT 66) | 11/11 | 0.2 | 11/11 | 0.1 |
| 36 | the parcel sort and gather-shaped deposition (SPEC-LIT 67) | 12/12 | 0.0 | 12/12 | 0.0 |
| 37 | two-way coupling of the dispersed phase (SPEC-LIT 68) | 13/13 | 2.7 | 13/13 | 2.6 |
| 38 | droplet heating and evaporation (SPEC-LIT 76) | 12/12 | 1.5 | **9/12** | 1.1 |
| 39 | the vapour into the gas (SPEC-LIT 77) | 14/14 | 1.4 | **13/14** | 1.4 |
| 40 | droplet-wall impact (SPEC-LIT 78) | 45/45 | 1.7 | **42/45** | 11.3 |
| 41 | Gate 95-D: the thick cylinder heated through the conduction solver (three meshes, SPEC-LIT 95.10) | 6/6 | 3.9 | **3/6** | 81.1 |
| 42 | Gate 95-E: the bimetal strip, two materials bonded in one region (three meshes, two bond treatments) | 3/3 | 163.7 | 3/3 | 442.3 |
| 43 | Gate 95-A: the end-loaded cantilever, block-coupled, at 2.5:1, 5:1 and 10:1 (four meshes each, the tip study on the finest three, SPEC-LIT 109.6) | **36/42** | 86.5 | **4/42** | 335.2 |
| 44 | Gate 95-G: the boundary-point fit on the Lame ring, and NAFEMS LE1, LE10 and LE11 from a restatement (LE1 and LE11 three meshes, the ring and LE10 four with their study on the finest three, SPEC-LIT 95.11) | 24/24 | 50.4 | NOT RUN | - |
| 45 | the imported region (SPEC-LIT 97) | 5/5 | 0.1 | 5/5 | 0.1 |
| 46 | the region layout (SPEC-LIT 97) | 3/3 | 0.0 | 3/3 | 0.1 |
| 47 | Gate 98-A: the straight fin, three meshes, against (S98.7) (SPEC-LIT 98.6) | 1/1 | 0.1 | 1/1 | 0.1 |
| 48 | Gate 98-B: the slab radiating to a surround, its Newton passes (SPEC-LIT 98.6) | 4/4 | 0.0 | **3/4** | 0.0 |
| 49 | Gate 98-C: an enclosure on a live conjugate flow, its balance, split and relaxation (SPEC-LIT 98.9) | 5/5 | 12.0 | **2/5** | 183.9 |
| 50 | Gate 100-B: silicon, air and the seven-term cp/R against their tables (SPEC-LIT 100.4) | 5/5 | 0.0 | **4/5** | 0.0 |
| 51 | Gate 100-A: the Kirchhoff slab, kappa linear in T, three meshes (SPEC-LIT 100.8) | 4/4 | 0.2 | **0/1** | 0.3 |
| 52 | Gate 100-C: Brinkman's plane Poiseuille with viscous heating, two walls, three meshes each (SPEC-LIT 100.15) | 16/16 | 27.4 | **5/16** | 850.7 |
| 53 | Gate 100-D: Qu & Mudawar's micro-channel with water's mu(T) live (SPEC-LIT 100.15) | 11/11 | 91.6 | **0/1** | 0.3 |
| 54 | Gate 105-A: space conservation on a moving mesh, euler and backward, 100 steps (SPEC-LIT 105.5) | 10/10 | 0.1 | 10/10 | 0.1 |
| 55 | Gate 105-B: the piston and the stroking outlet, euler and backward (SPEC-LIT 105.10) | 14/14 | 20.4 | **8/14** | 41.1 |
| 56 | Gate 105-C: Turek-Hron CFD1/CFD2 drag and lift over four meshes (the study on the finest three), CFD3 on a wobbling mesh (SPEC-LIT 105.14) | - | 0.0 | - | 0.0 |
| 57 | lid-driven cavity, Ghia, Ghia & Shin (1982), three meshes per Re (SPEC-LIT 94.4 Gate 94-D) | 8/8 | 388.7 | 8/8 | 462.8 |
| 58 | Gate 110-A: channel DNS, Moser-Kim-Mansour 1999, Re_tau 180/395/590 (SPEC-LIT 110.2) | - | 0.0 | - | 0.0 |
| 59 | Gate 110-B: backward-facing step, Driver & Seegmiller 1985 (SPEC-LIT 110.3) | 1/1 | 0.0 | 1/1 | 0.0 |
| 60 | Gate 110-C: buoyant plume, McCaffrey 1979 (SPEC-LIT 110.4) | 17/17 | 0.0 | 17/17 | 0.0 |
| 61 | Gate CMP-B: de Vahl Davis with Boussinesq at beta*dT = 0.03, centro-symmetry, and the density-ratio control (SPEC-LIT 121.8) | 13/13 | 226.8 | **4/10** | 374.6 |
| 62 | Gate 119-A: the wedge - Hagen-Poiseuille through the axis prisms and radial conduction in an annulus (SPEC-LIT 119.3) | 5/5 | 54.4 | 5/5 | 216.3 |
| 63 | Gate 145-A: the dam break against Martin & Moyce 1952 Table 2, three meshes (SPEC-LIT 145.2) | 9/9 | 79.8 | 9/9 | 84.5 |
| 64 | Gate 78-D measured leg: the two splash criteria against Riboux & Gordillo 2014's threshold speeds, eight liquids (SPEC-LIT 146.2) | 2/2 | 0.0 | 2/2 | 0.0 |
| 65 | Gate 132-L: the laminar flat plate against Blasius, three meshes (SPEC-LIT 144.2) | 7/7 | 52.7 | 7/7 | 24.2 |
| 66 | T3A leg 1 in 2-D: the leading-edge Tu of kOmegaSSTLM and kOmegaSSTGamma on the flat-plate harness, three meshes, replayed (SPEC-LIT 144.3) | 6/6 | 0.0 | 6/6 | 0.0 |
| 67 | Gates 88-T and 90-T leg 4: the 2-D C_f(Re_x) of kOmegaSSTLM and kOmegaSSTGamma against Roach & Brierley's T3A measurement, replayed (SPEC-LIT 144.4) | 4/4 | 0.0 | 4/4 | 0.0 |
| 68 | Gate 132-Z: the TMR zero-pressure-gradient plate, SA and SST against Karman-Schoenherr and Coles, three meshes, replayed (SPEC-LIT 144.5) | 5/5 | 0.0 | 5/5 | 0.0 |
| 69 | Gate 131-P: TG0's turbulent pipe, kOmegaSST on the true wedge against Prandtl, Blasius, the log law and the core defect, three radial levels, replayed (SPEC-LIT 143.5) | 13/13 | 0.0 | 13/13 | 0.0 |
| 70 | Gate CMP-SOD: Sod's shock tube against the exact Riemann solution, three meshes (SPEC-LIT 127.6) | **3/9** | 34.5 | **3/9** | 40.5 |
| 71 | Gate CMP-AC: a small acoustic pulse travels at c_0 = sqrt(gamma R_s T_0), 400 cells (SPEC-LIT 127.6) | **1/2** | 21.7 | **1/2** | 27.8 |
| 72 | Gate CMP-LM: the lid-driven cavity at Re 100, lid M 0.05, against Ghia (1982) (SPEC-LIT 127.6) | 3/3 | 27.8 | 3/3 | 27.0 |
| 73 | Gate CMP-COU: compressible Couette with viscous heating, adiabatic-wall rise and T profile (SPEC-LIT 127.6) | 3/3 | 63.4 | 3/3 | 56.5 |
| 74 | Gate CMP-NOZ: the quasi-1-D nozzle, subsonic, supersonic started and shocked, 200 cells (SPEC-LIT 127.6) | **1/7** | 89.5 | **1/7** | 101.5 |
| 75 | Gate CMP-OBL: the oblique shock on a 10-degree ramp at M 2, 90 x 60 cells (SPEC-LIT 127.6) | 3/3 | 216.2 | 3/3 | 217.4 |
| 76 | Gate CMP-RING: Ringleb's transonic flow on three meshes, h, h/2, h/4 (SPEC-LIT 127.6) | **3/4** | 128.1 | 3/3 | 129.8 |
| 77 | Gate CMP-RAE: RAE 2822 Case 6 with Spalart-Allmaras against AR-138, a live coarse smoke and the three recorded C-meshes, replayed (SPEC-LIT 127.6) | **2/13** | 48.4 | **0/13** | 0.7 |
| 78 | Gate 116-P: Hagen-Poiseuille on tetrahedra and prisms, the L1 pipe at Re 100, replayed (SPEC-LIT 116.17) | **2/5** | 0.0 | **2/5** | 0.0 |
| 79 | Gate 88-T and 90-T leg 3: T3B and T3A- with a per-case inflow fitted to ERCOFTAC's measured Tu(x), onset ordering and spread, replayed (SPEC-LIT 144.7) | 14/14 | 0.0 | 14/14 | 0.0 |
| 80 | Gate DST-LEEDS: the Leeds fan-stirred cornflour bomb, S_t,b against Bradley, Chen & Swithenbank 1989 via Huang et al. 2020 Fig. 9 (SPEC-LIT 157.5) | **18/23** | 960.7 | **2/23** | 20.6 |
| 81 | Gate DST-VES: closed vessels, thin-flame spheres against (157.4) and the REMBE closed stage against Huang et al. 2022 Fig. 10 (SPEC-LIT 157.6) | **7/21** | 107.1 | **0/9** | 61.7 |
| 82 | Gate CHR-13: a 1-D freely propagating CH4-air flame, BFER phi 1.0, S_L by the 1500 K isotherm, three meshes, replayed (SPEC-LIT 133.12) | **0/3** | 0.0 | **0/3** | 0.0 |
| 83 | Gate CHR-16: Sandia Flame D, EDC + DRM19, k-epsilon C1 1.6, REDUCED on the true wedge, replayed (SPEC-LIT 134.8) | **1/7** | 0.0 | **1/7** | 0.0 |
| 84 | Gate 110-C leg 2: the radiating plume, P1 kappa 1.0 1/m, McCaffrey 1979, replayed (SPEC-LIT 110.4) | 2/2 | 0.0 | 2/2 | 0.0 |
| 85 | Gate CHR-THERMO: NASA-7 thermochemistry of the ten TM-4513 records against TP-2002 Table B1 (SPEC-LIT 128.7) | 111/111 | 0.0 | 111/111 | 0.0 |
| 86 | Gate CHR-EQ: gas-phase equilibrium by Gibbs minimisation on the TM-4513 fixture against the 40-digit oracle (SPEC-LIT 131.4) | 94/94 | 0.0 | 94/94 | 0.0 |
| 87 | Gate CHR-ODE: RODAS3 and ROS3 on Robertson and HIRES, and the constant-pressure reactor (SPEC-LIT 132.6, 132.8) | 48/48 | 3.7 | 48/48 | 3.6 |
| 88 | Gate CHR-EDM: EDM and PaSR, the library and kernel gate of CHR-14, not a driver run (SPEC-LIT 134.5) | 33/33 | 0.0 | 33/33 | 0.0 |
| 89 | Gate CHR-SLAB: the isothermal grey slab, P1 against its closed form and S8 fvDOM against 1 - 2 E_3 (SPEC-LIT 135.5, 136.7) | **61/63** | 72.8 | **38/41** | 130.6 |
| 90 | Gate CHR-CUBE: fvDOM with scattering on the slab, and the isothermal cube at 24^3 (SPEC-LIT 136.8) | **53/54** | 49.2 | **53/54** | 44.3 |
| 91 | Gate CHR-WSGG: the weighted sum of grey gases on the slab, per-gas fvDOM and P1, and the grey mean (SPEC-LIT 136.9) | 33/33 | 15.8 | 33/33 | 20.9 |
| 92 | Gate DST-B-LAM: the Phase B cloud's 1-D laminar flame against the gas flame of its volatile, the fine-particle limit and the size order (SPEC-LIT 159.11) | **15/16** | 195.0 | **13/16** | 221.1 |
| 93 | Gate DST-B-VES: the Phase B cloud in a closed 20 L sphere, replayed from its records - the AICC end pressure and the thin-flame half time (SPEC-LIT 159.12) | **11/18** | 0.0 | **11/18** | 0.0 |
| 94 | Gate 68-C coupled leg: Theobald test 3 through ofgpu-lowmach with Rosin-Rammler sizes, replayed from its records (SPEC-LIT 147.2) | **9/10** | 0.0 | **9/10** | 0.0 |
| 95 | Gate DST-REMBE: the vented REMBE explosion, key J.3 NOT RUN, and a REDUCED smoke on the PLACEHOLDER vessel (SPEC-LIT 158.7, 158.10) | 9/9 | 35.8 | **7/9** | 42.7 |
<!-- validate-run:sections:ko end -->

실행이 등록한 게이트이며, MISSES가 먼저, OPEN이 다음이고 각 묶음 안에서는 실행이 이름을 부른 순서입니다(같은 이름이 되풀이되면 다리마다 한 번씩 등록된 것입니다):

<!-- validate-run:gates:ko begin -->
| # | 게이트 | f64 | f32 | 비교 대상: 판정 |
|---|---|---|---|---|
| 1 | SPEC-LIT S60.5 Gate 5 | MISSES | MISSES | Kaminski & Prakash (1986), conjugate natural convection in a square enclosure - through the open-access SECONDARY table of Belazizia et al. (2012), the primary being paywalled and never read - bar 3 % at every conductivity ratio: worst disagreement 7.11 %, and it is at the SMALLEST conductivity ratio - the CONDUCTION-DOMINATED end - shrinking to 0.07 % at Kr = 10; reference-limited at Kr <= 1, SPEC-LIT 115 (O1, D-VAL-4) |
| 2 | SPEC-LIT S68.12 Gate 68-C | MISSES | MISSES | Theobald (1981), 90 hose streams, maximum throw, at the shape of the FDS Validation Guide's own metric for this quantity - +-10 % bias, 30 % scatter: with the gas at rest this solver throws the stream 61.2858 % of the measured distance on average |
| 3 | Gate 95-A cantilever | MISSES | MISSES | Timoshenko & Goodier ch. 3, plane strain, block-coupled (SPEC-LIT 109.6), four meshes r = 2, orders on the finest pair, the tip study on the finest three: 5:1: converged 3/4, p_u 1.42, p_sigma 1.46 (n_y 16 -> 32), finest tip 9.76e-5 |
| 4 | Gate 95-A cantilever | MISSES | MISSES | Timoshenko & Goodier ch. 3, plane strain, block-coupled (SPEC-LIT 109.6), four meshes r = 2, orders on the finest pair, the tip study on the finest three: 10:1: converged 3/4, p_u 1.90, p_sigma 1.45 (n_y 16 -> 32), finest tip 1.52e-4 |
| 5 | Gate 95-A cantilever | MISSES | MISSES | Timoshenko & Goodier ch. 3, plane strain, block-coupled (SPEC-LIT 109.6), four meshes r = 2, orders on the finest pair, the tip study on the finest three: 5:1 GMRES(50): converged 4/4, p_u 1.42, p_sigma 1.46 (n_y 16 -> 32), finest tip 9.83e-5 |
| 6 | Gate 95-A cantilever | MISSES | MISSES | Timoshenko & Goodier ch. 3, plane strain, block-coupled (SPEC-LIT 109.6), four meshes r = 2, orders on the finest pair, the tip study on the finest three: 10:1 GMRES(50): converged 4/4, p_u 1.75, p_sigma 1.45 (n_y 16 -> 32), finest tip 1.67e-4 |
| 7 | SPEC-LIT S110.3 Gate 110-B backward-facing step reattachment (Driver & Seegmiller 1985) | MISSES | MISSES | Driver & Seegmiller (1985) x_r/H = 6.26 +- 0.10, reference/driver_seegmiller_1985/reattachment.txt: study not formed: the dx = 0.1H level stopped - run ended: refused \| ofgpu-lowmach: max Mach number 3.615 at cell 56 (step 1232); the dx = 0.05H level stopped - run ended: refused \| ofgpu-lowmach: max Mach number 1.920 at cell 917604 (step 30); on the finest level that converged, dx = 0.2H, x_r/H = 5.637 against 6.26, E = -0.623 (-9.9% of D), outside the datum's own +-0.10 at that level, with no u_num |
| 8 | SPEC-LIT S110.3 Gate 110-B backward-facing step reattachment (Driver & Seegmiller 1985) | MISSES | MISSES | Driver & Seegmiller (1985) x_r/H = 6.26 +- 0.10, reference/driver_seegmiller_1985/reattachment.txt: class R: the single build's dx = 0.2H run left no record - run ended: refused \| ofgpu-lowmach: max Mach number 10255882.000 at cell 57149 (step 0) |
| 9 | SPEC-LIT 146.2 Gate 78-D measured leg, mundo | MISSES | MISSES | Riboux & Gordillo, Phys. Rev. Lett. 113 (2014) 024507 (arXiv:1401.6943v3): the splash threshold speeds of Fig. 4(c), eight liquids on smooth dry glass in air, band +-20 % per row: 0 of 8 liquids inside the band (146.6), 6 needed; 0 of 17 rows inside (146.5), e from -0.734 to -0.214 |
| 10 | SPEC-LIT 146.2 Gate 78-D measured leg, baiGosman | MISSES | MISSES | Riboux & Gordillo, Phys. Rev. Lett. 113 (2014) 024507 (arXiv:1401.6943v3): the splash threshold speeds of Fig. 4(c), eight liquids on smooth dry glass in air, band +-20 % per row: 1 of 8 liquids inside the band (146.6), 6 needed; 3 of 17 rows inside (146.5), e from -0.396 to +0.690 |
| 11 | SPEC-LIT 144.4 Gate 88-T leg 4, kOmegaSSTLM, transition onset Re_x against Roach & Brierley | MISSES | MISSES | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3A key (SPEC-LIT 144.6): kOmegaSSTLM onset Re_x (least C_f) against (144.4a): E = -35.88 % at Re_x 8.8932e4 on the finest mesh, band 15 %, u_val 6.53 % (u_num 6.53 %, u_D 0 %) |
| 12 | SPEC-LIT 144.4 Gate 88-T leg 4, kOmegaSSTLM, laminar C_f against Blasius | MISSES | MISSES | Blasius' 0.664/sqrt(Re_x) (SPEC-LIT 88.7) over the part of the plate the key measures laminar: kOmegaSSTLM worst laminar C_f against Blasius (144.4c): E = +16.29 % at Re_x 7.5000e4 on the finest mesh, band 10 %, u_val 0.62 % (u_num 0.62 %, u_D 0 %) |
| 13 | SPEC-LIT 144.4 Gate 90-T leg 4, kOmegaSSTGamma, laminar C_f against Blasius | MISSES | MISSES | Blasius' 0.664/sqrt(Re_x) (SPEC-LIT 88.7) over the part of the plate the key measures laminar: kOmegaSSTGamma worst laminar C_f against Blasius (144.4c): E = +15.78 % at Re_x 1.0000e5 on the finest mesh, band 10 %, u_val 1.46 % (u_num 1.46 %, u_D 0 %) |
| 14 | SPEC-LIT 143.5 Gate 131-P turbulent pipe, kOmegaSST, Re_tau 576.69, the core defect | MISSES | MISSES | Nikuradse's core defect (U_axis - U_b)/u_tau = 4.07 (Schlichting NACA TM 1218 eq 16.22): kOmegaSST at Re_tau 576.69, defect against 4.07: E = -11.21 % on the finest level (n = 160), bar 10 %, u_val 0.01 % |
| 15 | SPEC-LIT 143.5 Gate 131-P turbulent pipe, kOmegaSST, Re_tau 2358, the log law | MISSES | MISSES | the log law u+ = 2.5 ln y+ + 5.5 over 30 <= y+ <= 0.2 Re_tau (Schlichting NACA TM 1218 eq 16.14): kOmegaSST at Re_tau 2358, log-law deviation: E = 1.230 in u+ on the finest level (n = 160), bar 1.0 in u+, u_val 0.173 |
| 16 | SPEC-LIT 143.5 Gate 131-P turbulent pipe, kOmegaSST, Re_tau 2358, the core defect | MISSES | MISSES | Nikuradse's core defect (U_axis - U_b)/u_tau = 4.07 (Schlichting NACA TM 1218 eq 16.22): kOmegaSST at Re_tau 2358, defect against 4.07: E = -18.39 % on the finest level (n = 160), bar 10 %, u_val 0.04 % |
| 17 | SPEC-LIT 127.6 Gate CMP-SOD Sod shock tube | MISSES | MISSES | the exact Riemann solution of Sod's (1978) shock tube, (127.5)-(127.7b), evaluated live (CMP-07): PISO, 800 cells: worst plateau rho*_L 45.98 % (bar 1 %), shock not found inside the tube (bar 0.5 % of the length); L1(rho) order 0.015 (bar >= 0.6) |
| 18 | SPEC-LIT 127.6 Gate CMP-AC acoustic pulse speed | MISSES | MISSES | linear acoustics: c_0 = sqrt(gamma R_s T_0), (122.4): the solver's default coupling (nOuterCorrectors 1, nCorrectors 2), 400 cells: the pulse travels at 1.1895 c_0 (bar: within 1 %) |
| 19 | SPEC-LIT 127.6 Gate CMP-NOZ quasi-1-D nozzle | MISSES | MISSES | the quasi-1-D keys (127.23)-(127.25), SPEC-LIT §127.2's isentropic and shocked nozzle relations evaluated live by compressible::exact (CMP-06): Gate CMP-NOZ subsonic: M(x) within 0.5 % of the key's M(xi), all 200 cells (SPEC-LIT 127.6): 6.0014e-3 against 0.005; Gate CMP-NOZ supersonic: M(x) within 0.5 % of the key's M(xi), all 200 cells (SPEC-LIT 127.6): no field, the march stopped on a bound; Gate CMP-NOZ supersonic: exit p within 0.5 % of the key at the last cell centre (SPEC-LIT 127.6): no field, the march stopped on a bound; Gate CMP-NOZ shocked: shock position within 1 % of the length, 200 cells (SPEC-LIT 127.6): no field, the march stopped on a bound; Gate CMP-NOZ shocked: p_t2/p_t1 within 1 % of (127.25)'s, the last cell (SPEC-LIT 127.6): no field, the march stopped on a bound; Gate CMP-NOZ shocked: exit p within 0.5 % of the key at the last cell centre (SPEC-LIT 127.6): no field, the march stopped on a bound |
| 20 | SPEC-LIT 127.6 Gate CMP-RING Ringleb flow | MISSES | - | the isentropic Ringleb field of (127.8)-(127.11a), SPEC-LIT section 127.3's smooth answer key evaluated live by compressible::exact (CMP-07): the L2 entropy error reads E = 1.0304e-2 / 1.4008e-2 / 1.9372e-2 (finest first, 96 x 24 / 48 x 12 / 24 x 6); pair orders 0.469 (h/2 vs h), 0.443 (h/4 vs h/2), fitted order 0.456 against the bar 1.5; iterations 10000 / 10000 / 10000 |
| 21 | SPEC-LIT 127.6 Gate CMP-RAE RAE 2822 | MISSES | MISSES | RAE 2822 Case 6 (the corrected conditions M 0.729, alpha 2.31 degrees, Re 6.5e6) of Cook, McDonald & Firmin, AGARD AR-138 (1979), as the NPARC archive reproduces it - the answer key rae2822-cp-case6: the smoke's residual did not fall to a tenth of its first chunk's; f64 L0 stopped on a bound at step 5968 (the first out-of-bounds cell 6312); f64 L1 stopped on a bound at step 5454 (the first out-of-bounds cell 21072); f64 L2 stopped on a bound at step 5376 (the first out-of-bounds cell 75232); single L1 stopped on a bound at step 210 (the first out-of-bounds cell 151) |
| 22 | Gate 116-P | MISSES | MISSES | Hagen-Poiseuille, 8 mu L Q / (pi R^4) (116.27): Hagen-Poiseuille on the L1 pipe at Re 100: f64 +15.49 %, single stopped (diverged), the solve 78.6 s, the single-build mesh refused (the known A28) |
| 23 | SPEC-LIT 144.7 Gate 88-T leg 3, kOmegaSSTLM, T3B laminar C_f against Blasius | MISSES | MISSES | Blasius' 0.664/sqrt(Re_x) (SPEC-LIT 88.7) over the part of the plate the key measures laminar: kOmegaSSTLM worst laminar C_f against Blasius (144.4c): E = +58.30 % at Re_x 3.9360e4 on the finest mesh, band 10 %, u_val 8.97 % (u_num 8.97 %, u_D 0 %) |
| 24 | SPEC-LIT 144.7 the T3A- inflow fit against ERCOFTAC's measured free-stream Tu(x) | MISSES | MISSES | the ERCOFTAC key's measured free-stream Tu(x), its TUFS column: fitted decay misses the measured Tu(x) by 3.63 % at Re_x 7.7200e5 (band 3 %); the run uses this inflow anyway (an input check, SPEC-LIT 144.7) |
| 25 | SPEC-LIT 144.7 Gate 88-T leg 3, kOmegaSSTLM, T3A- turbulent C_f against Roach & Brierley | MISSES | MISSES | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3A- key (ercoftac-t3am, SPEC-LIT 144.7): kOmegaSSTLM worst turbulent C_f against the station's (144.4b): E = +132.12 % at Re_x 2.0220e6 on the finest mesh, band 10 %, u_val 12.81 % (u_num 12.65 %, u_D 2 %) |
| 26 | SPEC-LIT 144.7 Gate 88-T leg 3, kOmegaSSTLM, T3A- laminar C_f against Blasius | MISSES | MISSES | Blasius' 0.664/sqrt(Re_x) (SPEC-LIT 88.7) over the part of the plate the key measures laminar: kOmegaSSTLM worst laminar C_f against Blasius (144.4c): E = +27.55 % at Re_x 1.2097e6 on the finest mesh, band 10 %, u_val 0.63 % (u_num 0.63 %, u_D 0 %) |
| 27 | SPEC-LIT 157.5 Gate DST-LEEDS Leeds fan-stirred cornflour bomb | MISSES | MISSES | Bradley, Chen & Swithenbank (1989) via Huang et al. (2020) Fig. 9, key huang2020-leeds-st-b: 5 of the gate's rows leave the bands of §157.5 |
| 28 | SPEC-LIT 157.6 Gate DST-VES closed vessels: thin-flame spheres and the REMBE closed stage | MISSES | MISSES | Huang, Bloching & Lipatnikov (2022) Fig. 10, key huang2022-rembe-p-closed: 14 of the gate's rows leave the bands of §157.6 |
| 29 | SPEC-LIT 133.12 Gate CHR-13 1-D freely propagating CH4-air flame, BFER phi 1.0 | MISSES | MISSES | the plan's S_L = U_in - dx_f/dt within 10 % of Konnov et al. 2018 (closed, so that comparison is NOT RUN); a level that stops before its end time gives no S_L at all: 3 of 3 levels stopped before 6 ms on the reactor's refusal (100 µm at 0.0006 s, 50 µm at 0.0006 s, 25 µm at 0.0006 s), so no S_L was measured |
| 30 | SPEC-LIT 134.8 Gate CHR-16 Sandia Flame D, EDC + DRM19, k-epsilon C1 1.6, REDUCED | MISSES | MISSES | the plan's REDUCED gate: 300 iterations, finite fields, element and mass budgets within 1e-6: 40 of 300 steps: chemistry: EDC refused 1 cell(s); the first is cell 244 with status 8 (BELOW_NEGATIVE_ATOL) (SPEC-LIT §134.8); the worst inventory budget is C at 2.354e-1 against 1e-6 |
| 31 | SPEC-LIT 136.7 Gate CHR-SLAB the isothermal grey slab | MISSES | MISSES | the exact slab 1 - 2 E_3(tau), (135.6) of SPEC-LIT 135.2: S8 N 256 tau 0.1: e_256 +2.1713e-2 outside 2 % on both walls, S8's own angular limit (136.13) e_lim +2.2191e-2 above 2 % |
| 32 | SPEC-LIT 136.8 Gate CHR-CUBE scattering slab and cube | MISSES | MISSES | (136.6), the cube's wall-centre flux by the exact angular integral, SPEC-LIT 136.3: S8 kL 0.1 24^3: e -3.1005e-2 outside 3 %, by the step scheme's spatial error |
| 33 | SPEC-LIT 159.11 Gate DST-B-LAM Phase B laminar dust flame: the fine-particle limit and the size order | MISSES | MISSES | (i) \|S_L,B(1 um)/S_L,gas - 1\| within 2 % and (ii) S_L(d_{j+1}) <= S_L(d_j) along d = 5, 10, 20, 40 um - the gate's own bands; no published measurement: 1 of the gate's rows leave the bands of 159.11 |
| 34 | SPEC-LIT 159.12 Gate DST-B-VES Phase B cloud in a closed 20 L sphere: the AICC end pressure and the thin-flame half time | MISSES | MISSES | the cloud's host AICC (159.21) and the thin-flame half time (159.30) with S_u(p) of the 1-D Phase B tube: 7 of the gate's rows leave the bands of 159.12 |
| 35 | SPEC-LIT 147.2 Gate 68-C coupled leg: Theobald test 3 through ofgpu-lowmach, replayed | MISSES | MISSES | Theobald (1981) test 3's measured maximum throw, 24.4 m, at the shape of the FDS Validation Guide's own metric for this quantity - +-10 % bias, 30 % scatter: 3 of the gate's rows leave the bars of 68.12 |
| 36 | SPEC-LIT S32.4 verdict 2 (Reynolds analogy), wall-function leg | OPEN | OPEN | Gnielinski (1976) evaluated at this leg's own MEASURED wall friction factor, +-10 % band: WF Nu is +34.4% of Gnielinski at the MEASURED f and +15.2% at the viscous form of it - outside the band on both |
| 37 | SPEC-LIT S32.4 verdict 2 (Reynolds analogy), wall-function leg, at PrtModel KaysCrawford (opt-in, S37) | OPEN | OPEN | Gnielinski (1976) evaluated at this leg's own MEASURED wall friction factor, +-10 % band, on the PrtModel KaysCrawford record: WF Nu is +34.0% of Gnielinski at the MEASURED f, at PrtModel KaysCrawford (opt-in, S37) - outside the band |
| 38 | SPEC-LIT S32.4 verdict 1 (absolute prediction), resolved leg | OPEN | OPEN | Gnielinski (1976) evaluated at the Petukhov smooth-PIPE f, +-10 % band: resolved leg Nu is +11.9% of it - outside the band by 1.9 points, against an energy-balance uncertainty of 0.00000% which is now round-off, so the gate does NOT close and the miss is DECISIVE with nothing left to hide any of it behind (S32.4's UNDECIDED clause does not apply) |
| 39 | SPEC-LIT S32.4 verdict 2 (Reynolds analogy), resolved leg | OPEN | OPEN | Gnielinski (1976) evaluated at this leg's own MEASURED wall friction factor, +-10 % band: resolved leg Nu is +14.9% of it at the MEASURED f = 2.3832e-02 - outside the band |
| 40 | SPEC-LIT S88 Gate 88-T | OPEN | OPEN | the ERCOFTAC T3A flat plate's measured transition, on the NASA/TMBWG 2D T3A rig (U = 69.44 m/s, Re/m = 2.00e5, Tu = 3.300 % at the leading edge): the model's own onset criterion fires at Re_x = 85251.4, where the local Tu has decayed to 2.24405 % and Re_theta~ = 234.21 - but no published digit-level onset Re_x for T3A was found to hold it against, so the comparison is not closed |
| 41 | SPEC-LIT S90 Gate 90-T | OPEN | OPEN | the ERCOFTAC T3A flat plate's measured transition, on the NASA/TMBWG 2D T3A rig (U = 69.44 m/s, Re/m = 2.00e5, Tu = 3.300 % at the leading edge): the 2015 correlation's onset switch fires at Re_x = 22826.1, where the local Tu_L has decayed to 54.579 % and Re_thetac reads 100 - but no published digit-level onset Re_x for T3A was found to hold it against, so the comparison is not closed |
| 42 | 78-D | OPEN | OPEN | Bai & Gosman SAE 950283 (1995) against Mundo, Sommerfeld & Tropea, Int. J. Multiphase Flow 21 (1995) 151: the two published splash criteria disagree by a factor of 4.78 in Weber number for the same 100 um water droplet - Mundo splashes it at 8.99 m/s (We 111), Bai-Gosman at 19.65 m/s (We 531) |
| 43 | SPEC-LIT 105.14 Gate 105-C Turek-Hron | OPEN | OPEN | Turek & Hron (2006), CFD1/CFD2 level 6+0 and CFD3 level 4+0: the Turek-Hron meshes are not generated, so no comparison was run |
| 44 | SPEC-LIT S110.2 Gate 110-A channel DNS (Moser, Kim & Mansour 1999) | OPEN | OPEN | Moser, Kim & Mansour (1999) DNS mean-velocity profiles, reference/mkm99/chan{180,395,590}.means, u_D = 0 (none stated): answer key missing: mkm99-chan180, mkm99-chan395, mkm99-chan590 - absent from reference/ or not its manifest digest (SPEC-LIT 110.1); no comparison was run |
| 45 | SPEC-LIT S110.4 Gate 110-C buoyant plume centreline (McCaffrey 1979) | OPEN | OPEN | McCaffrey (1979) NBSIR 79-1910 Table 1 plume-region correlations at the declared Q, reference/mccaffrey_1979/centreline_table1.txt: P3: the finest centreline dT decays as z^(-2.130) over the six stations, outside -5/3 +- 0.25 - the correlation's exponent, so the gate is left open (S110.1) |
| 46 | SPEC-LIT S110.4 Gate 110-C buoyant plume centreline (McCaffrey 1979) | OPEN | OPEN | McCaffrey (1979) NBSIR 79-1910 Table 1 plume-region correlations at the declared Q, reference/mccaffrey_1979/centreline_table1.txt: P1 worst +212.6% outside, P2 worst +76.1% undecided against the +-15% band at u_val/D = 1.422 - on the predicted high side but BEYOND what 20-40 % radiative loss accounts for (dT +16% to +41%, w +8% to +19%): radiation alone does not explain this miss (S110.4) |
| 47 | SPEC-LIT 144.3 T3A leg 1 in 2-D, kOmegaSSTLM, against the TMR's 3.300 % | OPEN | OPEN | the NASA/TMBWG T3A page's stated leading-edge Tu, 3.300 % (SPEC-LIT 88.10): Tu at the leading edge 3.35332 % on the finest mesh, 1.616 % from 3.300 % (bar 0.5 %); the 1-D closed form of the same model is 1.607 % from it |
| 48 | SPEC-LIT 144.3 T3A leg 1 in 2-D, kOmegaSSTGamma, against the TMR's 3.300 % | OPEN | OPEN | the NASA/TMBWG T3A page's stated leading-edge Tu, 3.300 % (SPEC-LIT 88.10): Tu at the leading edge 3.35332 % on the finest mesh, 1.616 % from 3.300 % (bar 0.5 %); the 1-D closed form of the same model is 1.607 % from it |
| 49 | SPEC-LIT 144.4 Gate 88-T leg 4, kOmegaSSTLM, turbulent C_f against Roach & Brierley | OPEN | OPEN | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3A key (SPEC-LIT 144.6): undecided, within u_val of the band edge (SPEC-LIT 32.4, 110.1): kOmegaSSTLM worst turbulent C_f against the station's (144.4b): E = -7.07 % at Re_x 3.0930e5 on the finest mesh, band 10 %, u_val 5.22 % (u_num 4.82 %, u_D 2 %) |
| 50 | SPEC-LIT 144.4 Gate 90-T leg 4, kOmegaSSTGamma, transition onset Re_x against Roach & Brierley | OPEN | OPEN | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3A key (SPEC-LIT 144.6): undecided, within u_val of the band edge (SPEC-LIT 32.4, 110.1): kOmegaSSTGamma onset Re_x (least C_f) against (144.4a): E = +12.72 % at Re_x 1.5633e5 on the finest mesh, band 15 %, u_val 6.99 % (u_num 6.99 %, u_D 0 %) |
| 51 | SPEC-LIT 144.5 Gate 132-Z ZPG plate, SpalartAllmaras, C_f against Karman-Schoenherr | OPEN | OPEN | the Karman-Schoenherr C_f(Re_theta) as the NASA TMR tabulates it (a correlation, SPEC-LIT 144.5): undecided, within u_val of the bar (SPEC-LIT 32.4, 110.1): SpalartAllmaras worst C_f against K-S over Re_theta 4000..13000: E = -4.455% at Re_theta 4000 on the finest mesh, bar 3 %, u_val 6.344% |
| 52 | SPEC-LIT 144.5 Gate 132-Z ZPG plate, kOmegaSST, C_f against Karman-Schoenherr | OPEN | OPEN | the Karman-Schoenherr C_f(Re_theta) as the NASA TMR tabulates it (a correlation, SPEC-LIT 144.5): outside the bar, against a correlation (SPEC-LIT 144.5): kOmegaSST worst C_f against K-S over Re_theta 4000..13000: E = -4.310% at Re_theta 4000 on the finest mesh, bar 3 %, u_val 0.626% |
| 53 | SPEC-LIT 143.5 Gate 131-P turbulent pipe, kOmegaSST, Re_tau 576.69, f against Prandtl | OPEN | OPEN | Prandtl's smooth-pipe law at the input Re_tau (Nikuradse's measured pipes, Schlichting NACA TM 1218 eq 16.26): undecided, within u_val of the band edge (SPEC-LIT 32.4, 110.1): kOmegaSST at Re_tau 576.69, f against Prandtl: E = +5.18 % on the finest level (n = 160), bar 5 %, u_val 0.66 % |
| 54 | SPEC-LIT 143.5 Gate 131-P turbulent pipe, kOmegaSST, Re_tau 576.69, the log law | OPEN | OPEN | the log law u+ = 2.5 ln y+ + 5.5 over 30 <= y+ <= 0.2 Re_tau (Schlichting NACA TM 1218 eq 16.14): undecided, within u_val of the band edge (SPEC-LIT 32.4, 110.1): kOmegaSST at Re_tau 576.69, log-law deviation: E = 1.177 in u+ on the finest level (n = 160), bar 1.0 in u+, u_val 0.237 |
| 55 | SPEC-LIT 144.7 Gate 88-T leg 3, kOmegaSSTLM, T3B onset Re_x against Roach & Brierley | OPEN | OPEN | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3B key (ercoftac-t3b, SPEC-LIT 144.7): undecided, within u_val of the band edge (SPEC-LIT 32.4, 110.1): kOmegaSSTLM onset Re_x (the first grid minimum (144.7e), the least face being the trailing edge) against (144.4a): E = -11.50 % at Re_x 4.9938e4 on the finest mesh, band 15 %, u_val 28.55 % (u_num 28.55 %, u_D 0 %) |
| 56 | SPEC-LIT 144.7 Gate 90-T leg 3, kOmegaSSTGamma, T3B onset Re_x against Roach & Brierley | OPEN | OPEN | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3B key (ercoftac-t3b, SPEC-LIT 144.7): no kOmegaSSTGamma T3B run recorded yet - run the record_t3fit_* tests of SPEC-LIT 144.7 |
| 57 | SPEC-LIT 144.7 Gate 90-T leg 3, kOmegaSSTGamma, T3B turbulent C_f against Roach & Brierley | OPEN | OPEN | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3B key (ercoftac-t3b, SPEC-LIT 144.7): no kOmegaSSTGamma T3B run recorded yet - run the record_t3fit_* tests of SPEC-LIT 144.7 |
| 58 | SPEC-LIT 144.7 Gate 90-T leg 3, kOmegaSSTGamma, T3B laminar C_f against Blasius | OPEN | OPEN | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3B key (ercoftac-t3b, SPEC-LIT 144.7): no kOmegaSSTGamma T3B run recorded yet - run the record_t3fit_* tests of SPEC-LIT 144.7 |
| 59 | SPEC-LIT 144.7 Gate 88-T leg 3, kOmegaSSTLM, T3A- onset Re_x against Roach & Brierley | OPEN | OPEN | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3A- key (ercoftac-t3am, SPEC-LIT 144.7): undecided, within u_val of the band edge (SPEC-LIT 32.4, 110.1): kOmegaSSTLM onset Re_x (least C_f) against (144.4a): E = -16.13 % at Re_x 1.2106e6 on the finest mesh, band 15 %, u_val 5.68 % (u_num 5.68 %, u_D 0 %) |
| 60 | SPEC-LIT 144.7 Gate 90-T leg 3, kOmegaSSTGamma, T3A- onset Re_x against Roach & Brierley | OPEN | OPEN | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3A- key (ercoftac-t3am, SPEC-LIT 144.7): no kOmegaSSTGamma T3A- run recorded yet - run the record_t3fit_* tests of SPEC-LIT 144.7 |
| 61 | SPEC-LIT 144.7 Gate 90-T leg 3, kOmegaSSTGamma, T3A- turbulent C_f against Roach & Brierley | OPEN | OPEN | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3A- key (ercoftac-t3am, SPEC-LIT 144.7): no kOmegaSSTGamma T3A- run recorded yet - run the record_t3fit_* tests of SPEC-LIT 144.7 |
| 62 | SPEC-LIT 144.7 Gate 90-T leg 3, kOmegaSSTGamma, T3A- laminar C_f against Blasius | OPEN | OPEN | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3A- key (ercoftac-t3am, SPEC-LIT 144.7): no kOmegaSSTGamma T3A- run recorded yet - run the record_t3fit_* tests of SPEC-LIT 144.7 |
| 63 | SPEC-LIT 144.7 Gate 88-T leg 3, kOmegaSSTLM, T3A onset Re_x against Roach & Brierley | OPEN | OPEN | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3A key (ercoftac-t3a, SPEC-LIT 144.6): no kOmegaSSTLM T3A run recorded yet - run the record_t3fit_* tests of SPEC-LIT 144.7 |
| 64 | SPEC-LIT 144.7 Gate 88-T leg 3, kOmegaSSTLM, T3A turbulent C_f against Roach & Brierley | OPEN | OPEN | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3A key (ercoftac-t3a, SPEC-LIT 144.6): no kOmegaSSTLM T3A run recorded yet - run the record_t3fit_* tests of SPEC-LIT 144.7 |
| 65 | SPEC-LIT 144.7 Gate 88-T leg 3, kOmegaSSTLM, T3A laminar C_f against Blasius | OPEN | OPEN | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3A key (ercoftac-t3a, SPEC-LIT 144.6): no kOmegaSSTLM T3A run recorded yet - run the record_t3fit_* tests of SPEC-LIT 144.7 |
| 66 | SPEC-LIT 144.7 Gate 90-T leg 3, kOmegaSSTGamma, T3A onset Re_x against Roach & Brierley | OPEN | OPEN | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3A key (ercoftac-t3a, SPEC-LIT 144.6): no kOmegaSSTGamma T3A run recorded yet - run the record_t3fit_* tests of SPEC-LIT 144.7 |
| 67 | SPEC-LIT 144.7 Gate 90-T leg 3, kOmegaSSTGamma, T3A turbulent C_f against Roach & Brierley | OPEN | OPEN | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3A key (ercoftac-t3a, SPEC-LIT 144.6): no kOmegaSSTGamma T3A run recorded yet - run the record_t3fit_* tests of SPEC-LIT 144.7 |
| 68 | SPEC-LIT 144.7 Gate 90-T leg 3, kOmegaSSTGamma, T3A laminar C_f against Blasius | OPEN | OPEN | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3A key (ercoftac-t3a, SPEC-LIT 144.6): no kOmegaSSTGamma T3A run recorded yet - run the record_t3fit_* tests of SPEC-LIT 144.7 |
| 69 | SPEC-LIT 144.7 Gate 88-T leg 3, kOmegaSSTLM, onset spread Re_x(T3A-)/Re_x(T3B) against Roach & Brierley | OPEN | OPEN | Roach & Brierley's T3 series, the keys' own (144.4a) onsets and their ratio: undecided, within u_val of the band edge (SPEC-LIT 32.4, 110.1): the onset spread (144.7d): S_sim = 2.4242e1 against S_key = 2.5578e1, E = -5.22 %, band 15 %, u_val 32.97 % (u_num from the two onsets' studies) |
| 70 | SPEC-LIT 144.7 Gate 90-T leg 3, kOmegaSSTGamma, onset ordering T3B < T3A < T3A- against Roach & Brierley | OPEN | OPEN | Roach & Brierley's T3 series, the keys' own (144.4a) onsets and their ratio: missing a T3B fine f64 run, a T3A- fine f64 run |
| 71 | SPEC-LIT 144.7 Gate 90-T leg 3, kOmegaSSTGamma, onset spread Re_x(T3A-)/Re_x(T3B) against Roach & Brierley | OPEN | OPEN | Roach & Brierley's T3 series, the keys' own (144.4a) onsets and their ratio: missing a T3B fine f64 run, a T3A- fine f64 run |
| 72 | SPEC-LIT S110.4 Gate 110-C leg 2 radiating plume centreline (McCaffrey 1979) | OPEN | OPEN | McCaffrey (1979) NBSIR 79-1910 Table 1 plume-region correlations at the declared Q, reference/mccaffrey_1979/centreline_table1.txt; the run's radiant fraction beside the report's 20-40 %: undecided: P1/P2 worst 1.539 is within u_val/D = 2.678 of the 15% edge (S32.4); the run's radiant fraction chi_r = 0.077 against McCaffrey's 0.20-0.40 |
| 73 | Gate 95-A cantilever | - | MISSES | Timoshenko & Goodier ch. 3, plane strain, block-coupled (SPEC-LIT 109.6), four meshes r = 2, orders on the finest pair, the tip study on the finest three: 5:1 GMRES(50): converged 0/4, p_u 0.59, p_sigma 4.61 (n_y 16 -> 32), finest tip 9.71e-1 |
| 74 | Gate 95-A cantilever | - | MISSES | Timoshenko & Goodier ch. 3, plane strain, block-coupled (SPEC-LIT 109.6), four meshes r = 2, orders on the finest pair, the tip study on the finest three: 10:1 GMRES(50): converged 0/4, p_u -53.62, p_sigma -inf (n_y 16 -> 32), finest tip 1.37e16 |
| 75 | SPEC-LIT 98.6 Gate 98-B radiating slab | - | MISSES | the steady balance of a slab radiating to a large surround, (S98.8), SPEC-LIT 98.6: T_b rel 6.56e-7, residual 0.00e0, worst Newton ratio 1.092e4 against 2C = 1.725e-3, 5 ratios |
| 76 | SPEC-LIT 98.9 Gate 98-C enclosure with a running flow | - | MISSES | the enclosure's own power balance, (S98.9), (S98.10), parallel_plate_flux and concentric_flux, SPEC-LIT 98.9: (a) power balance 3.39e-6 against 1e-10; (b) (S98.10) worst 5.77e-2 against 1e-9; (b) cell source rel 9.08e-8 against 1e-12; (b) closed form rel 5.02e-4 against 1e-10 over 100 faces; (c) not every radiationRelaxation converged within 4000 iterations |
| 77 | SPEC-LIT 100.4 Gate 100-B published property tables | - | MISSES | Ho, Powell and Liley (1972), J. Phys. Chem. Ref. Data 1, 279, p. 394: silicon, recommended values: silicon: knots to the bit false, worst 1.55 % between them against 5 % |
| 78 | SPEC-LIT 100.4 Gate 100-B published property tables | - | MISSES | McBride, Zehe and Gordon, NASA/TP-2002-211556 (2002), eq. (1), Appendix D and Table B1: cp/R: worst \|Cp(298.15) - Table B1\| 3.51e-4 J/(K mol) against 5e-4, worst jump at 1000 K 3.41e-7 against 1e-8 |
| 79 | SPEC-LIT 100.15 Gate 100-C Brinkman plane Poiseuille | - | MISSES | Brinkman's fully developed plane Poiseuille with viscous heating, (S100.15) and (S100.16), SPEC-LIT 100.15: both walls held, ny = 8: converged false, balance 5.60e-6; both walls held, ny = 16: converged false, balance 1.68e-4; both walls held, ny = 32: converged false, balance 1.86e-4; top adiabatic, ny = 8: converged false, balance 3.15e-5; top adiabatic, ny = 16: converged false, balance 1.53e-4; top adiabatic, ny = 32: converged false, balance 1.50e-4 |
| 80 | SPEC-LIT 158.7 Gate DST-REMBE: the vented REMBE explosion, REDUCED PLACEHOLDER smoke on the castellated vessel, key J.3 NOT RUN | - | MISSES | Huang, Bloching & Lipatnikov (2022) Fig. 11, key huang2022-rembe-p-vented: 2 row(s) of the REDUCED PLACEHOLDER smoke leave their bars (never compared with H22) |
<!-- validate-run:gates:ko end -->

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
- **압축성 유동은 새로 들어왔고, 그 게이트 몇 개가 미달입니다.** `ofgpu-compressible`가 전속도 압력 기반 방정식을 풉니다(SPEC-LIT §122-§127). Gate CMP-LM, CMP-COU, CMP-OBL은 통과합니다(CMP-OBL은 수렴 정지 조건을 채우지 못한 채). CMP-SOD의 PISO 행, CMP-AC, CMP-NOZ, CMP-RING, CMP-RAE는 미달이며 `ofgpu-validate`가 매 실행마다 이름을 댑니다(§127.6). 동봉 케이스는 `cases/sodShockTube.jsonc`와 `cases/nozzle` 두 개입니다.
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

격자 구조가 전혀 없는 polyMesh라도 단면(slice)과 평면(plane)은 거절되지 않습니다.
스튜디오 서버가 셀을 평면으로 정확히 자르고(SPEC-LIT §148.1) 뷰어는 각 단면
다각형을 그 셀의 값으로 칠합니다. 등값면, 유선, 화살표는 여전히 격자 구조가
필요해 그런 격자에서는 `NO_STRUCTURED_GRID`로 답하며, 셀 중심이 격자를 이루지
않는 VTU도 마찬가지입니다(자를 셀 면 정보가 없음).

이런 polyMesh에서도 월드 좌표 probe(`probe {point}`)가 됩니다. 서버가 그 점을 품은
셀을 찾습니다. 면이 평평한 볼록 셀은 면마다 정확히 판정하고, 뒤틀리거나 볼록하지
않은 셀만 그 점을 품을 수 있을 때는 중심이 가장 가까운 셀이 답하며 검사기(와
`gui_state`의 `selection.located`)가 `nearest-centre`라고 밝힙니다. 어느 셀에도
없는 점은 `MISS`입니다.

### 뷰어가 케이스를 옆으로 눕혀 그림

케이스에 `constant/g`가 없고, 바닥 패치 이름이 `bottomWall`/`floor`/`ground`가
아닙니다. 패치 이름을 맞추거나, 중력이 실제로 필요한 케이스라면 `constant/g`를
넣으십시오. **뷰어를 돌리려는 목적만으로 `g`를 넣지 마십시오** — 중력은
방정식에 실제로 들어가고 해가 달라집니다.

### 컷셀·경사 격자에서 `-output nvdb`가 거절됨

`nvdb`/`vdb`는 복셀 격자입니다. 균일 카테시안 상자에서는 셀이 곧 복셀이고, 그 밖의
격자에는 `output.visualisation.lattice`의 복셀 격자가 필요합니다(8장).
모든 드라이버가 그 격자를 깔고, 격자를 깔 수 없을 때만 루프에 들어가기 전에 그 이유를 대고
거절합니다.
컷셀 해석의 산출물은 `-output foam,vtu`로 받으십시오. `-permissive`를 주면 `nvdb`만
목록에서 빠지고 나머지 형식은 그대로 쓰입니다.

### `-output "U,p"` 가 거절됨

`-output`은 형식 목록입니다. `foam`, `vtu`, `nvdb`, `vdb`, `usda` 중에서
고르십시오.

### 결과가 어느 디렉터리에 쓰이는가

`ofgpu-lowmach`의 정상해석은 반복 횟수를 디렉터리 이름으로 씁니다(`-iters 3000` →
`3000/`; `-restartFrom`으로 이어 돌리면 이어진 횟수). `0/`은 절대 쓰지 않습니다. 다른
정상해석 드라이버(`ofgpu-k-epsilon` 등)는 `system/controlDict`의 `endTime`을 이름으로
씁니다(기본 `1/`). 이전 해에서 이어 가려면 `-restartFrom`을 쓰십시오 — 케이스를 다시
돌리면 `0/`의 초기장에서 시작합니다(SPEC-LIT §44.9).

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
