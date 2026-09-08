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

동봉된 경주차 샘플이 가장 빠른 길입니다. 형상 하나와 명령 세 줄입니다.

```powershell
cd cases
.\racecar.cmd
```

이 스크립트가 하는 일:

```powershell
# 1. 단위 풍동(1 m 정육면체)에 차체를 컷셀로 새깁니다.  20-60분 (CPU)
ofgpu-generate-mesh big racecar_case 128 -stl car=racecar.stl -cutcell

# 2. 압력과 온도 필드를 넣습니다.                         즉시
copy racecar.fields\p racecar_case\0\p
copy racecar.fields\T racecar_case\0\T

# 3. 운동량까지 풉니다.                                   1-2분 (GPU)
ofgpu-lowmach racecar_case -iters 3000 -check 250 -output foam
```

1단계가 오래 걸리는 이유는 **형상 분류가 CPU에서 돌기 때문**입니다. 210만 개
셀 각각을 STL 표면에 대해 안/밖으로 가르고, 표면에 걸친 셀을 잘라내는 작업입니다.
3단계는 GPU에 상주하는 루프라 몇 분이면 끝납니다.

2단계가 왜 필요한지는 [§6](#6-솔버-고르기--가장-많이-틀리는-곳)에서 설명합니다.
요약하면 메쉬 생성기는 **난류 전용 드라이버**가 읽는 필드만 쓰고, 운동량
드라이버가 필요로 하는 `p`와 `T`는 쓰지 않기 때문입니다.

끝나면 `racecar_case/0/`에 `U`, `p`, `T`, `k`, `omega`, `nut`, `rho`가 OpenFOAM
ASCII로 들어 있습니다. Studio의 3D 뷰어로 열거나 ParaView로 바로 읽힙니다.

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
                    [-wallModel standard|spalding|rough|lowRe [-Ks x [-Cs y]]]
                    [-cyclic x|y|z] [-permissive]
```

프리셋: `channel`, `cavity`, `step`, `big`, `plume`, `room`, `damBreak`.
`big`은 한 변 1 m의 정육면체 풍동이고 셀 수를 **하나만** 받습니다(`n³`).

생성되는 것은 바로 돌릴 수 있는 완전한 케이스입니다 — `constant/polyMesh`,
`constant/physicalProperties`, `constant/momentumTransport`,
`system/{controlDict,fvSchemes,fvSolution}`, 그리고 `0/`에 `U`, `k`, `epsilon`,
`omega`, `nut`.

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
| `lowRe` | `nutLowReWallFunction` | `kLowReWallFunction` | `zeroGradient` |

자세한 표(LES일 때의 축약 포함)는 [`cases/README.md`](../cases/README.md).

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

### 난류 전용 드라이버는 언제 쓰는가

정해진 속도장 위에서 난류량만 보고 싶을 때입니다 — 모델 자체의 검증, 벽함수
비교, 특정 필드에 대한 난류 응답 조사. `plume.jsonc`처럼 얼린 `U` 위의 두
방정식 수렴이 목적인 케이스가 그렇습니다.

### 운동량 드라이버에 필요한 것

`ofgpu-lowmach`는 `U`와 `p`를 함께 풀고, 저마하 루프가 `T`를 요구합니다.
메쉬 생성기는 그 둘을 쓰지 않으므로 직접 넣어야 합니다. 경주차 샘플은
`cases/racecar.fields/`에 균일장 두 개를 동봉해 두었습니다 — 내용이라 할 것은
경계조건뿐입니다(출구 `p = 0` 고정, 나머지 zeroGradient, 등온 293.15 K).

없이 돌리면 이렇게 거절합니다:

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

메쉬 생성기는 난류 전용 드라이버가 읽는 필드만 씁니다. `0/p`와 (저마하면)
`0/T`를 넣어야 합니다. 경주차 샘플의 `cases/racecar.fields/`가 그 예입니다.
[§6](#6-솔버-고르기--가장-많이-틀리는-곳) 참고.

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
초기장을 보존하려면 원본을 따로 두십시오(경주차 샘플의 `racecar.fields/`가
그 이유로 존재합니다).

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
