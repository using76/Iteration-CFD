# meteor-cfd — GPU-native finite volume CFD

**주식회사 메테오시뮬레이션** 개발. Rust 호스트 + CUDA 커널. 메쉬와 필드를 한 번 올린 뒤 시간 루프
전체가 device에 머무르는 것을 목표로 합니다.

```
rust/          호스트(Rust) + 커널(CUDA C++)
rust/SPEC-LIT.md   수치 명세 — 모든 수식이 논문/교과서로 인용됨
docs/          CFD 모델 카탈로그와 GPU 이식성 분류
cases/         테스트 케이스
tools/         볼륨 렌더러, 리포트 생성기
third_party/   AMGX (BSD-3-Clause)
reference/fds  FDS (NIST, 퍼블릭 도메인) — 참조용, 배포물 아님
```

---

## 라이선스

**소스는 공개되지만 오픈소스는 아닙니다.**

| | |
|---|---|
| 교육 목적 (수업·과제·실습·독학) | **무료** |
| 연구·논문·학위논문·보고서 | 유상 라이선스 |
| 상업·생산 이용 | 유상 라이선스 |
| 읽기·수정·포크·재배포 | 허용 (조건 명시 필요) |

**라이선스 문의: simul@msimul.com**

개인 연구자와 비영리 목적에는 무상 또는 감면 라이선스를 검토합니다.
문의해 주세요.

**2036년 8월 23일**부터 그 시점에 공개된 버전은 Apache-2.0이 됩니다.
전문은 [`LICENSE`](LICENSE), 서드파티 고지는 [`NOTICE`](NOTICE)에
있습니다.

> 이 라이선스는 Business Source License 1.1의 구조를 따르지만 **BUSL이
> 아닙니다.** BUSL은 모든 비생산 이용을 무료로 허용하고, 그 안에는 학술
> 연구가 포함됩니다. 여기서는 교육만 무료이고 연구는 아닙니다.

---

## 코드 출처

수치 코드는 전부 **문헌에서 직접 구현**했습니다.

- 모든 수식을 원논문으로 인용한 [`rust/SPEC-LIT.md`](rust/SPEC-LIT.md)를
  명세로 삼았고, 구현자는 그것만 봅니다.
- 파일별 출처는 [`rust/PROVENANCE.md`](rust/PROVENANCE.md)에 있고, 53개 파일이
  자기가 어느 논문에서 나왔는지와 `No GPL-licensed source was consulted.`를
  헤더에 적습니다.
- **검증은 다른 CFD 코드와 비교하지 않습니다.** 인위해(MMS), 해석해,
  공개 벤치마크뿐입니다.
- 의존성은 cudarc(MIT/Apache-2.0), thiserror, 선택적 AMGX(BSD-3절)뿐입니다.

OpenFOAM ASCII 케이스 형식을 읽고 씁니다 — ParaView·`foamToVTK` 같은 기존
도구를 그대로 쓰기 위한 상호운용이며, OpenFOAM의 어떤 부분과도 링크하지
않고 그 소스를 포함하지도 않습니다. 파일 형식은 저작물이 아닙니다.

---

## 설정은 동작하거나, 시끄럽게 실패합니다

조용한 대체가 없습니다. 케이스 파일의 모든 항목은 셋 중 하나입니다
(`rust/SPEC-LIT.md` §13.4):

```
인식되고 구현됨   -> 사용
인식되나 미구현   -> 설정 이름과 대안을 밝힌 에러
인식 안 됨        -> 설정 이름을 밝힌 에러
```

```
$ ofgpu-k-epsilon case            # fvSchemes 에 `div(phi,k) Gauss totalGarbage;`
error: divSchemes/div(phi,k): "Gauss totalGarbage" is not supported by ofgpu;
       available: Gauss linear, Gauss upwind, Gauss linearUpwind [grad],
       Gauss cubic, Gauss QUICK, Gauss QUICKUnlimited, Gauss Gamma <0.1..0.5>,
       Gauss blended <0..1>, Gauss linearUpwindBlended <0..1>,
       Gauss limitedLinear <1..2>, Gauss vanLeer, Gauss vanAlbada,
       Gauss Minmod, Gauss SuperBee, Gauss MUSCL
  (run with -permissive to substitute Gauss upwind and continue)
```

탈출구는 `-permissive` 하나뿐이고, **무엇으로 대체했는지 매번 출력합니다.**
그럴듯한 틀린 답이 답이 없는 것보다 나쁘기 때문입니다.

증거: 아래 여섯 스킴이 같은 케이스에서 각각 다른 결과를 냅니다. 예전에는
전부 upwind와 비트 단위로 같았습니다.

| fvSchemes | `0/k` 해시 |
|---|---|
| `Gauss upwind` | `dec2a499fd69` |
| `Gauss linear` | `4c774d8fd354` |
| `Gauss vanLeer` | `e3315377c41a` |
| `Gauss linearUpwind grad(U)` | `b9ce961dad61` |
| `Gauss QUICK` | `05413b401b03` |
| `Gauss totalGarbageScheme` | **에러, exit 1** |

---

## 무엇이 들어 있나

| 영역 | |
|---|---|
| **이산화** | Gauss linear / upwind / linearUpwind / cubic / QUICK / Gamma / blended, TVD 리미터 6종 (minmod, van Leer, van Albada, Superbee, MUSCL, Sweby), bounded 보정 |
| **기울기** | Green-Gauss, 최소제곱, cellLimited·faceLimited (Barth-Jespersen, Venkatakrishnan) |
| **snGrad** | uncorrected / corrected / limited α |
| **시간** | steadyState, Euler, BDF2 (가변 dt 포함), local time stepping. theta법은 구현됐으나 완화된 방정식에서 도달 불가 — 그 사실을 에러로 알립니다 |
| **압력-속도** | SIMPLE, SIMPLEC, PISO, PIMPLE (외부 보정자, 마지막 반복에서 완화 해제) |
| **난류 RAS** | k-ε, k-ω, **k-ω SST** |
| **난류 LES** | **Smagorinsky, WALE, Deardorff** + 필터 폭 (cube-root, max-edge, Scotti 이방성, van Driest 감쇠, 평활화) |
| **벽함수** | nutk, **nutU (Spalding 역해)**, **nutLowRe**, **거친 벽 (Cebeci-Bradshaw)**, epsilon, omega, kqR, **kLowRe** |
| **벽거리** | **Poisson 방법 (Tucker 1998)** — SST와 van Driest가 필요로 함 |
| **다상** | **VOF** — 계면 압축, Zalesak FCT 유계화, 서브사이클링, CSF 표면장력, p_rgh |
| **수송** | 온도, **화학종 N종 (합=1 강제)**, **체적 소스, Darcy-Forchheimer** |
| **부력** | 밀도비 `b = g(T_ref/T−1)`, **난류 생성항 G_b** |
| **선형 솔버** | PBiCGStab, PCG, **다색 DIC / DILU**, AMGX (feature), cuFFT 직접해 |
| **케이스 I/O** | OpenFOAM ASCII 형식, **정규식 패치 키** (`".*"`, `"(U\|k\|epsilon)"`) |

굵은 항목이 이번에 추가된 것입니다.

---

## 검증 — RTX 5070 Ti, double precision

```
ofgpu-validate     198 / 198 checks passed
cargo test         503 tests passed, 0 failed
```

다른 CFD 코드와 비교하는 검사는 **하나도 없습니다.**

**수렴 차수** — 인위해(MMS), 격자를 절반으로:

| 격자 | 조격자 L2 | 세격자 L2 | 관측 차수 |
|---|---|---|---|
| 3-D graded (10³ → 20³) | 7.943e-3 | 1.857e-3 | **2.10** |
| 3-D sheared (8³ → 16³) | 4.350e-3 | 1.154e-3 | **1.91** |
| 2-D empty patches (16² → 32²) | 4.075e-3 | 9.711e-4 | **2.07** |

**공개 벤치마크** — Ghia, Ghia & Shin, *JCP* 48 (1982) 387, 80×80 캐비티:

| Re | SIMPLE 반복 | 최대 \|Δu\| | 최대 \|Δv\| |
|---|---|---|---|
| 100 | 3,000 | 0.0046 | 0.0088 |
| 400 | 6,000 | 0.0067 | 0.0057 |

Table II의 Re=400, x=0.9063 한 점(−0.23827)은 **논문의 오식**이라 제외했습니다.
그 표 자체의 단조 구간을 깨고, Nilsson & Wallin (Uppsala 22015, 2022) §5.2도
같은 이유로 제외합니다. 원표는 편집 없이 두고 출력에 표시만 합니다.

**VOF** — 댐 붕괴 (6,000셀, 0.25초, 1,250스텝, 118초):

```
phase volume 1.256250e-05 -> 1.256250e-05   (상대 변화 1.35e-16)
alpha in [-4.163e-17, 1]
```

| 검사 | 결과 |
|---|---|
| Zalesak 회전 슬롯 원반: `min α ≥ 0` | 1.7e-18 |
| 같은 것: 상 체적 보존 | 3.9e-12 |
| 정지 액적 Laplace 압력 `σ/R` | 4.888 대 5.000 (2.2 %) |
| **밀폐 성층 탱크가 정지 유지** | max \|U\| 5.5e-11 m/s (√gH = 3.13) |

마지막 것이 `p_rgh` 정식화가 옳다는 유일한 결정적 증거입니다 — 안 쓰면 즉시
깨집니다.

**부력 생성항·소스·화학종**:

| 검사 | 결과 |
|---|---|
| G_b 부호: 안정 성층 (dT/dz > 0) → 음수 | 정확 |
| G_b 부호: 열원 위 (dT/dz < 0) → 양수 | 정확 |
| G_b 크기 | 1.6e-14 |
| 열원이 정확히 그 출력만큼 주입 | 2.3e-16 |
| 화학종 질량분율 합 = 1 | 0.0 |

**기계 정밀도** (발췌): 행렬 조립·완화·경계 folding·`Amul` 대 독립 CPU 구현
~2e-16; PCG/PBiCGStab 대 조밀 직접해 2.8e-15 / 1.1e-15; cuFFT Poisson 대 같은
행렬의 반복해 1.4e-15; 정수압 평형 6.6e-15.

---

## 무엇이 "GPU-native"인가

메쉬와 필드를 한 번 올린 뒤, 시간 루프 안에서:

- `cudaMalloc` 없음 — 모든 버퍼는 생성 시점에 한 번만 할당
- 필드 데이터의 `cudaMemcpy` 없음 — k, epsilon/omega, nut, U, phi, 행렬 전부 device 상주
- Krylov 솔버의 alpha/beta/omega/rho 같은 **제어 스칼라도 device에 있습니다.**
  1-스레드 커널이 계산하고 axpy 커널이 device 포인터로 바로 읽습니다.

host로 넘어가는 것은 두 가지뿐이고, 둘 다 필드가 아니라 **스칼라**입니다.

| What | Size | When | 끄는 법 |
|---|---|---|---|
| 선형 솔버 수렴 플래그 | 4 B | 솔버 반복 `checkInterval`회마다 | `-fixedIters N` |
| 잔차 로그 (`initRes`, `finalRes`, `converged`) | 3 × 8 B | 방정식을 다 푼 뒤 | `SolverControls::report_residuals = false` |
| 수렴 판정 `max dk/k` | 8 B | `-check N`회마다 | `-check`를 크게 |

`-fixedIters N`을 주면 앞의 둘이 사라집니다. 이 상태라야 시간 스텝 전체를
CUDA graph로 캡처할 수 있습니다.

면 루프에서 `diag[owner[f]] -= ...` 로 **흩뿌리는(scatter)** 연산은 전부
cell→face CSR을 통한 **모으기(gather)** 로 뒤집었습니다. double atomic이 필요
없고, 합산 순서가 고정되어 결과가 비트 단위로 재현됩니다.

## 압력 방정식 — 문제에 맞는 솔버를 골라냅니다

`PressureBackend` 트레이트 뒤에 세 가지 구현이 있고, 선택은 추측이 아니라
**측정**입니다.

| Backend | 적용 조건 | 비고 |
|---|---|---|
| PBiCGStab / PCG | 항상 | 기본값 |
| AMGX | 임의 비정렬 계 | BSD-3-Clause |
| cuFFT 직접 해 | 균일 직교 격자 + 상수계수 + 분리 가능한 BC | 조건을 만족할 때만 |

선택기는 (1) `applicable(&SystemProbe)`로 하드 필터를 걸고, (2) 정확도를
검증하고, (3) 실제 시간을 재서 고릅니다. cuFFT는 상수계수 Poisson에서 25.2배
빠르지만 SIMPLE의 압력 방정식은 `rAUf`가 셀마다 다르므로 선택기가 정확히
거부합니다 — 이것이 이 설계의 핵심입니다.

FFT 고유값은 연속형 `−k²`가 아니라 **이산형 `2(cosθ−1)/h²`** 를 씁니다. 그래야
같은 2차 라플라시안의 정확한 역이 되어 이산화 오차가 아니라 round-off까지
맞습니다 (`rust/SPEC-LIT.md` §8.5).

## 부력

Boussinesq는 `ΔT/T ≪ 1`을 요구하는데, 화재 플룸은 293 K 대 1173 K로
`ΔT/T ≈ 3`입니다. 그래서 쓰지 않고 밀도비를 그대로 남깁니다:

```
b = g·(T_ref/T − 1)
```

`g = (0,0,−9.81)`, `T_ref = 293.15 K`, `T = 1173.15 K` → `b = (0,0,+7.36)`.
`T = T_ref`에서 정확히 0입니다. 완전한 처리는 Rehm & Baum (1978)의 저마하
정식화이고, 그것을 구현한 FDS가 `reference/fds`에 있습니다 (퍼블릭 도메인).

## 성능 — RTX 5070 Ti (70 SM, 896 GB/s, double)

한 번의 outer iteration은 완전한 수송방정식 두 개입니다 — 조립, 완화,
벽함수 구속, Krylov 해.

| Mesh | ms / iter (k-ε) | ms / iter (k-ω) | Mcell-iter/s | Device memory |
|---|---|---|---|---|
| 80 k | 1.187 | 1.150 | 67 / 70 | 1.4 GB |
| 500 k | 3.427 | 3.400 | 146 / 147 | 1.9 GB |
| 2 M | 13.346 | 13.337 | 150 / 150 | 4.0 GB |

작은 메쉬에서는 커널 실행 오버헤드가 지배하고, 500 k 이상에서 메모리 대역폭
한계에 도달합니다.

### CUDA Graph — 24 k 셀, 200 iteration

| Mode | ms / iter | Mcell-iter/s |
|---|---|---|
| 적응형 (반복마다 4바이트 플래그) | 1.323 | 18.1 |
| 고정 반복, per-launch | 1.191 | 20.1 |
| 고정 반복, **CUDA graph** | **0.377** | **63.7** |

**3.16배**입니다. 캡처와 instantiate는 0.46 ms, 한 번뿐입니다. 그리고 결과는
per-launch 경로와 **24,000셀 전부가 비트 단위로 동일**합니다 — graph는 실행
순서를 바꾸지 않고 실행 오버헤드만 없애기 때문입니다.

작은 메쉬일수록 이득이 큽니다. 1.19 ms 중 0.81 ms가 순수 실행 오버헤드였다는
뜻이고, 이것이 `-fixedIters`로 host 전송을 0으로 만드는 이유입니다 — 전송이
남아 있으면 캡처가 불가능합니다.

### 압력 백엔드 선택 — 82,320셀 플룸 격자

선택기가 실제로 측정한 것:

```
uniform cartesian    (98, 42, 20), h = (0.1494, 0.1486, 0.15)
separable bcs        true      symmetric  true      constant coefficient  true

  PBiCGStab   applicable    51.13 ms   (reference)      residual 7.19e-12
  cuFFT       applicable     2.05 ms   agrees to 8.0e-11
  AMGX        unavailable              feature 'amgx' not enabled

chosen: cuFFT   —  25.0x
```

전송을 끈 cuFFT는 **0.86 ms**까지 내려갑니다. 두 해의 상대 차이는 1.5e-14 —
같은 행렬의 정확한 역이라는 뜻입니다.

이 25배는 **상수계수 Poisson에서만** 나옵니다. SIMPLE의 압력 방정식은 `rAUf`가
셀마다 다르므로 선택기가 cuFFT를 정확히 거부합니다. 그게 이 설계의 요점입니다.

## 런타임 선택은 공짜인가

측정했습니다 (`cargo run --release --bin ofgpu-dispatch-bench`).

| Granularity | 비용 |
|---|---|
| 커널 **실행**당 가상 호출 | 측정 불가 수준 (노이즈 내) |
| **요소**당 가상 호출 | 1.75–1.80배 느림 |

이 솔버의 모든 런타임 선택 — SIMPLE/PISO, k-ε/k-ω/SST, 솔버 백엔드 — 은 실행
단위에 있습니다. 조합마다 빌드할 필요가 없습니다.

---

## 문서

| File | Contents |
|---|---|
| [`rust/SPEC-LIT.md`](rust/SPEC-LIT.md) | **수치 명세.** 모든 수식이 논문/교과서로 인용됨. 구현자는 이것만 봅니다 |
| [`rust/PROVENANCE.md`](rust/PROVENANCE.md) | **파일별 출처.** 어느 파일이 어느 논문에서 나왔는지, 무엇이 우리 설계인지 |
| [`LICENSING.md`](LICENSING.md) | 라이선스 감사와 재작성 계획 |
| `docs/01-model-catalog.md` | CFD 모델 전수 목록 (1,823개 구성요소) |
| `docs/02-gpu-portability.md` | 같은 구성요소들의 GPU 이식 등급 (A~E)과 맞는 NVIDIA 라이브러리 |
| `docs/03-esi-vs-foundation.md` | 두 상류 배포판의 모델 구성 차이 |
| `cases/README.md` | 테스트 케이스별 기하·속도장 설명 |

`docs/01`과 `02`는 `docs/_build_catalog.py`가 조사 에이전트들의 구조화된
출력에서 **결정적으로** 생성합니다. 요약이 아니라
전수입니다 — 데이터의 모든 항목이 정확히 한 행이 됩니다.

### GPU 이식성 분류 — 구성요소 1,824개

| Tier | 뜻 | 맞는 라이브러리 | 개수 |
|---|---|---|---:|
| **A-trivial** | 요소별 또는 면 스텐실. 순수 CUDA 커널로 device 상주 가능 | 없음 (직접 커널) | 899 |
| **B-sparse-solve** | 비용이 sparse 선형해에 지배됨. 조립부는 A등급 | AMGX, cuDSS, cuSPARSE | 85 |
| **C-fft** | 스펙트럴/합성곱 | cuFFT | 7 |
| **D-hard** | 불규칙·직렬·위상변경·탐색 위주 | — | 214 |
| **E-cpu-only** | I/O, 딕셔너리, 런타임 선택, 분할 설정 | — | 619 |

**A + B = 984개 (54 %)** 가 완전 device 상주 경로에 올라가고, 난류 모델 전체와
압력 방정식이 그 안에 있습니다. E등급 619개는 애초에 시간 루프 안에 없습니다.
실제로 시간 루프를 막는 것은 **D등급 214개**, 대부분 Lagrangian 추적과 메쉬
위상 변경입니다.

`cuFFT`가 맞는 곳이 7개뿐인 것은 비정렬 격자 FV 코드의 핵심 루프에 FFT가 없기
때문입니다. `cuDSS`는 큰 압력계가 아니라 셀별 화학 반응 Jacobian 같은 작은
dense 계가 표적입니다.

---

## 빌드

필요한 것: Rust 1.85 stable, Visual Studio 2022 (C++ 워크로드), CUDA Toolkit 13.x.

```powershell
cd rust
cargo build --release
cargo test  --release
```

`build.rs`가 `vcvars64.bat`을 실행해 MSVC 환경을 잡고, `nvcc`로 각 `.cu`를
**CUBIN**(PTX 아님)으로 컴파일해 `OUT_DIR`에 넣습니다. 드라이버가 보고하는 CUDA
버전보다 툴킷이 새로우면 PTX는 `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`으로 죽기
때문입니다.

`-Xcompiler=/Zc:preprocessor`가 필요합니다 — CUDA 13의 CCCL 헤더가 MSVC의
전통적 전처리기에서 `fatal error C1189`를 냅니다.

| Binary | Purpose |
|---|---|
| `ofgpu-validate` | GPU 커널 대 독립 CPU 구현, MMS 수렴 차수 |
| `ofgpu-bench` | 처리량·메모리 벤치마크 |
| `ofgpu-graph-bench` | CUDA graph 캡처 대 스트림 실행 |
| `ofgpu-dispatch-bench` | 런타임 선택의 실제 비용 |
| `ofgpu-probe` | 장치 속성과 최소 커널 왕복 |
| `ofgpu-generate-mesh` | 바로 실행 가능한 케이스 생성 |
| `ofgpu-k-epsilon`, `ofgpu-k-omega` | 난류 모델 단독 실행 |
| `ofgpu-plume`, `ofgpu-buoyant` | 부력 플룸 |
| `ofgpu-vof` | 2상 VOF (댐 붕괴) |

## 실행

```powershell
cd rust
cargo run --release --bin ofgpu-generate-mesh -- channel ..\cases\channel 200 120 1
cargo run --release --bin ofgpu-k-epsilon    -- ..\cases\channel -iters 4000 -check 400
cargo run --release --bin ofgpu-k-omega      -- ..\cases\channelKW -iters 4000 -check 400
cargo run --release --bin ofgpu-generate-mesh -- damBreak ..\cases\damBreak 60 100 1
cargo run --release --bin ofgpu-vof          -- ..\cases\damBreak -endTime 0.25 -surge
cargo run --release --bin ofgpu-validate
```

| Flag | Meaning |
|---|---|
| `-iters N` | outer iteration 수 (기본값은 `controlDict`의 `endTime`) |
| `-fixedIters N` | 선형 솔버를 정확히 N번 돌리고 잔차를 읽지 않음 — host 전송 0 |
| `-check N` | N회마다 수렴 판정 |
| `-write NAME` | 결과를 쓸 시간 디렉터리 |
| `-noWrite` | 결과를 쓰지 않음 (타이밍용) |
| `-permissive` | 지원하지 않는 설정을 에러 대신 경고로 낮추고, **무엇으로 대체했는지 출력** |

생성 가능한 케이스: `channel`, `cavity`, `step`, `big`, `plume`, `damBreak`.

난류 모델은 `constant/momentumTransport`의 `RAS { model ...; }` 또는
`simulationType LES;`가 정합니다 — 어느 바이너리를 실행했느냐가 아니라.
없는 이름을 쓰면 있는 목록을 알려주는 에러가 납니다.

케이스는 **OpenFOAM ASCII 형식**으로 읽고 씁니다 — `constant/polyMesh`, `0/`,
`constant/physicalProperties`, `constant/momentumTransport`,
`system/{fvSolution,fvSchemes,controlDict}`. 기존 도구 체인(ParaView,
`foamToVTK`)을 그대로 쓸 수 있게 하려는 상호운용 목적이며, ofgpu는 OpenFOAM의
어떤 부분과도 링크하지 않습니다. 바이너리 케이스는 먼저 ASCII로 변환해야 합니다.

---

## 지금 하지 않는 것

- **MPI 다중 GPU는 없습니다.** 단일 GPU 전용입니다.
- **AMGX는 빌드해 두었으나 `amgx` Cargo feature가 기본 비활성입니다.**
  NVIDIA의 Windows 지원이 제한적이고 검증된 최신 툴킷이 CUDA 12.2인데 이
  기계는 13.3입니다. 꺼진 상태에서도 선택기는 AMGX를 "unavailable"로
  **보고합니다** — 후보가 아니었던 척하지 않습니다.
- **Crank-Nicolson은 구현됐지만 완화된 방정식에서 도달할 수 없습니다.**
  theta 가중과 암시적 완화가 조립의 같은 자리를 원하고, 완화는 가중되지 않은
  대각을 봐야 합니다. 조용히 Euler로 떨어뜨리지 않고 그 사실을 에러로
  알립니다.
- **압축성·천음속은 없습니다.** `fv.rs`에 밀도 가중 시간미분이 있고 VOF가
  쓰지만, 압력 방정식은 비압축성입니다.
- **화학반응·복사는 없습니다.** 체적 소스로 열 방출은 넣을 수 있습니다.
- **cyclic 패치의 비직교 보정 벡터가 없습니다.** 직교 메쉬에서는 영향 없음.

## 라이선스 문의

**simul@msimul.com**

주식회사 메테오시뮬레이션 / Meteo Simulation Co., Ltd.

교육 목적은 무료입니다. 연구·논문·상업 이용은 라이선스가 필요합니다 —
[`LICENSE`](LICENSE) 제2·3절. 개인 연구자와 비영리 목적에는 무상 또는 감면을
검토합니다.
