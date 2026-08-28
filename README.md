# meteor-cfd

**GPU 상주 유한체적 전산유체역학 솔버**

주식회사 메테오시뮬레이션 · Rust 호스트 + CUDA 커널

[English](README.en.md)

---

## 개요

meteor-cfd는 시간 적분 루프 전체가 GPU에 머무르도록 설계된 비정렬 격자 유한체적
CFD 솔버입니다. 메쉬와 필드를 한 번 업로드한 뒤에는 시간 루프 안에서 장치 메모리
할당도, 필드 데이터의 호스트 전송도 발생하지 않습니다.

수치 코어 전체를 공개 문헌으로부터 직접 구현했으며, 모든 수식은
[`rust/SPEC-LIT.md`](rust/SPEC-LIT.md)에 원논문 인용과 함께 명세되어 있습니다.
검증은 인위해법(MMS), 해석해, 공개 벤치마크만을 사용하며 다른 CFD 코드와
비교하지 않습니다.

| 항목 | |
|---|---|
| 언어 | Rust 1.85 (호스트) / CUDA C++ (커널) |
| 정밀도 | 배정밀도 기본, `single` 기능으로 단정밀도 |
| 대상 | NVIDIA GPU |
| 의존성 | cudarc, thiserror (선택적 AMGX) |
| 검증 | 790개 단위 시험, 250개 수치 검증 항목 |

---

## 라이선스

**소스는 공개하되 오픈소스는 아닙니다.**

| 용도 | 조건 |
|---|---|
| 교육 — 수업, 과제, 실습, 독학 | **무료** |
| **학술 연구** — 대학·학교 및 그 소속 연구소·연구실 | **무료** |
| 학술 연구 결과의 논문·학위논문·발표 | **무료** (출처 표기 조건) |
| 기업 내부 연구개발 | 유상 라이선스 |
| 학교에 속하지 않은 국가·정부출연 연구기관 | 유상 라이선스 |
| 컨설팅, 수탁 연구, 시험 대행 | 유상 라이선스 |
| 실제 제품·설비·시스템의 설계·인증·운용 | 유상 라이선스 |

기업이 연구비를 지원하더라도 **결과를 공개 발표하는 연구는 무료 범위**입니다.
결과가 비공개이거나 스폰서가 독점권을 갖는 수탁 연구만 유상 라이선스 대상입니다.

연구 결과를 발표하실 때는 다음과 같이 출처를 표기해 주십시오. 이는 라이선스
조건입니다.

```
meteor-cfd, 주식회사 메테오시뮬레이션, https://github.com/using76/meteor-cfd
```

국가연구소, 병원 연구팀, 비영리 기관, 대학 스핀오프 등 적용 범위가 불분명한
경우 문의해 주시기 바랍니다. 실질이 상업적이지 않은 이용에 대해서는 무상 또는
감면 라이선스를 검토합니다.

**라이선스 문의: simul@msimul.com**

2036년 8월 23일 이후, 해당 시점에 공개된 버전은 Apache License 2.0으로
전환됩니다. 전문은 [`LICENSE`](LICENSE), 서드파티 고지는 [`NOTICE`](NOTICE)를
참조하십시오.

> 본 라이선스는 Business Source License 1.1의 구조를 차용하였으나 BUSL이
> 아닙니다. BUSL은 모든 비생산 이용을 무상으로 허용하며 여기에는 기업 내부
> 연구개발이 포함됩니다. 본 라이선스에서 무상 범위는 교육과 학술 연구로
> 한정됩니다.

---

## 기능

### 이산화

| 구분 | 지원 |
|---|---|
| 대류 항 | Gauss linear, upwind, linearUpwind, cubic, QUICK, Gamma, blended |
| TVD 제한자 | minmod, van Leer, van Albada, Superbee, MUSCL, Sweby-φ |
| 기울기 | Green–Gauss, 최소제곱, cellLimited·faceLimited (Barth–Jespersen, Venkatakrishnan) |
| 면법선 기울기 | uncorrected, corrected, limited α |
| 시간 적분 | steadyState, Euler, BDF2 (가변 시간간격 포함), 국소 시간전진 |
| 확산 항 | 과이완(over-relaxed) 비직교 보정, 비직교 보정자 반복 |

### 압력–속도 결합

SIMPLE, SIMPLEC, PISO, PIMPLE을 지원합니다. Rhie–Chow 보간을 사용하며, 체적력은
셀 값을 보간하지 않고 면에서 직접 처리합니다.

### 난류 모형

| 구분 | 모형 |
|---|---|
| RANS | 표준 k-ε, Wilcox k-ω, Menter k-ω SST, Launder-Sharma 저레이놀즈수 k-ε (SPEC-LIT §33 — `wallTreatment lowRe`가 유효한 유일한 모형; 감쇠함수가 점성 서브레이어까지 적분되며, 해석적 `Re_t` 극한과 실제 채널 유동의 `u+`/`y+` 벽법칙 양쪽으로 검증됨) |
| LES | Smagorinsky, WALE, Deardorff |
| LES 여과폭 | 체적 세제곱근, 최대 모서리 길이, Scotti 이방성 보정, van Driest 감쇠 |
| LES 벽모형 | Werner–Wengle (1991) — 첫 셀 평균 속도로부터 적분·역해한 멱법칙, Newton 반복 없음; `standard`/`spalding` 프리셋이 LES에서는 이 모형 하나로 수렴, `lowRe`는 `nu_t,w = 0`, `rough`는 아직 없음(§13.4 오류로 명시) |
| 벽함수 | nutk, nutU (Spalding 역해), nutLowRe, 조도벽 (Cebeci–Bradshaw), epsilon, omega, kqR, kLowRe |
| 결합 솔버(`ofgpu-buoyant`, `ofgpu-fire`)의 난류 선택 | `ofgpu-buoyant`: `CoupledTurbulence` 트레이트로 케이스의 `RAS { model ...; }`/`simulationType`을 그대로 반영 — k-ε, k-ω, k-ω SST(벽거리 자동 계산), LES(Smagorinsky/WALE/Deardorff, §16 여과폭·van Driest 포함) 전부 실제로 그 모형을 구성함, 부력 생성 `G_b`도 모형별로 올바른 방정식에 배선됨(§17, §30.2). `ofgpu-fire`: 아직 k-ε만 지원 — 연소 혼합시간 종결식과 열 벽함수가 `epsilon`을 직접 요구하므로 다른 모형은 이름을 밝힌 §13.4 오류로 거부(조용한 대체 없음) |
| 벽 모형 프리셋 (`wallTreatment`) | `standard`/`spalding`/`rough`/`lowRe` — 설정 하나가 케이스 빌드 시점에 필드별(nut/k/epsilon/omega, 에너지 방정식을 풀 때는 T까지) 경계 타입의 일관된 한 행으로 전개됨; 서로 다른 행을 섞으면 이름을 명시해 거부, `-permissive`는 `nut` 선택이 함의하는 행으로 대체 (SPEC-LIT §29.1). `lowRe`는 추가로 벽 근처를 해석할 수 있는 난류 모형을 요구함 — 그 목록에 있는 모형은 `LaunderSharmaKE`(SPEC-LIT §33) 하나뿐이며, `kEpsilon`/`kOmega`/`kOmegaSST`는 여전히 해당하지 않으므로 이 셋 아래에서 `lowRe`는 발산하도록 두는 대신 이름을 명시해 거부함 (SPEC-LIT §32) |
| 열 벽함수 | Jayatilleke의 열 대수법칙 하위층 저항 보정 (`thermalWallFunction`, 별칭 `compressible::alphatJayatillekeWallFunction`) — `lowRe`를 제외한 모든 프리셋 행이 벽의 `T`에 적용 (`lowRe`는 해상된 하위층 자체의 분자 저항을 그대로 둠, SPEC-LIT §29.3); `ofgpu-fire`의 에너지 방정식에 배선됨. 고정 열유속 주기 덕트에서 Dittus-Boelter/Gnielinski 대비 **검증 완료** — 두 밴드 모두 안쪽인 +2%/−4% (SPEC-LIT §32) |
| 벽거리 | Poisson 방정식 기반 (Tucker 1998) |
| 부력 생성 | G_b 항 (Rodi 1987, Henkes et al. 1991) |

### 다상 및 수송

| 구분 | 지원 |
|---|---|
| VOF | 계면 압축, Zalesak FCT 유계화, 하위 순환, CSF 표면장력, p_rgh 정식화 |
| 스칼라 수송 | 온도, 다성분 화학종 (질량분율 합 = 1 강제) |
| 소스 항 | 체적 열원, 운동량 소스, Darcy–Forchheimer 다공성 저항 |
| 부력 | 비Boussinesq 밀도비 `b = g(T_ref/T − 1)` |

### 선형 해법

| 구분 | 지원 |
|---|---|
| Krylov | PBiCGStab, PCG |
| 전처리기 | 없음, Jacobi, 다색(multi-colour) DIC, 다색 DILU |
| 압력 backend | 반복법, cuFFT 직접해, AMGX (선택적 기능) |
| Backend 선택 | 적용 가능성 판정 → 정확도 검증 → 실측 시간 비교 |

### 메쉬 생성

| 구분 | 지원 |
|---|---|
| 블록 격자 | 케이스별 구조 격자 — grading, 경계 패치, `0/` 필드까지 한 번에 생성 |
| STL 장애물 | 이진·ASCII STL로 블록 격자를 계단식(castellated)으로 조각 — 닫힘 검증(열린 모서리 개수 보고 후 거부, `-permissive`는 패리티 투표로 진행), 열 단위 패리티 판정과 3축 다수결, 새 wall 패치에 기존 벽 경계조건 자동 부여 (`-stl [name=]path`, Aftosmis et al. 1998; Barill et al. 2018) |
| 잘림 셀 (cut cell) | 계단식 다음 단계 — 교차 셀을 제거하지 않고 부피/면적 분율과 닫힘식으로 정의된 절단면을 부여 (슈퍼샘플링, 기본 16³, SPEC-LIT §24); `theta_min` 미만인 얇은 셀은 가장 넓은 유체 면을 공유하는 이웃에 병합 |
| Gmsh `.msh` | v4.1 사면체·육면체·프리즘·피라미드 요소, `$PhysicalNames` 패치 읽기 — 공개 포맷 사양으로부터 구현 |
| Cyclic 패치 | 블록의 마주보는 두 면을 경계 대신 결합 — `ofgpu-generate-mesh -cyclic x\|y\|z` 또는 JSONC `mesh.cyclic` 항목으로 양쪽 이름과 변환(`translate`만 지원, `rotate`는 이름을 밝혀 거부)을 지정. 변환된 중심점이 가장 가까운 면끼리 짝짓고, 두 불변식으로 검증 — 모든 면이 정확히 한 번씩 짝지어지는지(전단사), 변환 후 `Sf_a == -Sf_b`인지(지정 허용오차) — 둘 중 하나라도 어긋나면 아무것도 보존하지 않는 격자를 조용히 만드는 대신 패치 쌍과 가장 나쁜 면을 이름으로 밝혀 거부합니다 |

### 화재 물리 (저-마하 가변밀도, 연소, 복사)

`ofgpu-fire`가 SPEC-LIT §25–28을 하나로 결합합니다.

| 구분 | 지원 |
|---|---|
| 저-마하 정식화 | `p = p0(t) + p~(x,t)` 분리, 발산 구속조건, 밀폐/개방 공간의 `p0(t)` 적분 (Rehm & Baum 1978) |
| 에너지 방정식 | 현열 엔탈피, `k_eff = k + rho cp nu_t/Prt`, 벽 열유속·고정온도 경계, `thermalWallFunction` 벽의 Jayatilleke 열 벽함수 (SPEC-LIT §29.3) |
| 연소 | 혼합제어 단일 스텝 EDM (Magnussen & Hjertager 1977) — `Y_F`, `Y_O2`, `Y_P` 수송, 연료 고갈 방지 클리핑, 소모된 연료질량과 방출열이 정확히 일치 (반올림 오차 수준) |
| 복사 | 회색 P1 근사 (Modest), Marshak 벽 경계, `chi_r` 복사분율 하한 — `fvDOM`은 §13.4 계약에 따라 이름을 명시하며 거부 |
| 검증 게이트 | 밀폐 상자 `dp0/dt` 램프(해석해), 버너 정확 열방출, 복사 평형, 컷셀 닫힘, msh 육면체 닫힘 — 전부 `ofgpu-validate`의 상시 항목 |
| 필드 출력·재시작 | `-output foam,vtu,nvdb,vdb,usda`·`-writeInterval`이 `ofgpu-buoyant`/`ofgpu-vof`와 같은 방식으로 `U`·`p`·`T`·난류 완결식·화학종 필드를 씀; `-restartWrite N`/`-restartFrom FILE`이 체크포인트를 쓰고 재개 — `U`/`p`/`T`뿐 아니라 `p0`, `dp0dt`, 화학종 질량분율까지 재시작에서 그대로 이어받습니다(저-마하 실행의 열역학 상태는 그 세 필드만이 아니므로). 연속 40스텝 실행과 20스텝+재시작+20스텝 실행이 재시작 직후 첫 압력 잔차·`p0`·전체 엔탈피에서 일치합니다 |
| 체적 소스 | `sources[]`(JSONC) 또는 `constant/fvSources`(OpenFOAM 케이스 디렉터리)로 운동량 방정식에 소스를 등록 — 전체 도메인에 걸친 균일 체적력으로, 유입 경계가 없어 질량유량을 지정할 수 없는 periodic(cyclic 패치) 케이스가 필요로 하는 바로 그것입니다 |

### 케이스 입력 형식과 재시작

| 구분 | 지원 |
|---|---|
| JSONC 케이스 | 주석·trailing comma를 허용하는 JSON 한 파일로 메쉬·물성·경계·수치·소스·화재 블록을 기술 — `schemars`로 스키마를 자동 생성해 리더와 스키마가 어긋날 수 없음 |
| 재시작 (`.mcr`) | 전체 배정밀도, `phi` 포함, 메쉬 해시 불일치 시 거부, 버전 관리 |
| 시각화/교환 출력 | VTU(부가 이진, 폴리헤드라 보존), NanoVDB/OpenVDB(`.vdb`/`.nvdb`), USD(`.usda`) 장면 참조 |

---

## GPU 상주 설계

메쉬와 필드를 한 번 업로드한 뒤 시간 루프 안에서 다음이 성립합니다.

- `cudaMalloc` 호출 없음 — 모든 버퍼는 생성 시점에 일괄 할당
- 필드 데이터의 `cudaMemcpy` 없음 — 필드, flux, 행렬 전부 장치 상주
- Krylov 해법의 제어 스칼라(α, β, ω, ρ, 잔차)도 장치 메모리에 상주하며, 단일
  스레드 커널이 갱신하고 axpy 커널이 장치 포인터로 직접 참조

호스트로 전송되는 것은 다음 두 스칼라뿐입니다.

| 항목 | 크기 | 시점 | 비활성화 |
|---|---|---|---|
| 선형 해법 수렴 플래그 | 4 B | `checkInterval` 반복마다 | `-fixedIters N` |
| 잔차 기록 | 3 × 8 B | 방정식 해석 완료 시 | `-fixedIters N` |

`-fixedIters` 지정 시 호스트 전송이 완전히 사라지며, 이 상태에서만 시간 단계
전체를 CUDA Graph로 캡처할 수 있습니다.

면 루프에서 대각 성분을 누적하는 연산은 모두 cell→face CSR을 통한 **gather**로
구현하였습니다. 배정밀도 atomic 연산이 불필요하고, 합산 순서가 고정되어 결과가
비트 단위로 재현됩니다.

---

## 케이스 설정 처리 원칙

지원하지 않는 설정을 임의의 다른 값으로 대체하지 않습니다. 케이스 파일의 모든
항목은 다음 세 가지 중 하나로 처리됩니다.

| 상태 | 동작 |
|---|---|
| 지원하는 설정 | 그대로 적용 |
| 인식되나 미구현 | 설정 이름과 사용 가능한 대안을 명시한 오류 |
| 인식 불가 | 설정 이름을 명시한 오류 |

```
error: divSchemes/div(phi,k): "Gauss totalGarbage" is not supported by ofgpu;
       available: Gauss linear, Gauss upwind, Gauss linearUpwind [grad],
       Gauss cubic, Gauss QUICK, Gauss QUICKUnlimited, Gauss Gamma <0.1..0.5>,
       Gauss blended <0..1>, Gauss linearUpwindBlended <0..1>,
       Gauss limitedLinear <1..2>, Gauss vanLeer, Gauss vanAlbada,
       Gauss Minmod, Gauss SuperBee, Gauss MUSCL
  (run with -permissive to substitute Gauss upwind and continue)
```

`-permissive` 옵션이 유일한 예외이며, 이 경우에도 대체한 내용을 매번 출력합니다.

동일 케이스에서 이산화 기법만 변경하였을 때 각각 다른 결과가 산출됨을
확인하였습니다.

| `divSchemes` 항목 | `0/k` 해시 |
|---|---|
| `Gauss upwind` | `dec2a499fd69` |
| `Gauss linear` | `4c774d8fd354` |
| `Gauss vanLeer` | `e3315377c41a` |
| `Gauss linearUpwind grad(U)` | `b9ce961dad61` |
| `Gauss QUICK` | `05413b401b03` |
| `Gauss totalGarbageScheme` | 오류 종료 |

같은 세 갈래 원칙은 항목 하나가 아니라 설정 조합 전체에도 적용됩니다 — `run.endTime`이
0보다 크고 `ddt`가 `steadyState`가 아닌 과도(transient) 케이스가 정상상태 알고리즘
`SIMPLE`을 지정하거나, 반대로 정상상태 케이스가 과도 알고리즘 `PISO`/`PIMPLE`을
지정하면, 존재하지 않는 정상상태를 향해 완화(under-relaxation)를 거는 대신 이름을
밝힌 오류로 거부합니다 — `cases/burnerPlume.jsonc`가 정확히 이 경로로 20스텝
근처에서 `Inf`에 도달했습니다. `endTime`, `ddt`, 알고리즘 딕셔너리가 각각은
개별적으로 유효했고 아무것도 경고하지 않았습니다.

```
error: numerics/algorithm: "SIMPLE (ddt "Euler", endTime 6)" is a steady
       algorithm on a transient case (endTime > 0 and ddt is not steadyState)
  available for a transient run: PISO, PIMPLE
  (run with -permissive to substitute PIMPLE with one outer corrector and continue)
```

---

## 검증

```
cargo test        724 passed, 0 failed (lib 크레이트 기준; 각 바이너리의 소소한 CLI 파싱 스위트까지
                  전부 합치면 776)
ofgpu-validate    240 / 240 checks passed
```

### 수렴 차수 — 인위해법 (MMS)

`−∇²ψ = f`, 격자 간격 1/2 세분화.

| 격자 | 조격자 L2 | 세격자 L2 | 관측 차수 |
|---|---|---|---|
| 3차원 비균일 (10³ → 20³) | 7.943 × 10⁻³ | 1.857 × 10⁻³ | **2.10** |
| 3차원 전단 (8³ → 16³) | 4.350 × 10⁻³ | 1.154 × 10⁻³ | **1.91** |
| 2차원 empty 패치 (16² → 32²) | 4.075 × 10⁻³ | 9.711 × 10⁻⁴ | **2.07** |

### 공개 벤치마크

**뚜껑구동 공동유동** — Ghia, Ghia & Shin (1982) Table I·II, 80 × 80 격자.

| Re | SIMPLE 반복 | 운동량 잔차 | 최대 \|Δu\| | 최대 \|Δv\| |
|---|---|---|---|---|
| 100 | 3,000 | 1.011 × 10⁻⁴ | 0.0046 | 0.0088 |
| 400 | 6,000 | 7.382 × 10⁻⁴ | 0.0067 | 0.0057 |

Table II의 Re = 400, x = 0.9063 지점(−0.23827)은 논문의 오식으로 판단하여
비교에서 제외하였습니다. 해당 값은 논문 자체 표의 단조 구간(x = 0.9453의
−0.22847에서 최소값 x = 0.8594의 −0.44993으로 이어지는 구간)을 위배하며,
Nilsson & Wallin (2022) §5.2도 동일한 사유로 제외합니다. 원표는 수정하지 않고
보존하며 출력 시 해당 지점을 표시합니다.

### VOF

**댐 붕괴** — 6,000 셀, 0.25초, 1,250 시간단계, 118초 소요.

```
phase volume 1.256250e-05 → 1.256250e-05    (상대 변화 1.35 × 10⁻¹⁶)
alpha in [-4.163e-17, 1]
```

| 검증 항목 | 결과 |
|---|---|
| Zalesak 회전 슬롯 원반: min α ≥ 0 | 1.7 × 10⁻¹⁸ |
| Zalesak 회전 슬롯 원반: 상 체적 보존 | 3.9 × 10⁻¹² |
| 정지 액적 Laplace 압력 σ/R | 4.888 대 5.000 (오차 2.2 %) |
| 밀폐 성층 탱크 정지 유지 | max \|U\| = 5.5 × 10⁻¹¹ m/s (√gH = 3.13) |

마지막 항목은 p_rgh 정식화의 타당성을 판정하는 결정적 시험입니다.

### 부력 생성, 소스, 화학종

| 검증 항목 | 결과 |
|---|---|
| G_b 부호 — 안정 성층 (dT/dz > 0)에서 음수 | 일치 |
| G_b 부호 — 열원 상부 (dT/dz < 0)에서 양수 | 일치 |
| G_b 크기 | 1.6 × 10⁻¹⁴ |
| 열원의 주입 열량 정확도 | 2.3 × 10⁻¹⁶ |
| 화학종 질량분율 합 = 1 | 0.0 |

### 기계 정밀도 검증

| 검증 항목 | 오차 |
|---|---|
| 행렬 조립 (대각·상·하·소스·경계계수) 대 독립 CPU 구현 | ~2 × 10⁻¹⁶ |
| 저이완, 경계항 병합, 행렬–벡터 곱 | ~3 × 10⁻¹⁶ |
| PCG / PBiCGStab 대 조밀 직접해 | 2.8 × 10⁻¹⁵ / 1.1 × 10⁻¹⁵ |
| cuFFT Poisson 직접해 대 동일 행렬의 반복해 | 1.4 × 10⁻¹⁵ |
| 정수압 평형 | 6.6 × 10⁻¹⁵ |

CPU 참조 구현은 장치 코드가 gather 방식인 것과 달리 의도적으로 scatter 루프로
작성하였습니다. 구조가 서로 다른 두 구현이 일치할 때 비로소 검증으로서 의미를
갖습니다.

### 벽 처리 (SPEC-LIT §29)

`wallTreatment` 프리셋과 Jayatilleke 열 벽함수에 대한 상시 `ofgpu-validate`
게이트 두 가지:

| 검증 항목 | 결과 |
|---|---|
| `Ks → 0`가 매끈한 `nutk` 벽함수를 항상 재현 | 0 (반올림 수준) |
| `Ks → 0`가 매끈한 `nutU` 벽함수를 항상 재현 | 0 (반올림 수준) |
| `P(Pr/Pr_t = 1) = 0` 정확히 성립 | 0 (반올림 수준) |
| `Pr = Pr_t`일 때 `T+ == Pr_t · u+` 어디서나 성립 | 1.3 × 10⁻¹⁶ |
| `thermalWallFunction`의 Robin 삼중값이 Jayatilleke 해석 열유속을 정확히 부호화 (한 셀 전도 항등식) | 0 (반올림 수준) |
| Werner-Wengle: 두 분기가 분기점에서 일치하고, 각 분기 자체의 닫힌식이 조작된 `tau_w`를 반올림 수준까지 재현 | 0 (반올림 수준) |
| 결합 솔버 선택: `ofgpu-buoyant`의 `build_coupled`로 부력 케이스에서 만든 `kOmegaSST`의 `nut` FNV 해시가 동일 케이스의 `kEpsilon`과 다름 | 해시 불일치 (결정적) |
| 열 벽함수 게이트, Nusselt 검증치(재현) — `cases/channelPeriodicFluxWF.jsonc` 자체 측정값을 Dittus-Boelter/Gnielinski와 비교 | +2% / −4% (±10% / ±20–25% 밴드 모두 안쪽) |

이 게이트들은 조도벽 법칙이 `Ks = 0`(조도를 전혀 언급하지 않는 케이스)에서
기존 매끈한 벽으로 붕괴함과, 열 벽함수 자체의 대수가 내부적으로 정확함과,
결합 솔버에서 `kOmegaSST`를 요청한 케이스가 실제로 그 모형을 구성함을
입증합니다. 이것만으로는 조(粗)격자 벽함수 메쉬가 독립적으로 발표된
상관식과 실제 유동의 벽 열유속에 합의한다는 것까지 입증하지는 않습니다 —
이 주장은 실제로 돌려야 확인할 수 있었고, SPEC-LIT §32의 재설계된 게이트가
비로소 그것을 검증 가능하게 만들었습니다: 이전 세 번의 고정벽온도 시도
(비율 0.095, 0.381, 0.107)는 사실 서로 다른 문제를 푼 두 실행을 비교하고
있었습니다 — 고정된 `T_w`는 벌크 온도를 자유롭게 두므로, 벽 근처 전도율이
다른 두 메쉬는 서로 다른 ΔT에 정착합니다.

**검증 완료.** 대신 두 메쉬에 동일한 벽 열유속 `q_w`를 고정하고 — 각자
자기 자신의 ΔT를 예측하게 하고 — 그 결과를 Dittus & Boelter (1930)와
Gnielinski (1976) 대비 Nusselt 수로 비교하면 벽함수 쪽 게이트가 닫힙니다.
`cases/channelPeriodicFluxWF.jsonc`(`standard` wallTreatment, y+ ≈ 41.7):
`q_w` = 500 W/m², `T_w` = 501.99 K(열 벽함수가 진단), `T_b` = 453.97 K,
`U_b` = 3.512 m/s. 가열 둘레 기준 수력직경(`D_h` = 0.08 m — 네 벽 중 두
벽만 가열되는 덕트에 대해 이 상관식들이 유도된 관례)으로 Re = 18,731이고,
측정된 Nu = 50.41은 **Gnielinski의 +2%**(±10%)와 **Dittus-Boelter의
−4%**(±20–25%) — 두 밴드 모두 안쪽입니다. 힘-균형 교차검증은
`U_b/u_tau` = 12.6을 주는데, 완전발달 평면 채널이 주는 15–17보다 낮습니다 —
이 덕트의 옆벽이 평면-채널 상관식이 모르는 저항을 더하기 때문이며, 숨기지
않고 그대로 보고합니다. `ofgpu-validate`의
`check_thermal_wall_function_gate_verdict_replay`가 이 측정치를 매 실행마다
영구적으로 재현합니다.

**`LaunderSharmaKE`가 도입된 지금(SPEC-LIT §33), 다른 이유로 여전히 열려
있음.** 모형 자체는 검증을 통과합니다 — 감쇠함수의 해석적 극한은
정확하며(`ofgpu-validate`), 벽이 해상된 페리오딕 채널에서 실제로 돌려 보면
점성 서브레이어 `u+ = y+`를 1% 이내로, 로그 법칙을 y+ ≈ 30–35에서 1% 이내로
재현합니다 — SPEC-LIT §33.3이 요구하는 벽법칙 검증입니다. 하지만 대응하는
해상 메쉬 `cases/channelPeriodicFluxLowRe.jsonc`(동일한 `q_w`, y+ ~ 1까지
격자 조정)는 자신의 3차원 덕트 형상 위에서는 여전히 쓸 만한 상태로 수렴하지
않습니다: 벌크 속도가 ~0.24 m/s로 무너지는데, 벽함수 짝은 3.51 m/s, 동일
메쉬의 층류 해는 14.8 m/s이고, 온도는 정상상태에 도달하지 못합니다. 이
붕괴는 벽 처리 행(row) 불일치를 제거해도, `lowRe`의 디리클레 조건을
제거해도, 격자 조정을 완화해도 사라지지 않으므로 그중 어느 하나가 원인은
아닙니다 — 가장 유력하지만 아직 검증되지 않은 단서는 `E`항의
`grad(grad U)` 경계 외삽이 두 벽인접 경계층이 만나는 덕트 모서리에서
일으키는 문제입니다. ofgpu는 이제 `LaunderSharmaKE` 아래에서 `wallTreatment
lowRe`를 받아들이며(`kEpsilon`/`kOmega`/`kOmegaSST`는 여전히 이름을 밝혀
거부, §13.4), 어느 쪽도 조용히 발산하도록 두지 않습니다. 전체 수치(앞선
대체된 시도들과 벽법칙 표 포함)는 `docs/07-fire-solver.md` §1.1을
보십시오.

---

## 성능

측정 환경: NVIDIA GeForce RTX 5070 Ti (70 SM, 896 GB/s), 배정밀도.

1회 외부 반복은 완전한 수송방정식 2개(조립, 저이완, 벽함수 구속, Krylov 해)에
해당합니다.

| 격자 | k-ε (ms/iter) | k-ω (ms/iter) | Mcell-iter/s | 장치 메모리 |
|---|---|---|---|---|
| 80 k | 1.187 | 1.150 | 67 / 70 | 1.4 GB |
| 500 k | 3.427 | 3.400 | 146 / 147 | 1.9 GB |
| 2 M | 13.346 | 13.337 | 150 / 150 | 4.0 GB |

소규모 격자에서는 커널 실행 오버헤드가 지배적이며, 500 k 셀 이상에서 메모리
대역폭 한계에 도달합니다.

### CUDA Graph

24,000 셀, 200회 외부 반복 기준.

| 모드 | ms/iter | Mcell-iter/s |
|---|---|---|
| 적응형 (반복마다 4바이트 플래그 전송) | 1.323 | 18.1 |
| 고정 반복, 커널 개별 실행 | 1.191 | 20.1 |
| 고정 반복, **CUDA Graph** | **0.377** | **63.7** |

**3.16배** 향상이며, 캡처 및 인스턴스화 비용은 0.46 ms로 1회에 그칩니다. 결과는
개별 실행 경로와 24,000 셀 전부가 비트 단위로 일치합니다. Graph는 실행 순서를
변경하지 않고 실행 오버헤드만 제거하기 때문입니다.

### 압력 backend 자동 선택

82,320 셀 플룸 격자에서 선택기가 실측한 결과입니다.

```
uniform cartesian    (98, 42, 20), h = (0.1494, 0.1486, 0.15)
separable bcs        true    symmetric  true    constant coefficient  true

  PBiCGStab   applicable    51.13 ms   (reference)      residual 7.19e-12
  cuFFT       applicable     2.05 ms   agrees to 8.0e-11
  AMGX        unavailable              feature 'amgx' not enabled

chosen: cuFFT   —  25.0x
```

전송을 비활성화한 cuFFT는 0.86 ms까지 단축됩니다. 두 해의 상대 차이는
1.5 × 10⁻¹⁴로, 동일 행렬의 정확한 역연산임을 의미합니다.

이 성능 이득은 상수계수 Poisson 방정식에서만 성립합니다. SIMPLE의 압력
방정식은 계수 `rAUf`가 셀마다 달라 선택기가 cuFFT를 배제합니다.

### 실행시간 분기 비용

| 분기 단위 | 비용 |
|---|---|
| 커널 실행당 가상 호출 | 측정 한계 이하 |
| 요소당 가상 호출 | 1.75 – 1.80배 |

본 solver의 모든 실행시간 선택(SIMPLE/PISO, 난류 모형, 해법 backend)은 커널 실행
단위에서 이루어지므로 조합별 빌드가 불필요합니다.

---

## 빌드

요구사항: Rust 1.85 이상, Visual Studio 2022 (C++ 워크로드), CUDA Toolkit 13.x.

```powershell
cd rust
cargo build --release
cargo test  --release
```

`build.rs`가 `vcvars64.bat`을 실행하여 MSVC 환경을 구성하고, 각 `.cu` 파일을
PTX가 아닌 **CUBIN**으로 컴파일합니다. 드라이버가 보고하는 CUDA 버전보다 툴킷이
최신인 경우 PTX는 `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`으로 실패하기 때문입니다.

CUDA 13의 CCCL 헤더가 MSVC 전통 전처리기에서 `fatal error C1189`를 발생시키므로
`-Xcompiler=/Zc:preprocessor`를 지정합니다.

### 실행 파일

| 실행 파일 | 용도 |
|---|---|
| `ofgpu-validate` | 수치 검증 (228개 항목) |
| `ofgpu-bench` | 처리량 및 메모리 벤치마크 |
| `ofgpu-graph-bench` | CUDA Graph 대 개별 실행 비교 |
| `ofgpu-dispatch-bench` | 실행시간 분기 비용 측정 |
| `ofgpu-probe` | 장치 속성 조회 |
| `ofgpu-generate-mesh` | 케이스 생성 |
| `ofgpu-k-epsilon`, `ofgpu-k-omega` | 난류 모형 단독 실행 |
| `ofgpu-plume`, `ofgpu-buoyant` | 부력 플룸 |
| `ofgpu-vof` | 2상 VOF |
| `ofgpu-fire` | 저-마하 연소·복사 (SPEC-LIT §25–28) |

---

## 실행

```powershell
cd rust
cargo run --release --bin ofgpu-generate-mesh -- channel  ..\cases\channel  200 120 1
cargo run --release --bin ofgpu-k-epsilon     -- ..\cases\channel -iters 4000 -check 400
cargo run --release --bin ofgpu-generate-mesh -- damBreak ..\cases\damBreak  60 100 1
cargo run --release --bin ofgpu-generate-mesh -- plume    ..\cases\plumeCol  60 40 30 -stl column=column.stl
cargo run --release --bin ofgpu-vof           -- ..\cases\damBreak -endTime 0.25 -surge
cargo run --release --bin ofgpu-fire          -- ..\cases\burnerPlume.jsonc -combustion -radiation -endTime 6.0 -deltaT 0.005
cargo run --release --bin ofgpu-validate
```

| 옵션 | 설명 |
|---|---|
| `-iters N` | 외부 반복 횟수 (기본값은 `controlDict`의 `endTime`) |
| `-fixedIters N` | 선형 해법을 정확히 N회 실행하고 잔차를 읽지 않음 (호스트 전송 없음) |
| `-check N` | N회마다 수렴 판정 |
| `-write NAME` | 결과 출력 시간 디렉터리 |
| `-noWrite` | 결과를 기록하지 않음 |
| `-permissive` | 미지원 설정을 오류 대신 경고로 처리하고 대체 내용을 출력 |

생성 가능한 케이스: `channel`, `cavity`, `step`, `big`, `plume`, `damBreak`.
`-stl [name=]path`(반복 가능)를 붙이면 어떤 케이스든 블록 격자를 STL 표면으로
조각합니다 — [cases/README.md](cases/README.md) 참조.

난류 모형은 `constant/momentumTransport`의 `RAS { model ...; }` 또는
`simulationType LES;`로 지정합니다. 미지원 모형명을 지정하면 사용 가능한 목록을
포함한 오류가 발생합니다.

케이스는 OpenFOAM ASCII 형식으로 읽고 씁니다 — `constant/polyMesh`, `0/`,
`constant/physicalProperties`, `constant/momentumTransport`,
`system/{fvSolution, fvSchemes, controlDict}`. 이는 ParaView, `foamToVTK` 등
기존 전·후처리 도구와의 상호운용을 위한 것이며, meteor-cfd는 OpenFOAM의 어떠한
부분과도 링크하지 않고 그 소스를 포함하지 않습니다. 바이너리 형식 케이스는
ASCII로 변환한 뒤 사용하십시오.

---

## 문서

| 파일 | 내용 |
|---|---|
| [`rust/SPEC-LIT.md`](rust/SPEC-LIT.md) | 수치 명세. 모든 수식의 원논문 인용 포함 |
| [`rust/PROVENANCE.md`](rust/PROVENANCE.md) | 파일별 출처 및 설계 결정 기록 |
| [`LICENSING.md`](LICENSING.md) | 라이선스 감사 기록 |
| `docs/01-model-catalog.md` | CFD 구성요소 목록 (1,823항목) |
| `docs/02-gpu-portability.md` | GPU 이식성 등급 분류 |
| `docs/03-esi-vs-foundation.md` | 상류 배포판 간 모형 구성 차이 |
| `cases/README.md` | 시험 케이스 형상 설명 |

---

## 제한사항

- **MPI 다중 GPU를 지원하지 않습니다.** 단일 GPU 전용입니다.
- **AMGX는 `amgx` Cargo 기능으로 제공되며 기본 비활성 상태입니다.** NVIDIA의
  Windows 지원 범위가 제한적이고 검증된 최신 툴킷이 CUDA 12.2인 반면 개발 환경은
  13.3입니다. 비활성 상태에서도 backend 선택기는 AMGX를 "unavailable"로 명시
  보고합니다.
- **Crank–Nicolson은 구현되었으나 저이완 방정식에서 사용할 수 없습니다.** θ
  가중과 암시적 저이완이 조립 과정의 동일 위치를 요구하며, 저이완은 가중되지 않은
  대각 성분을 참조해야 하기 때문입니다. 묵시적으로 Euler로 대체하지 않고 해당
  사유를 오류로 보고합니다.
- **압축성 및 천음속 해석을 지원하지 않습니다.** 밀도 가중 시간미분은 구현되어
  VOF에서 사용되나, 압력 방정식은 비압축성입니다.
- **연소는 혼합제어 단일 스텝(EDM)만 지원합니다.** 유한율 화학반응 메커니즘은
  없습니다. **복사는 회색 P1 근사만 지원합니다.** `fvDOM`(유한체적 이산종좌표법)은
  광학적으로 얇은 화재 경계에서 더 정확하지만 명세된 다음 단계이며, 요청 시
  §13.4 계약에 따라 이름을 명시하고 거부됩니다.
- **다중 cyclic 쌍을 지원하지 않습니다.** 축 하나에 대해서만 주기 경계를 선언할
  수 있어, 두 방향이 모두 주기인 진짜 평면 채널은 아직 표현할 수 없습니다.

---

## 참고문헌

수치 기법과 모형의 출처입니다. 절 번호는 [`rust/SPEC-LIT.md`](rust/SPEC-LIT.md)
기준입니다.

### 유한체적 이산화

- Jasak, H. (1996). *Error Analysis and Estimation for the Finite Volume Method
  with Applications to Fluid Flows.* PhD thesis, Imperial College London. — §2, §3
- Moukalled, F., Mangani, L., & Darwish, M. (2016). *The Finite Volume Method in
  Computational Fluid Dynamics.* Springer. — §2, §3, §11
- Ferziger, J. H., & Perić, M. *Computational Methods for Fluid Dynamics.*
  Springer. — §2.4, §3.3, §11.1, §11.5
- Patankar, S. V. (1980). *Numerical Heat Transfer and Fluid Flow.* Hemisphere.
  — §3.4, §5.2, §18

### 대류 이산화 기법

- Warming, R. F., & Beam, R. M. (1976). Upwind second-order difference schemes
  and applications in aerodynamic flows. *AIAA Journal*, 14(9), 1241–1249. — §11.2
- Leonard, B. P. (1979). A stable and accurate convective modelling procedure
  based on quadratic upstream interpolation. *Computer Methods in Applied
  Mechanics and Engineering*, 19(1), 59–98. — §11.3
- Khosla, P. K., & Rubin, S. G. (1974). A diagonally dominant second-order
  accurate implicit scheme. *Computers & Fluids*, 2(2), 207–209. — §11.1
- Jasak, H., Weller, H. G., & Gosman, A. D. (1999). High resolution NVD
  differencing scheme for arbitrarily unstructured meshes. *International Journal
  for Numerical Methods in Fluids*, 31(2), 431–449. — §11.6
- Sweby, P. K. (1984). High resolution schemes using flux limiters for hyperbolic
  conservation laws. *SIAM Journal on Numerical Analysis*, 21(5), 995–1011. — §7
- van Leer, B. (1977). Towards the ultimate conservative difference scheme IV.
  A new approach to numerical convection. *Journal of Computational Physics*,
  23(3), 276–299. — §7
- van Leer, B. (1979). Towards the ultimate conservative difference scheme V.
  A second-order sequel to Godunov's method. *Journal of Computational Physics*,
  32(1), 101–136. — §7
- van Albada, G. D., van Leer, B., & Roberts, W. W. (1982). A comparative study
  of computational methods in cosmic gas dynamics. *Astronomy and Astrophysics*,
  108, 76–84. — §7
- Roe, P. L. (1986). Characteristic-based schemes for the Euler equations.
  *Annual Review of Fluid Mechanics*, 18, 337–365. — §7
- Darwish, M., & Moukalled, F. (2003). TVD schemes for unstructured grids.
  *International Journal of Heat and Mass Transfer*, 46(4), 599–611. — §7

### 기울기 및 제한자

- Barth, T. J., & Jespersen, D. C. (1989). The design and application of upwind
  schemes on unstructured meshes. *AIAA Paper 89-0366.* — §12.2
- Venkatakrishnan, V. (1993). On the accuracy of limiters and convergence to
  steady state solutions. *AIAA Paper 93-0880.* — §12.2

### 시간 적분

- Crank, J., & Nicolson, P. (1947). A practical method for numerical evaluation
  of solutions of partial differential equations of the heat-conduction type.
  *Mathematical Proceedings of the Cambridge Philosophical Society*, 43(1),
  50–67. — §13.1

### 압력–속도 결합

- Patankar, S. V., & Spalding, D. B. (1972). A calculation procedure for heat,
  mass and momentum transfer in three-dimensional parabolic flows.
  *International Journal of Heat and Mass Transfer*, 15(10), 1787–1806. — §5.2
- Van Doormaal, J. P., & Raithby, G. D. (1984). Enhancements of the SIMPLE method
  for predicting incompressible fluid flows. *Numerical Heat Transfer*, 7(2),
  147–163. — §5.3
- Issa, R. I. (1986). Solution of the implicitly discretised fluid flow equations
  by operator-splitting. *Journal of Computational Physics*, 62(1), 40–65.
  — §5.4, §14
- Rhie, C. M., & Chow, W. L. (1983). Numerical study of the turbulent flow past
  an airfoil with trailing edge separation. *AIAA Journal*, 21(11), 1525–1532.
  — §5.1

### 난류 모형 — RANS

- Launder, B. E., & Spalding, D. B. (1974). The numerical computation of
  turbulent flows. *Computer Methods in Applied Mechanics and Engineering*, 3(2),
  269–289. — §6.1, §6.4
- Wilcox, D. C. *Turbulence Modeling for CFD.* DCW Industries. — §6.2
- Menter, F. R. (1994). Two-equation eddy-viscosity turbulence models for
  engineering applications. *AIAA Journal*, 32(8), 1598–1605. — §6.3
- Menter, F. R., Kuntz, M., & Langtry, R. (2003). Ten years of industrial
  experience with the SST turbulence model. *Turbulence, Heat and Mass Transfer*,
  4, 625–632. — §6.3
- Launder, B. E., & Sharma, B. I. (1974). Application of the energy-dissipation
  model of turbulence to the calculation of flow near a spinning disc. *Letters
  in Heat and Mass Transfer*, 1(2), 131–138. — §33
- Patel, V. C., Rodi, W., & Scheuerer, G. (1985). Turbulence models for
  near-wall and low Reynolds number flows: a review. *AIAA Journal*, 23(9),
  1308–1319. — §33

### 난류 모형 — LES

- Smagorinsky, J. (1963). General circulation experiments with the primitive
  equations. *Monthly Weather Review*, 91(3), 99–164. — §6.5
- Deardorff, J. W. (1970). A numerical study of three-dimensional turbulent
  channel flow at large Reynolds numbers. *Journal of Fluid Mechanics*, 41(2),
  453–480. — §16.1
- Deardorff, J. W. (1980). Stratocumulus-capped mixed layers derived from a
  three-dimensional model. *Boundary-Layer Meteorology*, 18(4), 495–527. — §6.5
- Nicoud, F., & Ducros, F. (1999). Subgrid-scale stress modelling based on the
  square of the velocity gradient tensor. *Flow, Turbulence and Combustion*,
  62(3), 183–200. — §6.5
- van Driest, E. R. (1956). On turbulent flow near a wall. *Journal of the
  Aeronautical Sciences*, 23(11), 1007–1011. — §16.4
- Scotti, A., Meneveau, C., & Lilly, D. K. (1993). Generalized Smagorinsky model
  for anisotropic grids. *Physics of Fluids A*, 5(9), 2306–2308. — §16.3

### 벽 처리

- Spalding, D. B. (1961). A single formula for the law of the wall. *Journal of
  Applied Mechanics*, 28(3), 455–458. — §6.4, §15.1
- Cebeci, T., & Bradshaw, P. (1977). *Momentum Transfer in Boundary Layers.*
  Hemisphere. — §15.3
- Jayatilleke, C. L. V. (1969). The influence of Prandtl number and surface
  roughness on the resistance of the laminar sub-layer to momentum and heat
  transfer. *Progress in Heat and Mass Transfer*, 1, 193–330. — §29.3
- Werner, H., & Wengle, H. (1991). Large-eddy simulation of turbulent flow
  over and around a cube in a plate channel. *8th Symposium on Turbulent
  Shear Flows.* — §30.1
- Tucker, P. G. (1998). Assessment of geometric multilevel convergence robustness
  and a wall distance method for flows with multiple internal boundaries.
  *Applied Mathematical Modelling*, 22(4–5), 293–305. — §6.6
- Dittus, F. W., & Boelter, L. M. K. (1930). Heat transfer in automobile
  radiators of the tubular type. *University of California Publications in
  Engineering*, 2, 443–461. (재인쇄: *International Communications in Heat
  and Mass Transfer*, 12(1), 3–22, 1985.) — §32.3
- Gnielinski, V. (1976). New equations for heat and mass transfer in
  turbulent pipe and channel flow. *International Chemical Engineering*,
  16(2), 359–368. — §32.3

### 부력

- Rehm, R. G., & Baum, H. R. (1978). The equations of motion for thermally driven
  buoyant flows. *Journal of Research of the National Bureau of Standards*,
  83(3), 297–308. — §9
- Spiegel, E. A., & Veronis, G. (1960). On the Boussinesq approximation for a
  compressible fluid. *Astrophysical Journal*, 131, 442–447. — §9
- Rodi, W. (1987). Examples of calculation methods for flow and mixing in
  stratified fluids. *Journal of Geophysical Research*, 92(C5), 5305–5328. — §17
- Henkes, R. A. W. M., van der Vlugt, F. F., & Hoogendoorn, C. J. (1991).
  Natural-convection flow in a square cavity calculated with low-Reynolds-number
  turbulence models. *International Journal of Heat and Mass Transfer*, 34(2),
  377–388. — §17

### 다상 유동

- Hirt, C. W., & Nichols, B. D. (1981). Volume of fluid (VOF) method for the
  dynamics of free boundaries. *Journal of Computational Physics*, 39(1),
  201–225. — §20.1
- Zalesak, S. T. (1979). Fully multidimensional flux-corrected transport
  algorithms for fluids. *Journal of Computational Physics*, 31(3), 335–362.
  — §20.2
- Brackbill, J. U., Kothe, D. B., & Zemach, C. (1992). A continuum method for
  modeling surface tension. *Journal of Computational Physics*, 100(2), 335–354.
  — §20.4
- Ubbink, O. (1997). *Numerical Prediction of Two Fluid Systems with Sharp
  Interfaces.* PhD thesis, Imperial College London. — §20.1
- Rusche, H. (2002). *Computational Fluid Dynamics of Dispersed Two-Phase Flows
  at High Phase Fractions.* PhD thesis, Imperial College London. — §20.1

### 선형 해법

- Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed. SIAM.
  — §8, §21
- van der Vorst, H. A. (1992). Bi-CGSTAB: a fast and smoothly converging variant
  of Bi-CG for the solution of nonsymmetric linear systems. *SIAM Journal on
  Scientific and Statistical Computing*, 13(2), 631–644. — §8.1
- Hestenes, M. R., & Stiefel, E. (1952). Methods of conjugate gradients for
  solving linear systems. *Journal of Research of the National Bureau of
  Standards*, 49(6), 409–436. — §8.2
- Swarztrauber, P. N. (1977). The methods of cyclic reduction, Fourier analysis
  and the FACR algorithm for the discrete solution of Poisson's equation on a
  rectangle. *SIAM Review*, 19(3), 490–501. — §8.5
- Stüben, K. (2001). A review of algebraic multigrid. *Journal of Computational
  and Applied Mathematics*, 128(1–2), 281–309. — §8.3

### 다공성 매질

- Ward, J. C. (1964). Turbulent flow in porous media. *Journal of the Hydraulics
  Division, ASCE*, 90(5), 1–12. — §18

### 검증 데이터

- Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for
  incompressible flow using the Navier–Stokes equations and a multigrid method.
  *Journal of Computational Physics*, 48(3), 387–411.
- Moser, R. D., Kim, J., & Mansour, N. N. (1999). Direct numerical simulation of
  turbulent channel flow up to Re_τ = 590. *Physics of Fluids*, 11(4), 943–945.
- Driver, D. M., & Seegmiller, H. L. (1985). Features of a reattaching turbulent
  shear layer in divergent channel flow. *AIAA Journal*, 23(2), 163–171.
- McCaffrey, B. J. (1979). *Purely Buoyant Diffusion Flames: Some Experimental
  Results.* NBSIR 79-1910, National Bureau of Standards.
- Martin, J. C., & Moyce, W. J. (1952). An experimental study of the collapse of
  liquid columns on a rigid horizontal plane. *Philosophical Transactions of the
  Royal Society A*, 244(882), 312–324.
- Nilsson, A., & Wallin, S. (2022). *Lid Driven Cavity Flow Using Finite
  Difference and Radial Basis Function Methods.* Uppsala University report 22015.

---

## 문의

**simul@msimul.com**

주식회사 메테오시뮬레이션 / Meteo Simulation Co., Ltd.

교육과 학술 연구는 무료입니다. 기업 연구개발, 학교에 속하지 않은 연구기관, 수탁
및 상업적 이용은 별도 라이선스가 필요합니다 — [`LICENSE`](LICENSE) 제2·3절.
적용 범위가 불분명한 경우 문의해 주시기 바랍니다.
