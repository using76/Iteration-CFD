# 19 — Verification & Validation 계획 (초안, 2026-10-27)

작성: Opus 5.5, 브랜치 `feat/t19-vv` (feat/t18 `1565179`에서). §7에 사용자 결정, §9에 W1 단위.

## 1. 사용자 요청 (원문)

> 지금 다른 세션에서 verification 을 하고있어. verification은 exact solution과 우리가만든 gpu core의 결과를
> 비교하는거야, 이 iteration-cfd도 이런 verification 할수있어야하는데, 구체적으로 수식들이 정리되서 할수있는
> 자료를 찾아주고 자체적으로도 만들어보자, 그후 validation쪽도 해야하는데 이건 실험결과가 있어야해, 한번
> 이것도 있는지 찾아서 연구해줘 일단은 계획이야

참고한 다른 세션: `../V&V수정안/verification` — FDS Verification Guide 6.11.1의 해석해 절 92개를 BULC GPU와
CPU FDS로 돌리고, Guide LaTeX에서 수식 234개를 `docs/exact_solutions.json`으로 뽑아 포털 페이지에 싣는다.
FDS는 이런 "해석해 모음집"이 공식 문서로 있지만, OpenFOAM 계열 유한체적 코드인 Iteration-CFD에는 그런
단일 모음집이 없다. 그래서 아래 §3의 문헌에서 수식을 모으고, §4에서 우리 VER 모음집을 직접 만든다.

## 2. 지금 있는 것 (2026-10-27 트리 기준)

### 2.1 도구
- `rust/src/vv.rs` (SPEC-LIT §94): Celik 2008 관측 차수·Richardson·GCI, Eça-Hoekstra 2014 최소제곱 불확도,
  Coleman-Stern/ASME V&V 20의 `E = S − D`, `u_val`. **차수·불확도 계산은 이미 완비.**
- `ofgpu-validate` (`rust/src/bin/validate.rs` + `validate_gates/` 37개) — 섹션별 게이트, `-json` 기록이
  `docs/validate-run/`에 남는다.
- `reference/PROVENANCE.md` — 정답 키 장부(파일/fetched/literal, sha256, 라이선스). 새 키는 전부 여기 행으로.

### 2.2 이미 있는 verification (해석해·정확해 대비)

| 분류 | 게이트 |
|---|---|
| MMS | 94-A 스칼라 이방성 확산 `-div(K grad ψ)=f` (sin·sin·sin), 94-B CHT 계면 MMS, §10 "Laplacian order" |
| 비압축 정확해 | Couette(CMP-COU 포함), Hagen-Poiseuille(119-A wedge, 116-P tet/prism), Brinkman 점성가열(100-C), Blasius(132-L) |
| 열전도 | 2층 slab(Gate 1-4), Kirchhoff slab(100-A), 핀(98-A), 두꺼운 원통(95-D) |
| 압축성 | Sod 정확 Riemann(CMP-SOD), 음향 펄스(CMP-AC), 준1차원 노즐(CMP-NOZ), 경사충격파(CMP-OBL), Ringleb(CMP-RING) |
| 기타 | 이동격자 GCL(105-A), 복사 slab/cube(CHR-SLAB/CUBE), ODE ROBER/HIRES, 평형(CHR-EQ), 고체 NAFEMS LE1/10/11 |
| 수치 벤치마크(정확해 아님) | Ghia 1982, de Vahl Davis 1983, Turek-Hron 2006 |

### 2.3 이미 있는 validation (실험값 대비)
Driver-Seegmiller 후향계단, McCaffrey 플룸, ERCOFTAC T3A/T3B/T3A-, RAE 2822 Case 6, Martin-Moyce 댐붕괴,
Riboux-Gordillo 비산, Qu-Mudawar 마이크로채널, Theobald 호스 분류, TNF Sandia Flame D, Leeds/REMBE 분진,
(DNS 기준) MKM 채널 180/395/590, TMR 평판 상관식.

### 2.4 빈 곳 (이 계획의 대상)
1. **운동량-압력 결합 NS에 MMS가 없다.** MMS는 스칼라 확산뿐이다. SIMPLE/PISO/lowmach의 대류항·압력보정·
   비직교 보정·경계조건 버그는 지금 게이트로는 차수 저하로 드러나지 않는다.
2. **대류 스킴별 차수 확인이 없다** (upwind 1차, linear/linearUpwind 2차, TVD 제한자).
3. **시간 차수 확인이 NS에 없다** (Euler 1, backward 2, CN 2 — `timescheme.rs`가 주장만 함).
4. **난류 모델 MMS 없음** (SA, k-ω SST).
5. **가변밀도(lowmach)·압축성 NS MMS 없음** (Ringleb은 Euler, 매끄러운 해 하나).
6. **VOF verification 없음** (계면 이송 정확도, 표면장력).
7. **비정렬 격자 차수** — tet/prism/poly 위의 NS 차수 확인 없음 (116-P는 단일 값 비교).
8. **f32 빌드의 차수 붕괴 지점**이 체계적으로 기록돼 있지 않다 (`single-build` 고정점 바닥 ~3e-6 은 알려짐).
9. 다른 세션처럼 **수식과 결과를 한곳에 모은 VER 카탈로그**(수식 LaTeX + 수렴 그래프)가 없다.

## 3. 수식이 정리된 외부 자료 (조사 결과)

라이선스 원칙은 기존과 같다: **논문·보고서의 수식은 옮겨 적고, 코드는 읽지 않는다**(GPL/LGPL 라이브러리
코드 비참조). 수식은 출처와 함께 `PROVENANCE.md` literal 행으로 남긴다.

### 3.1 MMS 방법론과 수식 세트
| 자료 | 주는 것 | 접근 |
|---|---|---|
| Salari & Knupp, SAND2000-1444 (2000), DOI 10.2172/759450 | MMS 절차 전체, 비압축·압축 NS 예제와 소스항, 차수 검사 절차 | OSTI 공개(미국 정부) |
| Roy, Nelson, Smith, Ober, IJNMF 44 (2004) 599 | 2D/3D 압축 Euler·NS 사인형 MMS (아음속/초음속 계수표) — 사실상 표준 | 저자 공개 PDF (vt.edu) |
| Roache, JFE 124 (2002) 4 / Oberkampf & Roy, *V&V in Scientific Computing* (2010) | MMS 설계 규칙(경계, 소스 적분, 반복오차 분리) | 이미 vv.rs 인용 / 교재 |
| Eça, Hoekstra, Hay, Pelletier, IJNMF 54 (2007) 119 + Lisbon "Workshop on CFD Uncertainty Analysis" MS1/MS2 | **SA·k-ω SST용 난류 MMS** (2D 경계층형 제조해) | NASA TMR `eca1.html`이 안내 |
| Shunn, Ham, Moin, JCP 231 (2012) 3801 | **가변밀도 저마하 MMS** — lowmach 솔버에 정확히 맞음 | 저널 |
| MASA (Malaya et al., Eng. Comput. 29 (2013) 487) | 다수 MMS의 수식 문서 (Euler/NS/RANS/열) | 수식만 참조 — 코드는 LGPL-2.1이라 읽지 않음 |
| ASME V&V 20-2009 | 검증 불확도 체계 (vv.rs가 이미 구현) | 유료 |

### 3.2 고전 정확해 (교재·논문 수식)
- **Kovasznay (1948)** — 정상 2D NS 정확해 (Re=40). 정상 SIMPLE의 공간 차수.
- **Taylor-Green 2D 감쇠 와류** — 비정상 주기 NS 정확해. PISO/PIMPLE의 공간·시간 차수.
- **Ethier & Steinman (1994) 3D Beltrami 흐름**, IJNMF 19 — 3D 비정상 NS 정확해. tet/poly 격자용.
- **Stokes 제2문제 / Womersley** — 진동 흐름, 시간 차수와 위상.
- **Lamb-Oseen 와류, Hiemenz 정체점, Jeffery-Hamel** — White, *Viscous Fluid Flow* 3장에 모여 있음.
- **등엔트로피 와류 이송** (Shu 1998; HiOCFD 워크숍 표준 케이스) — 압축 Euler의 차수·수치소산.
- 열전도: Cole et al. *Heat Conduction Using Green's Functions* / EXACT 해 목록(UNL) — 과도 전도 정확해.
- VOF: Zalesak 슬롯 원판, Rider-Kothe 단일 와류(시간 역전 → 초기형상이 정답), Prosperetti(1981) 감쇠 모세관파,
  정지 액적 Laplace 압력.
- 압축 추가: LANL **ExactPack**(SoftwareX 2023) 이 Noh/Sedov/Guderley/여러 충격관 해를 제공 — 라이선스 파일
  확인 전까지 "논문 수식만" 원칙.

### 3.3 코드 대 코드 verification (정확해가 아닌 격자수렴 기준값)
NASA TMR verification 케이스: 2DZP 평판, 2D bump-in-channel, 3D bump, NACA 0012, 2D coflowing jet,
airfoil near-wake. CFL3D·FUN3D의 격자 계열과 격자수렴 값이 공개(미국 정부, 공공영역). SA/SST 구현 일치를
정확해 없이 확인하는 표준 경로.

## 4. 자체 VER 모음집 만들기

### 4.1 구조
- `tools/mms/` — **SymPy(BSD)로 정확해 → 소스항을 기호 미분**하고, Rust 함수(정확해, 소스, 경계값)와
  수식 매니페스트(`ver_formulas.json`: LaTeX, 변수, 출처)를 생성한다. 생성 코드는 커밋하고 생성기는 재현용.
  매니페스트 형식은 다른 세션의 `exact_solutions.json`과 맞춰 포털을 공유할 수 있게 한다.
- 소스항은 **셀 평균(가우스 구적)**으로 넣는다 — 셀 중심 점값은 2차 이상에서 차수를 오염시킨다.
- 격자 계열: 직교 / 전단·비틀림(`blockgen`) / 무작위 섭동 / tet / prism / poly. 배율 2, 정식은 4–5 단계,
  단위 검증은 REDUCED 3단계(10분 규칙).
- 지표: 변수별 L1·L2·L∞ 오차 → `vv::observed_order`, GCI. 반복오차를 떼기 위해 f64에서 잔차 1e-12 까지.
- 게이트는 `ofgpu-validate`의 새 섹션 `VER`로 넣는다. 미달은 MISSED로 기록하고 진행(`docs/17-review-later.md`).

### 4.2 MMS 소스 주입 경로 (사용자 확인 필요 — §7-1)
`sources.rs`의 `SourceTerm`은 셀 집합에 상수 소스만 준다. 공간·시간 함수 소스가 없다. 두 길이 있다.
- **(a) 하네스 수준 (권장)**: Gate 94-A처럼 validate가 라이브러리(`fv::fvm_su`, 운동량·압력 조립)를 직접 불러
  셀별 소스 배열을 더한다. 솔버 수치는 그대로이고 드라이버 바이너리는 손대지 않는다.
- (b) 솔버에 `extraSource` 배열 훅을 추가: 드라이버 전체(SIMPLE/PISO/lowmach/compressible)를 그대로 검증하지만
  솔버 코드 경로가 늘어난다. 수치 변경은 아니나 솔버 수정이므로 사용자 결정.

### 4.3 VER 케이스 목록

| id | 내용 | 대상 경로 | 기대 차수 |
|---|---|---|---|
| VER-01 | 정상 대류-확산 MMS, Pe 스윕, 스킴별 | scalar_transport | upwind 1, linear/linearUpwind 2, TVD ≈2(매끄러운 해) |
| VER-02 | 비정상 스칼라 MMS `sin(ωt)` | timescheme | Euler 1, backward 2, CN 2 |
| VER-03 | Kovasznay Re=40 | simple (SIMPLE/SIMPLEC) | u, p 2 |
| VER-04 | 2D Taylor-Green, 주기경계 | PISO/PIMPLE | 공간 2, 시간 스킴 차수, KE 감쇠율 |
| VER-05 | 비압축 NS 삼각함수 MMS (Salari-Knupp형), Dirichlet/Neumann/유출 경계 혼합 | simple | 2 (경계 포함) |
| VER-06 | VER-05를 전단·섭동·tet·prism·poly 격자에서 | 비직교 보정, 구배 스킴 | 2 (차수 저하 지점 기록) |
| VER-07 | Ethier-Steinman 3D Beltrami | PISO, tet | 2 |
| VER-08 | Stokes 제2문제 / Womersley | 비정상 | 진폭·위상, 시간 2 |
| VER-09 | Boussinesq 결합 MMS (u-p-T) | energy + 부력 | 2 |
| VER-10 | Shunn-Ham-Moin 가변밀도 MMS | lowmach | 2 |
| VER-11 | Roy 2004 압축 Euler/NS MMS (아음속·초음속) | compressible | 스킴 정식 차수 |
| VER-12 | 등엔트로피 와류 이송 | compressible | 2, 소산 |
| VER-13 | SA, k-ω SST MMS (Eça 2007) | models | 2 (벽 근처 경계 포함) |
| VER-14 | TMR 2DZP·bump 코드 대 코드 | SA/SST 전체 | CFL3D/FUN3D 격자수렴값 대비 |
| VER-15 | Zalesak, Rider-Kothe, 감쇠 모세관파, 정지 액적 | vof | 형상오차 차수, 기생류 |
| VER-16 | 변형격자 MMS (GCL 확장) | ale_flow | 2 |
| VER-17 | VER-01/03/04를 single 빌드로 | f32 | 차수가 꺾이는 h와 오차 바닥 기록 |

## 5. Validation — 실험 자료 조사 결과

원칙: 실험값 D와 그 불확도 u_D를 함께 키에 넣고, `E = S − D`, `u_val = √(u_num² + u_input² + u_D²)`
(vv.rs (94.10))로 판정. u_num은 3격자 GCI. 공공영역·CC-BY는 tracked, 비상업(CC BY-NC-SA, ERCOFTAC)은
fetched(D-VAL-2 선례).

### 5.1 우선순위 A — 공개, 제품 축과 직결
| 자료 | 내용 | 맞는 제품 축 | 라이선스 |
|---|---|---|---|
| NASA TMR validation 확장: **2DWMH NASA hump**(Greenblatt), **ATB 축대칭 천음속 bump**(Bachalo-Johnson), **2DN00 NACA0012**(Ladson), 2DN44 NACA4412, 2DML 혼합층, ASJ 아음속 제트, ASBL(Driver) | 박리·재부착, 벽면 Cp/Cf, 속도 프로파일 | 외부 공력, 난류 모델 | 미국 정부, 공공영역 |
| **firemodels/exp** (NIST FDS Validation 실험 DB): `Askervein_Hill`(지형 위 바람), **`LNG_Dispersion`**(무거운 가스 확산), `Sandia_Pool_Fires`, `Heskestad_Flame_Height`, 구획 화재 | 현장 측정 | **NH3 부지 누출·확산**, 화재/플룸 | NIST — 저장소 LICENSE 확인 필요 |
| **Windsor body / DrivAer** (AutoCFD 2·3, Loughborough 기관 저장소 공개) | 힘, 표면압 209점, 속도 프로파일, PIV 평면 | 차량 공력(racecar) | 기관 저장소 조건 확인 |
| Ampofo & Karayiannis (2003) 저난류 자연대류 공동 | 속도·온도 표 | 데이터센터·열 | 논문 표 transcribe |
| SPHERIC Test 2 — Kleefsman (2005) MARIN 장애물 댐붕괴 | 수위·압력 시계열 | VOF | 공개 다운로드 |

### 5.2 우선순위 B — 비상업(fetched) 또는 무거움
- ERCOFTAC: Ahmed body(Lienhart-Becker 2003), 주기 언덕(Rapp-Manhart, UFR 3-30), 벽부착 큐브(Martinuzzi-Tropea),
  사각 실린더(Lyn 1995), 충돌 제트 열전달(Baughn-Shimizu 1989, 전자냉각 축). CC BY-NC-SA → fetched.
- Armaly (1983) 층류 후향계단, 원기둥 Cd/St(Williamson·Roshko) — 층류 비정상.
- 도시·플랜트 확산: CEDVAL(함부르크 풍동), MUST, **Jack Rabbit II**(염소 대량 방출), **Desert Tortoise**(NH3 방출
  실측) — NH3 부지에 가장 가깝지만 자료 접근 경로를 따로 조사해야 함.
- AIAA DPW CRM / HLPW — 대형 3D, 장기.

### 5.3 다음 조사 단위 (VAL-R)
각 후보마다: 데이터 파일 실제 존재, 형식, 불확도 기재 여부, 라이선스 문구, 형상(CAD/좌표) 확보 가능 여부를
확인해 `PROVENANCE.md` 후보 행으로 정리. 형상 없는 자료(REMBE 선례)는 NAMED/NOT RUN으로 둔다.

## 6. 단계안

| wave | 내용 | 코더 |
|---|---|---|
| W1 | `tools/mms` 생성기 + 매니페스트 형식, VER 섹션 골격, VER-01/02 (스칼라 — 기존 94-A 경로 재사용) | GLM, Opus 리뷰 |
| W2 | VER-03/04/05 (Kovasznay, Taylor-Green, NS MMS) — §7-1 결정에 따라 하네스 또는 훅 | GLM |
| W3 | VER-06/07/08/16 비정렬·3D·비정상, VER-17 single | GLM |
| W4 | VER-09/10/11/12 열·가변밀도·압축 | GLM |
| W5 | VER-13/14 난류, VER-15 VOF | GLM |
| V1 | VAL-R 조사(§5.3) → 키 행 확정 → 우선순위 A 케이스부터 게이트 | Opus 조사, GLM 코딩 |
| P | VER/VAL 결과 포털 (다른 세션 포털 형식 공유: 수식 LaTeX + 수렴 그래프 + 판정) | 별도 |

각 단위는 표적 테스트 + REDUCED 게이트 10분 이내, 미달은 MISSED로 기록하고 진행.

## 7. 결정 (2026-10-27)

사용자 원문: "a, 형식만 가져오고 따로 만들자 Iteration-CFD 폴더에, 나머진 알아서"

1. **MMS 소스 주입 = (a) 하네스 수준.** validate가 라이브러리를 직접 불러 셀별 소스 배열을 더한다. 솔버·드라이버
   바이너리는 수정하지 않는다. (b)의 훅은 만들지 않는다.
2. **포털 = 형식만 가져와서 이 폴더에 따로 만든다.** 다른 세션의 코드·페이지는 복사하지 않고 데이터 형식(§8)만
   맞춘다. 위치 `docs/vv/` (수식 `docs/vv/formulas/`, 결과 `docs/vv/results/`, 페이지 `docs/vv/site/`).
3. (권장으로 결정) **validation 첫 축 = NASA TMR 확장** (공공영역, 2D, 132-Z의 SA/SST·키 경로 재사용, 10분 규칙에
   맞음). 두 번째 = NH3 부지 확산(firemodels/exp `LNG_Dispersion`·`Askervein_Hill`), 세 번째 = 데이터센터 열.
4. (권장으로 결정) **지금 시작한다.** tranche 18 W1 은 10-25 이후 커밋이 없고 이 폴더에서 돌고 있는 작업이 없다.
   브랜치 `feat/t19-vv` (feat/t18 `1565179`에서), 단위는 한 번에 하나, GLM 코딩·Opus 리뷰. tranche 18은 이
   브랜치를 병합한 뒤 이어간다.

## 8. 데이터 형식 (다른 세션 형식을 따름)

수식 매니페스트 `docs/vv/formulas/<VER-id>.json`:
```
{ "id": "VER-01", "title": "...", "intro": "...(한국어 개요)",
  "equations": [{"env": "equation", "tex": "..."}],
  "parameters": {"이름": 값}, "sources": [{"cite": "...", "doi_or_url": "..."}] }
```
결과 `docs/vv/results/<VER-id>.json` (다른 세션 `tools/post/common.py` 머리말과 같은 키):
```
{ "section": "VER-01",
  "engines": {"f64": {"label", "comparable", "reason"}, "single": {...}},
  "field": null 또는 {...},
  "charts": [{"title", "x_label", "y_label", "x_log", "y_log", "caption",
              "series": [{"label", "engine": "f64"|"single"|"exact"|"ref", "style": "line"|"dash"|"dots", "x", "y"}]}],
  "metrics": [{"label", "f64", "single", "expect"}],
  "notes": ["..."] }
```
엔진 키만 우리 것(`f64`, `single`)으로 바꾼다. 수렴 그래프는 h–오차 log-log, 지표는 관측 차수.

## 9. W1 단위

| unit | 내용 | 산출물 |
|---|---|---|
| VER-00 | SymPy 생성기 `tools/mms/` — 정확해 정의 → 소스항 기호 미분 → Rust 모듈(정확해·소스·구배, 호스트 전용 f64)과 §8 수식 매니페스트 생성, 셀 평균 구적 도우미 | `tools/mms/*.py`, `rust/src/vv/mms.rs`(생성물), `docs/vv/formulas/VER-01.json`, `VER-02.json` |
| VER-01 | 정상 대류-확산 MMS, 스킴별 관측 차수, validate 섹션 + §8 결과 JSON | validate 섹션 `VER-01`, `docs/vv/results/VER-01.json` |
| VER-02 | 비정상 스칼라 MMS, Euler/backward/CN 시간 차수 | 섹션 `VER-02`, 결과 JSON |
