# meteor-cfd

**GPU 상주 유한체적 전산유체역학 솔버**

주식회사 메테오시뮬레이션 · Rust 호스트 + CUDA 커널 · [English](README.en.md)

---

## 개요

meteor-cfd는 시간 적분 루프 전체가 GPU에 머무르도록 설계된 비정렬 격자 유한체적 CFD 솔버입니다. 메쉬와 필드를 한 번 업로드한 뒤에는 시간 루프 안에서 장치 메모리 할당도, 필드 데이터의 호스트 전송도 발생하지 않습니다. **단일 GPU 전용**이며, 쓰임새는 비압축성·저마하 유동입니다 — RANS/LES/하이브리드 난류, 부력 플룸, 저마하 화재(연소·복사·그을음), 2상 VOF, 켤레 열전달, 팬과 다공성 점프를 갖춘 환기 및 데이터센터 기류. 수치 코어 전체를 공개 문헌으로부터 직접 구현했으며 모든 수식은 [`rust/SPEC-LIT.md`](rust/SPEC-LIT.md)에 원논문 인용과 함께 명세되어 있습니다. 검증은 인위해법(MMS), 해석해, 공개 벤치마크만을 사용하며 **다른 CFD 코드와 비교하지 않습니다.**

Rust 1.85 호스트에 CUDA C++ 커널, 배정밀도 기본(`single` 기능으로 단정밀도), 대상은 NVIDIA GPU 한 장, 의존성은 cudarc와 thiserror뿐입니다(AMGX는 선택 기능).

---

## 라이선스

| 용도 | 조건 |
|---|---|
| 개인 학습, 취미, 아마추어 연구 | **무료** |
| 교육기관 — 수업, 과제, 실습 | **무료** |
| 대학·학교 및 그 소속 연구소 | **무료** |
| 공공 연구기관, 정부기관 | **무료** (재원 출처 불문) |
| 공공 안전·보건 기관, 환경 보호 단체 | **무료** |
| 자선단체 | **무료** |
| 그 밖의 상업적 이용 | **30일 시험** 후 유상 라이선스 |

**30일 시험 기간은 사람이 아니라 회사 단위입니다.** 업무 목적으로 쓰신다면
인원 수와 무관하게 회사 전체에 시험 기간이 한 번 주어집니다. CFD는 자기
케이스·자기 격자·자기 하드웨어에서 돌려보지 않고는 도입을 결정할 수 없는
소프트웨어이므로, 이 기간은 평가를 위한 것입니다.

기여를 되돌려 주시는 경우 — Blue Oak 1.0.0, Apache 2.0, MIT, 2-clause BSD 중
하나로 — 그 개발은 상업적 이용으로 계산되지 않습니다.

적용 범위가 불분명한 경우 문의해 주시기 바랍니다. 실질이 상업적이지 않은
이용에 대해서는 무상 또는 감면 라이선스를 검토합니다.

**라이선스 문의: simul@msimul.com**

라이선서는 주식회사 메테오시뮬레이션(Meteo Simulation Co., Ltd.)입니다.
전문은 [`LICENSE`](LICENSE), 상업 라이선스의 범위와 절차는
[`LICENSING.md`](LICENSING.md), 서드파티 고지는 [`NOTICE`](NOTICE)를
참조하십시오.

> Prosperity는 비상업 라이선스이지 오픈소스 라이선스가 아닙니다. OSI 승인
> 라이선스가 아니며, 상업적 이용을 제한하므로 그렇게 부르지 않습니다.

---

## 현황

**2026-09-03 이 작업 트리에서 실제로 돌려 얻은 출력입니다.** NVIDIA GeForce RTX 5070 Ti (sm_120), CUDA 13.3, 배정밀도.

```
cargo test --release   1,853 passed, 0 failed, 6 ignored   (모든 타깃 합계, 18개 스위트)
                       1,699 passed, 0 failed, 4 ignored   (lib 크레이트만)

ofgpu-validate         901 / 901 checks passed
                       853개는 실시간 계산, 48개는 기록된 측정값 재생
                       이어서 MISSES 6개와 OPEN 5개를 이름으로 부르는 목록을 출력
```

그 목록은 손으로 유지되는 것이 아니라, 게이트가 자신의 판정을 보고하는 바로 그 지점에서 들어가는 레지스트리로부터 **생성**됩니다(SPEC-LIT §69). 판정을 출력하는 것과 등록하는 것이 같은 호출이므로 열한 개 전부가 매 실행 이름으로 불리며, 열두 번째가 추가되더라도 목록에서 빠질 수 없습니다. **`ofgpu-validate`가 실행하는 모든 항목은 통과합니다. 그것은 "이 프로젝트가 비교하는 모든 발표된 벤치마크를 재현한다"와 다른 진술이며, 둘을 혼동해서는 안 됩니다.**

---

## 빌드와 실행

요구사항: Rust 1.85 이상, Visual Studio 2022 (C++ 워크로드), CUDA Toolkit 13.x. `build.rs`가 `vcvars64.bat`으로 MSVC 환경을 구성하고 모든 `.cu`를 PTX가 아닌 CUBIN으로 컴파일합니다.

```powershell
cd rust
cargo build --release
cargo run --release --bin ofgpu-generate-mesh -- channel ..\cases\channel 200 120 1
cargo run --release --bin ofgpu-k-epsilon     -- ..\cases\channel -iters 4000 -check 400
```

검증 전체는 `cargo run --release --bin ofgpu-validate`입니다. 나머지 열세 개 실행 파일(`ofgpu-fire`, `ofgpu-vof`, `ofgpu-cht`, `ofgpu-datacentre`, `ofgpu-decompose`, 벤치마크들), 케이스 파일 형식, 명령행 옵션은 **사용자 안내서**에 있습니다. 케이스는 JSONC 한 파일 또는 OpenFOAM ASCII 케이스 디렉터리로 읽고 씁니다 — 후자는 ParaView·`foamToVTK` 같은 기존 도구와의 상호운용을 위한 것이며, meteor-cfd는 OpenFOAM의 어떤 부분과도 링크하지 않고 그 소스를 포함하지 않습니다.

---

## 할 수 있는 것

| 영역 | 지원 |
|---|---|
| 이산화 | Gauss linear·upwind·linearUpwind·cubic·QUICK·Gamma·blended, TVD 제한자 6종, Green–Gauss·최소제곱 기울기와 제한자(Barth–Jespersen, Venkatakrishnan), 과이완 비직교 보정, steadyState·Euler·BDF2·국소 시간전진 |
| 압력–속도 | SIMPLE, SIMPLEC, PISO, PIMPLE. Rhie–Chow 보간이며 체적력은 셀 값을 보간하지 않고 면에서 직접 처리 |
| 난류 | RANS: 표준·realizable·RNG k-ε, Wilcox k-ω, Menter SST, Launder–Sharma 저Re, Spalart-Allmaras(4변종). LES: Smagorinsky, WALE, Deardorff. 하이브리드: DES97·DDES·IDDES. 천이: k-ω SST-LM. 벽 처리: 표준·연속 벽함수, `lowRe` 적분, Jayatilleke 열 벽함수, 거칠기 |
| 다상·수송 | VOF(계면 압축, Zalesak FCT, CSF 표면장력, 정적·이력·동적 접촉각), 다성분 화학종, 일반화 뉴턴 점성 6종, Darcy–Forchheimer 다공성, 비Boussinesq 부력 |
| 화재 | 저마하 `p = p0(t) + p~`, EDM 단일 단계와 직렬 2단계, 임계화염온도 소염, 회색 P1과 fvDOM(level-symmetric S4 24 종좌표), WSGG 스펙트럼 복사, 그을음(연기점·지정수율), 개방 복사 경계 |
| 켤레 열전달 | 고체 영역, 접촉저항, 조화평균 계면 전도도, 유입구 하나와 유출구 하나까지의 유체 영역 |
| 환기·데이터센터 | 팬 성능곡선(AMCA 210 보정), 다공성 점프, 습공기(Hyland–Wexler), RCI·RTI·SHI·RHI 지표 |
| 라그랑주 파셀 | SoA 풀, Schiller–Naumann 항력, 양방향 결합, 증발과 증기 결합, Bai-Gosman 벽 충돌 지도 — **라이브러리 API로만** |
| 선형 해법 | PBiCGStab, PCG; Jacobi·다색 DIC·다색 DILU; cuFFT 직접 Poisson과 AMGX(선택 기능); 적용가능성·정확도·실측 시간으로 backend 자동 선택 |
| 메쉬·입출력 | 블록 격자와 grading, STL 계단식 조각, 컷셀, Gmsh v4.1, cyclic 다중 쌍, `empty`/`symmetry` constraint, 2:1 적응 세분화(어느 솔버에도 연결되지 않음); JSONC 케이스와 자동 생성 스키마, OpenFOAM ASCII, VTU·NanoVDB/OpenVDB·USD, 배정밀도 재시작 |

## 할 수 없는 것

- **MPI·다중 GPU 없음.** 분할·헤일로·분할 불변 축약·분산 PCG/PBiCGStab이 모두 구현되고 게이트되어 있으나(§71–§73) 한 장의 카드 위 한 프로세스에서 돕니다. 통신 라이브러리를 링크하지 않으며 강한 확장성 수치도 발표하지 않습니다.
- **압축성·천음속 없음.** 밀도 가중 시간미분은 VOF에서 쓰이지만 압력 방정식은 비압축성입니다.
- **유한율(아레니우스) 화학 없음** — 경직 ODE 적분기도, 야코비안도, 반응 메커니즘도 없습니다.
- **Crank–Nicolson은 저완화 방정식과 함께 쓸 수 없습니다** — 묵시적으로 Euler로 대체하지 않고 사유를 오류로 보고합니다.
- **면대면 복사와 라그랑주 스프레이는 케이스 형식이 없습니다.** 둘 다 명세되고 게이트되어 있으나 어느 드라이버 바이너리도 케이스 파일에서 밀폐공간이나 스프레이를 읽지 않습니다(§50.12, §13.4.2).
- **파셀은 스플래시 자식 액적을 만들지 않고, 막(film) 수송이 없으며, 복사를 흡수하지 않습니다**(§78.11, §68.13).
- **적응 세분화는 어떤 솔버에도 연결되어 있지 않습니다** — 면 플럭스를 옮기지 않고, 적응 후 압력 투영도 없습니다.
- **AMGX는 기본 비활성**이며, 비활성 상태에서도 선택기가 AMGX를 "unavailable"로 명시 보고합니다.
- **DES 계열이 발표된 박리 유동 통계를 재현한다고 주장하지 않으며**, Spalart-Allmaras의 TMR 평판 게이트는 실행하지 않습니다(§57.12, §56.11).

**그리고 빗나가는 게이트 여섯 개.** `ofgpu-validate`가 매 실행 이름으로 부르며, 아래는 그 이름 그대로입니다.

| 게이트 | 판정 |
|---|---|
| §42.8b Gate 2 — NIST Reduced Scale Enclosure 구획 스윕 (Bryner, Johnsson & Pitts, NISTIR 5568) | **MISSES.** 200 kW를 넘으면 예측 천장 CO가 최대 20배 낮음. 진단은 화학이 아니라 환기 — 연소 효율 15–58 % |
| §60.5 Gate 5 — 정사각 밀폐공간 켤레 자연대류 (Kaminski & Prakash 1986) | **MISSES**, 전도 지배 쪽에서 3 % 바를: `Kr = 0.1`에서 −7.11 %, `Kr = 10`에서 −0.07 %. 1차 문헌은 유료라 끝내 읽지 못했고 비교는 Belazizia 등(2012)이라는 **2차 출처**에 대한 것 |
| §62.12 Gate 1-E — WSGG 전방사율 대 RADCAL (Grosshandler, NIST TN 1402) | **MISSES.** 108점 중 58점이 ±10 % 밖, 최악 30.52 %. 편향이 온도에 대해 단조이며, 어느 모형도 진실이 아닙니다 |
| §61.8 Gate 61-A — 예측 그을음 수율 대 Tewarson (프로판 0.024 kg/kg) | **MISSES.** `cases/burnerPlumeResolved.jsonc`에서 262,144셀 중 296셀이 1375 K 창에 도달해 **0.0124 kg/kg — 1.94배**. 통과로 보고하지 않습니다: 그 leg가 2.949 kW 공급에 4.559 kW의 미연 연료를 내보냅니다 |
| §62.12 Gate 4 — NIST 37 cm 버너 복사분율 (Sung 등, NIST TN 2162r1) | **MISSES.** 케이스가 복사분율이 의미를 갖는 상태에 애초에 도달하지 못합니다 — 연소 효율 **226 %** |
| §68.12 Gate 68-C — Theobald(1981) 소방 수류 90회 | **MISSES**, 기체를 정지시킨 채로: 평균 측정 거리의 **61.29 %**. 항력 없는 진공 괄호가 198.65 %이므로 던지는 거리를 결정하는 것은 유입 공기입니다 |

**판정이 `OPEN`인 것이 다섯 개 더** 있으며 같은 목록의 둘째 그룹으로 출력됩니다. 셋은 §32.4의 평판 채널을 측정값이 아니라 **상관식**(Gnielinski 1976)에 견준 것이고, 넷째 `78-D`는 발표된 두 스플래시 기준이 Weber 수로 4.78배 **서로** 어긋나기 때문이며, 다섯째 §88.10 Gate 88-T는 T3A 평판의 측정된 개시 `Re_x`를 구하지 못해 비교를 닫지 못했기 때문입니다.

---

## 자세한 내용

| 문서 | 내용 |
|---|---|
| **사용자 안내서** (별도 페이지) | 빌드, 케이스 파일, 설정 계약, 실행, 출력, 못 하는 것 |
| **기술 안내서** (별도 페이지) | 이산화, 경계조건, 압력–속도, 난류, 저마하, 연소·복사, 검증, GPU 상주·메쉬 적응·성능 |
| [`rust/SPEC-LIT.md`](rust/SPEC-LIT.md) | 수치 명세 86개 절. 모든 수식의 원논문 인용 포함 — 두 안내서는 여기서 뽑아낸 것이며 이것을 대체하지 않습니다 |
| [`rust/PROVENANCE.md`](rust/PROVENANCE.md) · [`LICENSING.md`](LICENSING.md) · [`NOTICE`](NOTICE) | 파일별 출처와 설계 결정, 라이선스 감사 기록, 서드파티 고지 |
| [`cases/README.md`](cases/README.md) · [`docs/README.md`](docs/README.md) | 시험 케이스 형상, 그리고 `docs/`의 색인 — 모델 카탈로그, GPU 이식성, 입출력 재설계와 JSONC 스키마, `ofgpu-fire`의 정식화와 게이트 기록 |

---

## 참고문헌

수치 기법과 모형의 출처입니다. 절 번호는 SPEC-LIT 기준이며, 각 항목은 SPEC-LIT이 실제로 인쇄한 서지사항만 담습니다 — 제목·호수·면수가 없는 곳은 SPEC-LIT이 그것 없이 인용했기 때문입니다. 읽지 않은 출처(유료·접근 불가·의도적으로 열지 않은 것)와 `ofgpu-validate`가 이름으로 인쇄하는 판정은 해당 줄에 표시했습니다.

### 유한체적 이산화
- Jasak, H. (1996). *Error Analysis and Estimation for the Finite Volume Method with Applications to Fluid Flows.* PhD thesis, Imperial College London. `http://hdl.handle.net/10044/1/8335` — §2, §3, §74, §82
- Moukalled, F., Mangani, L., & Darwish, M. (2016). *The Finite Volume Method in Computational Fluid Dynamics.* Springer. — §2, §3, §11, §74, §82
- Ferziger, J. H., & Perić, M. *Computational Methods for Fluid Dynamics*, 3rd ed. Springer (2002). — §2.4, §3.3, §11.1, §11.5, §74, §82
- Patankar, S. V. (1980). *Numerical Heat Transfer and Fluid Flow.* Hemisphere. ISBN 0-89116-522-3. — §3.4, §5.2, §18, §46, §50, §56, §68
### 대류 이산화 기법
- Warming, R. F., & Beam, R. M. (1976). *AIAA Journal*, 14, 1241–1249. — §11.2
- Leonard, B. P. (1979). *Computer Methods in Applied Mechanics and Engineering*, 19, 59–98. — §11.3
- Leonard, B. P. (1991). *Computer Methods in Applied Mechanics and Engineering*, 88, 17–74. — §7 (the NVD framework)
- Khosla, P. K., & Rubin, S. G. (1974). *Computers & Fluids*, 2, 207–209. — §11.1
- Jasak, H., Weller, H. G., & Gosman, A. D. (1999). *International Journal for Numerical Methods in Fluids*, 31, 431–449. — §11.6
- Sweby, P. K. (1984). *SIAM Journal on Numerical Analysis*, 21, 995–1011. — §7 (the TVD framework)
- van Leer, B. (1977). *Journal of Computational Physics*, 23. — §7 (the limiter)
- van Leer, B. (1979). — §7 (the MUSCL limiter)
- van Albada, G. D., van Leer, B., & Roberts, W. W. (1982). *Astronomy and Astrophysics*, 108. — §7
- Roe, P. L. (1986). — §7 (minmod and Superbee)
- Darwish, M., & Moukalled, F. (2003). *International Journal of Heat and Mass Transfer*, 46, 599–611. — §7 (the gradient ratio on an unstructured mesh)
### 기울기 및 제한자
- Barth, T. J., & Jespersen, D. C. (1989). The design and application of upwind schemes on unstructured meshes. *27th Aerospace Sciences Meeting*, AIAA 89-0366. DOI `10.2514/6.1989-366` — §12.2, §75
- Venkatakrishnan, V. (1993). *AIAA Paper 93-0880.* — §12.2 (the smooth variant)
### 시간 적분
- Crank, J., & Nicolson, P. (1947). *Proceedings of the Cambridge Philosophical Society*, 43, 50–67. — §13.1
### 압력–속도 결합
- Patankar, S. V., & Spalding, D. B. (1972). — §5.2 (SIMPLE)
- Van Doormaal, J. P., & Raithby, G. D. (1984). — §5.3 (SIMPLEC)
- Issa, R. I. (1986). *Journal of Computational Physics*, 62, 40–65. — §5.4, §14
- Rhie, C. M., & Chow, W. L. (1983). *AIAA Journal*, 21, 1525–1532. — §5.1, §52
### 난류 모형 — RANS
- Launder, B. E., & Spalding, D. B. (1974). *Computer Methods in Applied Mechanics and Engineering*, 3, 269–289. — §6.1, §6.4
- Wilcox, D. C. *Turbulence Modeling for CFD.* DCW Industries. — §6.2 (the 1988 form); §5.4 there is the source of the Favre-averaged dilatation terms
- Menter, F. R. (1994). *AIAA Journal*, 32, 1598–1605. — §6.3
- Menter, F. R., Kuntz, M., & Langtry, R. (2003). *Turbulence, Heat and Mass Transfer*, 4. — §6.3 (the 2003 revision)
- Launder, B. E., & Sharma, B. I. (1974). *Letters in Heat and Mass Transfer*, 1, 131–138. — §33
- Patel, V. C., Rodi, W., & Scheuerer, G. (1985). *AIAA Journal*, 23, 1308. — §33 (the review of the low-Reynolds-number family)
- Shih, T.-H., Liou, W. W., Shabbir, A., Yang, Z., & Zhu, J. (1995). *Computers & Fluids*, 24, 227–238. Read as **NASA TM-106721 / ICOMP-94-21 (August 1994)**, `https://ntrs.nasa.gov/citations/19950005029`, a US government work in the public domain; **the journal version is paywalled and was not read**. — §40
- Yakhot, V., Orszag, S. A., Thangam, S., Gatski, T. B., & Speziale, C. G. (1992). *Physics of Fluids A*, 4, 1510–1520. Read as **ICASE Report 91-65 / NASA CR-187611 (1991)**, `https://ntrs.nasa.gov/citations/19910021152`, US government-sponsored, public domain via NTRS. — §41
- Yakhot, V., & Orszag, S. A. (1986). *Journal of Scientific Computing*, 1, 3–51. — §41 (the original renormalisation-group derivation)
- Reynolds, W. C. AGARD Report 755 (1987). — §40 (the realizability constraints — positivity of the normal stresses, the Schwarz inequality — that the variable `C_mu` is constructed to satisfy)
- Lumley, J. L. (1978). *Advances in Applied Mechanics*, 18, 123–176. — §40 (realizability as a modelling principle)
- Pope, S. B. *Turbulent Flows* (2000), §10.4. — §40
- Spalart, P. R., & Allmaras, S. R. *AIAA Paper* 92-0439 (1992); also *La Recherche Aérospatiale*, 1 (1994), 5–21. — §56 (the original)
- Allmaras, S. R., Johnson, F. T., & Spalart, P. R. (2012). Modifications and Clarifications for the Implementation of the Spalart-Allmaras Turbulence Model. *ICCFD7-1902.* `https://www.iccfd.org/iccfd7/assets/pdf/papers/ICCFD7-1902_paper.pdf` — a freely distributed conference paper, **the copy actually read**, and the implementation reference. — §56
- NASA / Turbulence Modeling Benchmarking Working Group. *Turbulence Modeling Resource — The Spalart-Allmaras Turbulence Model.* `https://tmbwg.github.io/turbmodels/spalart.html` — US government-authored DOCUMENTATION, not source; quoted to the printed digit. — §56
- Rumsey, C. L., & Spalart, P. R. (2009). *AIAA Journal*, 47, 982–993. — §56 (why the free-stream `nu~/nu` matters)
### 난류 모형 — LES
- Smagorinsky, J. (1963). *Monthly Weather Review*, 91, 99–164. — §6.5
- Deardorff, J. W. (1970). *Journal of Fluid Mechanics*, 41, 453–480. — §16.1
- Deardorff, J. W. (1980). *Boundary-Layer Meteorology*, 18, 495–527. — §6.5 (the model FDS uses)
- Nicoud, F., & Ducros, F. (1999). *Flow, Turbulence and Combustion*, 62, 183–200. — §6.5 (WALE)
- van Driest, E. R. (1956). *Journal of the Aeronautical Sciences*, 23, 1007–1011. — §16.4
- Scotti, A., Meneveau, C., & Lilly, D. K. (1993). *Physics of Fluids A*, 5, 2306–2308. — §16.3
### 난류 모형 — 하이브리드 RANS-LES
- Spalart, P. R., Jou, W.-H., Strelets, M., & Allmaras, S. R. (1997). Comments on the feasibility of LES for wings, and on a hybrid RANS/LES approach. In *Advances in DNS/LES*, Greyden Press, 137–147. — §57 (DES97)
- Shur, M., Spalart, P. R., Strelets, M., & Travin, A. (1999). *Engineering Turbulence Modelling and Experiments 4*, 669–678. — §57 (the `C_DES = 0.65` calibration on the SA background, at `Delta = h_max`)
- Strelets, M. *AIAA Paper* 2001-0879. — §57 (SST-DES, the `k`-equation dissipation form)
- Spalart, P. R., Deck, S., Shur, M., Squires, K. D., Strelets, M., & Travin, A. (2006). *Theoretical and Computational Fluid Dynamics*, 20, 181–195. — §57 (DDES: `r_d`, `f_d`, and the grid-induced separation they fix)
- Shur, M., Spalart, P. R., Strelets, M., & Travin, A. (2008). *International Journal of Heat and Fluid Flow*, 29, 1638–1649 — IDDES. **Paywalled and NOT read**; §57's IDDES equations come from the two open-access restatements below. — §57
- Gritskevich, M. S., Garbaruk, A. V., Schütze, J., & Menter, F. R. (2012). *Flow, Turbulence and Combustion*, 88, 431–449 — the SST-background recalibration. **Paywalled and NOT read**: `C_dt1 = 20`, `c_t = 1.87` and `c_l = 5.0` are carried from a design note's reading of it, defaulted, printed in the banner, and **not independently verified**. — §57
- Herr, F., Radespiel, R., & Probst, A. (2023). Improved Delayed Detached Eddy Simulation with Reynolds-Stress Background Modeling. *arXiv:2301.07223v2*; published in *Computers & Fluids*, 265 (2023) 106014. **Appendix A is a complete restatement of the IDDES formulation** and is where (57.9)–(57.16) come from, equation by equation. Open access, fetched and read in full. — §57
- Savino, A., Griffin, K., Lee, S., Vijayakumar, G., Wu, S., & Sprague, M. (2026). Improving boundary-layer separation prediction by an IDDES turbulence model using a pressure-gradient sensor. *arXiv:2603.08875*, arXiv non-exclusive distribution licence. **Section 2 states SST-IDDES** and is where `C_DES1 = 0.78`, `C_DES2 = 0.61`, `C_w = 0.15` and the simplified filter width (57.18) come from. Open access, read in full. — §57
- Nikitin, N. V., Nicoud, F., Wasistho, B., Squires, K. D., & Spalart, P. R. (2000). *Physics of Fluids*, 12, 1629–1632. — §57 (the log-layer mismatch `f_e` exists to remove)
- Spalart, P. R. (2009). *Annual Review of Fluid Mechanics*, 41, 181–202. — §57 (the review)
- Fröhlich, J., Mellen, C. P., Rodi, W., Temmerman, L., & Leschziner, M. A. (2005). *Journal of Fluid Mechanics*, 526, 19–66. — §57.12, the periodic-hill gate at `Re_b = 10 595`: **named and NOT run**
### 벽 처리
- Spalding, D. B. (1961). *Journal of Applied Mechanics*, 28, 455–458. — §6.4, §15.1
- Cebeci, T., & Bradshaw, P. *Momentum Transfer in Boundary Layers*, Hemisphere (1977). — §15.3 (rough-wall boundary layers; Nikuradse's sand-grain data underlies the constants)
- Jayatilleke, C. L. V. (1969). *Progress in Heat and Mass Transfer*, 1, 193–330. — §29.3 (the sublayer-resistance correction to the thermal log law)
- Werner, H., & Wengle, H. (1991). Large-eddy simulation of turbulent flow over and around a cube in a plate channel. *8th Symposium on Turbulent Shear Flows.* — §30.1
- Tucker, P. G. (1998). *Applied Mathematical Modelling*, 22, 293–305. — §6.6 (the Poisson wall-distance approach)
- Dittus, F. W., & Boelter, L. M. K. (1930). *University of California Publications in Engineering*, 2, 443, reprinted in *International Communications in Heat and Mass Transfer*, 12 (1985) 3. — §32.3. Conventionally quoted at ±20–25 %.
- Gnielinski, V. (1976). *International Chemical Engineering*, 16, 359. — §32.3, ±10 %. **OPEN × 3.** §32.4's three channel verdicts are held against this correlation and none closes at the shipped default: verdict 1 (absolute prediction, resolved leg, at the Petukhov smooth-pipe `f`), and verdict 2 (Reynolds analogy, at each leg's own MEASURED `f`) on both the wall-function leg and the resolved leg. Every such statement must name which `f` Gnielinski was evaluated at — §32.3's own rule.
- Kays, W. M. (1994). *ASME Journal of Heat Transfer*, 116, 284–295. — §32.5, §37 (that `Pr_t` rises towards a wall: a named hypothesis with a mechanism and a direction for the §32.4 verdicts, and nothing here has measured it)
### 부력
- Rehm, R. G., & Baum, H. R. (1978). The equations of motion for thermally driven, buoyant flows. *Journal of Research of the National Bureau of Standards*, 83, 297–308. — §9, §25, §77
- Majda, A., & Sethian, J. (1985). *Combustion Science and Technology*, 42, 185. — §25
- Spiegel, E. A., & Veronis, G. (1960). *Astrophysical Journal*, 131, 442. — §9 (the `ΔT/T << 1` requirement, which a fire plume does not meet)
- Rodi, W. (1987). *Journal of Geophysical Research*, 92, 5305–5328. — §17
- Henkes, R. A. W. M., van der Vlugt, F. F., & Hoogendoorn, C. J. (1991). *International Journal of Heat and Mass Transfer*, 34, 377–388. — §17
### 켤레 열전달
- Carslaw, H. S., & Jaeger, J. C. *Conduction of Heat in Solids*, 2nd ed., Oxford University Press (1959), ch. I. ISBN 0-19-853368-3. — §46 (the anisotropic solid, and the affine transformation that reduces `div(K grad T)` to `lap T`)
- Aavatsmark, I. (2002). An introduction to multipoint flux approximations for quadrilateral grids. *Computational Geosciences*, 6, 405–432. DOI `10.1023/A:1021291114475` — §46.4 (the rigorous full-tensor treatment, and therefore the reason §46.4 refuses rather than approximating)
- Lipnikov, K., Shashkov, M., Svyatskiy, D., & Vassilevski, Yu. (2007). *Journal of Computational Physics*, 227, 492–512. DOI `10.1016/j.jcp.2007.08.008` — §46.4 (the nonlinear monotone alternative, named in the same refusal)
- Yovanovich, M. M. (2005). *IEEE Transactions on Components and Packaging Technologies*, 28, 182–206. DOI `10.1109/TCAPT.2005.848483` — §46.3 (the layered-stack conductivities the Wiener pair homogenises), §47.12 (the review, and the gas-gap and elastic regimes §47.12 omits)
- Giles, M. B. (1997). *International Journal for Numerical Methods in Fluids*, 25, 421–436. DOI `10.1002/(SICI)1097-0363(19970830)25:4<421::AID-FLD557>3.0.CO;2-J` — §47 (the Godunov–Ryabenkii normal-mode analysis behind the classical "Dirichlet on the fluid, Neumann on the solid" rule)
- Meng, F., Banks, J. W., Henshaw, W. D., & Schwendeman, D. W. (2017). A stable and accurate partitioned algorithm for conjugate heat transfer. *Journal of Computational Physics*, 344, 51–85. DOI `10.1016/j.jcp.2017.04.052` — §47.7, **Theorem 1**: the amplification factor that is the reason Dirichlet–Neumann partitioning is not implemented here
- Henshaw, W. D., & Chand, K. K. (2009). *Journal of Computational Physics*, 228, 3708–3741. DOI `10.1016/j.jcp.2009.02.007` — §47 (Robin coefficients can always be chosen so the sub-time-step iteration converges)
- Verstraete, T., & Scholl, S. (2016). *International Journal of Heat and Mass Transfer*, 101, 852–869. DOI `10.1016/j.ijheatmasstransfer.2016.05.041` — §47 (the numerical Biot number, and FFTB's instability above `Bi = 1`)
- Gander, M. J. (2006). Optimized Schwarz methods. *SIAM Journal on Numerical Analysis*, 44, 699–731. DOI `10.1137/S0036142903425409` — §47 (the physical series conductance is the zeroth-order optimised-Schwarz weight; the optimal weight is a non-local operator)
- Cooper, M. G., Mikic, B. B., & Yovanovich, M. M. (1969). Thermal contact conductance. *International Journal of Heat and Mass Transfer*, 12, 279–300. DOI `10.1016/0017-9310(69)90011-8` — §47.12 (the plastic-deformation contact conductance correlation)
- de Vahl Davis, G. (1983). Natural convection of air in a square cavity: a bench mark numerical solution. *International Journal for Numerical Methods in Fluids*, 3, 249–264. DOI `10.1002/fld.1650030305` — §59.8, the fluid-only anchor run first, because a conjugate answer built on an unvalidated buoyant solver measures nothing. **The primary is paywalled**; its four numbers are quoted from Qi et al., *Nanoscale Research Letters*, 8 (2013) 56, DOI `10.1186/1556-276X-8-56`, Table 3 (open access).
- Kaminski, D. A., & Prakash, C. (1986). *International Journal of Heat and Mass Transfer*, 29(12), 1979–1988. DOI `10.1016/0017-9310(86)90017-7` — §47.12's Gate 5, the configuration §60.5 runs. **Paywalled; no open-access copy was found and the primary table was never read**, so no title is asserted for it here either. **MISSES — Gate 5.**
- Belazizia, A., Benissaad, S., & Abboudi, S. (2012). Effect of wall conductivity on conjugate natural convection in a square enclosure with finite vertical wall thickness. *Advanced Theoretical and Applied Mechanics*, 5, no. 4, 179–190. Open access at `m-hikari.com/atam/atam2012/atam1-4-2012/` — an independent published solution of the Kaminski–Prakash configuration, itself validated against it. **The SECONDARY source Gate 5 actually compares against.** — §60.5
- Qu, W., & Mudawar, I. (2002). Analysis of three-dimensional heat transfer in micro-channel heat sinks. *International Journal of Heat and Mass Transfer*, 45, 3973–3985. DOI `10.1016/S0017-9310(02)00101-1` — §47.12's Gate 6, the semiconductor gate. Read in full from the authors' own public copy. **§79.12 runs it and it PASSES**: both substrate temperatures fall inside the experimental uncertainty the paper draws. — §79
- Kawano, K., Minakami, K., Iwasaki, H., & Ishizuka, M. Micro channel heat exchanger for cooling electrical equipment. *ASME HTD-361-3/PID-3* (1998) 173–180. — the inlet and outlet thermal-resistance measurements Gate 6 is held against. **NOT OBTAINED** (an ASME conference volume; no copy found), so the comparison is a **digitisation of Qu & Mudawar's Fig. 4**, which §79.12's Disclosure 1 states. — §79
### 연소, 그을음, 복사
- Magnussen, B. F., & Hjertager, B. H. (1977). *Proceedings of the Combustion Institute*, 16, 719–729. — §27 (the eddy-dissipation model), §61.3 (its soot-burnout rate, the Khan–Greeves partner)
- McGrattan, K., McDermott, R., & Floyd, J. E. (2022). A simple two-step reaction scheme for soot and CO. *Proceedings of the Tenth International Seminar on Fire and Explosion Hazards (ISFEH10)*, Oslo, 23–27 May 2022. `https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=927294` — a NIST work, US public domain; fetched and read in full, and its Eqs. (1)–(5) are the model implemented. — §42
- McGrattan, K., Hostikka, S., McDermott, R., Floyd, J., Weinschenk, C., Overholt, K., Vanella, M., et al. *Fire Dynamics Simulator Technical Reference Guide*, NIST SP 1018-1. NIST, US public domain; read locally from `reference/fds/Manuals/` with `reference/fds/LICENSE.md` read verbatim. **No FDS source code was read for these sections.** — §25, §42, §43, §66, §68, §76
- Beyler, C. (2016). Flammability limits of premixed and diffusion flames. Chapter in *SFPE Handbook of Fire Protection Engineering*, 5th ed. — §43 (the critical-flame-temperature and auto-ignition values, as quoted by two independent NIST sources both read here)
- Morehart, J. H., Zukoski, E. E., & Kubota, T. NIST-GCR-90-585 (1991). — §43 (the self-extinction bracket, as quoted by the FDS Technical Reference Guide)
- Bryner, N., Johnsson, E., & Pitts, W. NISTIR 5568 (1994). — §42.8b, the NIST Reduced Scale Enclosure compartment sweep, 50–600 kW: measured ceiling CO volume fraction below `0.0005` under 100 kW, rising to `0.02–0.035` at 400–640 kW. **MISSES — Gate 2.** The published statistics for this model over the RSE 1994 / RSE 2007 / FSE 2008 set are model bias 1.08 and model relative standard deviation 0.50 against an experimental relative standard deviation of 0.19; that is the bar.
- Steckler, Quintiere & Rinkinen (1982) — the compartment doorway-flow measurements. Named in §42.8b as the prerequisite the Reduced Scale Enclosure miss points to, and **NOT run**; the paper itself has not been read here, so no report number or title is asserted for it.
- Lautenberger, C. W., de Ris, J. L., Dembsey, N. A., Barnett, J. R., & Baum, H. R. (2005). A simplified model for soot formation and oxidation in CFD simulation of non-premixed hydrocarbon flames. *Fire Safety Journal*, 40(2), 141–176. DOI `10.1016/j.firesaf.2004.10.002`; author manuscript free at `https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=101148` — §61 (the laminar-smoke-point model, and every constant in §61.3)
- Kent, J. H., & Honnery, D. (1990). *Combustion and Flame*, 79, 287. — §61 (the measured formation-rate map the smoke-point polynomials are shaped to)
- Tewarson, A. *SFPE Handbook*, ch. 36, Table A.40, as quoted in the FDS Validation Guide (NIST, public domain). — §61.8, propane post-flame soot yield `0.024 kg/kg`. **MISSES — Gate 61-A**: `0.000` here; `0.0124` (a factor of 1.94) on the resolved burner of §85.10.
- Modest, M. F. *Radiative Heat Transfer*, 3rd ed., Academic Press (2013), ch. 5, 11, 15–17. — §28, §36, §50, §62, §65
- Fiveland, W. A. Discrete-ordinates solutions of the radiative transport equation for rectangular enclosures. *Journal of Heat Transfer*, 106 (1984) 699. DOI `10.1115/1.3246741` — §36, §65
- Truelove, J. S. Discrete-ordinate solutions of the radiation transport equation. *Journal of Heat Transfer*, 109 (1987) 1048. DOI `10.1115/1.3248182` — §36, §65 (the S_N quadrature sets and the two conditions that fix their directions and weights; the S4 digits in §36.2 are this crate's own closed-form solve, not a copied table)
- Hottel, H. C., & Sarofim, A. F. *Radiative Transfer*, McGraw-Hill (1967), ch. 3, 5. — §50 (the net-radiation exchange method; the method of images, named in §50.9's refusal), §62 (the weighted-sum construction itself)
- Bordbar, M. H., Węcel, G., & Hyppänen, T. (2014). A line by line based weighted sum of gray gases model for inhomogeneous CO2–H2O mixture in oxy-fired combustion. *Combustion and Flame*, 161(9), 2435–2445. DOI `10.1016/j.combustflame.2014.03.013` — §62, the coefficient set implemented. **The paper is paywalled** (ScienceDirect returned HTTP 403 to every attempt); the 168 coefficients are transcribed from NIST's public-domain `reference/fds/Source/radi.f90`, and its own tabulated emissivities could not be obtained, which is why Gate 1-E measures against RADCAL instead.
- Grosshandler, W. *RADCAL: A Narrow-Band Model for Radiation Calculations in a Combustion Environment*, NIST Technical Note 1402 (1993). DOI `10.6028/NIST.TN.1402` — US public domain; NIST's own implementation ships at `reference/fds/Source/rcal.f90` and `tools/radcal_emissivity/` compiles it **unmodified** behind a standalone driver. — §62.12, the reference the total emissivity is measured against. **MISSES — Gate 1-E.** RADCAL is an independent model, not truth: a disagreement says which way and by how much, and convicts neither.
- Sung, Chen, Bundy, Fernandez & Hamins. NIST Technical Note 2162r1 (2021), as documented in the FDS Validation Guide (NIST, public domain). — §62.13, the NIST 37 cm gas burner's measured radiative fractions 0.23 / 0.30 / 0.33 at 20 / 34 / 50 kW. **MISSES — Gate 4.**
- Walton, G. N. *Calculation of Obstructed View Factors by Adaptive Integration.* NISTIR 6925, National Institute of Standards and Technology, November 2002. `https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nistir6925.pdf` — US Government, public domain. §49 (the double area integral and its dot-product form, the obstruction-elimination tests, the row-sum figure of merit, and the `BB104` benchmark)
- Shapiro, A. B. *FACET — A Radiation View Factor Computer Code for Axisymmetric, 2D Planar and 3D Geometries with Shadowing.* UCID-19887, Lawrence Livermore National Laboratory, 1983. DOI `10.2172/5607653` — US DOE, public domain. §49.8 (the shadowed configuration `F_12 = 0.115621`)
- Howell, J. R. *A Catalog of Radiation Heat Transfer Configuration Factors*, 3rd ed. `https://www.thermalradiation.net/` — entries **C-11** (identical parallel directly-opposed rectangles) and **C-14** (two rectangles of equal length sharing an edge at 90°), both tracing to Hottel (1931) and Hamilton & Morgan (1952). §49.8 (the two analytic view-factor gates)
- Gebhart, B. (1961). *International Journal of Heat and Mass Transfer*, 3(4), 341–346. DOI `10.1016/0017-9310(61)90048-5` — §50.2, the absorption-factor alternative: **named and not used**
- Balaji, C., & Venkateshan, S. P. *International Journal of Heat and Fluid Flow*, 14(3) (1993) 260–267, DOI `10.1016/0142-727X(93)90057-T`, and 15(3) (1994) 249–251, DOI `10.1016/0142-727X(94)90046-9`; Akiyama, M., & Chong, Q. P. *Numerical Heat Transfer A*, 32(4) (1997) 419–433, DOI `10.1080/10407789708913899` — the coupled convection-plus-surface-radiation cavity gate. — §50.12, **NOT run**: the tabulated `Nu_conv`/`Nu_rad` are paywalled and the fluid side has no case format for a radiating enclosure.
### 유변학과 접촉각
- Ostwald, W. (1925). *Kolloid-Zeitschrift*, 36, 99–117; de Waele, A. (1923). — §38 (the power law)
- Cross, M. M. (1965). *Journal of Colloid Science*, 20, 417–437. — §38
- Carreau, P. J. (1972). *Transactions of the Society of Rheology*, 16, 99–127; Yasuda, K., Armstrong, R. C., & Cohen, R. E. (1981). *Rheologica Acta*, 20, 163–178. — §38 (one formula serves both)
- Herschel, W. H., & Bulkley, R. (1926). *Kolloid-Zeitschrift*, 39, 291–300. — §38
- Casson, N. In Mill (ed.), *Rheology of Disperse Systems*, Pergamon (1959), 84–104. — §38
- Papanastasiou, T. C. (1987). *Journal of Rheology*, 31, 385–404. — §38.3 (the regularisation, in the **product** form)
- Bercovier, M., & Engelman, M. (1980). *Journal of Computational Physics*, 36, 313–326. — §38.3 (the alternative regularisation)
- Frigaard, I. A., & Nouar, C. (2005). *Journal of Non-Newtonian Fluid Mechanics*, 127, 1–26. — §38.3 (what regularisation costs)
- Bird, R. B., Armstrong, R. C., & Hassager, O. *Dynamics of Polymeric Liquids*, vol. 1, 2nd ed., Wiley (1987). — §38 (the family)
- Chhabra, R. P., & Richardson, J. F. *Non-Newtonian Flow and Applied Rheology*, 2nd ed. (2008). — §38.9 (Buckingham–Reiner)
- Young, T. (1805). *Philosophical Transactions of the Royal Society*, 95, 65–87. — §39 (the equilibrium angle)
- Huh, C., & Scriven, L. E. (1971). *Journal of Colloid and Interface Science*, 35, 85–101. — §39 (the moving contact-line singularity)
- Voinov, O. V. (1976). *Fluid Dynamics*, 11, 714–721; Cox, R. G. (1986). *Journal of Fluid Mechanics*, 168, 169–194. — §39.4 (the asymptotic matching)
- Hoffman, R. L. (1975). *Journal of Colloid and Interface Science*, 50, 228–241. — §39.4 (the master curve)
- Jiang, T.-S., Oh, S.-G., & Slattery, J. C. (1979). *Journal of Colloid and Interface Science*, 69, 74–77. — §39.4 (the explicit correlation used here; Kistler's fit is **deliberately absent**, its four constants coming from a book chapter this project has not read)
- Afkhami, S., Zaleski, S., & Bussmann, M. (2009). *Journal of Computational Physics*, 228, 5370–5389 — the mesh-dependent (numerical-slip) angle. §39.8, **named and deliberately not implemented** until the gate that would show it works exists
- Sui, Y., Ding, H., & Spelt, P. D. M. (2014). *Annual Review of Fluid Mechanics*, 46, 97–119. — §39 (the review)
- Washburn, E. W. (1921). *Physical Review*, 17, 273–283. — §39.7 (capillary rise)
### 라그랑주 입자
- Dukowicz, J. K. A particle-fluid numerical model for liquid sprays. *Journal of Computational Physics*, 35 (1980) 229–253. DOI `10.1016/0021-9991(80)90087-X` — §66 (the discrete droplet model: the parcel, and the real-valued weight `n_p`)
- Crowe, C. T., Sharma, M. P., & Stock, D. E. The particle-source-in-cell (PSI-CELL) model for gas-droplet flows. *Journal of Fluids Engineering*, 99 (1977) 325. DOI `10.1115/1.3448756` — §67, §68 (the per-cell sum every coupled source is)
- Crowe, C., Sommerfeld, M., & Tsuji, Y. *Multiphase Flows with Droplets and Particles*, CRC Press (1998). ISBN 0-8493-9469-4 — §66.2 (the equation of motion, and the regime argument for which of its terms survive)
- Maxey, M. R., & Riley, J. J. *Physics of Fluids*, 26 (1983) 883. DOI `10.1063/1.864230` — §66, §68 (the equation of motion: the added-mass coefficient, and the drag term §68 returns to the gas)
- Schiller, L., & Naumann, A. *Zeitschrift des Vereines Deutscher Ingenieure*, 77 (1933) 318, in the form compiled by Clift, R., Grace, J. R., & Weber, M. E. *Bubbles, Drops, and Particles*, Academic Press (1978). ISBN 0-12-176950-X — §66.3 (the drag correlation)
- Macpherson, G. B., Nordin, N., & Weller, H. G. *Communications in Numerical Methods in Engineering*, 25 (2009) 263. DOI `10.1002/cnm.1128` — §66.6, barycentric tracking: the **paper** was read and it is **not implemented** — the one case the face-crossing walk cannot do. Its OpenFOAM implementation is GPL-3.0 and was **not** opened.
- Elghobashi, S. On predicting particle-laden turbulent flows. *Applied Scientific Research*, 52 (1994) 309. DOI `10.1007/BF00936835` — §67, §68 (the coupling map: below `alpha_p ~ 1e-6` one-way coupling suffices, `1e-6`–`1e-3` needs §68, above `1e-3` collisions matter and are not here)
- Satish, N., Harris, M., & Garland, M. Designing efficient sorting algorithms for manycore GPUs. *IEEE IPDPS 2009.* DOI `10.1109/IPDPS.2009.5161005` — the **paper** was read; no implementation of it was opened. §67.4 (the three-phase radix pass)
- Merrill, D., & Grimshaw, A. *Parallel scan for stream architectures.* University of Virginia Technical Report CS2009-14. — §67.2 (the reduce-then-scan decomposition)
- Blelloch, G. E. *Prefix sums and their applications.* CMU-CS-90-190 (1990). — §67.2 (the exclusive scan and its work-efficiency argument)
- Hillis, W. D., & Steele, G. L., Jr. *Communications of the ACM*, 29(12) (1986) 1170. DOI `10.1145/7902.7903` — §67.2 (the in-block scan network, chosen for a property that is not speed)
- Steele, G. L., Jr., Lea, D., & Flood, C. H. Fast splittable pseudorandom number generators. *OOPSLA 2014*, ACM SIGPLAN Notices, 49(10) 453. DOI `10.1145/2660193.2660195` — §66.9 (the SplitMix64 finalising mix, used as a **bijection** and not as a generator)
- Ranz, W. E., & Marshall, W. R. Evaporation from drops. *Chemical Engineering Progress*, 48 (1952) 141–146 (Part I) and 173–180 (Part II). — §68.5 (the sensible-heat half of `Nu_0 = 2 + 0.6 Re^(1/2) Pr^(1/3)`), §76 (the mass-transfer half, and the 56 suspended-droplet experiments §76.12's first gate measures)
- Spalding, D. B. The combustion of liquid fuels. *4th Symposium (International) on Combustion* (1953) 847–864; and *Convective Mass Transfer: An Introduction*, Edward Arnold (1963). — §76.6 (`B_M`, and the Stefan-flow rate)
- Godsave, G. A. E. Studies of the combustion of drops in a fuel spray. *4th Symposium (International) on Combustion* (1953) 818–830. — §76.9 (the heat-limited rate at the boiling point)
- Abramzon, B., & Sirignano, W. A. Droplet vaporization model for spray combustion calculations. *International Journal of Heat and Mass Transfer*, 32 (1989) 1605–1618. DOI `10.1016/0017-9310(89)90043-4` — §76.6 (`B_T = (1 + B_M)^φ − 1`, the default)
- Sazhin, S. S. Advanced models of fuel droplet heating and evaporation. *Progress in Energy and Combustion Science*, 32 (2006) 162–214. DOI `10.1016/j.pecs.2005.11.001` — §76
- Watson, K. M. Thermodynamics of the liquid state. *Industrial & Engineering Chemistry*, 35 (1943) 398–406. — §76.4 (`h_v(T)`)
- Marrero, T. R., & Mason, E. A. Gaseous diffusion coefficients. *Journal of Physical and Chemical Reference Data*, 1 (1972) 3–118. DOI `10.1063/1.3253094` — §76.4
- Lewis, W. K. The evaporation of a liquid into a gas. *Transactions of the ASME*, 44 (1922) 325–340. — §76.13
- NIST Chemistry WebBook, SRD 69. US government, public domain. — §76.4 (the water-vapour specific heat and the critical constants)
- Theobald, R. C. The effect of nozzle design on the stability and performance of turbulent water jets. *Fire Safety Journal*, 4 (1981) 1–13. — §68.12, about 90 hose-stream experiments. **MISSES — Gate 68-C**, with the gas held at rest.
- Bai, C., & Gosman, A. D. *SAE 950283* (1995). — §78.1, §78.4 (the impact regime map, and the alternative splash threshold)
- Mundo, C., Sommerfeld, M., & Tropea, C. *International Journal of Multiphase Flow*, 21 (1995) 151. — §78.4 (`K = Oh Re^1.25`, `K_crit = 57.7`, the default). The experimental data itself was **not transcribed**, which is why Gate 78-D is open. **OPEN — Gate 78-D**: the two published splash criteria disagree by a measured factor in Weber number for the same droplet.
- Yarin, A. L. *Annual Review of Fluid Mechanics*, 38 (2006) 159. — §78.1 (the review that explains why neither threshold is better than approximately right)
- IAPWS R1-76 (2014). — §78.2 (`sigma = B tau^mu (1 + b tau)`, implemented verbatim and gated against the release's own table)
### 다상 유동
- Hirt, C. W., & Nichols, B. D. *Journal of Computational Physics*, 39 (1981) 201–225. — §20.1
- Zalesak, S. T. *Journal of Computational Physics*, 31 (1979) 335–362. — §20.2 (and §22's rotating-slotted-disc boundedness check)
- Brackbill, J. U., Kothe, D. B., & Zemach, C. *Journal of Computational Physics*, 100 (1992) 335–354. — §20.4, §87 (the continuum-surface-force regularisation)
- Ubbink, O. PhD thesis, Imperial College London (1997). — §20.1 (the interface-compressed finite-volume form on unstructured meshes)
- Rusche, H. PhD thesis, Imperial College London (2002). — §20.1 (the same)
### 선형 해법
- Saad, Y. *Iterative Methods for Sparse Linear Systems*, 2nd ed. SIAM (2003). DOI `10.1137/1.9780898718003` — §8, §21 (§6.7 PCG, §7.4.2 BiCGStab, §12.4 multicolour ILU, ch. 14 block Jacobi and additive Schwarz)
- van der Vorst, H. A. Bi-CGSTAB: A Fast and Smoothly Converging Variant of Bi-CG for the Solution of Nonsymmetric Linear Systems. *SIAM Journal on Scientific and Statistical Computing*, 13(2), 631–644 (1992). DOI `10.1137/0913035` — §8.1
- Hestenes, M. R., & Stiefel, E. Methods of conjugate gradients for solving linear systems. *Journal of Research of the National Bureau of Standards*, 49(6), 409 (1952). DOI `10.6028/jres.049.044` — §8.2. A US Government work, public domain.
- Swarztrauber, P. N. *SIAM Review*, 19 (1977) 490–501. — §8.5
- Stüben, K. *Journal of Computational and Applied Mathematics*, 128 (2001) 281–309; Ruge & Stüben (1987). — §8.3. Provided here by AMGX (BSD-3-Clause), **not reimplemented**.
### 다공성 매질
- Ward, J. C. Turbulent flow in porous media. *Journal of the Hydraulics Division, ASCE*, 90(5) (1964) 1–12. DOI `10.1061/JYCEAJ.0001096` — §18, §53 (the same Darcy–Forchheimer law integrated through a slab instead of over a cell)
- Idelchik, I. E. *Handbook of Hydraulic Resistance*, 4th ed., Begell House (2007). ISBN 978-1-56700-251-5, Diagrams 8-1 to 8-6 — perforated plates and screens, the source of `K(sigma)`. **Not opened for §53**; the thin-plate form used is the one published in the open literature, and §53.7 gates it against its own limits. — §53
- Karki, K. C., Radmehr, A., & Patankar, S. V. Use of computational fluid dynamics for calculating flow rates through perforated tiles in raised-floor data centers. *HVAC&R Research*, 9(2) (2003) 153–166. DOI `10.1080/10789669.2003.10391062` — §53.8, the per-tile flow-rate gate: **NOT run** — the paper was not reachable from this environment.
- Karki, K. C., & Patankar, S. V. Airflow distribution through perforated tiles in raised-floor data centers. *Building and Environment*, 41(6) (2006) 734–744. DOI `10.1016/j.buildenv.2005.03.005` — §53
### 환기, 습공기, 데이터센터 지표
- AMCA 210 / ASHRAE 51, *Laboratory Methods of Testing Fans for Certified Aerodynamic Performance Rating.* — §52 (what a manufacturer's curve **is** — a static-pressure rise against volumetric flow at a stated density and shaft speed — and therefore why §52.5 carries a density and a speed correction rather than treating the table as absolute)
- NIST, *Fire Dynamics Simulator* 6 verification suite, `Verification/HVAC/fan_test.fds` and `qfan_test.fds` with their published `.csv` reference values — US Government public domain. **The case files and their reference numbers** are the external cross-check of §52.12 Gate 52-B. `Source/hvac.f90` was read for the DISCIPLINE only: that a fan curve is scaled by `rho/rho_curve` at every evaluation, and that its tabulated branch resolves the operating point by a bisection with a data-dependent trip count, which is correct for a CPU code and uncapturable here (§52.7).
- Buzbee, B. L., Dorr, F. W., George, J. A., & Golub, G. H. The direct solution of the discrete Poisson equation on irregular regions. *SIAM Journal on Numerical Analysis*, 8(4) (1971) 722–736. DOI `10.1137/0708066` — §52.9, the capacitance-matrix path: **named and refused**
- ASHRAE. *ASHRAE Handbook—Fundamentals*, Chapter 1, "Psychrometrics", ASHRAE (2021). — §54.2 (whose equation numbering is used), §54.8 (Table 2, the external comparison, at 101.325 kPa)
- Hyland, R. W., & Wexler, A. Formulations for the thermodynamic properties of the saturated phases of H2O from 173.15 K to 473.15 K. *ASHRAE Transactions*, 89(2A) (1983) 500–519. — §54.2 (the `C1`–`C13` coefficients), §76.4 (reused rather than re-fitted)
- Herrmann, S., Kretzschmar, H.-J., & Gatley, D. P. Thermodynamic properties of real moist air, dry air, steam, water, and ice (RP-1485). *HVAC&R Research*, 15(5) (2009) 961–986. DOI `10.1080/10789669.2009.10390874` — §54.3, **named and not implemented**; the enhancement factor it carries is what makes the ideal relations 0.44 % low in `W_s` at 25 °C, which §54.8 prints rather than tolerates.
- Gatley, D. P., Herrmann, S., & Kretzschmar, H.-J. A twenty-first century molar mass for dry air. *HVAC&R Research*, 14(5) (2008) 655–662. DOI `10.1080/10789669.2008.10391032` — §54 (where `M_a = 28.966 g/mol`, and hence `eps = 0.621945`, come from)
- Herrlin, M. K. Rack cooling effectiveness in data centers and telecom central offices: the Rack Cooling Index (RCI). *ASHRAE Transactions*, 111(2) (2005) 725–731. *(ASHRAE Transactions of this vintage carries no DOI; stable record `https://www.semanticscholar.org/paper/99b942df4aa448a1e06f77d36b48d5d52a40c6e0`.)* — §55.1
- Herrlin, M. K. Airflow and cooling performance of data centers: two performance metrics. *ASHRAE Transactions*, 114(2) (2008) 182–187. *(No DOI; same caveat.)* — §55.2 (RTI)
- Sharma, R. K., Bash, C. E., & Patel, C. D. Dimensionless parameters for evaluation of thermal design and performance of large-scale data centers. *AIAA 2002-3091* (2002). DOI `10.2514/6.2002-3091` — §55.3 (SHI and RHI)
- ASHRAE Technical Committee 9.9. *Thermal Guidelines for Data Processing Environments*, 5th ed., ASHRAE (2021). ISBN 978-1-947192-90-4 — §55.1 (the Class A1–A4 **recommended**, 18–27 °C, and **allowable** envelopes RCI is measured against)
- Wibron, E., Ljung, A.-L., & Lundström, T. S. *Energies*, 12(8) (2019) 1473. DOI `10.3390/en12081473` — **CC-BY-4.0, licence verified live through the Crossref REST API**, but **the full text was not reachable from this environment**, so §55.8's six-configuration ranking gate is **NOT run** and only the one relation the abstract states is gated. — §55.8
### 검증 데이터
- Ghia, U., Ghia, K. N., & Shin, C. T. *Journal of Computational Physics*, 48 (1982) 387–411. — the lid-driven cavity's tabulated centreline profiles
- Moser, R. D., Kim, J., & Mansour, N. N. *Physics of Fluids*, 11 (1999) 943. — DNS channel profiles at `Re_tau` 180 / 395 / 590, and the sublayer `k+ ≈ C_v (y+)^2` with `C_v ≈ 0.07` of §15.2
- Driver, D. M., & Seegmiller, H. L. *AIAA Journal*, 23 (1985) 163–171. — the backward-facing step's reattachment length, `x_r/h = 6.26 ± 0.10`. Named in §22 and **NOT run** by §41's section.
- McCaffrey, B. J. **NBS TN 910** (1979). — the buoyant plume's centreline temperature and velocity decay correlations, `ΔT ~ z^{−5/3}` in the plume region
- Martin, J. C., & Moyce, W. J. *Philosophical Transactions of the Royal Society A*, 244 (1952) 312. — the dam break's surge-front position against time

---

## 문의

**simul@msimul.com** · 주식회사 메테오시뮬레이션 / Meteo Simulation Co., Ltd.

교육과 학술 연구는 무료입니다. 기업 연구개발, 학교에 속하지 않은 연구기관, 수탁
및 상업적 이용은 별도 라이선스가 필요합니다 — [`LICENSE`](LICENSE) 제2·3절.
적용 범위가 불분명한 경우 문의해 주시기 바랍니다.
