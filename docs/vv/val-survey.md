# VAL-R 조사 — validation 실험 자료 후보 (2026-10-27)

조사 범위: docs/19-vv-plan.md §5.3 VAL-R. 저장소는 건드리지 않았고 cargo도 돌리지 않았다.
읽은 것은 모두 실제로 연 페이지와 파일이다(WebFetch, curl, gh api, figshare API). 내려받은 사본은
`scratchpad/ver/dl/` 아래에 있다(git 밖). sha256은 요청대로 계산하지 않았다.

이미 있는 키(reference/PROVENANCE.md)와 겹치지 않게 했다: `driver-seegmiller1985`(TMR backstep),
`mccaffrey1979-table1`, `ercoftac-t3a/t3b/t3am`, `rae2822-*`, `tmr-ks-cf`, `tmr-coles-uplus`,
`mkm99-chan*`, `tnf-flameD-*`, `martin-moyce-1952` 등. 아래 후보는 모두 새 id다.

## 0. 공통 결론 먼저

1. **TMR 라이선스가 이제 글로 확인된다.** TMR은 GitHub(TMBWG/turbmodels)로 옮겨졌고 저장소의
   `LICENSE`는 CC0 1.0이다. 실제로 읽은 첫 줄: *"Creative Commons Legal Code / CC0 1.0 Universal"*.
   README: *"This is a public access version of what was originally created by and known as the NASA
   Langley Research Center"*. 홈페이지 면책: *"They are provided as-is and accuracy is not guaranteed;
   questions should be directed to the sources of the data provided."* 개별 validation 페이지에는
   라이선스 문구가 없다. → TMR 파일은 `kind = file`(git 추적)이 맞다. 기존 행은 "US Government
   work, public domain"이라고만 적었는데, 앞으로는 CC0 문구를 인용하면 된다.
   - 주의: CC0는 TMBWG가 가진 권리만 포기한다. Delville 혼합층(원본 ERCOFTAC C.34 /
     AGARD AR-345)과 Coles–Wadcock NACA 4412처럼 NASA 밖에서 온 데이터는 출처 문제가 남는다.
     숫자 표(사실)라서 위험은 낮지만 provenance 행에 "TMR이 재배포한 사본"이라고 적어야 한다.
2. **격자 입력이 공통 선행 작업이다.** TMR 격자는 Plot3D(`.p2dfmt/.p3dfmt.gz`), CGNS, UGRID로
   배포된다. 이 트리에는 `rust/src/io/polymesh.rs`, `io/msh.rs`, `mesh/wedge.rs`가 있지만 Plot3D와
   CGNS reader는 없다. validate.rs:4667은 TMR 평판을 *"its grid family is a curvilinear CGNS C-grid,
   and blockgen builds axis-aligned graded blocks"* 이유로 돌리지 않는다고 적고 있다.
   → **2D Plot3D 구조격자를 한 셀 두께 polyMesh로 바꾸는 변환기**(tools/ 아래 Python 스크립트가
   가장 싸다. 축대칭 케이스는 wedge로 회전) 하나만 있으면 아래 TMR 케이스가 모두 열린다. 이것을
   W1의 첫 유닛으로 둔다.
3. **실험 불확도가 파일에 있는 곳은 드물다.** 수치로 확인한 것은 hump의 Cf(열 하나), hump의
   reattachment 1.11 ± 0.003, Windsor/DrivAer의 yaw ±0.1° 정도다. 나머지 TMR 페이지에는 "No
   experimental uncertainty"라고 되어 있다. 게이트 허용오차는 우리가 정해야 하고, 그 결정은
   gate tolerance이므로 사용자 몫이다(docs/17의 decided items 규칙).

---

## 1. 우선순위 A — NASA TMR validation 케이스

공통: 베이스 URL `https://tmbwg.github.io/turbmodels/`. 데이터 파일은 Tecplot ASCII 형식
(`variables=`, `zone t=`), 주석은 `#`.

### 1.1 2DWMH — NASA wall-mounted hump (nasahump_val.html) ★

| 파일 | 내용 | 점 수 |
|---|---|---|
| `Nasahump_validation/noflow_cp.exp.dat` | centerspan Cp(x/c) | ~125 |
| `Nasahump_validation/noflow_cf.exp.dat` | Cf(x/c) **+ "Uncertainty(C_f)" 열**, x/c −0.07…(분리 구간 포함) | ~135 |
| `Nasahump_validation/noflow_u_inflow.exp.dat` | x/c = −2.14 유입 BL 프로파일, "Y, mm" / "U, m/s" | ~100 |
| `Nasahump_validation/noflow_vel_and_turb.exp.dat` | x/c, y/c, u/U∞, v/U∞, uu, vv, uv /U∞² — 8 zones: x/c = 0.65, 0.66, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3 | ~500 |
| `…/noflow-piv-exp.zip` (nasa.gov) | 2D PIV 원자료 | 미확인 |
| `Other_exp_Data/CFDVAL2004/case3expdata.html` | 전체 Case 3 자료(흐름 제어 포함) | — |

- 조건: Re_c = 936,000(워크숍 값; Greenblatt 2006은 929,000), U∞ ≈ 34.6 m/s, M = 0.1, chord 420 mm,
  x/c = −2.14에서 BL 두께 ≈ 35 mm. no-flow-control 케이스는 공동 벽을 통한 질량 유량이 없다.
- 불확도(실제로 읽은 문구): Cf 파일 머리말 *"#value range = Cf +- Uncertainty(C_f)"*. 값 예:
  x/c = −0.07에서 0.00123 ± 0.00009, x/c = 0.62에서 0.00505 ± 0.00030. case3 페이지:
  *"Flow reattachment location was x/c=1.11 +/- 0.003 for the no-flow-control condition"*. PIV는
  *"2-3% in the mean at x/c=1.3"*, u-rms는 통계 수렴 대역 16%. Cf 부호는 x/c = 1.09와 1.15 사이에서
  바뀌고(파일에 그 사이 점은 없다), 이것은 1.11과 맞는다.
- 격자: `nasahump_grids.html` no-plenum 패밀리 `hump2newtop_noplenumZ{1633x433, 817x217, 409x109,
  205x55, 103x28}`(p2dfmt/p3dfmt/cgns.gz). 위쪽 경계는 end-plate blockage를 흉내 낸 contoured slip
  wall이고 격자에 이미 들어 있다. 경계조건: 입구 Pt/Tt, 출구 p(배압으로 맞춤). 값은 그림에만 있다.
  비압축 코드라면 입구 속도 + 출구 압력으로 바꿔 쓰면 된다. 유입 BL 프로파일 파일이 있어서 입구
  위치에서 BL 두께를 확인할 수 있다.
- 라이선스: 페이지에 문구가 없다 → 저장소 CC0(§0).
- REDUCED 비용: 2D, 205x55 ≈ 11k 셀(또는 409x109 ≈ 44k), SA/SST 정상 상태. 기존 132-Z 평판과
  같은 규모라 한 GPU에서 10분 안에 들어갈 것으로 본다(추정, 측정 안 함). 알려진 위험: RANS는
  기포를 길게 예측한다(페이지: *"tend to predict too long a separation bubble"*, SA는 대략 1.25
  수준). 그래서 reattachment 게이트는 실험 ±0.003으로는 MISSED가 날 수 있다.
  "misses proceed and log" 규칙대로 기록한다.

### 1.2 ATB — axisymmetric transonic bump, Bachalo–Johnson (axibump_val.html)

- 파일: `Axi_Bump_validation/cp.exp.dat`(Cp(x/c), M = 0.875, 약 33점), `exp.uprofile.dat`(u/U∞
  대 y'/c, 8 zones x/c = −0.25, 0.688, 0.813, 0.938, 1.0, 1.125, 1.25, 1.375), `exp.upvpprofile.dat`
  (u'v'), `exp.kprofile.dat`(TKE), `exp.alldata.875.dat`, 표면 `Axi_Bump_grids/bumpgrd.surfpts_orig.dat`.
  Sandia가 다시 한 실험(Cf 포함)은 `Other_exp_Data/sandia_transonicaxibump_exp.html`.
- 조건: M = 0.875, 원호 bump(h 1.905 cm, 길이 20.32 cm), 원통 지름 15.24 cm, BL ≈ 1 cm, Tu ≈ 0.6%.
  Re는 본문에 없고 BC 그림에만 있다.
- 불확도: 없음. 터널 간 shock 위치 차이 0.01–0.015 c는 비교값이고 불확도가 아니다.
- 격자: `bump_newaxi_{1441,721,361,181,91}`(2 x 1441x641 … 2 x 91x41). rotated periodic wedge,
  far-field Riemann.
- REDUCED: wedge 181x81 ≈ 15k 셀. 하지만 **천음속 압축성 RANS + shock**가 필요하다. 압축성
  경로에 SA/SST가 연결되어 있는지 따로 확인해야 하므로 첫 5개에서는 뺀다.

### 1.3 2DN00 — NACA 0012 (naca0012_val.html) ★

- 파일(`NACA0012_validation/`):
  - `CLCD_Ladson_expdata.dat` — Ladson NASA TM 4074(1988), Re = 6e6, transition tripped, M = 0.15;
    columns `alpha, deg`, `cl`, `cd`; 3 zones(80/120/180 grit), 약 17점씩.
  - `CP_Ladson.dat` — Ladson, Hill & Johnson TM 100526, M = 0.3, 9 zones(Re 6e6 α = 0.0169/10.0254/15.0299
    free transition; Re 3e6 fixed transition …). 파일 머리말: *"may not be sufficiently 2D, because
    aspect ratio of model only 1.333"*.
  - `CP_Gregory_expdata.dat`, `CL_Gregory_expdata.dat`(digitized, *"digitizing only approximate"*),
    `0012.abbottdata.cl/cd.dat`, `0012.mccroskeydata.cl.dat`(CL-α best fit).
  - Cf 실험값은 없다(페이지: no known experimental skin friction data).
- 조건: Re = 6e6/chord, "essentially incompressible"(압축성 코드는 M = 0.15), fully turbulent.
  수정된 형상 정의(TE를 닫은 식)는 페이지에 있다.
- 불확도: 없음. 원자료끼리 차이가 있다(McCroskey: tripped CD0는 Re 3e6이 6e6보다 약 10% 높다).
- 격자: `n0012_{1793-513, 897-257, 449-129, 225-65, 113-33}`(p2dfmt/p3dfmt/cgns/hex cgns),
  farfield ≈ 500c. 대안은 point-vortex farfield 보정(Thomas & Salas 1986).
- REDUCED: 2D C-grid 225x65 ≈ 14.6k 셀, α = 0, 10, 15 세 번 정상 SA 계산. 각 3분 안쪽이면 10분 안에
  들어간다(추정).

### 1.4 2DN44 — NACA 4412 trailing-edge separation (naca4412sep_val.html)

- 파일: `NACA4412sep_validation/naca4412.cp.expt.dat`(x, y, cp, 약 50점, α = 13.87°, Re = 1.52e6),
  `exp.profiles.new.dat`(x, y, u, v, uv — 6 lines x/c = 0.6753…0.9528, d/c = 0.002 간격, /Uref 정규화),
  flowfield zip.
- 페이지 경고(실제 문구): *"all comparisons for this case should only be viewed in a qualitative
  sense"*. Uref 보정 계수 약 0.93이 필요하다.
- 격자: `n4412_{1793-513 … 113-33}`, farfield 100c.
- 판단: 정량 게이트로는 약하다. 후순위.

### 1.5 2DML — Delville plane mixing layer (delvilleshear_val.html) ★

- 파일(`Delvilleshear_grids/`):
  - `delville_exp_u.dat` — single-wire, X(mm), Y(mm), U(m/s), Y/δω, (U−U1)/ΔU; 5 zones x = 1, 50, 200,
    650, 950 mm; 머리말 *"U1=22.40 m/s, deltaU=19.14 m/s"*, δω(x)가 주석에 있다.
  - `delville_exp_delomega.dat` — vorticity thickness δω(x) 약 23점(x = 40…950 mm).
  - `delville_exp_turb.dat` — X-wire u'v', u'u', v'v', w'w'(m²/s² 및 /ΔU²), 3 zones x = 200, 650, 950 mm.
- 조건: U_upper = 41.54, U_lower = 22.40 m/s, Re = 2900 per mm(격자 길이 1 = 1 mm), Tu ≈ 0.3%,
  splitter TE 0.3 mm, x = −10 mm에서 BL δ ≈ 9.6 / 6.3 mm. 위·아래 경계는 contoured slip wall.
- 불확도: 없음.
- 격자: `Delville_mmfinal4_{737,369,185,093,047}`(p2dfmt/p3dfmt/cgns). 입구는 Pt/Tt 그림.
- 출처 주의: 원본은 ERCOFTAC C.34(CC BY-NC-SA)와 AGARD AR-345. TMR 사본을 쓰면 CC0 저장소 안의
  파일이지만, provenance에 원 출처를 적는다(§0).
- REDUCED: 2D, 185 패밀리(수만 셀). 자유 전단층이라 벽 해상도 부담이 작다. 비교는 성장률
  dδω/dx(선형 구간 x = 200…950 mm)와 정규화 U 프로파일.

### 1.6 ASJ — axisymmetric subsonic jet, Bridges–Wernet Set Point 3 (jetsubsonic_val.html)

- 파일: `Jetsubsonic_validation/ARN-jet-lines-expt.dat` — "Consensus PIV Dataset", variables x/Dj,
  y/Dj, u/Uj, v/Uj, u'v'/Uj², k/Uj²; 12 zones(x/Dj = 2, 5, 10, 15, 20, centerline 등, 한 줄에 139점).
  머리말: *"k … values not meaningful for x/Dj>22.2"*. 그 밖에 `arn-jet-flowfield-expt-dat.zip`,
  `consensus1.zip`(NASA/TM-2011-216807 전체).
- 조건: Mjet ≈ 0.51, acoustic M ≈ 0.5, coflow Mref = 0.01, 노즐 반지름 1 in. NPR/TTR은 그림에만 있다.
- 불확도: 페이지에는 없다(TM-2011-216807 원문에 있을 수 있지만 읽지 않았다).
- 격자: `arn_3zone_{513,257,129,65}`(3 zone, p2dfmt/cgns/quad cgns), rotated periodic wedge.
- REDUCED: 65 또는 129 레벨 wedge(수만 셀 이하). **multi-zone Plot3D**라 변환기가 zone 접합을
  처리해야 한다. plume·jet 계열과 관련이 커서 상위 5위 안에 넣는다.

### 1.7 ASBL — Driver axisymmetric separated BL (driver_val.html) ★

- 파일(`Driver_validation/`): `cp.exp.dat`(x[m], cp, 약 30점, 기준 x = −0.457 m), `cf.exp.dat`(x, cf
  10점, 분리 구간에서 음수), `exp.uprofile.dat`(u/uinf 대 y'[m], 13 zones x = −0.4572…+0.3048 m),
  `exp.upvpprofile.dat`.
- 조건: case C.S0, 원통 D = 0.140 m, 벽 발산으로 area ratio ≈ 1.6, 기포 ≈ 0.2 m, x ≈ −0.3에서
  δ ≈ 0.012 m, "essentially incompressible". 위쪽 경계는 실험자가 준 streamline(inviscid)이고
  격자에 들어 있다. 실험의 측벽 suction은 CFD에서 주지 않는다.
- 불확도: 없음.
- 격자: `driver-{1425,713,357,179,90}-1zn-narrow2d.p2dfmt.gz`(2 x 1425x385 … 2 x 90x25).
  **단일 zone이고 2D 평면판(`narrow2d`)도 있다** → 변환이 가장 쉽다.
- 원 데이터 출처는 Driver(NASA Ames 직원, AIAA 91-1787). 공공영역 성격이 가장 분명하다.
- REDUCED: wedge 179x49 ≈ 8.8k 셀, 비압축 SA/SST 정상 상태. 10분 안에 충분히 들어간다(추정).

### 1.8 TMR verification 참조값 (VER-14 재료, validation 아님)

- 2DZP 평판(flatplate_sa.html / flatplate_sst.html): M = 0.2, Re = 5e6/L, 격자 5개(finest 545x385).
  - `FlatPlate/SA/cf_convergence.dat` — N, h, Cf(x = 0.970084071). CFL3D finest 0.270562e-2,
    FUN3D 0.270540e-2.
  - `FlatPlate/SA/drag_convergence.dat` — CD. CFL3D finest 0.285985e-2, FUN3D 0.2852469e-2.
  - **`FlatPlate_validation/cf_incomp_results_sa.dat`** — *"incompressible codes using SA model"*:
    FUN3D incompressible 2.729090e-3, SC/TETRA 2.72937e-3(finest). **비압축 코드인 우리에게 맞는
    기준값이다.** validate.rs:4667의 "the case is compressible" 걱정을 덜어 준다. SST 판:
    `cf_incomp_results_sstv.dat`, `FlatPlate/SST/cf_convergence_sstv.dat` 등.
  - 이 밖에 `cf_plate.dat`, `mut_0.97.dat`, `flatplate_u.dat`, `flatplate_u+y+.dat`, `find_peak_mut.dat`.
  - 페이지의 GCI: Cf(0.97) CFL3D p = 1.98, GCI_fine 0.017%; FUN3D p = 1.34, 0.028%.
- 2D bump-in-channel(bump_sa.html / bump_sst.html): M = 0.2, Re = 3e6, finest 1409x641.
  - `Bump/SA/cf_convergence_sa.dat` — Cf at x = 0.6321975, 0.75, 0.8678025(CFL3D/FUN3D 각 zone).
    CFL3D finest Cf(0.75) = 0.614937e-2.
  - `Bump/SA/force_convergence.dat` — CL, CD, CDp, CDv. CFL3D finest CL = 0.2490011e-1,
    CD = 0.3572382e-2.
  - `cp_bump.dat`, `cf_bump.dat`, `bump_u.dat`(x = 0.75, 1.20148), `mut_0.75.dat`; SST는 `Bump/SST/*_sst.dat`.
- eca1.html(난류 MMS): Lisbon 워크숍(Eça) 자료.
  - `Eca1/test_case_description.pdf`, `Eca1/ValidationProcedure.pdf`, `Eca1/Report_D72_34_EMP_RT_2005_8.pdf`,
    `Eca1/Report_D72_36_EMP_RT_2006_2.pdf`(manufactured solution 보고서), `Eca1/Fortran_functions_Lisbon_2006.pdf`,
    `fortranfilesvv.zip`(nasa.gov; MS 함수 Fortran 소스).
  - 페이지 설명: wall-bounded incompressible turbulent flow, *"one- and two-equation eddy-viscosity
    turbulence models"*. 식은 페이지가 아니라 보고서 PDF와 Fortran에 있다. 근거 논문은 IJNMF 54(2)
    119–154(2007) doi 10.1002/fld.1387 등. 페이지 문구: *"NASA assumes no responsibility for the
    accuracy of this information"*. PDF와 zip은 이번에 열지 않았다(VER 쪽 조사 대상).
  - Fortran은 GPL이 아니지만(CC0 저장소) 그래도 식은 보고서에서 옮기는 편을 권한다.

---

## 2. 우선순위 B — NIST firemodels/exp

### 2.1 라이선스 (실제로 읽음)

`https://raw.githubusercontent.com/firemodels/exp/master/LICENSE.md`(GitHub API: spdx NOASSERTION):
> "NIST-developed software is provided by NIST as a public service. You may use, copy and distribute
> copies of the software in any medium, provided that you keep intact this entire notice. … Please
> explicitly acknowledge the National Institute of Standards and Technology as the source of the
> software. … The software developed by NIST employees is not subject to copyright protection within
> the United States."

→ 재배포는 허용되고 조건은 고지 유지와 NIST 출처 표기다. 다만 여기 들어 있는 숫자는 LLNL(Burro,
Coyote), Shell(Maplin Sands), Askervein 보고서(Taylor & Teunissen 1983) 같은 제3자 실험을 NIST가
옮긴 것이다. 숫자(사실)는 `kind = file`로 추적하되 NOTICE에 NIST 문구를 남기기를 권한다.
FDS 입력 파일(`firemodels/fds/Validation/...`)도 같은 NIST 고지 아래에 있고 GPL이 아니다(spdx
NOASSERTION, 같은 문구 체계). 그래도 우리 규칙상 setup 수치(source 질량유량, 측정점 좌표)만 읽는다.

### 2.2 LNG_Dispersion

- 파일(모두 수십~백 바이트 CSV): Burro3/7/8/9, Coyote3/5/6, Falcon1/3/4(+Falcon2/5 profile만),
  MaplinSands27/34/35. 각 시험마다 두 파일이 있다.
  - `<trial>_exp.csv`: 단위 행 `m,%`, 헤더 `x,X_CH4`. **풍하 거리별 최대 CH4 체적분율(%)**, 4–8점
    (예: Burro3 57 m 28.2%, 140 m 9.0%, 400 m 1.2%, 800 m 0.9%).
  - `<trial>_profile.csv`: `m,m/s,C` / `z,u,T` — 유입 풍속·온도 연직 프로파일 6–7점.
- 셋업: `firemodels/fds/Validation/LNG_Dispersion/FDS_Input_Files/Burro3.fds` 등에 Monin-Obukhov
  (L, u*, z0, θ*), 원천 반지름과 질량유량(Burro3: RADIUS 12.94, MASS_FLUX 0.167 kg/m²s, T = −161.55 °C),
  센서 좌표가 들어 있다.
- 불확도: 파일에는 없다. FDS Validation Guide에 실험 불확도 표가 있지만 이번에 읽지 않았다.
- REDUCED: 3D. FDS 원판 메시는 100x100x32 여러 블록. 우리 lowmach + 무거운 가스 + 대기
  경계층이 필요하고, 1 km 영역을 성기게 잡아도 수십만 셀이 된다. 10분 안에 넣으려면 짧은 영역
  (≤ 400 m, 2–3점 비교)과 정상 상태 근사가 있어야 한다. NH3 부지 일과 직접 관련이 있으므로 다음
  웨이브 후보로 둔다.

### 2.3 Askervein_Hill (run TU03-A)

- 파일: `ASW-ANE_TU03A_10m.csv`(능선을 따라 DIST[m] −100…, DIRECTION, UPWASH, SPEED, σu/v/w, INT,
  uv/uw/vw, SIGM_SPEED; 10 m 높이), `AASW-AANE_TU03A_10m{,_MF,_TU}.csv`, `BNW-BSE_TU03A_10m.csv`,
  `ASW60_TU03A_UK30m_tower.csv`, `CP_TU03A_FRG17m_tower_{MF,TU}.csv`, `RS-HT_TU03A_50m_tower.csv`
  (기준점 RS와 정상 HT의 연직 SPEED 프로파일, DZ 1–49 m), `README.md`.
- README가 각 값의 출처(Askervein 1983 보고서 Table A1.3/A1.4/A1.7, 쪽수)를 밝힌다. 일부는
  graphreader로 디지타이즈했다는 문구: *"Extracted with http://www.graphreader.com/ (in 2021)"*.
- 지형: FDS 입력 README — *"Thanks to Javier Sanz Rodrigo of the Windbench project for providing the
  topology data"*, 5 m 해상도 `Askervein_5m.gingeom`(firemodels/cad). **지형 데이터의 라이선스는
  확인하지 않았다**(Windbench 출처).
- 비교 지표: 정상에서 fractional speed-up ΔS = (S − S_ref)/S_ref(HT 10 m ≈ 0.8), 능선 단면 속도.
- REDUCED: 3D 지형 RANS(중립 ABL, k-ε 또는 SST). 성기게 잡으면 ~10⁵ 셀, 정상 상태. 10분은
  빠듯하다. 터레인 STEP→메시 경로(automesher / site-mesh)를 시험하는 데 가치가 있다.

### 2.4 Sandia_Pool_Fires / Heskestad_Flame_Height

- Sandia: `{ethane,ethylene,isopentane,propane}_pool.csv`(헤더 `Exp,HF1,HF2`; R1–R40 반경별 복사
  열유속, NaN 다수, 단위 행 없음 — kW/m²로 추정하지만 확인 안 함), `*_pool_flame.csv`(Time, SEP,
  L_F 화염 길이, Tilt). 복사가 지배적인 대형 화재라 우리 범위 밖이다. 후순위.
- Heskestad: `flame_lengths.csv`, `Heskestad_Correlation_2.csv`는 **상관식이 계산한 값**이다(Q*별
  L_F/D: Steward, Heskestad, Cetegen…). 측정값이 아니다. `Tamanini_HRR_Data.csv`만 실험값이다.
  → 실험 validation으로 쓰지 말고 상관식 대 코드 비교로만 쓴다.
- `Atmospheric_Dispersion/plume_rise.csv`(58 B)도 있다. 열지 않았다.

### 2.5 Desert Tortoise(NH3, 1983) / Jack Rabbit II(Cl2)

- firemodels/exp 최상위에는 Desert Tortoise, Jack Rabbit, NH3 디렉터리가 **없다**(gh api로 목록 확인;
  dispersion 계열은 `Atmospheric_Dispersion`, `LNG_Dispersion`뿐).
- Desert Tortoise: ADMLC 데이터셋 페이지(https://admlc.com/datasets)에 따르면 Chang–Hanna의
  **Modelers' Data Archive(MDA)** 에 들어 있고, 얻으려면 이메일을 보내야 한다. 문구: *"These datasets
  can be obtained by contacting Joe."* 라이선스·이용 조건은 쓰여 있지 않다. ADMLC 면책:
  *"ADMLC does not take responsibility for the quality of the data referenced here."* 공개 다운로드
  URL은 찾지 못했다. (웹검색 도구가 "Desert Tortoise ammonia" 질의 하나를 정책상 거부했다. 다른
  질의로 MDA 경로를 확인했다.)
- Jack Rabbit II: ADMLC — *"Datasets can be requested by emailing: Jack.Rabbit@st.dhs.gov"*. 조건은
  쓰여 있지 않다. AIChE 2016/2017 논문들은 "made available"이라고만 하고 경로를 주지 않는다.
- → 둘 다 **요청 후 받는 자료**이고 조건을 모른다. 재배포 여부는 받은 뒤 조건을 보고 정해야
  하므로 사용자 결정 사항이다. 지금 게이트 후보는 아니다.

---

## 3. 우선순위 C

### 3.1 Windsor body (Loughborough) — figshare API로 확인

- 레코드 "Windsor Model Experimental Aerodynamic Dataset", DOI 10.17028/rd.lboro.13161284,
  **license: "CC BY-NC 4.0"**(API의 license 필드). 25% 축척 square-back, LU Large Wind Tunnel
  (2.5 m², closed, fixed ground). 설명: *"No corrections (for example blockage) have been applied to
  the data."* 영점 yaw는 *"yawing the model in 0.1 degree increments until a symmetric base pressure
  was achieved"*. 형상(모델·마운트·터널) **STEP/STL 포함**(`Geometry.zip` 176 MB). `Set A - Baseline
  Flow.zip` 3.9 GB, `Set B - Yawed Flow.zip` 2.8 GB.
- 개별 파일 레코드 10.17028/rd.lboro.24762966(CC BY-NC 4.0): `FCAV0D_NW_0_0yaw_Force.csv`(492 B —
  U = 39.67 m/s, ρ = 1.260, A = 0.112 m², **CD = 0.3070**, CL = −0.0296 …), `SquareBack_*_Pressures_Averages.csv`
  (3.5 KB, MeanCp/RMSCp), `SquareBack_Pressure_Tapping_Map.csv`, PIV 평면 평균 `*_FlowField.csv`(2–6 MB).
  NW = no wheels, WW = with wheels.
- 불확도: 수치로 적힌 것은 없다.
- → **fetched**(NC). REDUCED: 3D bluff body, RANS로 CD/기저 Cp. 10분 안에는 수백만 셀이
  불가능하므로 50만 셀 이하 거친 SST 정상 상태로 CD만 본다. 정확도가 낮다. 후순위.

### 3.2 DrivAer (Loughborough)

- DOI 10.17028/rd.lboro.12881213, **CC BY-NC 4.0**: 25% DrivAer(estate/fast/notch), *"The CAD
  geometry of the DrivAer model has not been duplicated for this dataset"*(차체 형상은 TUM 배포본을
  따로 받아야 한다), yaw *"+/-0.1 degrees"*. `Mean Results.zip` 53 MB, `Instantaneous Results.zip` 8.1 GB,
  `Tunnel_CAD.zip`.
- 2024–2026 레코드 10.17028/rd.lboro.26096761.v5(CC BY-NC 4.0): 단면별 `*_Pressure_Mean.dat`,
  `*_Force_Mean.dat`, Tapping_Map.
- → fetched. 3D 실차라 10분 REDUCED에 맞지 않는다. 후순위.
- AutoCFD 쪽 WindsorML(CFD 데이터, CC-BY-SA로 보고됨)은 실험값이 아니어서 제외했다.

### 3.3 Ampofo & Karayiannis 2003 저난류 자연대류 캐비티

- 논문: "Experimental benchmark data for turbulent natural convection in an air filled square cavity",
  IJHMT 46(19) 3551–3572, doi 10.1016/S0017-9310(03)00147-9. Crossref 라이선스는 Elsevier TDM
  userlicense(유료 논문). 0.75 x 0.75 x 1.5 m, 50/10 °C, Ra ≈ 1.58e9.
- 공개 데이터 파일은 figshare 검색과 웹검색에서 **찾지 못했다**. 표는 논문 본문에 있을 가능성이
  있지만 읽지 않았다.
- → 지금은 불가. 사용자가 논문을 확보하면 `literal` 또는 전사 `file`(논문 표의 숫자)로 갈 수 있다.
  2D 비압축 Boussinesq k-ω SST 저Re라 계산 비용은 낮다.

### 3.4 SPHERIC Test 2 (Kleefsman/MARIN 3D dam break)

- 페이지 https://www.spheric-sph.org/tests/test-02, zip:
  `https://9449af45-2363-44f6-8b03-03b6b7c2aee5.usrfiles.com/archives/9449af_6ce73873d6d04315853e481c337dbfca.zip`
  (2.63 MB, 받아서 열어 봤다): `test_case_2_v1p1.pdf`(Issa & Violeau, EDF, ERCOFTAC SPHERIC SIG,
  March 2006), `test_case_2.f`, `test_case_2_exp_data.xls`(5 MB; xlrd가 없어서 열 구조는 확인하지 못했다).
- PDF에 적힌 내용: 탱크 3.22 x 1 x 1 m, 물기둥 1.228 m 폭, 높이 0.55 m, 박스 0.161 x 0.403 x 0.161 m,
  H1–H4 연직 수위 probe 위치(x 간격 0.496/0.496/0.496/1.150 m), P1–P8 박스 압력 센서(앞면 4개,
  윗면 4개; 0.021/0.04 m 간격), 물 ρ = 1000, ν = 1e-6, 6 s. 출력: *"Vertical water height history at
  probes H1 to H4 / Pressure time histories at pressure sensors P1 to P8"*.
- 라이선스: 페이지 footer *"© 2025 by SPHERIC"* 외에는 조건 문구가 없다 → **fetched**(재배포 안 함).
  불확도는 없다.
- REDUCED: 3D VOF. 기존 `martin-moyce-1952` 2D 붕괴 게이트의 3D 확장이다. 거친 메시(예: 64x20x20
  ≈ 26k 셀, 박스 포함)로 H2/H4 수위와 P1 첫 피크 시각만 비교하면 10분 안에 가능할 것으로 본다(추정).
  이번 1차 5개에는 넣지 않았다(TMR 우선 결정).

---

## 4. 후보 요약표

| 후보 | kind | 2D/3D | 불확도 | 형상/BC | REDUCED 셀(추정) | 판정 |
|---|---|---|---|---|---|---|
| TMR 2DWMH hump | file (CC0) | 2D | Cf 열, x_r ±0.003 | 격자 5종, 유입 프로파일 | 11k–44k | **1차** |
| TMR 2DN00 NACA0012 | file | 2D | 없음 | 격자 5종 | 14.6k | **1차** |
| TMR ASBL Driver | file | 축대칭 wedge | 없음 | 격자 5종, 2D판 | 8.8k | **1차** |
| TMR 2DML Delville | file(출처 주의) | 2D | 없음 | 격자 5종 | ~수만 | **1차** |
| TMR ASJ jet | file | wedge, 3-zone | 없음 | 격자 4종 | ~수만 | **1차** |
| TMR ATB | file | wedge, 천음속 | 없음 | 격자 5종 | 15k | 2차(압축성 RANS 확인 후) |
| TMR 2DN44 | file | 2D | "qualitative" | 격자 | 14.6k | 후순위 |
| TMR 2DZP/bump 참조값 | file | 2D | GCI 제공 | 격자 | 3k–14k | VER-14로 |
| firemodels LNG | file(NIST 고지) | 3D | 미확인 | FDS 입력에 셋업 | 10⁵+ | 2차(NH3 관련) |
| firemodels Askervein | file(+지형 라이선스 미확인) | 3D | 미확인 | 5 m 지형 | ~10⁵ | 2차 |
| Sandia pool / Heskestad | file | 3D / 상관식 | — | — | — | 제외/보조 |
| Desert Tortoise / JR II | 요청제 | 3D | — | — | — | 사용자 결정 |
| Windsor / DrivAer | fetched (CC BY-NC 4.0) | 3D | yaw ±0.1° | STL 포함(Windsor) | 10⁵–10⁶ | 후순위 |
| Ampofo 2003 | 논문 유료 | 2D | — | 논문 | 수만 | 자료 없음 |
| SPHERIC T2 | fetched | 3D VOF | 없음 | PDF 치수 | ~26k | 2차 |

---

## 5. 권장 — 처음 만들 validation 게이트 5개 (TMR 우선)

선행 유닛 **VAL-R0**: TMR Plot3D(p2dfmt, 단일·다중 zone) → 한 셀 두께 polyMesh 변환기
(축대칭은 wedge 회전), 그리고 TMR Tecplot `.dat` zone 파서(키 파일 읽기). 다섯 게이트가 모두
이것을 쓴다. 공통: kind = file, 라이선스 칸에 "TMBWG/turbmodels LICENSE: CC0 1.0 Universal"을
인용한다. 허용오차는 실험 불확도가 있으면 그 값을 쓰고, 없으면 사용자 결정 사항으로 남긴다.
결과가 기준을 벗어나도 MISSED로 기록하고 진행한다.

1. **`tmr-hump-cf` / `tmr-hump-cp`** (+ literal `tmr-hump-reattach` = 1.11 ± 0.003, case3expdata.html)
   - 비교: 벽 Cf(x/c)와 Cp(x/c). 분리점(Cf 첫 음수)과 재부착점(Cf 마지막 음수→양수 교차).
   - 지표: |x_r,CFD − 1.11|와 0.003의 비교(RANS는 놓칠 가능성이 크다 → MISSED 기록). Cf는 파일의
     Uncertainty(C_f) 대역 안에 드는 점의 비율. Cp는 x/c ∈ [0.6, 1.3] RMS.
   - REDUCED: no-plenum 205x55(≈11k)와 409x109 두 레벨, 비압축 SA(+SST), U∞ = 34.6 m/s 상당
     Re_c = 936k, 입구 BL은 `noflow_u_inflow`로 확인.
2. **`tmr-naca0012-ladson-clcd`** (+ `tmr-naca0012-ladson-cp`)
   - 비교: 80-grit zone의 CL(α), CD(α)를 α = 0, 10, 15에서. Cp(x/c)는 α ≈ 0/10/15 zone(파일 경고
     문구를 provenance에 남긴다).
   - 지표: CL 상대오차, CD0 상대오차, Cp RMS. 허용오차는 원자료 간 산포를 근거로 사용자가 정한다
     (McCroskey 10% 언급).
   - REDUCED: 225x65 C-grid(≈14.6k), 비압축 SA, Re = 6e6, α 3개.
3. **`tmr-driver-asbl-cf` / `-cp` / `-u`**
   - 비교: Cf(x) 10점(음수 구간 = 기포), Cp(x), x = −0.33…0.30 m의 u/uinf 프로파일 13개.
   - 지표: 기포 시작·끝 x(Cf 부호), Cp RMS, 프로파일 L2.
   - REDUCED: `driver-179-1zn-narrow2d`(≈8.8k)를 wedge로, 비압축 SA/SST. 첫 번째로 해 볼 만큼 가장
     싸다. **변환기 검증용으로 1번보다 먼저 돌리는 것을 권한다.**
4. **`tmr-delville-2dml-delomega`** (+ `-u`, `-turb`)
   - 비교: δω(x) 성장(x = 200…950 mm 선형 구간 기울기), (U−U1)/ΔU 대 y/δω 자기상사 프로파일,
     x = 650/950 mm의 u'v'/ΔU² 최대값.
   - 지표: dδω/dx 상대오차, 프로파일 L2.
   - REDUCED: `Delville_mmfinal4_185` 2D, 입구 BL 두께를 맞춤, SA/SST. 자유 전단층이라 수렴이 쉽다.
     provenance에 원 출처(ERCOFTAC C.34 / AGARD AR-345, Delville et al. 1989)를 적는다.
5. **`tmr-asj-centreline`** (+ `tmr-asj-lines`)
   - 비교: `ARN-jet-lines-expt.dat`의 centerline u/Uj(x/Dj)와 x/Dj = 5, 10, 15 반경 프로파일.
   - 지표: potential core 길이(u/Uj = 0.95 위치)의 오차, 감쇠 기울기, 반폭 r½ 성장률.
     k는 x/Dj ≤ 22.2에서만 쓴다(파일 문구).
   - REDUCED: `arn_3zone_65` 또는 `_129`, wedge, 비압축 또는 lowmach(Mjet 0.51 → 압축성이 약간
     있다. 비압축으로 돌리면 그 사실을 note로 남긴다), coflow M = 0.01. 3-zone 접합이 변환기의 추가
     요구다. plume·jet 키(McCaffrey)와 이어진다.

다음 순서(6위 이후): VER-14(평판·bump code-to-code, 특히 `cf_incomp_results_sa.dat`), ATB(압축성
RANS 확인 후), SPHERIC T2(fetched), firemodels LNG Burro3 단축판(NH3 관련), Askervein(지형 라이선스
확인 후).
