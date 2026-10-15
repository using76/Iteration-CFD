# Tranche 17 — 추후 검토 목록 (기준 미달·미수렴·미실행)

**규칙 (사용자, 2026-10-14):** "수렴못해도 진행해 기준못해도 일단 진행하자 문서로 남겨만놔줘 추후 검토하자".

이 규칙에 따라 단위 작업은 다음처럼 처리한다.

- 기준을 못 맞추거나 풀이가 수렴하지 않아도 단위 작업을 멈추지 않는다.
- 게이트와 허용오차는 쓰인 그대로 둔다. 느슨하게 하지 않는다.
- 테스트는 미달을 `MISSED`로 단언하고, 측정값과 원인을 출력한다.
- 단위는 그대로 반영(land)하고, 미달 항목은 이 문서에 적는다.
- 고치는 방법이 수치 기법을 바꾸는 것이면 만들지 않고 이 목록에 남긴다.

각 항목에는 근거가 되는 SPEC-LIT 절과 커밋을 붙였다. 웨이브가 끝날 때마다 병합할 때 새 항목을 덧붙인다.

---

## A. 기준 미달·미수렴 (검토 대기)

| # | 단위 | 내용 | 측정값·원인 | 근거 |
|---|---|---|---|---|
| A1 | CHR-20 | `--features single`에서 f32 PBiCGStab가 fvDOM step-scheme 행을 긴 upwind 경로에서 풀지 못한다 | 1셀 폭 slab, diagonal/무전처리: N=64는 일부 방향만 수렴, N=128부터 모든 방향 실패, N=256은 5000회 반복 끝에 NaN. DILU는 약 N회 후 발산. f64는 같은 행을 푼다. 경로가 64셀 이상인 실제 f32 fvDOM 계산은 모두 이 문제에 걸린다. 해결책 후보(upwind 순서 sweep, 다른 f32 Krylov)는 수치 변경이라 만들지 않았다 | SPEC-LIT 136.7, `c009734` |
| A2 | CHR-21 | 등온 정육면체 κL = 0.1 행 미달 (양쪽 세트) | S8, 24³: −3.10 % (기준 3 %). step scheme의 공간 오차가 원인이며, 32³은 −2.63 %, 48³은 −2.08 %. S4, 24³: +4.45 %로 통과하지만 공간 오차가 각도 오차를 상쇄한 결과다. S4 자체 한계는 +6.45 %로 기준 6 % 밖이다. 두 미달 모두 테스트에서 단언한다 | SPEC-LIT 136.8, `e381e17` |
| A3 | BC-13 | single에서 ABL 압력계(PBiCGStab + Diagonal)가 첫 outer 반복부터 수렴하지 않는다 | 500회 반복 뒤 잔차 38 (초기 1). SIMPLE 8회째에 NaN. f64는 수렴한다. 게이트는 양쪽 빌드에서 수렴하는 PCG + DIC(첫 반복 171회)로 돈다 | SPEC-LIT 120.9, `bab6396` |
| A4 | BC-07/08/10/14 | single SIMPLE가 고정점에 머물지 못하고 2-cycle에 걸린다 (f32 SIMPLE floor 계열) | BC-14 cyclic/recycled channel: 1e-14 변화에 닿지 못하고 4000회 상한까지 돈다. f64는 2037회와 303회에 멈춘다. 프로파일 차이는 정점 대비 1.557e-4. 입력(relax, SIMPLEC, nCorr)으로는 풀리지 않는다 | SPEC-LIT 118, 120.10, `23c4fd2` |
| A5 | CHR-21 | `DomSolve`의 정지 규칙 `change < tol`은 마지막 한 스텝의 크기만 제한한다 | 산란 slab(수축률 약 0.6/sweep)에서 남는 오차는 약 1.5 tol. f32에서 G를 tol32 이내로 원하면 더 작은 tol을 줘야 한다 | SPEC-LIT 136.8 |
| A6 | DST-02 | 단열선 Ψ가 논문 closure 값의 약 57 % | 단열선을 잡는 방법에 따라 Ψ가 약 2배까지 달라진다. DST-06의 논문 비교에는 논문에 실린 식을 쓴다 | SPEC-LIT 153.8, `d337a4c` |
| A7 | CHR-07b | device Rosenbrock 성능이 CPU 1코어의 약 4.5배에 그친다 | 정확도 게이트는 통과. 성능 목표는 미달 | `0758690` |
| A8 | VAL-70 | Gate 78-D 측정 레그: 두 splash 기준 모두 Riboux & Gordillo 2014 임계 속도에 미달 (`MISSES`) | 띠 ±20 %, 액체 8개 중 6개 이상이 들어와야 통과. Mundo: 17행 중 0, 액체 8개 중 0 (e −0.734 ~ −0.214). Bai & Gosman: 17행 중 3, 액체 8개 중 1(에탄올) (e −0.396 ~ +0.690). 4.78 행은 OPEN으로 남는다 | SPEC-LIT 146.2, `1280c6f` |
| A9 | AMG-04c | **해결됨 (AMG-06a `f13bebf`: single에서 7.68e-9로 통과, ignore 해제)** single에서 K-cycle FCG의 POI(32) 답이 Jacobi-PBiCGStab과 2.37e-8 차이 (기준 1e-8) | 11회 반복에 수렴은 한다. 이 테스트는 파일 관례(AMG-06a)에 따라 single에서 ignore 상태다. AMG-06a가 켜야 한다. f64는 1.61e-9~5.38e-9로 통과 | SPEC-LIT 141.6, `a887eb3` |
| A10 | AMG-07b | **해결됨 (AMG-06a `f13bebf`: `spmd_gamg`도 IR로 감쌈, tol 1e-11·답 1e-8로 양쪽 빌드 통과)** single의 SPMD GAMG(정상 상태 cycle)는 반복 정밀화가 없어서 tol 1e-6이면 정체된다 | POI 32³, 500 cycle 뒤 참 잔차 4.3e-6. 게이트는 기존 SPMD 게이트와 같은 solver tol 1e-5로 돈다 (게이트 허용오차 tol32 1e-5는 그대로). F32-07의 IR이 `spmd_gamg`도 감싸야 한다 | SPEC-LIT 141.11, `3f110b3` |
| A11 | CMP-13 | LTS 정상 해가 CFL에 따라 달라진다 (기준: CFL 2와 20의 해 차이 1e-8) | u 차이 2.85e-6, p는 max\|p\|의 7.0e-2 (2.2e-3 Pa), T 5.5e-9. Rhie-Chow 면 flux의 rAU = V/a_P가 시간항 대각 V r ρ를 지니기 때문이다. 기존 정상 SIMPLE도 alpha_U에 따라 같은 식으로 달라진다 (0.9와 0.7: 8.4e-7). 해결책(시간 간격에 무관한 rAU, 또는 면 flux의 ddt 보정)은 Rhie-Chow 수치 변경이다. 감독 권고는 미달 수용 | SPEC-LIT 125.7, `3c16286` |
| A12 | DST-04 | single: 확산 지배 화염관에서 연소 쪽 plateau가 1 위로 떠오른다 | σ=1 평면 1000셀에서 최대 1.5e-4, 층류에서 5e-5. float 대각 ρV/dt가 eps(1+2D)까지만 유지되기 때문이다 (D = Γdt/dx² ≈ 40). solver tol과 IR로는 바뀌지 않는다. 해결책(delta form, ddt 재스케일)은 수치 변경이다. DST-06의 긴 계산에서 누적될 수 있다 | SPEC-LIT 155.9, `4875b9d` |
| A13 | DST-04 | 양쪽 빌드: σ=5.06에서 δ_t 오차가 메시를 세분해도 줄지 않는다 | 250/500/1000셀에서 −2.2e-4, −6.2e-4, −4.5e-4 (모두 2 % 띠 안). 원인 미정: H20의 σ≠1 정확해 주장 또는 σ에 따른 이산화 하한. single에서는 x_f와 Linf도 1000셀에서 더 줄지 않는다 (float 하한 약 4e-4) | SPEC-LIT 155.9, `4875b9d` |
| A14 | DST-12 | f64: 스텝 안에서 T_b에 닿는 입자(4096개 중 258개)가 host 대비 m_w와 누적값에서 6 ulp 차이 (기준 4 ulp) | single은 3 ulp 이내. t_hit = −ln(1−r) m C/λ가 r 변화의 1/(1−r)배로 움직여, T_eq가 위에서 T_b에 다가갈수록 교차 시각이 ill-conditioned이다. device 초월함수(pow, cbrt, log1p)의 1 ulp 차이가 수 ulp로 커진다. 다른 분기를 잡도록 16×4 ulp 상한은 계속 단언한다 | SPEC-LIT 159.7, `26685a0` |
| A15 | VAL-12 | Gate 95-A: 네 번째 메시(n_y=32)를 넣어도 5:1, 10:1이 미달 (21/25) | 5:1: n_y=4가 outer 673회에서 발산, p_u 1.423 (기준 1.9). 10:1: n_y=4 정체 (q_obs 0.992), p_u 1.896 (0.0037 부족). 2.5:1은 통과. n_y=4 수렴은 VAL-11(GMRES outer)의 몫이다. 5:1의 1.42는 아직 수렴한 차수가 아니며, 다섯 번째 메시가 있어야 더 알 수 있다 | SPEC-LIT 109.6-109.7, `f0a0df2` |
| A16 | TET-09 | 리메시하지 않은 입력에서 feature가 아닌 segment 사이 각이 작으면 Delaunay refinement가 끝나지 않는다 | 면마다 삼각형 2개(대각선 45°)인 정육면체, h=0.4: 91 pass 뒤 67 289점 (예측 약 930 tet), 상한에서만 멈춘다. feature edge 각만 보호하므로 Shewchuk 1998의 90° 가정이 깨진다. 파이프라인은 항상 먼저 리메시하므로 기록만 했다 | SPEC-LIT 116.13, `0a2249d` |
| A17 | TET-10b | 30° 쐐기 게이트(최소 이면각 ≥ 10°, 최대 ≤ 165°)를 측정할 수 없다 | TET-09 refinement가 쐐기에서 끝나지 않는다: h=0.1에서 187 pass 뒤 20 000점 상한. h=0.04~0.15와 상한 60 000에서도 상한, 240 s 제한 또는 refine/facet에 걸린다. 리메시한 정육면체−원기둥(24면)도 h=0.1에서 520 pass 뒤 60 000점 상한, h=0.0625에서는 끝난다 (71 074 tet, 346 pass). TET-10b는 h=0.0625를 쓴다 | SPEC-LIT 116.14, `7e57528` |
| A18 | F32-07 | single: 반복 정밀화(IR)를 쓴 PBiCGStab 기준해가 section 6의 밀폐 all-Neumann 상자(216셀)에서 모든 셀 NaN | float로 반올림한 source가 O(n·u32)만큼 비일관적이어서, 정밀화 뒤 보정 방정식이 대부분 그 성분이 되고 float PBiCGStab가 붕괴하는 것으로 읽힌다 (추적은 아님). 기존 행 'cuFFT == iterative, sealed box'는 validate의 `max_abs_diff`(f64/f32::max)가 NaN을 버려서 0으로 통과해 왔다. 새 refined 행은 NaN을 FAIL로 낸다. single에서는 선택기도 밀폐 상자 계를 거부한다 (기준 잔차가 NaN). BC-07의 pure-Neumann NaN과 같은 계열로 보인다 | SPEC-LIT 118.12, `acbcfb0` |
| A19 | BC-19 | cyclic 결합면을 가로지르는 device leastSquares 기울기 (선형장) | 3:2 imprint 계면에서 5.1e-2 오차. 커널(`cuda/fv.cu` boundarySeparation)이 d = Sf/(\|Sf\| Δ) = n(n·d)를 써서 d의 접선 성분 −k/Δ를 버린다. device = host twin(2.4e-16)으로 원인을 단언한다. 한 줄 해결책(bNonOrthCorr로 d = Sf·sc − k/Δ)은 비직교 cyclic의 기존 결과를 모두 바꾼다 | SPEC-LIT 120.13, `afa6ab1` |
| A20 | BC-19 | 3:2 cyclicAMI 계면을 가로지르는 층류 채널이 단일 블록과 peak의 6.7 %, 7.5 % 차이 (기준 0.5 %) | bulk 0.959/0.963 대 1.014/1.031. 같은 reader·imprint·결합을 쓰는 conformal AMI 쌍은 단일 블록과 3e-16으로 같으므로, 기울어진 결합면의 이산화가 원인이다 (면 값에 skewness 보정이 없다). 0.5 % 띠는 단일 블록 자신의 Poiseuille 오차(0.69 %, 1.56 %)보다 작다 | SPEC-LIT 120.13, `afa6ab1` |
| A21 | (기존, CMP-14/CHR-10a가 발견) | single 빌드가 `cases/channelThermalWF.jsonc`에서 11스텝째 발산 (Mach 거부, 셀 3199에서 M 1.8e9) | f64는 정상. f32 SIMPLE floor 계열(A4)로 보인다. CHR-10a의 해시 고정은 channelPeriodicFluxWF를 쓴다 | `5a79bf0` |
| A22 | AUT-04 | quick set 날개 12개 중 4개가 RM-TE 뒤에도 F3를 통과하지 못한다 (게이트 목표 6/12는 8/12로 통과) | A-1-024: F3e, capture 0.357. A-1-049: bump 없음, F3e만, capture 0.284, NO-REMEDY. 둘 다 뭉툭한 뒷전 밑면이 0.46-0.66 h_f로, 한 셀보다 가까운 두 feature 사슬을 (92.62)가 예산 안의 어떤 level에서도 함께 잡지 못한다. A-1-098/099(6 % 날개): RM-TE로 max/h_f 0.21-0.23이지만 pin이 5.4-6.5 % (F3a, F3b). automesher snap 쪽 항목이며 수치 변경은 만들지 않았다 | `9a6862b` |
| A23 | AUT-07 | R-ECON 뒤에도 G-COST 미달 | quick set 중앙값 B0-template의 6.18배 (기준 1.5배). layer 셀과 R-WIN/R-CURV 벽 level이 남는다. AUT-06이 이미 layer 8개로는 1.5배가 불가능함을 보였다. D-8은 사용자 결정으로 남는다 | `9e5d05e` |

## B. 측정·검증을 미룬 것 (10분 규칙 또는 하드웨어)

| # | 항목 | 이유 | 언제 |
|---|---|---|---|
| B1 | 전체 release lib 테스트 스위트 | 10분 초과. 매 단위에서 이름 필터 테스트만 돌았다 | 긴 슬롯에서 한 번 |
| B2 | clippy | 트리 전체 baseline이 이미 실패한다 (오류 2, 경고 983). 새 파일은 clippy를 거치지 않았다 | baseline 정리와 함께 |
| B3 | `tools/f64_identity.py` SASS 동일성 | CHR-20(커널 추가), CHR-08b(`chemFitPick` 비교 1개 추가), BC-11b(f64 빌드에서는 전처리로 제외) | 긴 슬롯에서 한 번 |
| B4 | `cht::` 필터 전체 (242개) | 10분 초과 (BC-12) | B1과 함께 |
| B5 | 필터 없는 `ofgpu-validate` 전체 | 10분 초과 (BC-13, BC-14) | B1과 함께 |
| B6 | `ofgpu-lowmach` / k-epsilon의 OUT_VDB 쌍 비교 테스트 | 시간이 오래 걸려 VIO-08에서 미실행 | B1과 함께 |
| B7 | BC-11b `-forceSplitDouble`의 효과 | 사용자 규칙으로 측정하지 않았다. p_rgh가 f32로 저장되므로 효과가 제한적일 수 있다 | 원하면 single 빌드로 정지 기둥 1회 |
| B8 | PAR-05 `peer_placement_two_devices` | GPU 2개가 필요하다 (사용자가 보류) | 두 번째 GPU가 생기면 |
| B9 | PAR-06/16 MPI | MS-MPI 설치가 보류됐다. `-comm mpi`와 PMI 환경은 이름으로 거부한다 | MS-MPI를 설치하면 |
| B10 | `ofgpu-validate` Gate 110-B 기록 계산 (VAL-52 준비) | VAL-50 가격 산정에서 backstep 레시피가 그대로는 돌지 않는다. 생성된 케이스의 inlet U가 10 m/s가 아닌 0.518 m/s다 (blockgen의 power-law 프로파일이 y=0, h=1 기준인데 `-extent`가 inlet을 y∈[1,9]에 놓는다). 1400×180은 24스텝(M 0.730), 2800×360은 152스텝(M 0.490)에서 Mach guard로 멈춘다 | VAL-52가 inflow를 입력으로 바로잡은 뒤 |
| B11 | AUT-05 이후 autonomy selftest | AUT-02a/b/c의 stl_repair 변경 이후 campaign.py, prior.py, baseline.py selftest가 실패한다 (G-1-026이 non-manifold edge 12개로 SURFACE-OPEN이라고 가정). 그 뒤 그룹(replay, hooks, live)은 실행되지 않는다 | 오늘의 repair 결과로 기준을 다시 잡는 작은 fixture 단위 |
| B12 | VIO-06 드론 레그 (VIO-11도 같은 해 필요) | t = 0.5 s 드론 해가 디스크에 없다. 지시대로 다시 풀지 않았다 | `tools/drone/solve.py`로 해를 다시 만든 뒤 `ofgpu-sample volume`으로 1회 |
| B13 | single 빌드의 드라이버 비교 (AMG-05, AMX-02) | single에서는 라이브러리와 통합 테스트만 돌았다. `ofgpu-buoyant -backend gamg/amgx` 20회 비교는 f64에서만 했다 ('amgx single' bin 빌드 약 20분) | B1과 함께 |
| B14 | STU-18 | GLM 주간 할당량이 2026-10-16까지 막혀 건너뛰었다 (live GLM-5.3-Flash studio 호출 필요) | 할당량이 돌아오면 G 레인에 다시 넣는다 |
| B15 | AUT-04, AUT-07 효과의 family별 측정 | RM-TE는 bump가 있는 모든 sharp body에서 F3 remedy 중 먼저 걸린다 (가상 F3c에서 sharp 276행 중 169행). R-ECON quick set에는 sharp body가 없어 A, D, F family와 K = 4 remedy 캠페인 효과를 재지 않았다 | AUT-12의 L-run에서 family별 실패, BLC_8, F3를 L5 규칙과 비교 |

## C. 모델·설계 선택 (권고대로 진행, 검토 대기)

| # | 단위 | 선택 | 근거 |
|---|---|---|---|
| C1 | CHR-22 | Bordbar 2021 WSGG 가중치를 논문 그대로 쓴다 (적합 범위 안 72600번 평가 중 2243번이 음수, 최저 −0.1358). 잘라내면 모델이 바뀐다 | SPEC-LIT 136.9 |
| C2 | CHR-22 | 회색가스별 경계 방출 가중치는 기체 자신의 a_i를 쓰고, transparent window는 풀지 않는다. CHR-23이 결정해야 한다 | SPEC-LIT 136.4, 136.9 |
| C3 | BC-17 | jump 부호 규약 J = ψ(이웃) − ψ(owner). OpenFOAM 소스를 읽지 않으므로 같은 규약인지 확인하지 않았다 | SPEC-LIT 120.2 |
| C4 | BC-17 | fan/porous flux law는 lagged Picard. 강한 baffle은 §53.2의 implicit 형태가 맞다 | SPEC-LIT 120.2 |
| C5 | VIO-08 | `.vdb` 바이트가 의도적으로 바뀌었다 (버전 224, bufferCount, 격자 이름). 이전 파일은 실제 OpenVDB에서 빈 격자로 읽혔다. Cartesian SHA 기준은 `69d21f7` 이후로 새로 잡는다 | SPEC-LIT 117.8 |
| C6 | DST-12 | 건조·탈휘발 동안 d_p 고정. 건조는 §76의 Ranz-Marshall, 응축은 0으로 자른다. h_v(T_p)는 ISL22의 상수 2.25 MJ/kg 대신 Watson 식. 끓음은 대류 부분에 Bird 인자 ln(1+B)/B. 휘발 단계는 m_v ≤ 입자 질량의 1e-6에서 끝난다. C_p 하나, 구간별 속도 동결, 복사는 T_0 주위 선형화 | SPEC-LIT 159.7 |
| C7 | DST-13 | char 단계에는 blowing 보정이 없다 (f = 1). ISL22 본문이 없어 논문의 처리를 확인하지 못했다 | SPEC-LIT 159.8 |
| C8 | DST-13 → DST-15 | DST-12의 수분·휘발 누적값은 구간 안의 dm을 더하므로, single에서는 저장된 m_w/m_v의 f32 반올림만큼 질량이 어긋난다. DST-15의 1e-10 질량 게이트를 single에서 맞추려면 char처럼 저장값 차이 규칙이 필요할 수 있다 (DST-12 산술이 바뀜) | SPEC-LIT 159.8 |
| C9 | TET-09 | `refine.size_factor` 기본값을 sqrt(6)/4에서 0.8로 바꿨다 (sqrt(6)/4에서는 평균 edge/h 0.797, 개수 +134 %로 두 게이트 모두 미달) | SPEC-LIT 116.3, 116.13 |
| C10 | D-VAL-5 | 1-D oracle에서 Wilcox 2006이 TG0 띠를 모두 닫지 못해 트리거가 발동하지 않았다 (log-law 1.134/1.179 대 1.0, Re_tau 2358 defect −11.8 %). 그래서 VAL-22a/b/c(Wilcox 2006 device·case)를 만들지 않는다. VAL-23은 SST만 기록한다 | SPEC-LIT 143.2 |
| C11 | VIO-04 | 117.3의 기본 voxel (min V_c)^(1/3)이 얇거나 비등방 메시에서는 영역 깊이보다 거칠다. 등급 채널 8×6×4 (깊이 0.1 m): h = 0.0697 m, n_z = 2로 0.139 m를 덮어 유체 비율 0.491, volume_error −0.301. 가이드북은 이런 메시에서 voxel을 직접 지정하라고 쓴다. 기본값 변경은 사용자 결정 | SPEC-LIT 117.3, 117.12 |
| C12 | BC-21 | 부력 압력 이름 4개(prghPressure 등)의 rho_f를 스텝마다 갱신하지 않았다. 갱신하면 face alpha가 움직이는 기존 prghPressure 계산이 바뀐다. 원하면 `Vof::refresh_vof_family`에 짧게 추가 | SPEC-LIT 120.7, 120.14 |
| C13 | BC-16 | DFSEM 와류 모양 F(q) = (1−q)², 상수 G = 512π/45045, 강도 규칙 A = (tr R/2 I − R)/G는 자체 설계다 (Poletto et al. 2013은 유료라 식을 보지 못했다) | SPEC-LIT 120.12 |
| C14 | AMG-05, AMX-02 | `-backend auto`가 이제 GAMG와 AMGX도 잰다. 큰 메시(gb_800000)에서는 auto가 GAMG나 AMGX를 골라 결과가 PBiCGStab와 solver 허용오차만큼 달라진다. AMGX 구조 재사용은 기본 켜짐 (`OFGPU_AMGX_REUSE=0`으로 끔) | SPEC-LIT 141.8, 141.21 |
| C15 | AUT-10 | D-6 비공허 기준을 권고로 정했다: real − rules ≥ 1, real − shuffled 평균 ≥ 1.0 (기하 개수). 이 기준으로는 AM-L 2026-10-02 통과(241/241/240.667)가 FAIL, AM-13과 oracle은 통과(oracle은 margin 1.000으로 경계). 계획 §11 item 5에 따라 **사용자 승인이 필요**하다 | `c71a9e8` |

## D. 연결·기능이 남은 것 (API만 있거나 후속 작업 필요)

- BC-06, BC-09, BC-14, BC-15, BC-17의 경계 이름: 모든 드라이버가 아직 이름으로 거부한다.
  - BC-14 `mapped`는 lowmach/buoyant의 U, p만 연결돼 있다.
  - `turbulentInlet`은 transient 경로가 필요하다.
- BC-10, BC-11: 드라이버 연결이 제한적이다.
- BC-12: lowmach end-to-end 테스트가 없다 (BC-22 몫).
- AUT-15: y+를 계산하는 드라이버가 `ofgpu-lowmach` 하나뿐이다.
- DST-05 → DST-06: FlowState에 μ(T)가 필요하다.
- AMG-04c/07b: 케이스에서 FCG, K-cycle, SPMD GAMG를 고를 수 없다. `io/case.rs`는 S 레인 파일이며, 연결은 AMG-04b 영역의 후속 작업이다. `ofgpu-par`도 rank별 `DistGamg`를 아직 붙이지 않는다.
- AMG-07a: `build_spmd` 안에서 한 rank만 `SpmdHalo::from_level`에 실패하면 나머지 rank가 ThreadComm 타임아웃까지 기다린다. `coarsen`이 만든 계획에서는 일어나지 않지만, 투표를 한 번 더 넣으면 막을 수 있다.
- PAR-17: `ofgpu-par`가 rank별 잔차를 출력하지 않으므로, 실행 패널은 rank별 Krylov 반복 수와 전체 잔차만 보여준다.
- STU-06b: 메시 초안에서 텍스트 필드를 비울 수 없다. 한 번 켠 automesher `-check`가 계속 남는다.
- STU-07a: gui의 patchRules 매칭은 `^...$`로 고정되지만, 솔버(`resolve_patch_rule`)는 고정하지 않는다. 예를 들어 `wall` 규칙과 `wallXMin` 패치에서 둘의 결과가 다를 수 있다.
- STU-07b: `set_patch`는 명령 한 번에 필드 하나만 쓴다. T를 푸는 케이스에서 wall을 velocity-inlet으로 바꾸려면 U와 T를 동시에 바꿔야 해서 항상 `PATCH_FIELD`로 거부된다.
- STU-13: 볼록하지 않거나 뒤틀린 셀은 `nearest-centre`로 답한다. snappy 계열 실제 메시에서는 대부분의 셀이 이렇게 답할 수 있다.
- STU-17: boolean 결과 파일을 덮어써도, 이미 열린 탭은 다시 열 때까지 이전 파일을 보여준다.
- VAL-70: README와 GUIDEBOOK이 splash 기준을 "실험과 비교하지 않았다"고 적고 있어서 새 MISSES 게이트와 맞지 않는다.
- VAL-30: JSONC `cartesian` 메시는 축마다 GradedAxis가 하나라서 내부의 앞전 쪽으로 양쪽에서 조밀하게 할 수 없다. 그래서 `cases/flatPlateLaminar.jsonc`는 x 방향이 균일하다. 등급 메시를 JSONC에서 쓰려면 `case_json`(S 레인)에 flat-plate 메시 종류나 축별 segment가 필요하다.
- TET-08 → TET-12/14: 내부 유동(TET-14 파이프)은 유체 쉘을 안쪽으로 감고 seed가 필요한데, surfprep(TET-05)이 cavity 쉘(V_s < 0)을 거부한다. 쉘을 뒤집거나 그 거부를 풀어야 한다.
- TET-07a: 매 pass가 Delaunay를 처음부터 다시 만든다 (`Delaunay::insert`가 비공개였다). 큰 표면에서는 증분 삽입이 필요하다. TET-09가 `insert_point`를 열었다.
- AUT-05: gui 카드 템플릿(`templates.json`)에 R-SNAP 항목이 없어 AUT-05 이후 캠페인 기록에서 `explainGeometry`가 'no template for rule id R-SNAP'를 돌려준다 (G 레인 vitest).
- VAL-12/VAL-14: README의 validate 집계(1021/1026, 2026-09-27 기준)가 낡았다. Gate 95-A는 행이 3개 늘었고 실패 행은 여전히 4개다. Gate 95-G의 ring 차수 행은 이제 통과한다 (nr 48→96에서 p = 0.926).
- DST-04: dust_progress가 공유 GPU에서 빌드마다 5~6분 걸린다 (계획 예산 8분). harness의 host 전송을 공유하면 빨라진다.
- F32-07: validate의 `max_abs_diff`가 NaN을 버리므로, single에서 다른 행들도 NaN으로 통과하고 있을 수 있다. CHT의 영역별 잔차(`residual_ranged`, `region_met`)는 아직 좁혀진 float 장을 재서 single에서 float 하한에 머문다 (F32-13/14 몫).
- AMG-04b: PbicgstabBackend 밖의 방정식(ofgpu-vof p_rgh, 압축성 p, 에너지, 난류, U, dist_solve)에서 GAMG를 고르면 첫 solve에서야 내부 API(`set_gamg`) 이름으로 거부된다. reader 단계에서 방정식 이름으로 거부하는 편이 낫다.
- AMX-02: 이 기계에는 AMGX_DLL/AMGX_DIR/PATH 설정이 없다. 모든 AMGX 실행은 다른 프로젝트(FDS_GPU_RUST_AmgX2)의 `amgxsh.dll`을 읽기 전용으로 썼다. 자체적으로 돌리려면 AMGX_DLL을 시스템에 설정하거나 DLL을 복사해야 한다.
- BC-16: 어떤 드라이버도 `BcInputs::turbulent_inlet`을 넘기지 않아 turbulentInlet, turbulentDigitalFilterInlet, turbulentDFSEMInlet이 모두 이름으로 거부된다. transient 경로를 여는 드라이버 단위가 필요하다.
- BC-16, BC-21: gui pick-list(registry)에 새 BC 이름 6개가 없다 (BC-22 몫).
- CHR-25: 모든 드라이버가 parcel 케이스를 거부한다. CHR-26이 lowmach 경로에서 이 거부를 풀어야 한다.
- AUT-07: optimise.py의 L4 상자가 R-ECON 이전 L1(`l1_of(r_econ=False)`)을 중심으로 한다. optimiser를 켜면 제안이 §B의 세 band 모양(×0.5..2)과 그 셀을 되살린다. R-ECON 캠페인(AUT-12 L-run, 그다음 AUT-09/10/11)으로 surrogate를 다시 맞춰야 한다. optimiser는 꺼진 채로 배포된다.
- AUT-11: AM-14 기록 `optimise/`(PASS, AUC 0.986)는 AUT-11 이전에 gate돼 gate 파일 sha가 없어서 꺼져 있다. sha를 나중에 채우면 켜질 수 있으므로 커밋된 모델에는 절대 채우지 않는다. 새 gate 디렉터리로 옮기려면 `prior.REPORT_DIR`/`optimise.SHIP_DIR`을 바꿔야 한다 (AUT-12 L-run 다음 단계).
- F32-19: studio의 single 빌드 실행은 F32-18 전까지 preview다 (`SINGLE_PRECISION_PREVIEW = true`, `gui/shared/src/protocol.ts`). F32-18이 들어오면 한 줄을 false로 바꾸고 dispatch.test의 header 기대값 두 개에서 접미사를 지운다. G 트리 `rust/target/single/release`의 single exe 21개는 오래됐을 수 있다 (단위 검증은 `--lib`만 빌드).
- STU-15: tet 100 000개 넘게 지나거나 부피 0인 tet(중심에 대해 star-shaped가 아닌 셀)을 만나면 그 유선은 'lost'로 끝난다. 볼록하지 않은 영역에서 나갔다 들어오는 선은 첫 출구에서 끝난다.
- gui registry: `MESH_KINDS`에 VAL-30의 `plate`가 없어 studio에서 평판 preset을 고를 수 없다 (플래그는 오케스트레이터가 `b17d194`에서 맞췄다).

## E. 기존 gui 실패 (이번 트랜치가 원인 아님)

- server `registry/schema.test` 'merges the enum pick-lists from $defs'
- server `http/routes.test` 'health, hello, registry and the schema'
- shared `dc.seed.test` CAP-BUOY 앵커 (`momentum.rs:203` 대신 `:117`, CMP-01이 옮김)
- `import.test`의 221-runs 케이스

---

## 이미 사용자가 결정한 것 (2026-10-14, 추가 테스트 없음)

1. CHR-07b의 device-host T 게이트는 dt 1e-6 s에서 수용한다 (SPEC-LIT 132.10).
2. CHR-08b: TM-4513 수송 하한을 300 K에서 200 K로 내린다. 200–300 K는 외삽이며 게이트가 없다 (SPEC-LIT 133.7, `64089eb`).
3. CHR-20의 τ = 0.1 S8 미달을 수용한다 (SPEC-LIT 136.5, 136.7).
4. BC-11b: f64 face force split을 기본값 꺼짐인 옵션 `-forceSplitDouble`로 둔다 (SPEC-LIT 120.7, `92c93a9`).
