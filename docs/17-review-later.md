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

## C. 모델·설계 선택 (권고대로 진행, 검토 대기)

| # | 단위 | 선택 | 근거 |
|---|---|---|---|
| C1 | CHR-22 | Bordbar 2021 WSGG 가중치를 논문 그대로 쓴다 (적합 범위 안 72600번 평가 중 2243번이 음수, 최저 −0.1358). 잘라내면 모델이 바뀐다 | SPEC-LIT 136.9 |
| C2 | CHR-22 | 회색가스별 경계 방출 가중치는 기체 자신의 a_i를 쓰고, transparent window는 풀지 않는다. CHR-23이 결정해야 한다 | SPEC-LIT 136.4, 136.9 |
| C3 | BC-17 | jump 부호 규약 J = ψ(이웃) − ψ(owner). OpenFOAM 소스를 읽지 않으므로 같은 규약인지 확인하지 않았다 | SPEC-LIT 120.2 |
| C4 | BC-17 | fan/porous flux law는 lagged Picard. 강한 baffle은 §53.2의 implicit 형태가 맞다 | SPEC-LIT 120.2 |
| C5 | VIO-08 | `.vdb` 바이트가 의도적으로 바뀌었다 (버전 224, bufferCount, 격자 이름). 이전 파일은 실제 OpenVDB에서 빈 격자로 읽혔다. Cartesian SHA 기준은 `69d21f7` 이후로 새로 잡는다 | SPEC-LIT 117.8 |

## D. 드라이버 연결이 남은 것 (API만 있음)

- BC-06, BC-09, BC-14, BC-15, BC-17의 경계 이름: 모든 드라이버가 아직 이름으로 거부한다.
  - BC-14 `mapped`는 lowmach/buoyant의 U, p만 연결돼 있다.
  - `turbulentInlet`은 transient 경로가 필요하다.
- BC-10, BC-11: 드라이버 연결이 제한적이다.
- BC-12: lowmach end-to-end 테스트가 없다 (BC-22 몫).
- AUT-15: y+를 계산하는 드라이버가 `ofgpu-lowmach` 하나뿐이다.
- DST-05 → DST-06: FlowState에 μ(T)가 필요하다.

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
