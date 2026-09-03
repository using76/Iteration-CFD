# 폐기된 케이스 — 기록이지 케이스가 아닙니다 / Retired cases — records, not cases

이 디렉터리의 `.jsonc` 파일은 **실행되지 않습니다.** 실행하라고 두는 것이
아니라, 이미 발표된 측정값이 어떤 입력에서 나왔는지 확인할 수 있도록 남겨
둡니다. 살아 있는 후속 케이스는 `cases/` 최상위에 있습니다.

The `.jsonc` files in this directory **do not run**, and are not meant to. They
are kept so that a reader can see the exact input that produced a measurement
this repository has published. Their live successors are in `cases/`.

## 왜 실행되지 않는가 / Why they do not run

두 파일 모두 `wallTreatment lowRe`와 `turbulence.model kEpsilon`을 함께
이름합니다. 나중에 들어온 **SPEC-LIT §33** 규칙이 그 조합을 §13.4 오류로
거부합니다 — `kEpsilon`은 근벽 감쇠 함수가 없는 고-레이놀즈수 완결식이라
y+ ~ 30 아래에서는 메쉬가 아무리 조밀해도 유효하지 않기 때문입니다.

    error: patches: rule "inlet" T wallTreatment ("lowRe" together with
    turbulence.model): "kEpsilon" is not supported by ofgpu;
    available: LaunderSharmaKE

`-permissive`도 이 기록을 재현하지 못합니다 — 그 플래그는 `standard`(벽함수
행)로 대체하지 `lowRe`로 돌리지 않으므로, 대체된 실행은 애초에 이 케이스가
측정하려던 것과 다른 것을 측정합니다.

Both files name `wallTreatment lowRe` together with `turbulence.model
kEpsilon`. The later **SPEC-LIT §33** rule refuses that combination as a §13.4
error, because `kEpsilon` is a high-Reynolds-number closure with no near-wall
damping and is invalid below y+ ~ 30 however fine the mesh is. `-permissive`
does not recover the record either: it substitutes `standard` (the wall-function
row), not `lowRe`, so the substituted run measures something else.

## 왜 되살리지 않았는가 / Why they were not revived

토큰 하나(`kEpsilon` → `LaunderSharmaKE`)를 바꾸면 파일은 실행됩니다. 그렇게
하지 않은 이유는 **그 순간 아래 표의 숫자가 전부 그 파일의 것이 아니게 되기**
때문입니다. 되살리려면 다시 측정해야 하고, 다시 측정한다면 그것은 이 폐기된
게이트가 아니라 이미 그 자리를 대신하고 있는 후속 케이스입니다.

Changing one token (`kEpsilon` → `LaunderSharmaKE`) would make these files run.
That was deliberately not done: the moment it happens, none of the numbers below
belong to the file any more, and re-measuring them would be re-running a gate
that has already been replaced — not restoring a record.

## 무엇이 대신하는가 / What replaced them

| 폐기 / Retired | 대체 / Live successor |
|---|---|
| `channelThermalLowRe.jsonc` — SPEC-LIT §29.3 게이트의 해상 leg (비주기 2.0 m 덕트) | `cases/channelPeriodicFluxLowRe.jsonc` |
| `channelPeriodicLowRe.jsonc` — SPEC-LIT §31 페리오딕 재시도의 해상 leg | `cases/channelPeriodicFluxLowRe.jsonc` |

후속 케이스는 `LaunderSharmaKE`/`lowRe`(SPEC-LIT §33)로 실행되며, SPEC-LIT
§34가 진짜 2차원 평면 채널로 재구성하고 §35의 thermostat으로 구동합니다.
명령은 `cases/README.md`에 있고, 현재 판정은 `docs/07-lowmach-solver.md` §1.1에
있습니다.

The successor runs as `LaunderSharmaKE`/`lowRe` (SPEC-LIT §33), rebuilt by
SPEC-LIT §34 as a genuine 2-D plane channel and driven by §35's thermostat. Its
command is in `cases/README.md`; its current verdict is in
`docs/07-lowmach-solver.md` §1.1.

## 이 파일들이 낸 숫자 — 두 번 superseded / What they measured — superseded twice

아래 값은 (1) SPEC-LIT §33 규칙 **이전**에, 그리고 (2) SPEC-LIT §13.4.1의
numerics 수정 **이전**에 측정됐습니다. (2)는 특히 중요합니다: 그 당시 드라이버는
케이스의 `numerics` 블록을 전혀 읽지 않고 운동량을 `bounded Gauss upwind`로
돌렸습니다. 그러므로 이 숫자들은 **역사 기록**이며 현재 솔버의 예측이 아닙니다.

These were measured (1) **before** the SPEC-LIT §33 rule and (2) **before** the
SPEC-LIT §13.4.1 numerics fix. (2) matters most: the driver of the day did not
read the case's `numerics` block at all and ran momentum as `bounded Gauss
upwind`. They are **historical records**, not predictions of the current solver.

| | `channelThermalLowRe` | `channelPeriodicLowRe` |
|---|---|---|
| 측정 y+ (min/mean/max) | 0.43 / 0.89 / 1.02 | 0.302 / 0.310 / 0.318 |
| 벽법선 격자 | 50칸, two-sided `expansion: 200` | 50칸, two-sided `expansion: 200` |
| 구동 | 유입구동, `Tw = 373.15 K` | `momentumSource` + `-heaterPower -60` |
| 결과 | 두 메쉬 벽 열유입 비율 0.381 (게이트 열림) | `\|U\|` 잔차 ~1e-5에서 정체, 코어가 비물리적으로 냉각 |

전체 진단은 `docs/07-lowmach-solver.md` §1.1의 접힌 블록
("The second attempt", "The first attempt")에 있습니다.

The full diagnosis is in the collapsed blocks of `docs/07-lowmach-solver.md` §1.1
("The second attempt", "The first attempt").
