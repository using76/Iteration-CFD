# Test cases

`ofgpu-generate-mesh` writes a complete, ready-to-run case: `constant/polyMesh`,
`constant/physicalProperties`, `constant/momentumTransport`, `system/{controlDict,
fvSchemes,fvSolution}` and a `0/` directory with `U`, `k`, `epsilon`, `omega` and `nut`.

```powershell
cargo run --release --bin ofgpu-generate-mesh -- <case> <outputDir> [nx ny nz] [-stl [name=]path]... [-wallModel standard|spalding|rough|lowRe [-Ks x [-Cs y]]] [-permissive]
```

### `-wallModel` — 벽 처리 프리셋 (SPEC-LIT §29.1)

한 케이스의 모든 벽에 `nut`/`k`/`epsilon`/`omega`(그리고 에너지 방정식을 풀 때는
`T`까지) 경계 타입을 **일관된 한 행**으로 채워 넣습니다 — 필드마다 따로 골라
서로 모순되는 조합(예: 서브레이어를 직접 해상하는 `nutLowReWallFunction`과
벽함수로 강제하는 `epsilonWallFunction`을 같이 씀)을 만드는 실수를 막기 위한
것입니다.

| 프리셋 | `nut` | `k` | `epsilon`/`omega` | `T` (에너지 방정식을 풀 때) |
|---|---|---|---|---|
| `standard` (기본값) | `nutkWallFunction` | `kqRWallFunction` | `epsilonWallFunction`/`omegaWallFunction` | `thermalWallFunction` |
| `spalding` | `nutUWallFunction` | `kqRWallFunction` | `epsilonWallFunction`/`omegaWallFunction` | `thermalWallFunction` |
| `rough` | `nutkRoughWallFunction`(`-Ks` 필요, `-Cs` 기본 0.5) | `kqRWallFunction` | `epsilonWallFunction`/`omegaWallFunction` | `thermalWallFunction` |
| `lowRe` | `nutLowReWallFunction` | `kLowReWallFunction` | `zeroGradient` (분자 점성만) | 손대지 않음 — 해상된 서브레이어 자체의 분자 저항을 그대로 둠 |

`simulationType LES;`인 케이스는 `k`/`epsilon`/`omega` 완결식이 없으므로 위 표는
`nut` 열 하나로 줄어듭니다 — SPEC-LIT §30.1의 자체 표(§29.1의 유일한 생존
멤버, 다른 답으로): `standard`/`spalding` 둘 다 `wernerWengleWallFunction`
(Werner & Wengle 1991 — 첫 셀 평균 속도로 적분·역해한 멱법칙, Newton 반복 없음)
으로 읽히고, `lowRe`는 그대로 `nutLowReWallFunction`(`nu_t,w = 0`)이며, `rough`는
아직 LES 벽모형이 없어 둘 중 하나를 이름으로 밝힌 §13.4 오류로 거부됩니다(조용한
대체 없음 — 미래 작업).

`-wallModel`을 주지 않으면 기존 동작(레거시 기본값) 그대로입니다. `rough`는
`-Ks`(모래알 거칠기, m)가 필수이고 `-Cs`(거칠기 상수, 기본 0.5)는 선택입니다.
예:

```powershell
cargo run --release --bin ofgpu-generate-mesh -- plume ..\cases\plumeRough 30 20 10 -wallModel rough -Ks 0.003
```

`Ks -> 0`은 매끈한 벽함수를 반올림 오차 수준까지 정확히 재현합니다
(`ofgpu-validate`의 상시 게이트, 최상위 README "검증" 절 참고). 필드별
명시적 패치 타입 > 패치별 `wallTreatment` 오버라이드 > 케이스 기본값 순으로
우선하며, 한 벽 패치의 네(다섯) 필드가 서로 다른 행에 걸쳐 있으면 이름을
밝힌 오류로 거부합니다 — `-permissive`는 `nut`의 선택이 함의하는 행으로
대체하고 무엇으로 바꿨는지 출력합니다.

`-stl [name=]path` (반복 가능) — 어느 케이스든 블록 격자를 STL 표면으로 계단식
(castellated)으로 조각합니다 (SPEC-LIT §23). 표면 안에 셀 중심이 놓인 셀은 제거되고,
그 자리에 STL의 `solid` 이름(이진 STL은 파일 이름, `name=`으로 재지정)을 딴
새 wall 패치가 생깁니다. 새 패치는 blockgen이 벽에 쓰는 것과 같은 벽 경계조건을
받으므로 생성된 케이스는 그대로 실행됩니다. 표면은 닫혀 있어야 하며, 열린
표면은 열린 모서리 개수를 보고하고 거부됩니다 — `-permissive`는 패리티 투표로
대체하고 계속합니다. 예:

```powershell
cargo run --release --bin ofgpu-generate-mesh -- plume ..\cases\plumeCol 60 40 30 -stl column=column.stl
```

케이스별 설명입니다. 어느 것도 운동량 방정식을 풀지 않습니다 — **U와 phi는 고정**이고,
난류 모델만 그 위에서 수렴시킵니다. 그래서 각 케이스의 관건은 "고정된 U가 물리적으로
말이 되는가"와 "phi가 이산적으로 보존되는가"입니다.

| Case | Default size | Geometry | Frozen velocity field |
|---|---|---|---|
| `channel` | 200 x 120 x 1 | 평면 채널, 양 벽으로 expansion 20 grading | 1/7 멱법칙 프로파일 `U = (Uref (1-\|y\|)^(1/7), 0, 0)` |
| `cavity` | 128 x 128 x 1 | 정사각 공동, 네 면 모두 wall | 유선함수 `psi = A sin(pi x/Lx) sin(pi y/Ly)` 에서 나온 단일 재순환 셀 |
| `step` | 300 x 100 x 1 | 후향계단의 **하류 박스만** (계단면 없음) | 1/7 멱법칙 |
| `big` | 160^3 | 균일 정육면체, 벤치마크용 | 1/7 멱법칙 |

## Notes that matter

**`channel`** — `U`가 `y`에만 의존하므로 `phi = interpolate(U) & Sf`가 **정확히**
이산 보존입니다 (`max |sum_f phi| = 8.7e-19`). 두 모델 모두 몇백 회에 잔차 1e-8까지
떨어집니다. 첫 셀 중심이 y+ ≈ 11.5에 놓이기 때문에, 이 케이스는 omega 벽함수의
분기 전환 문제를 그대로 드러냅니다 — 그래서 생성되는 `constant/momentumTransport`가
`blended yes`를 씁니다. 최상위 README의 해당 절을 보세요.

**`cavity`** — 유선함수에서 만든 속도장이라 해석적으로는 발산이 0이지만,
면으로 보간한 `phi`는 이산적으로 완전 보존은 아닙니다 (`max |sum_f phi| ≈ 1e-6`).
이건 결함이 아니라 의도된 것입니다: `bounded Gauss` 스킴의
`- fvm::Sp(fvc::div(phi), psi)` 보정이 실제로 필요한 상황을 만들어 줍니다.
관통류가 없는 재순환 유동이라 수렴은 느립니다 — 2만 회에 1e-8입니다.

> 정지한 유체에 뚜껑만 움직이는 진짜 lid-driven cavity를 **고정 U**로 돌리면,
> 전단이 맨 윗줄 셀에만 있어서 나머지는 순수 확산 문제가 됩니다.
> 난류 모델을 시험하는 케이스로는 의미가 없어서 재순환장을 넣었습니다.

**`step`** — 이름과 달리 진짜 후향계단이 아닙니다. `blockgen`은 직육면체 블록
하나만 만들 수 있어서 계단면도, 입구 채널 오프셋도 없습니다. 실제 BFS 메쉬와
같은 셀 수·종횡비를 갖는 박스라 성능 측정과 경계조건 배관 확인에는 쓸 만하지만,
그 해는 박리 유동이 아니라 발달하는 평면 채널입니다.

**`big`** — 3-D 균일 격자. 벤치마크 전용이고 물리적 의미는 없습니다.

## Running

```powershell
cd ..\rust

# k-epsilon
cargo run --release --bin ofgpu-k-epsilon -- ..\cases\channel -iters 4000 -check 400

# k-omega  (momentumTransport의 model을 kOmega로 바꾼 뒤)
cargo run --release --bin ofgpu-k-omega   -- ..\cases\channelKW -iters 4000 -check 400
```

결과는 `<caseDir>/1/`에 OpenFOAM ASCII 형식으로 쓰이므로 ParaView나 `foamToVTK`로
바로 열립니다.

## JSONC 케이스 (`*.jsonc`)

`ofgpu-fire`와 `ofgpu-k-epsilon`은 OpenFOAM 케이스 디렉터리 대신 주석과
trailing comma를 허용하는 JSON 파일 하나로도 케이스를 읽습니다
(`docs/05-io-redesign.md` §4.1; 스키마는 `docs/schema/case-1.json`,
`ofgpu::io::case_json::emit_schema`로 자동 생성). 디렉터리 대신 `.jsonc`/`.json`
파일 경로를 넘기면 됩니다 — 출력은 `<stem>_jsonc/`에 씁니다.

**정정.** 이 문단은 오랫동안 "`ofgpu-buoyant`, `ofgpu-fire` 등 부력/화재 계열
드라이버"라고 적혀 있었지만 `ofgpu-buoyant`는 JSONC를 읽지 않습니다 —
`read_poly_mesh`만 호출하며 `.jsonc` 경로를 주면 "OpenFOAM 케이스가 아니다"로
실패합니다. `ofgpu-plume`, `ofgpu-k-omega`, `ofgpu-vof`도 마찬가지로 디렉터리
전용입니다. 문서가 실제와 어긋난 것 자체가 SPEC-LIT §13.4.1이 말하는 결함의
문서 쪽 절반이라 여기 남겨 둡니다.

**JSONC 케이스가 이름은 대지만 어떤 드라이버도 구현하지 않는 블록**
(SPEC-LIT §13.4.2): `output` 블록 전체, `run.adjustTimeStep: true`,
`run.maxCo`. 셋 다 §13.4 오류이며 오류 메시지가 대안(`-output`,
`-writeInterval`, `-restartWrite`, `-deltaT`, 그리고 이 크레이트에서 유일하게
스텝을 조절하는 `ofgpu-vof`)을 이름으로 밝힙니다. `-permissive`는 무엇으로
대체했는지 출력하고 계속 진행합니다. `run.endTime`/`run.deltaT`와
`adjustTimeStep: false`는 그대로 지켜집니다.

`mesh.grading` — 축별 격자 조밀화(선택). 기본은 균일 격자이며, 이 키가 아예
없거나 축 항목이 비어 있으면 이전과 **비트 단위로 동일한** 메쉬가 나옵니다.
`blockgen`의 `GradedAxis`(`expansion`, `two_sided`)를 그대로 JSON으로 옮긴
것입니다 — `expansion`은 `twoSided: true`일 때 중앙 셀/벽 셀의 비, `false`
(기본값, 벽 하나만 조밀화)일 때는 `hi`쪽 셀/`lo`쪽 셀의 비이며 반드시 0보다
커야 합니다. `<= 0`인 `expansion`이나 인식하지 못하는 키는 SPEC-LIT §13.4
오류(대상 축을 이름으로 밝힘)이고, `-permissive`는 그 축을 균일 격자로
되돌리고 무엇으로 바꿨는지 출력합니다:

```jsonc
"mesh": {
  ...,
  "grading": { "y": { "expansion": 6.0, "twoSided": true } },
},
```

예시는 [`../docs/case-example.json`](../docs/case-example.json)의 주석 처리된
`grading` 줄을, 실제로 켜서 쓰는 예는 아래 `channelPeriodicFluxLowRe.jsonc`(벽법선
방향 y+ ≈ 1을 노리는 two-sided grading)를 보십시오.

`mesh.cyclic` — SPEC-LIT §31.1의 cyclic 패치 쌍, §34.2에서 여러 쌍으로
일반화됨. `mesh.boundaries`의 여섯 이름 중 마주보는 두 개(`xmin`/`xmax`,
`ymin`/`ymax`, `zmin`/`zmax` 중 하나)를 `a`/`b`로 지정하면 그 두 면이
경계가 아니라 결합된 한 쌍이 됩니다. `transform`은 `"translate"`만 지원 —
축의 길이만큼 평행이동한다는 뜻이며, 회전 쌍(`"rotate"`)은 면 매칭·벡터
변환이 아직 없어 `translate`를 이름으로 밝힌 SPEC-LIT §13.4 오류입니다.
`mesh.cyclic` 배열은 이제 축마다 하나씩, 개수 제한 없이 받습니다 — 두
방향 모두 주기인 평면 채널, 세 방향 모두 주기인 완전 주기 박스까지
오늘 선언할 수 있습니다(아래 `channelPeriodicFluxWF.jsonc`/
`channelPeriodicFluxLowRe.jsonc`가 x 하나만 쓰는 예입니다). 같은 축을 두
쌍이 동시에 주장하면 축을 이름으로 밝힌 오류이고, cyclic으로 지정된
패치는 `patches[]`의 구체적 규칙(`empty`/`symmetry` 같은 constraint
포함)으로 또 지정할 수 없습니다(양쪽 이름을 밝힌 오류) — 다만 만능
규칙(`".*"`)은 상관없고, 오히려 `resolve_patch_rule`이 모든 패치 이름에
대해 규칙 하나를 요구하므로 cyclic 패치들을 받아줄 만능 규칙이 case
끝에 있어야 합니다(값 자체는 cyclic 패치에는 적용되지 않습니다 — 모든
필드가 자동으로 `cyclic` 타입이 됩니다):

```jsonc
"mesh": {
  ...,
  "cyclic": [
    { "a": "streamwiseMin", "b": "streamwiseMax", "transform": "translate" },
  ],
},
```

`blockgen`의 `-cyclic x|y|z` 플래그(반복 지정 가능, OpenFOAM 형식 케이스)와
같은 짝짓기·같은 두 불변식(전단사, `Sf_a == -Sf_b`, 쌍마다 검증)을 씁니다 —
자세한 내용은 SPEC-LIT §31.1/§34.2와 `../README.en.md`의 "Cyclic
patches" 행을 보십시오. 실제로 켜서 쓰는 예는 아래
`channelPeriodicWF.jsonc`/`channelPeriodicFluxWF.jsonc`입니다.

`"kind": "empty"`/`"symmetry"` — SPEC-LIT §34.1의 constraint 패치. JSONC가
전에는 `wall`/`inlet`/`open`만 표현할 수 있어 2차원 케이스를 아예 쓸 수
없었습니다; 이제 `mesh.boundaries`에서 한 축을 1칸으로 만들고 그 두 면을
`empty`로 선언하면 됩니다:

```jsonc
"boundaries": { ..., "zmin": "back", "zmax": "front" },
"cells": [8, 50, 1],
```

```jsonc
"patches": [
  { "match": "(back|front)", "kind": "empty" },
  { "match": ".*", "kind": "wall" },
],
```

경계조건이 아니라 CONSTRAINT입니다 — `empty`/`symmetry` 규칙이 `U`/`p`/`T`
등 필드별 BC까지 같이 지정하면 그 필드를 이름으로 밝힌 오류이고, `empty`는
셀이 두 개 이상인 축에서는 슬롯과 실제 셀 개수를 밝혀 거부합니다. 아래
`channelPeriodicFluxWF.jsonc`/`channelPeriodicFluxLowRe.jsonc`가 이제 이
방식으로 옆벽 없는 진짜 평면 채널입니다.

`sources` — SPEC-LIT §18/§31.1의 체적 소스, JSONC 쪽 입구입니다. 오늘은
두 종류를 지원합니다. `momentumSource`: 전체 도메인에 걸친 균일 체적력(단위
질량당, m/s²) — periodic 케이스처럼 유입 경계가 없어 질량유량을 지정할 수
없는 케이스가 유동을 몰아가는 유일한 방법입니다. `field`는 반드시 `"U"`여야
하고(체적력은 벡터라 운동량 방정식에만 뜻이 있음), 다른 값은 이름을 밝힌
오류입니다. `thermostat`(SPEC-LIT §35.1): 도메인의 체적평균(mixed-mean이
아님) 온도에 거는 균일 비례 제어기, `q = -rho_cp (T_mean - target)/tau` —
언제나 `T` 방정식 전체 도메인(`selection`은 생략하거나 `"all"`만 허용)에
걸립니다. 닫힌 주기 도메인에서 모든 열 경계가 Neumann이면(양쪽 벽
fixed-flux, 스트림방향 cyclic, 앞뒤 empty) 정상상태 온도 방정식이 순수
Neumann이 되어 더하는 상수만큼 부정확해지는데(§8.5가 압력에 대해 이미
하는 것과 같은 null space, `T`로 읽은 것), 이 제어기가 그 상수를 고정합니다.
`tau`는 생략하면 도메인 자체의 flow-through 시간(`V^(1/3)/U_ref`)으로
기본값이 잡히고, 명시적으로 줄 수도 있습니다.

`weighting`(SPEC-LIT §35.3): 제어기가 요구하는 TOTAL 파워를 어떻게
분배하는가. 균일 체적 싱크는 상수만 고정하는 것이 아니라 프로파일까지
바꿉니다 — 주기 완전발달 덕트에서 온도장을 주기적으로 만드는 보정항은
국소 스트림방향 질량유속 `rho u . e_hat`에 비례하고, 균일 형태는 그 항의
슬러그류 극한(`rho u . e_hat -> rho_bar U_b`)입니다. 균일 싱크는 근벽
(`u . e_hat < U_b`)에서 필요보다 많은 열을 빼고 코어에서는 적게 빼므로
`(T_w - T_b)`를 줄이고 `Nu`를 높게 편향시킵니다. `"weighting": "massFlux"`가
올바른 분배이고, `"direction"`이 스트림방향 `e_hat`입니다(생략하면 메쉬의
단 하나뿐인 cyclic 쌍의 축에서 가져오고, cyclic 쌍이 없거나 둘 이상이면
추측하지 않고 §13.4 오류). 기본값은 의도적으로 `"uniform"`입니다 —
`docs/07-fire-solver.md` §1.1에 기록된 모든 측정이 균일 형태로 이루어졌고
비트 단위로 재현되어야 하므로, `massFlux`는 명시적 opt-in입니다. `uniform`에
`direction`을 같이 주는 것은 읽고 무시하는 셈이라 §13.4 오류입니다.
두 종류 모두 같은 `crate::sources::SourceSpec` 레지스트리로 내려가지만
(OpenFOAM 형식 케이스의 `constant/fvSources`는 box/sphere 선택과 함께 이
둘을 포함해 일곱 종류의 항을 지원), 이 배열은 그 전체 표면을 복제하지
않고 페리오딕/닫힌 케이스에 필요한 이 두 가지만 추가합니다:

```jsonc
"sources": [
  { "type": "momentumSource", "field": "U", "bodyForce": [3.9, 0, 0] },
  { "type": "thermostat", "target": 293.15, "tau": 0.02 },
  // SPEC-LIT §35.3의 질량유속 가중 (opt-in):
  // { "type": "thermostat", "target": 293.15, "tau": 0.02,
  //   "weighting": "massFlux", "direction": [1, 0, 0] },
],
```

| Case | 드라이버 | 설명 |
|---|---|---|
| `dieStack.cht.jsonc` | `ofgpu-cht` | **SPEC-LIT §46/§47** — 켤레 열전달(conjugate heat transfer)의 첫 케이스이자, 이 저장소에서 처음으로 **여러 영역(region)** 을 하나의 열 메쉬로 이어 붙여 푸는 케이스입니다. 100 W를 소산하는 10 x 10 x 0.7 mm 실리콘 다이 → 솔더 TIM1 → 구리 스프레더 → 그리스 TIM2 → 등온 냉각판, 접촉저항 3개(§47.5)와 메쉬축 대각 이방성 전도도 (§46.3, 면내 120 / 두께방향 30 W/(m K))를 포함합니다. **확장자 `.cht.jsonc`** 는 이 문서의 다른 모든 케이스가 쓰는 유동 케이스 형식(`io::case_json`)이 아니라 다영역 전도 형식(`io::case_cht`)임을 뜻하며, 두 형식의 shipped-case 검사도 서로 나뉘어 있습니다. 스택 전체가 1차원이므로 **닫힌 해**가 있고 실측이 그것을 재현합니다: 접합부 온도 **649.7118 K**(닫힌 해 `300 + q(3.380452e-4) + 11.6667`, 상대오차 1e-8 이내), 세 계면 각각 **−100.000000 / +100.000000 W**, §47.12 Gate 4 보존 불균형 **0.000e0**, 최대 계면 온도 **점프** 40.000000 K = `q * 4e-5`(§47.3의 Robin 삼중항이 불연속을 실을 수 있기 때문 — cyclic 보간으로는 불가능). **§13.4.1 짝 시험**: 접촉저항 하나를 `2.5e-5` → `7.5e-5`로 바꾸면 접합부가 정확히 `q dRc` = 50 K 올라갑니다. 실행: `cargo run --release --bin ofgpu-cht -- ..\cases\dieStack.cht.jsonc -csv dieStack.csv` |
| `plume.jsonc` | `ofgpu-k-epsilon -permissive` | `plumeB`(OpenFOAM 형식)의 JSONC 재현 — 두 형식이 같은 필드를 만든다는 B3 게이트. **`-permissive`가 필요해졌습니다(SPEC-LIT §13.4)**: 이 케이스는 `physics.gravity [0, 0, -9.81]`을 이름하는데(짝인 `plumeB`의 `constant/g`도 같습니다), `ofgpu-k-epsilon`은 얼린 `U` 위에서 난류 두 방정식만 풀고 온도장을 읽지 않으므로 §17의 부력 생성 `G_b = (nu_t/Pr_t) g·grad(T)/T`를 만들 재료가 없습니다. `KEpsilon::set_buoyancy`는 처음부터 있었고 아무도 부르지 않았습니다 — 즉 이 명령은 **예전에도 `G_b`를 0으로 두고 돌았고, 그 사실을 말하지 않았을 뿐입니다**. `-permissive`는 무엇을 대체했는지 출력하고 예전과 **비트 단위로 같은** 결과를 냅니다(측정: `cases/channel`·`cases/channelKW`·`plumeB` 모두 수정 전후 필드 파일이 완전히 동일). 부력이 실제로 필요하면 `ofgpu-fire`(JSONC)나 `ofgpu-plume`/`ofgpu-buoyant`(디렉터리)를 쓰십시오 |
| `burnerPlume.jsonc` | `ofgpu-fire -combustion -radiation` | 프로판 버너 화재 데모 — SPEC-LIT §25(저-마하)·§26(에너지)·§27(연소)·§28(P1 복사)를 한 케이스에서 결합. 바닥 창(`Y_F = 1` 고정)으로 연료가 들어가고, 열은 전부 연소의 `q'''_c`에서 나옵니다(입구 자체는 상온) |
| `burnerPlume_fvDOM.jsonc` | `ofgpu-fire -combustion -radiation` | `burnerPlume.jsonc`와 메쉬·유동·연소가 완전히 동일한 SPEC-LIT §36 fvDOM 쌍둥이 케이스 — `physics.fire.radiation.model`만 `"fvDOM"`으로 다름. 같은 1200스텝(`-endTime 6.0 -deltaT 0.005`)에서 복사 분율 P1 **14.97%** 대 fvDOM **13.79%**(둘 다 도메인 열방출 대비) — 벽시계 시간은 P1 **19.22 s** 대 fvDOM **121.5 s**(RTX 5070 Ti, 32768 셀, 배수 6.3) — SPEC-LIT §36.6가 명시한 "N_ordinates배 비용"이 실측치로 확인됨. **재실행(SPEC-LIT §13.4.1)**: 예전 기록은 P1 15.08% / fvDOM 13.35%, 18.8 s / 119 s였다. 그 숫자들은 이 케이스의 `numerics` 블록을 전혀 읽지 않는 드라이버가 낸 것이다(운동량이 `bounded Gauss upwind`로 돌았고 케이스는 `Gauss linearUpwind grad(U)`를 요구한다). 두 모델 모두 고쳐진 드라이버로 다시 돌렸고, 이 행이 주장하는 **P1 대 fvDOM 비교 자체는 그대로다**(fvDOM이 여전히 더 적게 복사하고, 차이는 1.73포인트에서 1.15포인트로). 비교의 대상이 되는 상태 자체는 크게 움직였다 — `docs/07-fire-solver.md` §6 참고 **재실행(SPEC-LIT §26.1)**: §25.1의 발산 구속조건에 전도항 `div(k_eff grad T)`를 넣은 뒤 같은 명령으로 다시 돌렸다. P1 14.9825% → **14.9725%**, fvDOM 13.8303% → **13.7893%** — 두 모델 모두 0.05포인트 미만으로 움직였고 P1 대 fvDOM 차이는 1.15 → 1.18포인트로 이 행이 주장하는 비교 자체는 그대로다. 화재의 팽창은 연소의 `q'''_c`가 지배하는데 그 항은 원래부터 `Q`에 있었기 때문이다 |
| `nistRSE1994.jsonc` | `ofgpu-fire` | **SPEC-LIT §42.8 Gate 2** — NIST 축소 규모 밀폐공간(RSE) 1994 환기제한 화재. 0.98 x 1.46 x 0.98 m 구획, 0.48 x 0.81 m 출입구, 바닥 중앙 버너 — 메쉬 영역(region)을 **두 개** 쓰는 첫 케이스입니다(`blockgen::BlockSpec`이 슬롯당 창 하나를 지원하도록 §42.8에서 일반화됨). 직렬 2단계 스킴(§42) + 산소 기반 소염(§43), 메탄. 50–600 kW 스윕. **이 게이트는 실패합니다**: 천장 CO 체적분율이 측정값(Bryner, Johnsson & Pitts, NISTIR 5568, 1994)보다 최대 20배 낮습니다. 산소 고갈 시점은 한 스텝 이내로 맞지만 연소효율이 15–58 %에 그쳐 연료 대부분이 미연소로 빠져나갑니다 — 화학이 아니라 **환기**의 문제이며, SPEC-LIT §42.8b가 그 진단과 숫자를 기록합니다. 실행: `-endTime 30 -deltaT 0.005 -probe 0.30,0.10,0.88 -probe 0.30,1.16,0.88` |
| `channelThermalWF.jsonc` | `ofgpu-fire` | SPEC-LIT §29.3/§30.3의 게이트, 메쉬 (a): 2.0 m(L/Dh = 50) 입구구동 평면 채널/덕트, 위아래 벽 `Tw = 373.15 K`, 중력 0. 벽법선 방향 균일 격자(6칸) — 측정 y+ 21.95/37.90/40.29(목표 30-60), `standard` 프리셋, `thermalWallFunction`이 실제로 일을 해야 하는 조건. **재실행(SPEC-LIT §13.4.1)**: 고쳐진 드라이버로 6,000 반복을 다시 돌려 y+ **21.95/37.76/40.31**, 총 벽 열유입 **157.035 W**, `T_b` 308.039 K, `U_b` 3.00469 m/s, `\|U\|` 잔차 9.4e-10. 위 y+는 사실상 변하지 않았다 |
| `retired/channelThermalLowRe.jsonc` | — (실행되지 않음) | **폐기됨. 케이스가 아니라 기록입니다** — SPEC-LIT §29.3 게이트의 해상 leg. `wallTreatment lowRe`와 `turbulence.model kEpsilon`을 함께 이름하는데 나중에 들어온 SPEC-LIT §33 규칙이 그 조합을 §13.4 오류로 거부하므로 그대로는 실행되지 않으며, 되살리지 않았습니다. 살아 있는 후속 케이스는 `channelPeriodicFluxLowRe.jsonc`입니다. 이 파일이 낸 숫자(y+ 0.43/0.89/1.02, 벽 열유입 비율 0.381)와 왜 되살리지 않았는지는 `cases/retired/README.md` |
| `channelPeriodicWF.jsonc` | `ofgpu-fire` | SPEC-LIT §31의 페리오딕 재시도, 메쉬 (a): 위 두 케이스와 같은 단면(0.04 m × 0.04 m)·`Tw`를 스트림방향 **cyclic**(`mesh.cyclic`, 0.08 m, 8칸)으로 바꾸고, 유입 대신 `sources[]` `momentumSource`(체적력 3.9 m/s²)로, 에너지는 `-heaterPower -6`(균일 도메인 열싱크)로 구동 — 벽법선 균일 6칸, `standard`/`thermalWallFunction`. 측정 y+ 40.25/41.73/43.41, 완전 수렴(`\|U\|` 잔차 1.3e-10). **재실행(SPEC-LIT §13.4.1)**: 고쳐진 드라이버로 3,000 반복을 다시 돌려 y+ **40.14/41.80/43.64**, 총 벽 열유입 **6.00555 W**(싱크 −6 W와 §31의 에너지 균형이 여전히 닫힘), `T_b` 314.243 K, `U_b` 3.53146 m/s, `\|U\|` 잔차 1.0e-9 — y+도 열균형도 실질적으로 변하지 않았다 |
| `retired/channelPeriodicLowRe.jsonc` | — (실행되지 않음) | **폐기됨. 케이스가 아니라 기록입니다** — SPEC-LIT §31 페리오딕 재시도의 해상 leg. 위와 같은 이유로 실행되지 않습니다(같은 §33 규칙). 측정값은 y+ 0.302/0.310/0.318이며 `\|U\|` 잔차가 ~1e-5에서 정체했습니다. 살아 있는 후속 케이스와 전체 사유는 `cases/retired/README.md` |
| `channelPeriodicFluxWF.jsonc` | `ofgpu-fire` | SPEC-LIT §32의 재설계된 게이트, SPEC-LIT §34로 진짜 2차원 평면 채널로 재구성한 메쉬 (a): 스트림방향 cyclic, 앞뒤 `empty`(옆벽 없음), 위아래 가열벽만 — 고정 열유속(`fixedFluxTemperature`, `q = 500 W/m2`, 양쪽 가열벽), SPEC-LIT §35.1의 `sources[]` thermostat(`target 293.15`, `tau 0.02`, 예전 `-heaterPower -3.2`를 대체)으로 균형을 맞춤(§32.2). `standard`/`thermalWallFunction`, 측정 y+ 56.8/57.7/58.5, bit 단위로 고정된 상태로 수렴(`\|U\|` 잔차 1.4e-10). 평면 채널이라 가열-둘레와 젖은-둘레가 일치하여 `D_h` = 2H = 0.08 m 하나뿐 — Re = 28,638, Nu = 65.24(변화 없음) — Gnielinski −4.5%, Dittus-Boelter −11.5%, 둘 다 안쪽(**게이트 닫힘, 유지**). 열평형: thermostat 출력 −3.2 W 대 측정 벽열 3.2 W, 차이 2.8e-7 W(반올림 수준). **나중 보정(SPEC-LIT §32.4/§32.5)**: 위 Gnielinski 값은 Petukhov의 매끄러운 **원관** 마찰계수 `f`로 평가한 것이므로 §32.4가 말하는 **절대 예측** 판정이다. 이 leg가 실제로 실현하는 `f` = 0.02162(원관 상관식보다 −9.6%)로 평가하면 Nu_Gn = 61.30, +6.4% — **레이놀즈 유사** 판정으로도 밴드 안쪽. 단, 이 `f`는 아직 벽면 전단력의 직접 측정이 아니라 체적력 균형에서 추론한 값이다(§32.5.3) **재실행(SPEC-LIT §32.5.3/§35.3)**: 이 케이스는 이제 thermostat에 `"weighting": "massFlux"`를 명시한다. `uniform`으로의 통제 재실행은 위 숫자를 마지막 자릿수까지 그대로 재현했고, `massFlux`에서는 Nu = 64.32(−1.41%), T_w = 317.567 K, T_b = 293.256 K, Re = 28,622, thermostat 출력 −3.20335 W(에너지 불균형 0.105%). 절대 예측 판정은 여전히 닫힌다 — Gnielinski −5.8%, Dittus-Boelter −12.8%. 하지만 `f`를 벽에서 직접 측정하자 0.017247(`rho u_tau^2` 형태) / 0.019960(점성 형태)로, 예전에 추론했던 0.02162보다 8–25% 낮다 — **레이놀즈 유사 판정은 더 이상 닫히지 않는다**(+33.8%, 점성 `f`로도 +14.4%). 예전의 “+6.4%, 두 leg 모두 통과”는 추론된 `f`가 만든 허상이었다. 힘 균형은 이 leg에서 운동학적 단위로 +0.000% 정확히 닫힌다(SPEC-LIT §32.5.2의 보정) **재실행(SPEC-LIT §13.4.1/§32.5.5 — 케이스가 실제로 요구하는 numerics로)**: 위 블록의 모든 숫자는 이 파일의 `numerics` 블록을 **하나도 읽지 않는** 드라이버가 낸 것이다 — 운동량이 `MomentumControls::default()`, 즉 `bounded Gauss upwind`로 돌았고 이 파일은 `bounded` 없는 `Gauss linearUpwind grad(U)`를 이름한다. 고쳐서 다시 돌린 결과: `T_b` = 293.251 K \| `T_w` = 317.483 K \| ΔT = 24.2318 K \| `U_b` = 5.3972 m/s \| Re = 28,785 \| **Nu = 64.5257**(64.3168에서 +0.32%) \| y+ 56.89/57.78/58.59 \| thermostat 출력 −3.2034 W 대 벽열 3.2 W(+0.106%) \| `f` 측정값 0.017129(`rho u_tau^2`) / 0.019760(점성). 판정은 실질적으로 그대로다 — **절대 예측 닫힘**(Gnielinski at 원관 `f` = 0.023878 → 68.5979, −5.9%; Dittus-Boelter 74.0568, −12.9%), **레이놀즈 유사 열림**(+34.4%, 점성 `f`로도 +15.4%). 움직인 것은 힘 균형이다: 운동학적 불일치가 −0.113% → **−0.005%**. 통제 실험으로 `div(phi,U)`를 손으로 `bounded Gauss upwind`로 되돌리면 옛 기록이 유효숫자 5자리까지 재현된다(Nu 64.3136 대 64.3168) — 즉 이것은 같은 케이스의 재실행이지 다른 케이스가 아니다 **재실행(SPEC-LIT §26.1 — 에너지 불균형의 원인을 찾아 닫음)**: §25.1의 발산 구속조건 `div u = Q/(rho cp T)`가 `Q`에서 전도항 `div(k_eff grad T)`를 빠뜨린 채 구현되어 있었다. 넣고 다시 돌린 결과(이 파일 그대로, 40,000 반복): `T_w` = 317.497 K \| `T_b` = 293.251 K \| ΔT = 24.2454 K \| `U_b` = 5.39407 m/s \| Re = 28,768 \| **Nu = 64.4894**(−0.06%) \| thermostat 출력 −3.20056 W 대 벽열 3.2 W(**+0.106% → +0.0174%**) \| `f` 측정값 0.017140 \| 힘 균형 −0.005%(그대로) \| `contErr` 2.90e-8 → 1.99e-8. 판정은 그대로다 — 절대 예측 −5.9%(Gnielinski at 원관 `f` = 0.0238816 → 68.5672), Dittus-Boelter −12.9%, 레이놀즈 유사 +34.3%(열림). 이 leg은 대조군이고 대조군답게 거의 움직이지 않았다 |
| `channelPeriodicFluxLowRe.jsonc` | `ofgpu-fire` | `channelPeriodicFluxWF.jsonc`의 해상 짝, 같은 방식으로 재구성 — 동일 `q_w`, 동일 thermostat(`target 293.15`, `tau 0.02`), 앞뒤 `empty`, 벽법선만 y+ ~ 1을 노리는 two-sided grading(`expansion: 200`, 50칸). `LaunderSharmaKE`/`lowRe`(SPEC-LIT §33)로 실행: 옛 덕트의 속도 붕괴는 완전히 사라짐. **SPEC-LIT §35의 thermostat 적용 후: 에너지 방정식의 표류가 완전히 사라짐** — T0 = 293.15 K와 T0 = 400 K에서 각각 40,000 반복을 돌리면 T_mean(293.574 K)·T_b(292.817 K)·U_b(4.84388 m/s)·thermostat 출력(−3.28977 W)까지 마지막 출력 자릿수까지 동일하게 수렴(§35.2가 요구하는 바로 그 회귀 검증). 열평형은 반올림 수준이 아니라 2.8%의 차이(−3.28977 W 대 3.2 W) — 이 메쉬 고유의 압력 솔버 허용오차 바닥(`contErr` 9.2e-8, `relTol`을 1e-4로 조이면 3,317 반복째에 NaN으로 발산)에서 비롯된 것으로 추적, 조정하지 않고 그대로 보고. 게이트: Re = 25,834, Nu = 73.40 — Dittus-Boelter +8.1%(±20-25% 밴드 안쪽), Gnielinski +16.3%(±10% 밴드 밖) — 두 상관식 모두를 요구하는 §32.4 기준으로는 **게이트가 완전히 닫히지 않음**(WF 짝은 완전히 닫힘), 두 메쉬 Nu 비율 1.125. 예전의 "+31%/+41%, 표류 계속"과는 질적으로 다른, 훨씬 작은 미스 — Launder-Sharma의 근벽 열전달 예측 자체를(§33.3이 검증한 것은 운동량 로그법칙뿐) 시사. **나중 보정(SPEC-LIT §32.4/§32.5)**: 위 +16.3%는 Petukhov의 매끄러운 **원관** 마찰계수 `f`로 평가한 **절대 예측** 판정이며, 그 판정에서 게이트는 여전히 열려 있다. 이 leg가 실제로 실현하는 `f` = 0.02653(원관 상관식보다 +8.2%)로 평가하면 Nu_Gn = 68.72, +6.8% — **레이놀즈 유사** 판정으로는 두 leg 모두 밴드 안쪽이다. 두 메쉬는 같은 체적력에서 서로 22.7% 다른 `f`를 실현하며, 그 두 `f`로 Gnielinski가 예측하는 Nu 비는 1.121로 측정값 1.125와 거의 일치한다 — 즉 1.125라는 두 메쉬 비율의 대부분은 열전달이 아니라 **운동량** 차이로 설명된다(재실행 전의 가설이지 측정 결과가 아님). 두 `f` 모두 체적력 균형에서 추론한 값이고, 벽면 전단력의 직접 측정은 `ofgpu-fire`에 구현·단위시험됐으나 두 케이스 중 어느 쪽도 아직 재실행하지 않았다. 파일 자체 헤더와 `docs/07-fire-solver.md` §1.1에 전체 진단이 있음 **재실행(SPEC-LIT §32.5.3/§35.3, 결정적 실험)**: 이 케이스도 이제 `"weighting": "massFlux"`를 명시한다. 같은 메쉬에서 토큰 하나만 바꿔 40,000 반복씩 두 번 돌렸다(`uniform` 쪽은 위 기록을 마지막 자릿수까지 재현). 결과: `T_w − T_b`가 21.2703 → 22.1503 K(+4.14%)로 **넓어지고** Nu는 73.4006 → 70.4707(**−3.99%**)로 **낮아졌다** — §35.3.2의 예측 방향과 정확히 일치하며, WF 쌍둥이는 같은 변경에 −1.41%만 움직여 오차가 해상 메쉬에 더 많이 실린다는 예측까지 확인됐다(두 메쉬 Nu 비 1.125 → 1.096). 그래도 게이트는 열려 있다: 절대 예측 판정 +16.3% → **+11.8%**(±10% 밴드 밖 — 단 이 leg의 에너지 불균형 3.26%가 Nu에 그대로 불확실성으로 실린다), Dittus-Boelter +3.9%(밴드 안). 직접 측정한 `f` = 0.023870은 추론값 0.02653보다 11% 낮고 원관 상관식보다도 2.7% 낮아 레이놀즈 유사 판정도 +15.2%로 **닫히지 않는다**. 또 이 leg은 힘 균형이 −3.9% 어긋나는데(WF 쌍둥이는 +0.000%), 가열을 끄고 같은 메쉬를 돌리면 −0.00%로 정확히 닫힌다 — 메쉬나 모델이 아니라 에너지 방정식과의 결합이 원인이라는 뜻이다 **재실행(SPEC-LIT §13.4.1/§32.5.5), 그리고 이 leg에서는 판정이 움직였다**: 위의 모든 숫자도 이 파일의 `numerics` 블록을 읽지 않는 드라이버가 낸 것이다. 고쳐서 다시 돌린 결과: `T_b` = 292.800 K \| `T_w` = 314.186 K \| ΔT = 21.3862 K \| `U_b` = 4.92909 m/s \| Re = 26,288 \| **Nu = 72.9988**(70.4707에서 +3.59%) \| 최악 벽인접 y+ 0.00185363, y+<20 셀 192/400 \| thermostat 출력 −3.29963 W 대 벽열 3.2 W(**+3.11%**) \| `f` 측정값 0.023936. 판정: **절대 예측은 더 밖으로 나갔다** — Gnielinski at 원관 `f` = 0.024416 → 63.9587, **+11.8% → +14.1%**이며 이 leg의 ±3.1% 에너지 불균형 불확실성(Nu ∈ [70.7, 75.3])으로도 밴드 가장자리에 닿지 않는다. Dittus-Boelter는 68.8722, +6.0%로 여전히 안쪽. 레이놀즈 유사도 +16.6%로 열림. **대신 −3.79%였던 힘 균형이 −0.000%로 완전히 닫혔다.** `div(phi,U)`만 4가지로 바꿔 돌린 고립 실험이 원인을 특정한다: 1차 → 2차 정확도는 Nu의 +0.07%밖에 되지 않고, `bounded` 토큰 하나가 Nu의 +3.5%와 힘 불균형 3.79% 전부를 설명한다(SPEC-LIT §3.1 — 저-마하에서 `div u`는 수렴 오차가 아니라 규정된 물리량이므로, 운동량 방정식의 bounded 보정은 실제 운동량을 지운다). 두 메쉬 Nu 비율 1.096 → **1.131** **재실행(SPEC-LIT §26.1 — 3년 묵은 에너지 불균형이 닫혔다)**: 위 블록이 남긴 유일한 미해결 항목인 **+3.11% 에너지 불균형**의 원인은 §25.1의 발산 구속조건이었다. `Q = q'''_c + div(k_eff grad T) − div(q_r)`에서 전도항이 빠져 있어서, 팽창이 0이어야 할 이 채널에 −0.07 s⁻¹의 가짜 팽창장이 규정되고 있었다. 넣고 다시 돌린 결과(이 파일 그대로, 40,000 반복): `T_b` = 292.773 K \| `T_w` = 314.549 K \| ΔT = 21.7767 K \| `U_b` = 4.93682 m/s \| Re = 26,330 \| **Nu = 71.683**(−1.80%) \| 최악 벽인접 y+ 0.00179449, y+<20 셀 192/400 \| **thermostat 출력 −3.20000 W 대 벽열 3.2 W, 차이 −2.84e-6 W(+3.11% → +0.000089%)** \| `f` 측정값 0.023832 \| 힘 균형 −0.000%(그대로) \| **`contErr` 1.101e-7 → 6.7253e-14**. 판정: 절대 예측 +14.1% → **+11.9%**(여전히 밴드 밖이지만 이제 불확실성이 ±0.0001%라 미스가 결정적이다), Dittus-Boelter +4.0%, 레이놀즈 유사 +14.9%. `PrtModel KaysCrawford`에서는 +6.4% → **+4.3%**로 더 깊이 안쪽에 들어간다. 부수적으로 두 가지가 폐기된다 — 이 메쉬의 `contErr` 바닥은 압력 솔버 허용오차가 아니라 그 가짜 팽창장이었고, `div(phi,U)`의 `bounded` 토큰도 이제 이 케이스에서 힘 균형을 +0.000%로 남긴다(예전 −3.787%). SPEC-LIT §3.1의 규칙 자체는 그대로다: 팽창이 실제로 0이 아닌 화재 플룸에서는 여전히 틀린 보정이다 |

```powershell
cd ..\rust
cargo run --release --bin ofgpu-fire -- ..\cases\burnerPlume.jsonc -combustion -radiation -endTime 6.0 -deltaT 0.005 -check 200

# SPEC-LIT §36 fvDOM 짝 케이스 — 위와 완전히 같은 셋업, 복사 모델만 다름.
# "radiated fraction of heat release" 줄을 위 P1 실행과 비교합니다.
cargo run --release --bin ofgpu-fire -- ..\cases\burnerPlume_fvDOM.jsonc -combustion -radiation -endTime 6.0 -deltaT 0.005 -check 200

# SPEC-LIT §29.3/§30.3의 게이트 (비주기, 첫 두 시도) — 정상상태까지 돌리고
# "integrated wall heat flux" 줄을 비교합니다.
cargo run --release --bin ofgpu-fire -- ..\cases\channelThermalWF.jsonc    -iters 6000 -check 2000

# SPEC-LIT §31의 페리오딕 재시도 — 두 케이스의 -heaterPower 값이 다른 이유는
# cases/retired/channelPeriodicLowRe.jsonc 헤더와 docs/07-fire-solver.md §1.1을 보십시오.
cargo run --release --bin ofgpu-fire -- ..\cases\channelPeriodicWF.jsonc    -iters 3000  -check 3000 -heaterPower -6

# SPEC-LIT §29.3/§31의 해상(lowRe) leg 두 개는 이 목록에서 빠졌습니다. 실행되지
#   않는 명령을 싣느니 빼는 편이 낫기 때문입니다 — 두 케이스는 `cases/retired/`로
#   옮겨졌고, 무엇이 대신하는지와 그 숫자가 어떤 조건에서 나왔는지는
#   `cases/retired/README.md`에 있습니다. 살아 있는 해상 leg는 아래
#   `channelPeriodicFluxLowRe.jsonc`입니다.

# SPEC-LIT §32의 재설계된 게이트 — 고정 열유속. -heaterPower는 SPEC-LIT §35.1의
# sources[] thermostat(target 293.15, tau 0.02, 두 케이스 동일)로 대체됐으므로
# CLI 플래그가 필요 없습니다.
cargo run --release --bin ofgpu-fire -- ..\cases\channelPeriodicFluxWF.jsonc    -iters 40000 -check 5000
cargo run --release --bin ofgpu-fire -- ..\cases\channelPeriodicFluxLowRe.jsonc -iters 40000 -check 5000
# SPEC-LIT §35: thermostat 덕에 에너지 방정식의 표류가 사라졌습니다 - T0 = 293.15 K와
# T0 = 400 K에서 각각 위 LowRe 명령을 돌리면(초기 T만 바꿔서) 이제 마지막 출력
# 자릿수까지 동일한 T_mean/T_b/U_b/thermostat 출력으로 수렴합니다(§35.2의 회귀
# 검증).
#
# 재실행(SPEC-LIT §32.5.3/§35.3): 두 케이스 모두 thermostat에
# "weighting": "massFlux" 를 명시하게 바뀌었으므로 위 두 명령은 이제
# 올바른 분포의 보상 소스로 동작합니다. 예전 기록을 그대로 재현하려면
# 그 토큰을 "uniform"으로만 바꾸면 되며, 그것이 이번 재판정에 쓴 통제
# 실험입니다(마지막 자릿수까지 예전 기록과 일치).
#
# 재실행(SPEC-LIT §13.4.1/§32.5.5): 위 두 명령이 내는 숫자도 바뀌었습니다. 예전
# 기록은 이 두 파일의 numerics 블록을 전혀 읽지 않는 드라이버가 낸 것이고, 지금은
# 두 케이스가 이름하는 Gauss linearUpwind grad(U)로 실제로 돕니다.
#
# 판정(자세한 것은 docs/07-fire-solver.md §1.1의 마지막 소절):
#   * WF leg    - 절대 예측 판정에서 여전히 닫힌다(Gnielinski -5.9%, Dittus-Boelter -12.9%)
#   * LowRe leg - Dittus-Boelter는 통과(+6.0%), Gnielinski는 +14.1%로 밴드 밖이며
#     예전(+11.8%)보다 더 벌어졌습니다 - 이 leg의 +-3.1% 에너지 불균형으로도
#     밴드에 닿지 않으므로 이제 판정이 확정적입니다(열림)
#   * f를 벽에서 직접 측정한 뒤로는 레이놀즈 유사 판정이 두 leg 모두
#     닫히지 않습니다(+34.4%, +16.6%) - 예전의 "+6.4%/+6.8%"는 추론된 f가
#     만든 허상이었습니다.
#   * 반면 힘 균형은 두 leg 모두 닫혔습니다(-0.005%, -0.000%) - LowRe leg의
#     -3.79% 불일치는 div(phi,U)의 `bounded` 토큰 하나가 전부였습니다(SPEC-LIT §3.1).
# ofgpu-fire는 이제 벽면 전단력에서 f를 직접 측정하고, 힘 균형을 이 크레이트가
# 실제로 조립하는 운동량 방정식과 같은 운동학적 단위(m4/s2)로 비교해
# 출력합니다(SPEC-LIT §32.5.2의 보정).
# docs/07-fire-solver.md §1.1에 전체 표와 진단이 있습니다.
```

자세한 화재 솔버 설명과 네 채널 케이스가 실제로 낸 벽 열유속·비율은
[`../docs/07-fire-solver.md`](../docs/07-fire-solver.md)를 보십시오.

## 직접 만든 OpenFOAM 케이스 쓰기

그대로 넣으면 됩니다. 제약은 하나뿐입니다 — **ASCII 형식이어야 합니다.**
바이너리로 쓰인 케이스는 먼저 변환하세요.

```bash
foamFormatConvert -constant -time 0
```

지원하는 경계조건: `fixedValue`, `zeroGradient`, `fixedGradient`, `mixed`,
`inletOutlet`, `calculated`, `empty`, `symmetry`/`symmetryPlane`/`slip`, `cyclic`,
`noSlip`, 그리고 벽함수 `nutkWallFunction`, `nutUWallFunction`,
`nutLowReWallFunction`, `nutkRoughWallFunction`/`nutURoughWallFunction`
(`Ks`/`Cs` 항목 필요, SPEC-LIT §15.3), `epsilonWallFunction`, `omegaWallFunction`,
`kqRWallFunction`, `kLowReWallFunction`, 그리고 `T`의 `thermalWallFunction`
(Jayatilleke 열 벽함수, SPEC-LIT §29.3 — OpenFOAM의
`compressible::alphatJayatillekeWallFunction`도 별칭으로 인식하며 무엇으로
해석했는지 출력). 모르는 타입은 `calculated`로 처리합니다.
