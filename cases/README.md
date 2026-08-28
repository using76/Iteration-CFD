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
| `channel` | 200 x 120 x 1 | 평면 채널, 양 벽으로 expansion 20 grading | 1/7 멱법칙 프로파일 `U = (Uref (1-|y|)^(1/7), 0, 0)` |
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

`ofgpu-buoyant`, `ofgpu-fire` 등 부력/화재 계열 드라이버는 OpenFOAM 케이스
디렉터리 대신 주석과 trailing comma를 허용하는 JSON 파일 하나로도 케이스를
읽습니다 (`docs/05-io-redesign.md` §4.1; 스키마는 `docs/schema/case-1.json`,
`ofgpu::io::case_json::emit_schema`로 자동 생성). 디렉터리 대신 `.jsonc`/`.json`
파일 경로를 넘기면 됩니다 — 출력은 `<stem>_jsonc/`에 씁니다.

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
`grading` 줄을, 실제로 켜서 쓰는 예는 아래 `channelThermalLowRe.jsonc`(벽법선
방향 y+ ≈ 1을 노리는 two-sided grading)를 보십시오.

`mesh.cyclic` — SPEC-LIT §31.1의 cyclic 패치 쌍. `mesh.boundaries`의 여섯
이름 중 마주보는 두 개(`xmin`/`xmax`, `ymin`/`ymax`, `zmin`/`zmax` 중 하나)를
`a`/`b`로 지정하면 그 두 면이 경계가 아니라 결합된 한 쌍이 됩니다.
`transform`은 `"translate"`만 지원 — 축의 길이만큼 평행이동한다는 뜻이며,
회전 쌍(`"rotate"`)은 면 매칭·벡터 변환이 아직 없어 `translate`를 이름으로
밝힌 SPEC-LIT §13.4 오류입니다. `BlockSpec`에 cyclic 축 슬롯이 하나뿐이라
`mesh.cyclic` 배열도 항목이 정확히 하나여야 하고(`-permissive`는 첫 항목만
쓰고 계속), cyclic으로 지정된 패치는 `patches[]`의 구체적 규칙으로 또 지정할
수 없습니다(양쪽 이름을 밝힌 오류) — 다만 만능 규칙(`".*"`)은 상관없고,
오히려 `resolve_patch_rule`이 모든 패치 이름에 대해 규칙 하나를 요구하므로
cyclic 패치 두 개를 받아줄 만능 규칙이 case 끝에 있어야 합니다(값 자체는
cyclic 패치에는 적용되지 않습니다 — 모든 필드가 자동으로 `cyclic` 타입이
됩니다):

```jsonc
"mesh": {
  ...,
  "cyclic": [ { "a": "streamwiseMin", "b": "streamwiseMax", "transform": "translate" } ],
},
```

`blockgen`의 `-cyclic x|y|z` 플래그(OpenFOAM 형식 케이스)와 같은 짝짓기·같은
두 불변식(전단사, `Sf_a == -Sf_b`)을 씁니다 — 자세한 내용은 SPEC-LIT §31.1과
`../rust/README.en.md`의 "Cyclic patches" 행을 보십시오. 실제로 켜서 쓰는
예는 아래 `channelPeriodicWF.jsonc`/`channelPeriodicLowRe.jsonc`입니다.

`sources` — SPEC-LIT §18/§31.1의 체적 소스, JSONC 쪽 입구입니다. 오늘은
`momentumSource` 한 종류만 지원합니다: 전체 도메인에 걸친 균일 체적력(단위
질량당, m/s²) — periodic 케이스처럼 유입 경계가 없어 질량유량을 지정할 수
없는 케이스가 유동을 몰아가는 유일한 방법입니다. `field`는 반드시 `"U"`여야
하고(체적력은 벡터라 운동량 방정식에만 뜻이 있음), 다른 값은 이름을 밝힌
오류입니다. OpenFOAM 형식 케이스의 `constant/fvSources`(box/sphere 선택,
여섯 종류의 항)와 같은 `crate::sources::SourceSpec` 레지스트리로 내려가지만,
이 배열은 그 전체 표면을 복제하지 않고 페리오딕 케이스에 필요한 이 한 가지만
추가합니다:

```jsonc
"sources": [
  { "type": "momentumSource", "field": "U", "bodyForce": [3.9, 0, 0] },
],
```

| Case | 드라이버 | 설명 |
|---|---|---|
| `plume.jsonc` | `ofgpu-k-epsilon` 등 | `plumeB`(OpenFOAM 형식)의 JSONC 재현 — 두 형식이 같은 필드를 만든다는 B3 게이트 |
| `burnerPlume.jsonc` | `ofgpu-fire -combustion -radiation` | 프로판 버너 화재 데모 — SPEC-LIT §25(저-마하)·§26(에너지)·§27(연소)·§28(복사)를 한 케이스에서 결합. 바닥 창(`Y_F = 1` 고정)으로 연료가 들어가고, 열은 전부 연소의 `q'''_c`에서 나옵니다(입구 자체는 상온) |
| `channelThermalWF.jsonc` | `ofgpu-fire` | SPEC-LIT §29.3/§30.3의 게이트, 메쉬 (a): 2.0 m(L/Dh = 50) 입구구동 평면 채널/덕트, 위아래 벽 `Tw = 373.15 K`, 중력 0. 벽법선 방향 균일 격자(6칸) — 측정 y+ 21.95/37.90/40.29(목표 30-60), `standard` 프리셋, `thermalWallFunction`이 실제로 일을 해야 하는 조건 |
| `channelThermalLowRe.jsonc` | `ofgpu-fire` | 위와 동일한 형상·유입 조건·벽온도, `mesh.grading`으로 벽법선 방향을 양쪽 벽 모두를 향해 two-sided grading(`expansion: 200`, 50칸, 첫 셀 높이 ≈ 2×10⁻⁵ m)하여 y+ 목표 ≈ 1(측정 0.43/0.89/1.02)을 노리는 `lowRe` 프리셋 — 벽은 순수 분자 저항의 평범한 `fixedValue`(SPEC-LIT §29.3: "lowRe는 해상된 서브레이어 자체의 분자 저항을 그대로 둔다"). 두 케이스의 벽 열유입 비율은 0.381(첫 시도의 0.095에서 4배 개선, 여전히 게이트는 열려 있음) — 자세한 내용은 `docs/07-fire-solver.md` §1.1 |
| `channelPeriodicWF.jsonc` | `ofgpu-fire` | SPEC-LIT §31의 페리오딕 재시도, 메쉬 (a): 위 두 케이스와 같은 단면(0.04 m × 0.04 m)·`Tw`를 스트림방향 **cyclic**(`mesh.cyclic`, 0.08 m, 8칸)으로 바꾸고, 유입 대신 `sources[]` `momentumSource`(체적력 3.9 m/s²)로, 에너지는 `-heaterPower -6`(균일 도메인 열싱크)로 구동 — 벽법선 균일 6칸, `standard`/`thermalWallFunction`. 측정 y+ 40.25/41.73/43.41, 완전 수렴(`\|U\|` 잔차 1.3e-10) |
| `channelPeriodicLowRe.jsonc` | `ofgpu-fire` | 위와 같은 페리오딕 덕트, 벽법선만 `channelThermalLowRe.jsonc`와 같은 two-sided grading(`expansion: 200`, 50칸) — `lowRe`/`fixedValue`. 같은 체적력, 그러나 **다른** 열싱크(`-heaterPower -60`) — 해상된 y+~0.3 서브레이어가 같은 싱크로는 `Tw` 근처까지 데워질 만큼 전도가 좋아서(§1.1의 상세 설명 참고), `-900`까지 올려 두 메쉬를 같은 ΔT로 맞춰 보려 했으나 도메인 코어가 160 K까지 식는 비물리적 결과가 나와 포기 — 측정 y+ 0.302/0.310/0.318, `\|U\|` 잔차는 ~1e-5에서 정체(양은 안정) |
| `channelPeriodicFluxWF.jsonc` | `ofgpu-fire` | SPEC-LIT §32의 재설계된 게이트, 메쉬 (a): 위 페리오딕 덕트와 같은 형상·체적력이지만, 고정 벽온도 대신 고정 열유속(`fixedFluxTemperature`, `q = 500 W/m2`, 양쪽 가열벽)으로 바꾸고 `-heaterPower -3.2`(닫힌 형태 `-2 q_w A_wall`, 계산으로 나오는 값이지 맞춘 값이 아님)로 균형을 맞춤 — 이러면 두 메쉬가 각자의 ΔT를 스스로 예측하게 되어 Nu를 Dittus-Boelter/Gnielinski와 비교할 수 있음(§32.2). `standard`/`thermalWallFunction`, 측정 y+ 40.3/41.7/43.4, Nu = 50.41 — Gnielinski +2%, Dittus-Boelter −4%, 둘 다 안쪽(**게이트 닫힘**) |
| `channelPeriodicFluxLowRe.jsonc` | `ofgpu-fire` | `channelPeriodicFluxWF.jsonc`의 해상 짝 — 동일 `q_w`, 동일 `-heaterPower -3.2`, 벽법선만 y+ ~ 1을 노리는 two-sided grading(`expansion: 200`, 50칸). `LaunderSharmaKE`/`lowRe`(SPEC-LIT §33)로 실행: 벽 근처 폭주(옛 진단, k가 336 m²/s²까지 발산)는 사라졌지만, 이 특정 3차원 덕트에서는 여전히 수렴하지 않음 — 벌크 속도가 0.24 m/s로 무너지고(짝 케이스는 3.51 m/s, 동일 메쉬의 층류 해는 14.8 m/s), 온도가 정상상태에 이르지 못함. 모형 자체의 벽법칙(u+ = y+, 로그 법칙)은 별도의 2차원 채널에서 1% 이내로 검증됨 — 파일 자체 헤더와 `docs/07-fire-solver.md` §1.1에 전체 진단이 있음 (**게이트 열려 있음, 새로운 이유**) |

```powershell
cd ..\rust
cargo run --release --bin ofgpu-fire -- ..\cases\burnerPlume.jsonc -combustion -radiation -endTime 6.0 -deltaT 0.005 -check 200

# SPEC-LIT §29.3/§30.3의 게이트 (비주기, 첫 두 시도) — 정상상태까지 돌리고
# "integrated wall heat flux" 줄을 비교합니다.
cargo run --release --bin ofgpu-fire -- ..\cases\channelThermalWF.jsonc    -iters 6000 -check 2000
cargo run --release --bin ofgpu-fire -- ..\cases\channelThermalLowRe.jsonc -iters 6000 -check 1000

# SPEC-LIT §31의 페리오딕 재시도 — 두 케이스의 -heaterPower 값이 다른 이유는
# cases/channelPeriodicLowRe.jsonc 헤더와 docs/07-fire-solver.md §1.1을 보십시오.
cargo run --release --bin ofgpu-fire -- ..\cases\channelPeriodicWF.jsonc    -iters 3000  -check 3000 -heaterPower -6
cargo run --release --bin ofgpu-fire -- ..\cases\channelPeriodicLowRe.jsonc -iters 40000 -check 5000 -heaterPower -60

# SPEC-LIT §32의 재설계된 게이트 — 고정 열유속, 두 케이스 모두 같은 -heaterPower.
cargo run --release --bin ofgpu-fire -- ..\cases\channelPeriodicFluxWF.jsonc    -iters 3000  -check 1000 -heaterPower -3.2
cargo run --release --bin ofgpu-fire -- ..\cases\channelPeriodicFluxLowRe.jsonc -iters 20000 -check 2000 -heaterPower -3.2
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
