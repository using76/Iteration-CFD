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

| Case | 드라이버 | 설명 |
|---|---|---|
| `plume.jsonc` | `ofgpu-k-epsilon` 등 | `plumeB`(OpenFOAM 형식)의 JSONC 재현 — 두 형식이 같은 필드를 만든다는 B3 게이트 |
| `burnerPlume.jsonc` | `ofgpu-fire -combustion -radiation` | 프로판 버너 화재 데모 — SPEC-LIT §25(저-마하)·§26(에너지)·§27(연소)·§28(복사)를 한 케이스에서 결합. 바닥 창(`Y_F = 1` 고정)으로 연료가 들어가고, 열은 전부 연소의 `q'''_c`에서 나옵니다(입구 자체는 상온) |

```powershell
cd ..\rust
cargo run --release --bin ofgpu-fire -- ..\cases\burnerPlume.jsonc -combustion -radiation -endTime 6.0 -deltaT 0.005 -check 200
```

자세한 화재 솔버 설명은 [`../docs/07-fire-solver.md`](../docs/07-fire-solver.md)를
보십시오.

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
