# Test cases

`ofgpu-generate-mesh` writes a complete, ready-to-run case: `constant/polyMesh`,
`constant/physicalProperties`, `constant/momentumTransport`, `system/{controlDict,
fvSchemes,fvSolution}` and a `0/` directory with `U`, `k`, `epsilon`, `omega` and `nut`.

```powershell
cargo run --release --bin ofgpu-generate-mesh -- <case> <outputDir> [nx ny nz]
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

## 직접 만든 OpenFOAM 케이스 쓰기

그대로 넣으면 됩니다. 제약은 하나뿐입니다 — **ASCII 형식이어야 합니다.**
바이너리로 쓰인 케이스는 먼저 변환하세요.

```bash
foamFormatConvert -constant -time 0
```

지원하는 경계조건: `fixedValue`, `zeroGradient`, `fixedGradient`, `mixed`,
`inletOutlet`, `calculated`, `empty`, `symmetry`/`symmetryPlane`/`slip`, `cyclic`,
`noSlip`, 그리고 벽함수 `nutkWallFunction`, `nutUWallFunction`,
`nutLowReWallFunction`, `epsilonWallFunction`, `omegaWallFunction`,
`kqRWallFunction`, `kLowReWallFunction`. 모르는 타입은 `calculated`로 처리합니다.
