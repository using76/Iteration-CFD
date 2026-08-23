# ofgpu — Rust + CUDA

`../gpu/`의 CUDA C++ 구현을 **Rust 호스트 + CUDA C++ 커널**로 옮긴 것입니다.

커널을 CUDA C++로 남긴 이유는 단순합니다. 커널 본문에서는 Rust가 얻는 게 거의 없습니다 —
raw device 포인터 인덱싱은 어느 쪽이든 `unsafe`이고 경계 검사도 없습니다. 반면 그
**바깥**(메모리 소유권, 스트림·모듈 수명, OpenFOAM 파서, 메쉬, 모델 오케스트레이션)이
실제로 메모리 버그가 사는 곳이고, 그게 전체의 87 %입니다.

## 검증

```
cargo test  ->  187 passed; 0 failed
ofgpu-validate  ->  58/58 checks passed
```

`ofgpu-validate`는 C++판과 같은 검사를 합니다 — 메쉬 항등식, 명시적 연산자를 해석해와
독립 CPU 구현 양쪽에 대조, 행렬 조립·완화·경계 반영·Amul·PBiCGStab 대조, CSR export,
벽함수, 그리고 인위해 수렴 차수. 3-D 격자와 2-D(empty 패치) 격자 양쪽에서 돌립니다.

관측 수렴 차수 **2.00**, L2 오차 n=12에서 2.026e-03, n=24에서 5.053e-04 — C++판과
같은 값입니다.

### C++ 구현과의 대조

같은 케이스에 두 바이너리를 돌려 필드를 비교했습니다.

| 비교 | k | epsilon / omega | nut |
|---|---|---|---|
| **결정론적** (`-iters 20 -fixedIters 3`) | 24000/24000 동일 | 24000/24000 동일 | 24000/24000 동일 |
| 적응형 (수렴까지) | 174셀이 마지막 자리 차이 | 427셀 | 409셀 |

결정론적 모드에서 **한 셀도 다르지 않습니다** (출력 정밀도 6자리 기준). 조립, 연산자,
벽함수, Krylov 반복이 전부 일치한다는 뜻입니다.

적응형 모드의 미세한 차이는 이식 오류가 아니라 **리덕션 순서** 때문입니다. C++은 CUB를
쓰고 Rust판은 직접 짠 2단계 리덕션(warp shuffle + shared memory)을 씁니다 — CUB는
host 템플릿 라이브러리라 device 전용 `.cu`에 넣을 수 없기 때문입니다. 부동소수점 덧셈은
결합법칙이 성립하지 않으므로 잔차 노름이 ~1e-9 수준에서 갈리고, 그 결과 어느 outer
iteration에서 솔버가 몇 번째 스윕에 멈출지가 달라집니다. 차이는 전부 기록된 마지막
자리 한 칸입니다.

메쉬 생성기는 더 강합니다 — `points`/`faces`/`owner`/`neighbour`/`boundary`가
**바이트 단위로 동일**합니다.

## 성능

C++판과 오차 범위 내에서 같습니다.

| Mesh | C++ | Rust |
|---|---|---|
| 500 k, k-epsilon | 3.335 ms/iter | 3.344 ms/iter |
| 500 k, k-omega | 3.323 ms/iter | 3.330 ms/iter |
| 2 M, k-epsilon | 13.334 ms/iter | 13.291 ms/iter |
| 2 M, k-omega | 13.642 ms/iter | 13.643 ms/iter |
| 2 M, 상주 메모리 | 4327 MiB | 4273 MiB |

언어 전환 자체는 속도에 영향이 없습니다. 예정된 이득은 CUDA Graph와 커널 융합에서 나옵니다.

## 안전성

| | |
|---|---|
| production 코드의 `unwrap()`/`expect()` | **0** (147개 전부 `#[cfg(test)]` 안) |
| `unsafe` 블록 | 91개, **전부 커널 런치** |

타입 수준에서 강제되는 것들:

- `DevBuf<T>`는 drop 시 해제되고 이중 해제가 불가능하며, 할당한 컨텍스트보다 오래 살 수 없습니다.
- `Graph`가 `!Send`/`!Sync`를 물려받아, "그래프 객체는 스레드 안전하지 않다"는 CUDA 문서의 경고가 **컴파일 에러**가 됩니다. C++에서는 주석으로만 존재하던 규칙입니다.
- `types::tests::layout_matches_device`가 `Vec3`/`Tensor` 크기를 커널 쪽과 못 박아, 한쪽만 바뀌면 컴파일 타임에 잡힙니다.

C++판의 결함 하나도 고쳤습니다 — `setValues`가 파일 지역 공유 스크래치를 써서 두 행렬을
동시에 제약하면 충돌했는데, 이제 스크래치가 행렬에 소속됩니다.

## 빌드

필요한 것: Rust stable (1.85+), CUDA Toolkit 13.x, Visual Studio 2022 C++ 워크로드.
**nightly는 필요 없습니다.**

```bash
cargo build --release
cargo test --release
```

`build.rs`가 나머지를 합니다 — CUDA 툴킷 탐색, `vswhere` → `vcvars64.bat`로 MSVC 환경
캡처(Developer Prompt를 열 필요가 없습니다), `cuda/*.cu`를 CUBIN으로 컴파일해 바이너리에 임베드.

| 환경변수 / feature | 뜻 |
|---|---|
| `OFGPU_CUDA_ARCH` | 대상 아키텍처, 기본 `120` (RTX 50xx) |
| `--features single` | f32로 전환. 커널까지 같이 바뀝니다 |

> **PTX가 아니라 CUBIN을 내보냅니다.** 드라이버는 자기가 아는 ISA 버전의 PTX만
> JIT합니다. 툴킷이 드라이버보다 새로우면 — nvcc 13.3에 드라이버가 CUDA 13.2를 보고하는
> 이 머신이 정확히 그렇습니다 — 모듈 로드가 `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`으로
> 실패합니다. CUBIN은 JIT을 건너뛰고, 첫 실행 컴파일 지연도 없앱니다. 대가는 아키텍처
> 고정이라 `OFGPU_CUDA_ARCH`를 카드에 맞춰야 합니다.

## 실행

```bash
cargo run --release --bin ofgpu-generate-mesh -- channel ../cases/ch 200 120 1
cargo run --release --bin ofgpu-k-epsilon    -- ../cases/ch -iters 4000 -check 400
cargo run --release --bin ofgpu-k-omega      -- ../cases/chKW -iters 4000 -check 400
cargo run --release --bin ofgpu-validate
cargo run --release --bin ofgpu-bench        -- 2000 1000 1 -iters 30

# 2상 유동 (SPEC-LIT 20). Martin & Moyce (1952)의 댐 브레이크:
cargo run --release --bin ofgpu-generate-mesh -- damBreak ../cases/dam
cargo run --release --bin ofgpu-vof           -- ../cases/dam -endTime 0.25 -surge
```

플래그는 C++판과 같습니다 (`-iters`, `-fixedIters`, `-check`, `-write`, `-noWrite`,
k-omega의 `-blended`), 나란히 돌려 비교할 수 있게 하기 위해서입니다.

## 구조

```
rust/
├── build.rs              nvcc 탐색 · MSVC 환경 캡처 · CUBIN 임베드
├── PORT.md               모듈 계약 (토크나이저 · 딕셔너리 · 필드 파일 API)
├── cuda/                 device 전용 코드. host 코드 없음
│   ├── ofgpu_device.cuh      ofscalar / ofvec3 / oftensor
│   ├── ldu.cu  field.cu  fv.cu  solver.cu  wallfunctions.cu  turbulence.cu
│   ├── vof.cu               Zalesak FCT · 계면 압축 · 곡률 · 중력/표면장력 face flux
│   └── probe.cu              툴체인 수직 슬라이스
└── src/
    ├── types.rs  device.rs  error.rs        기반
    ├── mesh.rs  mesh/{geometry,topology}.rs  HostMesh · GpuMesh · cell→face CSR
    ├── field.rs  ldu.rs                      범용 mixed BC · fvScalarMatrix
    ├── {ldu_ops,field_ops,fv,solver,         커널 런처
    │    wallfunctions,turbulence}.rs
    ├── io/{tokenizer,dict,polymesh,          OpenFOAM ASCII 파서/작성기
    │       fields,case}.rs
    ├── field_setup.rs                        BC 타입 문자열 → (fr, refValue, refGrad)
    ├── blockgen.rs                           격자 생성기
    ├── reference.rs                          독립 CPU 구현 (검증 전용)
    ├── models/{k_epsilon,k_omega}.rs
    ├── vof.rs                                2상 VOF: alpha 방정식(FCT) · 혼합 물성 ·
    │                                         CSF 표면장력 · p_rgh 압력 방정식
    └── bin/                                  실행 파일 5개 + probe
```

## 알려진 제약

`../README.md`의 "지금 하지 않는 것"이 그대로 적용됩니다 — 운동량·압력 방정식 미이식,
`limitedLinear`/`linearUpwind`가 upwind로 대체, DIC/DILU가 Jacobi로 대체, cyclic 패치의
비직교 보정 벡터 없음, 단일 GPU 전용, AMGX/cuDSS 미연결(`CsrPattern`이 LDU→CSR 순열을
미리 만들어 두어 연결은 준비돼 있습니다).

Rust판 고유 사항 하나: 토크나이저가 `Tok::Num(f64)`로 숫자의 원본 표기를 버립니다.
`tolerance 1e-06`이 `"0.000001"`로, `FoamFile/version`이 `"2.0"` 대신 `"2"`로 되읽힙니다.
값은 정확히 왕복하고 `dimensions [0 1 -1 0 0 0 0]`은 바이트 단위로 같습니다. 대안은
토큰마다 `String`을 두는 것인데 `points` 리더가 감당할 수 없어 이쪽을 택했습니다.

## 다음

CUDA Graph 배선은 이미 있습니다 — `Gpu::capture()`가 스트림 캡처를 감싸고 `Graph`가
`upload()`/`launch()`를 제공합니다. `-fixedIters`와 `report_residuals = false`가
캡처 구간에서 host 왕복을 없애 주므로, 구조적으로는 준비된 상태입니다.
