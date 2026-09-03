# ofgpu — Rust + CUDA

meteor-cfd의 솔버 크레이트입니다. 호스트는 Rust, 커널은 CUDA C++이며, 시간 적분
루프 전체가 GPU에 상주합니다.

커널을 CUDA C++로 남긴 이유는 단순합니다. 커널 본문에서는 Rust가 얻는 게 거의
없습니다 — raw device 포인터 인덱싱은 어느 쪽이든 `unsafe`이고 경계 검사도
없습니다. 반면 그 **바깥**(메모리 소유권, 스트림·모듈 수명, 케이스 파서, 메쉬,
모델 오케스트레이션)이 실제로 메모리 버그가 사는 곳입니다.

영어 문서는 저장소 최상위 [`../README.en.md`](../README.en.md)를 보십시오.

## 이 코드가 어디에서 왔는가

수치 코어 전체가 **공개 문헌으로부터 직접 구현**되었습니다. 모든 이산화와 모델은
[`SPEC-LIT.md`](SPEC-LIT.md)에 원논문 인용과 함께 명세되어 있고, 파일별 출처는
[`PROVENANCE.md`](PROVENANCE.md)에 있습니다. 105개 소스 파일 전부가 저작권 헤더와
"No GPL-licensed source was consulted." 줄을 담고 있으며, 이는 산문이 아니라
시험으로 강제됩니다:

```bash
cargo test --release --lib provenance_audit
```

> **이 파일 자체가 오랫동안 틀린 말을 하고 있었습니다.** 이 README는 예전에 이
> 크레이트를 "`../gpu/`의 CUDA C++ 구현을 옮긴 것"이라고 소개하고, 성능과 필드를
> 그 C++ 판과 대조한 표를 실었습니다. 그 `gpu/` 트리는 OpenFOAM을 옮겨 적은
> GPL 파생물이었고 phase 0에서 저장소에서 **제거**되었으며, 그로부터 파생됐던
> 약 24,500줄의 Rust·CUDA 수치 코드도 함께 제거되고 문헌으로부터 다시
> 작성되었습니다 — 전말은 [`../LICENSING.md`](../LICENSING.md)의 "What phase 0
> actually removed"에 있습니다. 지금 이 트리에 그 계보를 가진 코드는 없습니다.
> 예전 표현과 C++ 대조표·성능 비교표는 사실과 달라졌으므로 철회하고 삭제했습니다.
> 남길 이유가 있는 기록이 아니라, 이미 존재하지 않는 산출물에 대한 주장이었기
> 때문입니다.

검증은 인위해법(MMS), 해석해, 공개 벤치마크만 사용하며 **다른 CFD 코드와
비교하지 않습니다**(SPEC-LIT §10/§22).

## 검증

```
cargo test --release   814 passed, 0 failed, 2 ignored (lib 크레이트)
                       905 passed, 0 failed, 4 ignored (모든 타깃 합계)
ofgpu-validate         314 / 314 checks passed
                       (279개는 실시간 계산, 35개는 기록된 측정값 재생)
```

재생(replay) 항목은 GPU를 수만 반복 돌려야 나오는 측정값이라 `cargo test` 안에서
다시 계산하지 않고, 기록된 숫자를 회귀 잠금으로만 씁니다. 요약 줄이 실시간 계산과
재생을 구분해 세는 이유가 그것입니다 — 재생은 증거가 아니라 회귀 방지입니다.
각 재생 항목은 어떤 케이스를 몇 반복 돌려 얻은 값인지, 어떤 토큰 하나를 바꾼
쌍인지 주석에 적혀 있습니다.

## 안전성

| | 측정값 | 다시 재는 법 |
|---|---|---|
| `unsafe {` 블록 | 254개 (전부 커널 런치 계열) | `grep -ro 'unsafe {' --include=*.rs src/ \| wc -l` |
| `unsafe fn` | 0개 | `grep -ro 'unsafe fn' --include=*.rs src/ \| wc -l` |
| `unwrap()`/`expect()` | 1367개 중 1352개가 `#[cfg(test)]` 안, 15개가 밖 | 아래 주 참고 |

`#[cfg(test)]` 밖의 15개는 대부분 `src/bin/validate.rs`(검증 전용 바이너리)에
있고, 라이브러리 쪽 소수는 바로 위에서 삽입을 마친 키를 다시 꺼내는 자리처럼
불변식이 지역적으로 증명되는 곳이며 `expect`에 그 불변식을 문장으로 적어
두었습니다. **예전 이 표는 "production 코드에 0개"라고 적고 있었고, 그것은 사실이
아니었습니다.**

타입 수준에서 강제되는 것들:

- `DevBuf<T>`는 drop 시 해제되고 이중 해제가 불가능하며, 할당한 컨텍스트보다 오래
  살 수 없습니다.
- `Graph`가 `!Send`/`!Sync`를 물려받아, "그래프 객체는 스레드 안전하지 않다"는
  CUDA 문서의 경고가 **컴파일 에러**가 됩니다.
- `types::tests::layout_matches_device`가 `Vec3`/`Tensor` 크기를 커널 쪽과 못 박아,
  한쪽만 바뀌면 컴파일 타임에 잡힙니다.

## 빌드

필요한 것: Rust stable (1.85+), CUDA Toolkit 13.x, Visual Studio 2022 C++ 워크로드.
**nightly는 필요 없습니다.**

```bash
cargo build --release
cargo test --release
```

`build.rs`가 나머지를 합니다 — CUDA 툴킷 탐색, `vswhere` → `vcvars64.bat`로 MSVC
환경 캡처(Developer Prompt를 열 필요가 없습니다), `cuda/*.cu`를 CUBIN으로 컴파일해
바이너리에 임베드.

| 환경변수 / feature | 뜻 |
|---|---|
| `OFGPU_CUDA_ARCH` | 대상 아키텍처, 기본 `120` (RTX 50xx) |
| `--features single` | f32로 전환. 커널까지 같이 바뀝니다 |
| `--features amgx` | AMGX 압력 backend. 기본 비활성 — `../README.md` 제한사항 참고 |

> **PTX가 아니라 CUBIN을 내보냅니다.** 드라이버는 자기가 아는 ISA 버전의 PTX만
> JIT합니다. 툴킷이 드라이버보다 새로우면 모듈 로드가
> `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`으로 실패합니다. CUBIN은 JIT을 건너뛰고 첫
> 실행 컴파일 지연도 없앱니다. 대가는 아키텍처 고정이라 `OFGPU_CUDA_ARCH`를 카드에
> 맞춰야 합니다.

## 실행

바이너리는 12개입니다(`Cargo.toml`의 `[[bin]]` 항목이 전부).

```bash
cargo run --release --bin ofgpu-generate-mesh -- channel ../cases/ch 200 120 1
cargo run --release --bin ofgpu-k-epsilon     -- ../cases/ch -iters 4000 -check 400
cargo run --release --bin ofgpu-validate
cargo run --release --bin ofgpu-bench         -- 2000 1000 1 -iters 30

# 저-마하 가변밀도 솔버 (SPEC-LIT §25/§26). JSONC 케이스를 읽습니다:
cargo run --release --bin ofgpu-lowmach -- ../cases/channelPeriodicFluxWF.jsonc -iters 40000 -check 5000

# 2상 유동 (SPEC-LIT §20). Martin & Moyce (1952)의 댐 브레이크:
cargo run --release --bin ofgpu-generate-mesh -- damBreak ../cases/dam
cargo run --release --bin ofgpu-vof           -- ../cases/dam -endTime 0.25 -surge
```

케이스 목록과 각 케이스가 무엇을 재는지는 [`../cases/README.md`](../cases/README.md)에
있습니다.

## 구조

```
rust/
├── build.rs              nvcc 탐색 · MSVC 환경 캡처 · CUBIN 임베드
├── SPEC-LIT.md           수치 명세 (구현은 전부 여기에서 나옵니다)
├── PROVENANCE.md         파일별 출처
├── cuda/                 device 전용 코드. host 코드 없음
│   ├── ofgpu_device.cuh      ofscalar / ofvec3 / oftensor
│   ├── ldu.cu field.cu fv.cu solver.cu precon.cu timescheme.cu
│   ├── momentum.cu simple.cu pressure.cu      운동량 · SIMPLE · 압력
│   ├── turbulence.cu sst.cu les.cu wallfunctions.cu
│   ├── energy.cu species.cu sources.cu
│   ├── s2s.cu                                 면대면 복사
│   ├── vof.cu                                 Zalesak FCT · 계면 압축 · CSF
│   └── probe.cu                               툴체인 수직 슬라이스
└── src/
    ├── types.rs device.rs error.rs            기반
    ├── mesh.rs mesh/{geometry,topology}.rs    HostMesh · GpuMesh · cell→face CSR
    ├── field.rs ldu.rs field_setup.rs         범용 mixed BC · fvScalarMatrix
    ├── fv.rs solver.rs precon.rs walldistance.rs
    ├── momentum.rs simple.rs scalar_transport.rs timescheme.rs
    ├── pressure/{mod,cartesian,fft,amgx}.rs   압력 방정식 backend 선택
    ├── energy.rs species.rs sources.rs
    ├── radiation.rs s2s.rs                    §13.4 선택기 · 면대면 복사
    ├── turbulence.rs les.rs
    ├── models/{registry,coupled,k_epsilon,k_omega,
    │           k_omega_sst,launder_sharma,les}.rs
    ├── io/{tokenizer,dict,polymesh,fields,writer,case}.rs   OpenFOAM ASCII
    ├── io/{case_json,contract,schemes,regex,output_types}.rs JSONC 케이스 · §13.4 계약
    ├── io/{msh,vtu,vdb,nvdb,usda}.rs        메쉬 입력·출력 형식
    ├── surface/{classify,cutcell,stl,obj}.rs  STL 컷셀
    ├── vof.rs                                 2상 VOF
    ├── blockgen.rs restart.rs potential_flow.rs
    ├── reference.rs                           독립 CPU 구현 (검증 전용)
    └── bin/                                   바이너리 16개
```

## 알려진 제약

[`../README.md`](../README.md)의 "제한사항"이 그대로 적용됩니다 — 단일 GPU 전용,
AMGX 기본 비활성, 저이완 방정식에서의 Crank–Nicolson, 압축성 미지원,
참여 매질 복사(P1 · fvDOM) 미지원 — 이름으로 거부됩니다.

Rust판 고유 사항 하나: 토크나이저가 `Tok::Num(f64)`로 숫자의 원본 표기를 버립니다.
`tolerance 1e-06`이 `"0.000001"`로, `FoamFile/version`이 `"2.0"` 대신 `"2"`로
되읽힙니다. 값은 정확히 왕복하고 `dimensions [0 1 -1 0 0 0 0]`은 바이트 단위로
같습니다. 대안은 토큰마다 `String`을 두는 것인데 `points` 리더가 감당할 수 없어
이쪽을 택했습니다.
