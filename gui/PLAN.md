# Iteration CFD Studio — 구현 계획

AI 채팅으로 meteor-cfd(Iteration CFD) 솔버를 구동하는 Cursor/Codex 스타일 GUI. 새 폴더 `gui/`에만 코드를 두며, 기존 Rust/CUDA 트리는 건드리지 않는다.

이 문서는 설계 패널(3개 관점 설계안 → 관점별 적대적 심사 2회 → 통합 누락 점검)을 거쳐 확정한 결정 기록이다. 심사에서 나온 "반드시 고칠 것"은 §12에 모았다.

---

## 1. 목표

| 요구 | 구현 방향 |
|---|---|
| Cursor/Codex 같은 채팅 중심 개발 환경 | VS Code형 셸(활동바·탐색기·탭·하단 패널·상태바) + 우측 AI 어시스턴트 패널. 첨부 목업을 그대로 따른다. |
| 기존 프로그램의 각 모델·요소를 **툴 콜링**으로 사용 | 16개 `ofgpu-*` 바이너리와 난류 모델·벽처리·알고리즘·출력 형식을 **레지스트리**로 정형화하고, Claude 도구(tool)로 노출. 케이스 JSONC 편집·검증, 메쉬 생성, 솔버 실행·모니터링·중단, 결과 탐색, 뷰어 조작, 잔차 플롯, 저장소 검색, SPEC-LIT 조회, 사용자 정의 도구 등록. |
| 결과를 볼 수 있는 가벼운 **사실적 3D 렌더러** | three.js r185 `WebGPURenderer` + TSL(WebGPU 우선, WebGL2 자동 폴백). PBR 재질 + 환경광(PMREM) + ACES 톤매핑 + MSAA + 소프트 섀도. 컬러맵 표면, 절단면, 등가면, 유선, 글리프, 범례, 스크린샷. |
| 계획 후 진행 | 이 문서 → 공유 계약 → 병렬 구현 → 데모 모드 E2E 검증 → 문서화. |

## 2. 기존 프로그램 분석 (요약)

- **빌드/실행 환경**: Rust 1.85 + CUDA 13, 단일 NVIDIA GPU, Windows 개발기. 이 샌드박스에는 GPU/nvcc가 없으므로 **데모 모드**(실제 바이너리 대신 동일한 출력 형식을 내는 모의 솔버 + 대본형 모의 LLM)가 필수다. 실제 바이너리가 있으면 그것을 우선 호출한다.
- **바이너리**(`rust/Cargo.toml`): generate-mesh, k-epsilon, k-omega, sa, plume, buoyant, vof, lowmach, cht, datacentre, decompose, validate, bench, graph-bench, dispatch-bench, probe.
- **케이스 입력**: OpenFOAM ASCII 디렉터리(`constant/polyMesh`, `0/`, `system/`) 또는 JSONC 한 파일(`docs/schema/case-1.json`, schemars 생성). JSONC를 직접 읽는 드라이버는 **k-epsilon, lowmach, decompose, cht, datacentre**뿐이며 k-omega·sa·plume·buoyant·vof는 OpenFOAM 디렉터리만 받는다.
- **출력**: JSONC 케이스는 `<stem>_jsonc/` 아래. `foam`(기본) → `<root>/<time>/{U,p,k,epsilon,...}` ASCII, `vtu` → `<root>/VTK/<tag>_NNNNNN.vtu`(+`.pvd`, 부가 원시 이진, LittleEndian, UInt64 헤더, `VTK_POLYHEDRON`=42, 면은 면중심 기준 사각형 **근사**), `vdb/nvdb` → `<root>/VDB/`, `usda` 장면. JSONC 실행은 디스크에 polyMesh를 쓰지 않는다 → 기하는 케이스의 `mesh` 블록(bounds·cells·grading·regions)에서 재구성한다.
- **셀 순서**: `cell(i,j,k) = i + nx*(j + ny*k)`. 축 그레이딩은 `blockgen::graded_nodes`/`fill_graded` 공식 그대로 이식.
- **표준출력 형식**(잔차 파서·모의 솔버의 근거, `scratchpad/logformats.md`에 원문 정리):
  - k-epsilon/k-omega/sa/plume(정상): `{it:>7}  epsilon|omega|nuTilda res X (n)  k res X (n)  [T res X (n)]  max dk/k X`
  - lowmach: `iter {n:6}  |U| res X  |p| res Y  contErr Z  T [a, b] K  rho [a, b] kg/m3  p0 P Pa  dp0/dt D Pa/s`
  - buoyant: `iteration N   wall W s` 또는 `t = T s   step N   wall W s` + `    res  Ux X (n)  Uy ...  p X (n)` + `         k X (n)  epsilon X (n)  T X (n)   T[min,max] a b K   max |sum_f phi| X m3/s`
  - vof: `step {n:>5}  t = T  dt D  alphaCo C  xN sub  p_rgh A -> B in N iters  continuity C  alpha [a, b]`
  - datacentre: `  iter {n:5}  fan1: Q = 0.1234 m^3/s, dp = 12.3 Pa  |  ...`(잔차가 아닌 팬 운전점)
  - 수렴: `converged: ...` 또는 (plume) 맨몸 `converged`; 기록: `written to <dir>`(선행 공백 가능, 여러 번); 오류: stderr `error: <msg>`, `ofgpu-<name>: <msg>`, `benchmark aborted: <msg>`; 발산: `*** NaN/Inf ***`.

## 3. 아키텍처

```
 브라우저 (Vite + React 19 + zustand)                 Node 22 서버 (@cfd/server, node:http + ws)
 ┌──────────────────────────────────────┐            ┌──────────────────────────────────────────┐
 │ 셸: 활동바·탐색기·탭·하단패널·상태바   │  WS /ws    │ agent/   Claude 스트리밍 수동 루프 + 승인   │
 │ Monaco(case.jsonc 스키마) · xterm     │◄──────────►│ tools/   레지스트리 기반 도구 20여 개       │
 │ uPlot 잔차 · AI 패널(도구 카드)        │  REST /api │ runs/    스폰·로그 링버퍼·잔차 파서·큐·킬  │
 │ viewer/ three.js WebGPU + Worker      │◄──────────►│ datasets/ foam/polyMesh/VTU 리더 → 매니페스트│
 └──────────────────────────────────────┘  blob(f32) │ mock/    모의 솔버(CLI) · 모의 LLM(대본)     │
                                                     └───────────┬──────────────────────────────┘
                                                                 │ spawn (argv 허용목록, 셸 없음)
                                            OFGPU_BIN_DIR/ofgpu-* ─┴─ rust/target/release ─ cargo run ─ mock
```

- 공유 계약은 `gui/shared/src/*`(protocol, viewerCommands, viewerDataset, residuals, registry, tools)에 단일 소스로 두고 서버·웹이 함께 사용한다. WS 양방향 모두 zod로 검증.
- 대용량 필드 데이터는 WS가 아니라 HTTP로 `Float32Array` 원시 바이트(LE)로 전달하고, WS는 알림만 나른다.
- LLM 호출은 `@anthropic-ai/sdk` 0.124: `client.beta.messages.stream(...)`, 모델 `claude-opus-5`, `thinking:{type:'adaptive', display:'summarized'}`, `output_config:{effort}`, `betas:['server-side-fallback-2026-07-01']`, `fallbacks:'default'`, `max_tokens: 64000`. 프리필 없음. 병렬 tool_use는 **한 user 메시지**에 모든 tool_result를 담아 반환.

### 스택 결정과 이유
| 결정 | 이유 |
|---|---|
| Node/TypeScript 서버 (Rust 서버 아님) | Anthropic SDK·스트리밍·zod가 TS에서 가장 성숙. 솔버는 별도 프로세스로 호출하므로 CUDA 의존 없음. 프런트와 타입 공유. |
| three.js `three/webgpu` + TSL | 같은 노드 그래프가 WGSL/GLSL 둘 다로 컴파일되어 폴백이 진짜로 자동. PBR·PMREM·MSAA·섀도·fat line이 내장. 순수 WebGPU 직접 구현은 "가볍지 않다". |
| react-three-fiber 미사용 | r185 WebGPU 경로에서 r3f 지원이 뒤처짐. 렌더러를 명령형으로 직접 제어. |
| Vite 7 / vitest 3 / Playwright 1.56.1 | 설치된 브라우저(chromium-1194)와 일치. |
| node:http + ws (Fastify 없음) | 의존성 최소. 라우터 30줄이면 충분. |
| 데스크톱 셸은 후순위(Tauri 2 권장) | 브라우저로 먼저 완성. 사이드카는 Node SEA/bun compile이 필요하므로 별도 단계. |

## 4. 폴더 구조

```
gui/
  PLAN.md  README.md  package.json (workspaces)  tsconfig.base.json  e2e/
  shared/src/     protocol.ts  viewerCommands.ts  viewerDataset.ts  residuals.ts  registry.ts  tools.ts  index.ts
  server/src/     main.ts  config.ts  http/(router, api routes)  ws/(hub)  workspace/(경로 안전, fs, search, git)
                  runs/(manager, runner, dispatch, kill, ring, registry)  gpu/  formats/(foam, polymesh, vtu, pvd, cartesian)
                  datasets/(service, manifest, blobs, worker)  agent/(loop, llm, mockLlm, prompt, session, policy, quick)
                  tools/(각 도구)  mock/(mock-cli, fields, scripts)  __tests__/
  web/src/        main.tsx  app/  state/  ws/  components/(shell, explorer, tabs, bottom, assistant, statusbar)
                  editor/  chart/  terminal/  i18n/  viewer/(engine, data, gpu, layers, worker, api, ui)  styles/
  runs/  sessions/  .cache/   (gitignore)
```

## 5. 공유 계약 (먼저 쓴다)

- `protocol.ts` — WS 메시지 유니온. 서버 → 클라: `session.state`(모드·GPU·워크스페이스·대기 승인 포함), `turn.start/done/error/refusal/warning`, `msg.block_start`, `msg.delta{blockIndex}`, `tool.start/input_delta/input/approval_request/result`, `run.started/log{seq}/residual{seq}/metric/written/exit`, `viewer.command{requestId}`, `dataset.progress`, `fs.changed`, `problems`, `gpu`, `pong`. 클라 → 서버: `session.open/list/new/delete`, `user.message{context}`, `turn.cancel`, `tool.approve/deny`, `viewer.result`, `run.subscribe{fromSeq}`, `run.stop`, `quick`, `settings.set`, `ping`.
- `viewerCommands.ts` — zod 판별 유니온(`load/setField/setRepresentation/addSlice/addPlane/addIsoSurface/addStreamlines/addGlyphs/remove/clear/setClipBox/setTime/setCamera/setQuality/screenshot/getState`). 도구 입력·WS·스토어가 모두 이 스키마를 쓴다.
- `viewerDataset.ts` — 매니페스트(JSON) + blob 참조(`/api/datasets/:id/blob/:key`, f32 LE). `source:'cartesian'|'polymesh'|'vtu'`, `geometryFidelity`, `grid`(구조격자 노드 좌표), `surface`(패치별 삼각형, `cellOfTri`), `fields[].perTime`, `times`, `up`.
- `residuals.ts` — 바이너리별 스타일 매처(println! 원문 기반) + 정규화 필드명(`U,p,k,epsilon,omega,nuTilda,T,continuity,dk_k,...`), `converged`/`written to`/오류/`NaN` 분류. 서버와 웹이 같은 구현을 쓴다.
- `registry.ts` — 바이너리 카탈로그(positional/flag 타입·기본값, `accepts`, `builds`, `residualStyle`, `writes`), 모델 목록(모델 → 드라이버), 벽처리·알고리즘·메쉬 종류·패치 종류·출력 형식·생성 메쉬 프리셋.
- `tools.ts` — 도구 이름 상수, 정책(auto/ask/never), 도구 카드 요약 함수.

## 6. 서버

### 6.1 레지스트리 동기화
- 시작 시 `docs/schema/case-1.json`을 `Ajv2020`(+formats, schemars의 `double/uint/int32/uint32` 등록)으로 컴파일하고 `$defs` 열거값(`oneOf`의 `enum`+`const` 형태 포함)으로 정적 목록을 덮어쓴다.
- vitest 동기화 테스트: `Cargo.toml [[bin]]` 이름·`path` 집합 = 레지스트리; `fn usage()`가 있는 8개는 `{}` 자리표시자 확장 후 **플래그 토큰 집합** 비교, `const USAGE` 3개(cht/datacentre/decompose)는 그 문자열, 나머지(bench/probe/validate/graph-bench/dispatch-bench)는 존재만 확인.

### 6.2 프로세스 러너
- 디스패치 순서: `OFGPU_BIN_DIR/<name>[.exe]` → `rust/target/release/<name>[.exe]` → (`nvcc` 있을 때) `cargo run --release --manifest-path rust/Cargo.toml --bin <name> --` → 데모 모드면 `node --import tsx mock/mock-cli.ts <name> ...`. `CFD_DEMO=1`이면 항상 모의.
- `spawn(argv, {cwd: workspace, detached: posix})`, 줄 단위 읽기 → 5,000줄 링버퍼 + `gui/runs/<id>/log.txt` + 50 ms 배치 WS 전송 + 잔차 파서. `written to` 수집(여러 번), 오류 정규식 `^(error|ofgpu-[\w-]+|benchmark aborted):\s*(.+)$`(stderr 포함), `*** NaN/Inf ***` → `diverged`.
- 중단: POSIX `process.kill(-pid, SIGTERM)` → 3 s 후 SIGKILL, Windows `taskkill /T /F`. 단일 GPU → 솔버 실행 큐 직렬화(메쉬·분석은 병렬 허용). 재시작 시 이전 실행은 절대 `running`으로 복원하지 않는다.
- GPU 정보: `nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader`를 실행 중 5 s 폴링; 없으면 `absent`; 데모는 값을 꾸며낸다. `ofgpu-probe`는 준비 확인용 옵션.

### 6.3 데모 모드 (`CFD_DEMO=1`)
- **모의 솔버 CLI**: 레지스트리로 argv를 파싱(잘못된 플래그면 실제 드라이버처럼 exit 1 + `error:`). 배너·계수·`iterating N times` 출력 후 `-check N`마다 **그 바이너리의 정확한 형식**으로 잔차 줄 출력(`r(it)=r0·e^(-it/τ)·(1+0.15·noise)`), 수렴/요약/`written to`. `CFD_MOCK_SPEED`로 속도 조절(E2E는 수 초).
- 결과 파일: 해석적 필드(채널 멱법칙 속도 + 제트, 선형 압력, `k/epsilon/nut`, 부력 케이스 `T`)를 **foam ASCII**(실제 `fields.rs` 형식), OpenFOAM 디렉터리 케이스에는 blockgen 형식 **polyMesh**, `-output vtu`면 실제 레이아웃(타입 42, `faces/faceoffsets`) **VTU + PVD**로 쓴다. 세 리더 모두 데모에서 실행된다.
- `ofgpu-generate-mesh` 모의: `[mesh]` 로그와 함께 polyMesh/`0/`/`system/` 생성.
- **모의 LLM** (`CFD_LLM=mock`, 데모 모드 기본): 실제 루프·승인·tool_result 조립을 그대로 타는 대본 클라이언트. 사용자 문장의 키워드로 분기(메쉬 → `mesh_generate`; 실행/solver → `run_start`+`run_wait`; 3D/뷰어 → `viewer_command load/addSlice/addStreamlines`; 오류 → `run_log`; 케이스 편집 → `case_edit`(승인 카드) 등). 거부·max_tokens·재시도 가능 오류 시나리오도 대본에 포함해 UI 카드를 전부 검증한다.
- `GET /api/health`. `CFD_ALLOW_NO_API_KEY=1`이면 키 없이 기동(모의 LLM).

### 6.4 데이터셋 리더 (`formats/`, `datasets/`)
- `cartesian.ts`: JSONC `mesh`(bounds·cells·grading·regions)에서 노드 좌표·셀 중심·경계면(패치 = 6면 + `regions` 창) 생성. `fill_graded`/`graded_nodes` 정확 이식(정수 나눗셈, 균일 폴백 조건 포함).
- `foam.ts`: `FoamFile` 헤더의 `class`로 `surfaceScalarField`(phi) 제외; `uniform x;`(셀 수로 확장), `nonuniform 0()`, `List<scalar|vector>` 줄바꿈 무관 토크나이즈; 스트리밍으로 `Float32Array` 직접 채움; 따옴표 패치명 허용.
- 시간 디렉터리: `0/` 포함 전체를 스캔, 필드별 **OpenFOAM 시간 폴백**(t 이하 최신 디렉터리에서 그 필드). 지수형(`1e-05`)·임의 이름(`-write NAME`) 디렉터리 허용.
- `polymesh.ts`: points/faces/owner/neighbour/boundary 파서, 패치별 삼각분할, owner 셀 값으로 면 색칠, 점 격자 감지(정렬된 고유 x/y/z 곱 = nPoints)로 구조격자 복원, 아니면 비정렬.
- `vtu.ts`: XML 헤드 파싱 후 부가 블록을 **오프셋 청크 읽기**(1M 셀 ≈ 1 GB이므로 readFile 금지), 타입 42 + faces/faceoffsets, 면 중심 해시로 경계면 추출, `geometryFidelity:'proxy'`. `VTK/*.pvd` 글롭, `.vtp` 무시, 정상 드라이버의 `TIME`은 시간축으로 쓰지 않음. 격자 복원 불가 시 절단면/등가면은 `NO_STRUCTURED_GRID`.
- `datasets/service.ts`: `POST /api/datasets/open {path}` → 즉시 `{id, status:'loading'}` + `dataset.progress`; 매니페스트의 필드 범위는 시간별로 지연 계산; blob 캐시(`gui/.cache`, mtime 키); worker_threads 풀(≤2).
- `up` 축: JSONC `physics.gravity`(plume은 z-up), OpenFOAM `constant/g`, 기본 z. 2D(nz=1) 케이스는 `empty` 면을 기본 색칠면으로, 등가면 비활성, 유선 평면화.

## 7. 에이전트 루프와 도구

### 7.1 루프 (`agent/loop.ts`)
1. user 메시지 추가(텍스트 + `@경로` 첨부) → **그 뒤에** 휘발성 컨텍스트 `role:'system'` 메시지(워크스페이스·모드·GPU·열린 케이스·활성 실행·등록된 사용자 도구·시각)를 마지막 항목으로 붙인다(캐시 프리픽스 보존; 400이면 user 텍스트 블록으로 폴백).
2. 스트림 열기 → `msg.block_start/delta`, `tool.start/input_delta` 전송 → `finalMessage()`.
3. `stop_reason`: `end_turn` 종료; `max_tokens` 경고 + 완결 블록만 저장하고 미완 tool_use엔 `is_error` 결과 보강; `refusal` 빈 메시지는 저장하지 않고 `turn.refusal`; `tool_use` → assistant 내용을 그대로 저장(thinking 포함, 히스토리는 append-only).
4. 모든 tool_use 수집 → zod 검증(실패 → `is_error`) → 정책 분류(auto/ask/never) → `ask`는 한 번의 `tool.approval_request`로 묶어 대기(10 분 초과 → DENIED) → 승인·자동 도구 동시 실행(도구별 120 s, `run_wait` 예외; 뷰어 도구는 클라 없으면 5 s 내 `NO_VIEWER`) → **한 user 메시지**에 순서대로 tool_result → 반복(최대 40회, 마지막엔 "도구 없이 요약" user 지시).
5. 취소: turn별 AbortController; 결과 없는 tool_use엔 "cancelled" `is_error` 보강. 실행 중인 솔버는 취소해도 죽이지 않는다.
6. 실행 종료 알림: 세션이 유휴면 `role:'user'` 텍스트("[실행 알림] ...")로 새 턴.
7. 세션 파일 `gui/sessions/<id>.json`: SDK 형식 `messages`(원본) + UI 투영(blocks) + 도구 호출 기록 + 실행 ID + 설정. 원자적 저장. `finalMessage().model` 기록(폴백 고정 라우팅 대비).
8. 후속 제안 칩: `suggest_followups{items:[≤3]}` 자동 도구를 시스템 프롬프트가 마지막에 호출하도록 요청(프리필 불가 대안).

### 7.2 도구 목록 (이름은 서버 기준으로 통일)
| 도구 | 역할 | 정책 |
|---|---|---|
| `case_read`, `case_validate`, `case_create`, `case_edit` | JSONC 읽기/스키마+의미 검증(모델↔드라이버, lowRe↔모델, 출력 열)/템플릿 생성/포인터 편집(주석 보존, `valueJson`, `dryRun` diff) | 읽기 auto, 쓰기 ask |
| `mesh_generate` | `ofgpu-generate-mesh` 프리셋 + 출력 디렉터리 + 셀 수/STL/컷셀/벽모델/cyclic | ask |
| `run_start`, `run_wait`, `run_status`, `run_log`, `run_stop` | 솔버·분석 실행(`args:[{flag,value}]`를 레지스트리로 검증, 즉시 `{runId}`), 대기(≤120 s), 상태, 로그 창, 중단 | start/stop ask, 나머지 auto |
| `results_discover`, `field_stats`, `residuals_get` | 시간 디렉터리·VTK 목록, 필드 통계(min/max/mean/rms/히스토그램), 잔차 시계열 | auto |
| `viewer_command`, `plot_residuals` | 뷰어 명령(공유 zod 유니온; `load`는 비차단; 스크린샷은 `image` 블록으로 반환), 잔차 패널 | auto (UI 전용) |
| `file_read`, `file_list`, `file_search`, `file_write` | 워크스페이스 파일(경로 안전), 정규식 검색 | 읽기 auto, 쓰기 ask |
| `spec_lookup` | `rust/SPEC-LIT.md` §절 조회/검색 | auto |
| `gpu_info` | nvidia-smi/데모 | auto |
| `custom_tool_create`, `custom_tool_run` | 사용자 정의 도구(argv 명령 또는 `node:vm` JS — "신뢰된 로컬"로 표시). 고정 디스패처라 tools 배열이 바뀌지 않는다 | ask |
| `shell_exec` | 임의 명령 | never(설정에서 ask로 변경 가능) |
| `suggest_followups` | 후속 제안 칩 | auto |

도구 스키마는 zod → JSON Schema. `strict`는 쓰지 않고(any 타입·범위 제약 문제 회피) 런타임 zod 검증으로 오류를 모델에 되돌린다. 도구 배열 순서 고정(캐시).

### 7.3 시스템 프롬프트
정적 블록(`cache_control: ephemeral`): 역할, 안전·승인 규칙, 레지스트리 다이제스트(바이너리·플래그·모델→드라이버·출력 레이아웃), JSONC 편집 규칙("전체 재작성 금지, `case_edit` 포인터 사용"), 잔차 규약, "폴링 대신 `run_wait`", 응답 간결성·범위 준수 지침(Opus 5 권고문). 휘발성 정보는 §7.1의 mid-conversation system 메시지로.

## 8. 프런트엔드 셸 (목업 대응)

- **레이아웃**: `[활동바 | 사이드패널 | 중앙(탭 + 하단패널) | AI 패널] / 상태바`, `react-resizable-panels`. 활동바: Explorer·Search·Source Control(읽기 전용 `git status`)·Run & Debug(실행 목록)·Extensions(스텁)·CFD Tools.
- **탐색기**: `WORKSPACE_ROOT` 트리(명시적 숨김목록: `.git`, `node_modules`, `rust/target`, `reference`; `.gitignore`는 쓰지 않음 — 결과 폴더가 사라진다). 결과 디렉터리·시간 디렉터리·VTK 배지, 클릭 시 뷰어 탭. OUTLINE(JSONC 최상위 키), SIMULATION TASKS(Generate Mesh, Run Solver, Post-Process, Validate Results, Export Data).
- **탭**: `case.jsonc`(Monaco, `loader.config({monaco})`로 번들 워커 사용, 스키마 `fileMatch: cases/**/*.jsonc` 또는 `$schema` 일치), `.rs`(읽기 전용 강조), 3D Viewer, Residuals, Diff. 저장은 `PUT /api/fs/file{baseHash}` → 409 처리.
- **하단 패널**: Terminal(xterm, 실행별 탭, 경량 색칠), Logs(가상 목록+필터), Problems(스키마 오류 + 솔버 거부/발산 분류, 클릭 시 로그 위치로), Output. 우측 분할에 Residuals 미니 차트.
- **잔차 차트**: uPlot, 로그 스케일(≤0 값은 null), 고정 계열 상위집합(continuity, U, p, k, epsilon, omega, nuTilda, T) + 동적 추가, 100 ms 배치, CSV 내보내기(`GET /api/runs/:id/residuals.csv`).
- **AI 패널**: 스트리밍 마크다운(코드 강조), 도구 카드(✓/✗/스피너/방패, 연속 성공 카드는 목업처럼 묶음), 실행 진행 카드(runId 바인딩, 진행률·잔차·경과·중단·"뷰어 열기"), 승인 카드(허용/세션 동안 허용/거부, diff 미리보기), Diff 카드, 제안 칩, 빠른 동작(Generate Mesh·Run Solver·Explain Error·Create Tool), 대화 목록/새 채팅.
- **상태바**: 프로젝트·브랜치·문제 수·GPU 상태(Ready/Busy/Demo/Absent + 메모리)·Ln/Col·인코딩·언어·버전.
- **단축키**(브라우저 안전): Ctrl/⌘+K 팔레트, Ctrl+B 사이드바, Ctrl+J 하단, Ctrl+S 저장, Ctrl+Enter 전송, Esc 취소, Ctrl+Shift+N 새 채팅, Alt+1..4 탭.
- **테마/i18n**: `styles/tokens.css` 라이트(목업)·다크; 한국어 기본 + 영어 사전.
- **WS 클라이언트**: 지수 백오프 재접속, `ping/pong`, 재접속 시 `run.subscribe{fromSeq}` 재생 + REST 재조정, 오프라인 시 입력 비활성.

## 9. 3D 뷰어

- **엔진**: `three/webgpu`(vite alias `three → three/webgpu`로 addons도 같은 빌드 사용). `await renderer.init()`; 초기화·첫 프레임(PMREM 포함)을 try/catch로 감싸 실패 시 `forceWebGL:true`로 재생성; `?renderer=webgl`로 강제. `RenderPipeline`(구 `PostProcessing`)은 선택, GTAO는 WebGPU+high 품질에서만. 오버레이에 활성 백엔드 표시. 주의: three r185의 WebGPU 경로는 Chrome 141에서 텍스처 뷰 스위즐 오류가 있어 더 새 Chromium/Electron이 필요하며, 그 경우 자동으로 WebGL2로 내려간다.
- **사실적 룩**: `MeshStandardNodeMaterial`(roughness 0.55) + TSL `colorNode = texture(lut, vec2(s, 0.5))`, `RoomEnvironment`→PMREM, ACES, `antialias` MSAA, `PCFSoftShadowMap` 그림자 받는 바닥, 거리 페이드 격자, 축 트라이어드(ViewHelper 검증 후 실패 시 자체 구현), `shading:'flat'` 토글(정량 판독용).
- **데이터**: 서버 매니페스트 → 경계면 지오메트리 → 현재 필드/시간 blob → 인접 시간 프리페치(LRU 512 MB). 표면 색은 `cellOfTri`로 워커에서 정점 스칼라 생성(경계면은 셀 수 대비 작아 저렴).
- **절단면**: 워커가 격자 해상도로 평면을 샘플링(구조격자 노드 좌표로 이진 탐색) → 컬러맵 적용 RGBA8 텍스처(부동소수 3D 텍스처 필터링 제약 회피, WebGL2/WebGPU 동일). 임의 평면은 박스 클리핑 다각형 위에 같은 샘플링.
- **등가면**: 워커 마칭큐브(셀 중심 격자), 결과 `Transferable`. **유선**: 워커 RK4, 국소 셀 크기 기반 적응 스텝, 삼선형 보간, 양방향, `Line2`(fat line, 속도 색) 또는 튜브. **글리프**: `InstancedMesh` 화살표(≤50k). 경계선(`EdgesGeometry`), 와이어프레임(외곽 셀 모서리), 클립 박스, 시간 스크럽, 카메라 프리셋, 스크린샷(오프스크린 MSAA + 범례 합성 → PNG; 도구 결과엔 base64 `image` 블록 ≤1568 px).
- **명령 브리지**: `viewer_command` → WS `viewer.command` → 클라 실행 → `viewer.result{state|error}`. 병렬 뷰어 명령은 순서대로 직렬 실행.

## 10. 보안

서버는 `127.0.0.1` 바인딩(원격은 `CFD_ALLOW_REMOTE=1` + 토큰). 경로가 들어가는 모든 입력은 `resolveInWorkspace()`(realpath, `..`·심링크 차단) 하나를 통과. 스폰은 16개 `ofgpu-*` 이름 허용목록 + 레지스트리 플래그 검증, 셸 없음. 쓰기 도구는 승인 필요, `shell_exec`는 기본 금지. API 키는 서버만 보유. WS origin 검사. 세션·실행 기록은 `gui/sessions`, `gui/runs`(gitignore). `custom_tool` JS는 보안 경계가 아님을 UI에 표시.

## 11. 단계와 검증

| 단계 | 산출물 | 검증 |
|---|---|---|
| 0 계약 | `shared/src/*` 전부, 레지스트리 데이터, 서버/웹 내부 인터페이스 스텁 | tsc, zod 라운드트립·잔차 파서 fixture 테스트 |
| 1 병렬 구현 | (A) 서버 코어+러너+모의 솔버+GPU+fs API, (B) formats+datasets 리더/라이터, (C) 에이전트 루프+도구+모의 LLM, (D) 웹 셸, (E) 3D 뷰어 | 각 모듈 vitest, tsc |
| 2 통합 | `main.ts` 배선, 웹 ↔ 서버 연결, 데모 시나리오 | `npm run typecheck && npm test && npm run build` |
| 3 E2E | 데모 모드: 채팅으로 메쉬 생성 → 솔버 실행 → 잔차 → 뷰어 로드/절단면/유선 → CSV → 승인 카드 → 재접속 재생 | Playwright(WebGL2 강제) + 스크린샷 |
| 4 리뷰 | 적대적 코드 리뷰 워크플로 → 수정 | 재실행 |
| 5 문서/푸시 | `gui/README.md`(설치·환경변수·데모·도구 목록·실제 GPU 기기 체크리스트), 커밋·푸시 | — |
| 이후 | 실제 GPU 기기 검증(Windows), Tauri 셸, Search/SC 패널 고도화, GTAO, 볼륨 렌더링(VDB) | — |

## 12. 심사에서 확정한 결정 (must-fix 반영)

1. 프로토콜은 `shared/protocol.ts` 하나. 이름·필드는 서버 설계안 기준, UI가 필요로 한 `blockIndex`·`seq`·`fromSeq` 재생·`ping/pong`·`pendingApprovals`·`quick`·`fs.changed`·`problems`·`gpu` 추가.
2. mid-conversation `system` 메시지는 **user 메시지 뒤**에만; 실행 알림은 `user` 텍스트.
3. 거부(빈 메시지)는 저장 금지; `max_tokens`/취소 시 미완 tool_use 보강; `fallback` 블록 에코 규칙 준수; 턴별 모델 기록.
4. `strict` 미사용; `.nullable()` 사용; any 값은 JSON 문자열(`valueJson`).
5. `accepts` 정정: JSONC = k-epsilon·lowmach·decompose·cht·datacentre, 나머지는 OpenFOAM 디렉터리. `case_validate.suggestedDriver`와 `run_start`가 불일치를 거부.
6. 잔차 파서는 바이너리별 매처 + 사전 치환(`max dk/k`→`dk_k`, `|U|`→`U`, `dp0/dt`→`dp0_dt`, `ε`→`epsilon`), 대소문자 무시 SKIP, 맨몸 `converged`, `*** NaN/Inf ***`→diverged, datacentre 팬 줄은 `run.metric`.
7. JSONC 실행엔 polyMesh가 없으므로 기하는 `cartesian.ts`(그레이딩·regions 포함)에서; VTU는 명시적으로 열 때만, `proxy` 표시.
8. 모의 VTU는 타입 42 + faces/faceoffsets(실제 레이아웃); 모의 polyMesh도 생성해 세 리더 모두 데모에서 실행.
9. 시간 폴백(0/ 포함), 지수형·임의 이름 디렉터리, `phi` 제외, `uniform` 확장.
10. `Ajv2020` + formats; Monaco는 번들 워커 + `cases/**/*.jsonc` 스키마 매칭; 탐색기 숨김목록 방식.
11. Playwright 1.56.1 고정, e2e는 `?renderer=webgl`; `/api/health`; 환경변수는 `CFD_DEMO`, `CFD_PORT`, `CFD_ALLOW_NO_API_KEY`, `CFD_LLM`, `CFD_MOCK_SPEED`, `OFGPU_BIN_DIR`, `CFD_WORKSPACE`.
12. 스폰: `--manifest-path rust/Cargo.toml`, `.exe` 접미사, 프로세스 그룹 킬; 종료코드·오류 접두 정규식 정정.
13. GPU 메모리는 nvidia-smi; `mesh_generate`는 "활성 케이스"가 아니라 프리셋 + 출력 디렉터리.
14. 스크린샷 결과는 URL이 아니라 `image` 블록; `load`는 비차단; 뷰어 명령 직렬 실행.
15. 브라우저 예약 단축키 회피; 볼륨 텍스처 대신 워커 샘플링 절단면.

## 13. 알려진 제약

- 이 샌드박스에는 GPU·nvcc·API 키가 없다. 실제 솔버·실제 Claude 경로는 코드 리뷰와 SDK 타입 검사로만 검증하고, 실행 검증은 Windows GPU 기기에서 README 체크리스트로 한다.
- three r185 WebGPU 백엔드는 브라우저 버전에 민감하다. 폴백이 있으므로 기능은 유지되지만 WebGPU 전용 품질(GTAO 등)은 환경에 따라 꺼진다.
- VOF·datacentre·bench 계열은 잔차가 아닌 지표를 출력하므로 차트엔 `metric` 계열로 표시된다.
- 라이선스: `gui/`도 저장소의 Prosperity 라이선스를 따른다. 사용한 오픈소스(three, monaco, uplot, xterm, react 등)는 MIT이며 README에 목록을 둔다.
