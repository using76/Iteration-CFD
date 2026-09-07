# 다음 세션 인수인계 — Iteration CFD Studio

2026-09-07 시점. 이 문서 하나만 읽으면 이어서 작업할 수 있게 정리했습니다.
설계 결정과 배경은 [`PLAN.md`](PLAN.md), 사용법은 [`README.md`](README.md)에 있습니다.

---

## 1. 지금 상태 (검증된 것)

브랜치 `claude/cfd-ai-chat-gui-k296dq`, 마지막 커밋 `61e1242`.

| 항목 | 결과 |
|---|---|
| `npm run typecheck` | 통과 (shared / server / web) |
| `npm test` | 279개 통과 |
| `npm run build` | 통과 (`web/dist` 70 MB, `server/dist`) |
| `npm run e2e` | 6/6 통과 — 데모 모드로 메쉬 생성 → 솔버 실행·잔차 → 3D 뷰어(절단면+유선) → CSV |
| 서버 부팅 | `CFD_DEMO=1 npx tsx src/main.ts` → `/api/health`, `/api/hello` 정상, 정적 서빙·SPA 폴백·경로 탈출 403 확인 |

구현이 끝난 모듈:

- `shared/` — WS 프로토콜, 뷰어 명령/상태, 데이터셋 매니페스트, 잔차 파서(바이너리별 매처), 바이너리·모델 레지스트리, 도구 카탈로그, i18n
- `server/` — node:http 라우터 + REST, WS 허브(실행 리플레이·뷰어 요청/응답), 실행 관리자(레지스트리 검증·단일 GPU 큐·프로세스 그룹 kill·로그 링·잔차 파싱), 실제/모의 디스패치, GPU 모니터, 파일 감시, 문제 수집, Ajv2020 스키마 + Cargo/usage 동기화 테스트, 모의 솔버 CLI(실제 로그 형식 + foam/polyMesh/VTU 출력), 에이전트 루프(스트리밍·승인·세션 영속·UI 투영·대본형 모의 LLM), 도구 25종
- `web/` — VS Code형 셸(탐색기·탭·Monaco·xterm·uPlot 잔차·문제 패널), 어시스턴트 패널(도구/실행/승인/diff 카드), zod 검증 WS 클라이언트, three r185 WebGPU 뷰어(WebGL2 자동 폴백, TSL LUT 색칠, 절단면·등가면·유선·글리프·범례·스크린샷)

**아직 못 한 검증**: 실제 GPU 기기(Windows + CUDA)에서 실제 `ofgpu-*` 바이너리 실행, 실제 Claude API 키로 에이전트 루프 실행, 실제 브라우저에서 WebGPU 백엔드. 세 가지 모두 이 샌드박스에 GPU·API 키·최신 Chromium이 없어 코드 리뷰와 타입 검사로만 확인했습니다.

---

## 2. 다음에 할 일 — 적대적 리뷰에서 확인된 결함 27건

영역별 리뷰어 5명이 찾고, 항목마다 회의적 검증자 2명이 **둘 다 반박에 실패한** 것만 남겼습니다. 원본(재현 시나리오·검증자 근거 포함)은 [`docs/review-2026-09-07.json`](docs/review-2026-09-07.json)에 있습니다.

권장 순서: **2.1 (보안 3건) → 2.2 (서버 안정성) → 2.3 (에이전트 루프) → 2.4 (웹/캐시)**.

### 2.1 높음 — 보안·데이터 정합성 (4건)

| # | 파일:줄 | 문제 | 고칠 것 |
|---|---|---|---|
| H1 | `server/src/http/server.ts:63` | REST에 Origin/Host/Content-Type 검사가 없어(WS만 검사) 기본 로컬 설정에서 임의 웹페이지가 교차 사이트 POST로 `/api`를 호출할 수 있고, DNS 리바인딩으로 워크스페이스 전체를 읽고 쓸 수 있음 | `onRequest`에서 `/api` 요청의 Host가 루프백이 아니면 거부, Origin이 있으면 `originAllowed()`로 검사, `ctx.json()`은 `content-type: application/json`을 요구 |
| H2 | `server/src/tools/shell.ts:33` | `spawnCapture`가 `env: process.env`를 그대로 물려줘 `ANTHROPIC_API_KEY` 등이 도구 출력 → 모델 컨텍스트 → 세션 파일로 새어나감 (PLAN §10 위반) | 자식 프로세스용 env를 정화(`*_KEY`/`*_TOKEN`/`*_SECRET` 삭제)하고 `runs/dispatch.ts`의 솔버 실행에도 같은 env 사용 |
| H3 | `server/src/datasets/manifest.ts:85` | 데이터셋 지문이 시간 **디렉터리** mtime만 보므로, 같은 시간 디렉터리를 덮어쓰는 재실행 후에도 낡은 필드 blob이 계속 제공됨 | `listVolFields`/`listTimeDirs`에서 파일별 `name@mtimeMs:size`를 지문에 포함(`vtuSeries`가 이미 하는 방식), OpenFOAM 케이스는 `controlDict`/`fvSolution`/`0/`도 포함 |
| H4 | `web/src/editor/CodeEditor.tsx:146` | 탭을 닫았다 다시 열면 Monaco가 살아 있는 낡은 모델을 재사용해 옛 내용을 보여주고, 그대로 편집·저장하면 baseHash가 맞아 409 없이 낡은 내용을 디스크에 씀 | 탭을 닫을 때 모델을 `dispose()`(또는 `keepCurrentModel` 제거), `onMount`에서 `model.getValue()`와 버퍼 내용이 다르면 `setValue`, 더티 탭 닫기 전 확인 |

### 2.2 중간 — 서버 안정성 (5건)

| # | 파일:줄 | 문제 | 고칠 것 |
|---|---|---|---|
| S1 | `server/src/runs/store.ts:59` | 실행별 `log.txt`/`residuals.jsonl` 쓰기 스트림에 `error` 리스너가 없어 I/O 오류가 서버 프로세스를 죽임 | 두 스트림에 `.on('error')` 부착 + `main.ts`에 `uncaughtException`/`unhandledRejection` 가드 |
| S2 | `server/src/runs/manager.ts:370` | `launch()`가 `store.openRun()`/spawn 전에 상태를 `running`으로 바꾸고 try/catch가 없어, 실패 시 GPU 큐를 영구 점유하는 유령 실행이 남고 `stop()`이 멈춤 | openRun/startProcess를 try/catch로 감싸 실패 시 `failed`로 마감하고 GPU 슬롯 해제·`exit` 발행·`pump()` 계속 |
| S3 | `server/src/runs/manager.ts:313` | `RE_WRITTEN`(shared/residuals.ts:91)이 앵커가 없어 실제 드라이버의 `restart checkpoint written to ...` 줄도 잡아 엉뚱한 디렉터리로 `viewer.open`을 쏘고 `run_wait untilWritten`을 조기 만족 | 패턴을 `/^\s*written to\s+(.+?)\s*$/`로 앵커하고 체크포인트 줄은 별도 이벤트로, 픽스처 추가 |
| S4 | `server/src/tools/files.ts:35` | `file_read`가 크기 제한 없이 파일 전체를 메모리에 읽음(REST 쪽은 8 MB 상한이 있음) | stat으로 상한(8–32 MB) 검사, 이진 판정은 앞 8 KB만, 요청 줄 범위만 스트림으로 |
| S5 | `server/src/tools/files.ts:123` | 모델·사용자 정규식(`file_search regex`, `run_log grep`, REST 검색)이 이벤트 루프에서 그대로 실행돼 파국적 백트래킹 한 방에 서버가 멈춤 | 안전 정규식 검사로 중첩 수량자 거부, 또는 워커 스레드 + 하드 타임아웃, 줄 길이·줄 수 상한 |

### 2.3 중간 — 에이전트 루프 (5건)

| # | 파일:줄 | 문제 | 고칠 것 |
|---|---|---|---|
| A1 | `server/src/agent/loop.ts:360` | `finalMessage()`와 `approvals.request()` 사이에 취소가 들어오면 승인 대기자가 취소 **이후에** 생겨 10분 TTL까지 턴이 멈춤(그동안 새 메시지 거부) | `approvals.request()` 전에 `signal.aborted` 검사해 CANCELLED로 마감, 각 outcome을 abort 신호와 race(또는 `ApprovalManager.request`가 signal을 받게), `service.cancelTurn`은 abort → cancelAll 순서로 |
| A2 | `server/src/agent/loop.ts:267` | tool_use를 담은 어시스턴트 메시지를 tool_result 없이 먼저 저장하고 로드 시 복구가 없어, 도구 실행 중 프로세스가 죽으면 그 세션의 이후 모든 요청이 400 | 세션 로드 시(그리고 요청 조립 전) 짝 없는 tool_use를 찾아 `is_error` tool_result를 합성해 붙이기 |
| A3 | `server/src/agent/loop.ts:249` | 스트림 중간 거부(refusal)에 완성된 tool_use가 섞여 있으면 tool_result 없이 저장돼 세션이 영구히 깨짐 | 거부 시 부분 내용을 저장하지 않거나, 최소한 tool_use 블록 제거/`is_error` 결과 부착 |
| A4 | `server/src/agent/ui-projection.ts:108` | 프로젝터가 모르는 블록 타입(특히 `fallback`)을 버려서, max_tokens/취소 복구 경로가 폴백 경계 블록이 빠진 내용을 저장 → 다음 턴 요청 거부 | 알 수 없는 블록을 원본 그대로 보관해 `completeContent()`에서 그대로 방출, max_tokens 경로는 `final.content` 우선 |
| A5 | `server/src/agent/anthropic.ts:19` | 캐시 브레이크포인트가 정적 시스템 블록에만 있고 최상위 `cache_control`이 없어 대화 이력 전체가 매 요청 재처리됨 | `buildStreamParams`에 최상위 `cache_control: { type: 'ephemeral' }` 추가(정적 블록의 명시적 마커는 유지) |

### 2.4 낮음 — 나머지 (13건)

**서버/공유**
- `server/src/workspace/fs.ts:76` — 탐색기 트리가 심링크를 따라가 워크스페이스 밖 내용을 나열. `lstat`으로 심링크는 내려가지 않기.
- `server/src/ws/hub.ts:178` — sessionId를 담은 아무 프레임(`session.delete`/`rename` 포함)이나 클라이언트의 열린 세션을 바꿔버림. `session.open`(과 `session.new` 응답)에서만 설정.
- `server/src/tools/mesh.ts:37` — `-stl` 값만 `resolveInWorkspace`를 안 거쳐 임의 파일 지정 가능. `name=` 접두를 분리해 경로만 해석.
- `shared/src/registry.ts:518` — `checkArgValue`가 int/float에 `null`/`true`/`''`을 통과시킴(`Number(null)===0`). 숫자 타입/숫자 문자열만 허용.
- `server/src/agent/service.ts:168` — `userMessage()`가 await 앞에서만 `rt.active`를 검사해, 두 프레임이 창을 통과하면 두 번째 메시지가 응답 없이 이력에 남음. await 전에 동기적으로 예약.

**웹**
- `web/src/app/hotkeys.ts:45` — Ctrl+Enter가 컴포저의 `canSend` 가드를 우회해 턴 진행 중에도 전송하고 초안을 지워 입력이 사라짐. 분기 제거 또는 같은 게이트 적용.
- `web/src/app/hotkeys.ts:56` — 컨텍스트 메뉴·기록 메뉴·잔차 필드 팝오버를 Esc로 닫으면 전역 핸들러까지 내려가 진행 중인 턴이 취소됨. 팝오버가 Esc를 소비할 때 `stopPropagation()`.
- `web/src/app/hotkeys.ts:24` — Ctrl+Shift+N은 Chromium 예약키(시크릿 창)라 죽은 바인딩. Ctrl+Alt+N 등으로 교체하고 팔레트 안내도 수정.
- `web/src/ws/client.ts:42` — 프레임을 rAF에서만 처리해 백그라운드 탭에서 큐가 무한히 쌓임. `document.hidden`일 때 타이머 폴백 + 지연 민감 프레임 즉시 처리.
- `web/src/ws/client.ts:102` — hello에 담긴 **모든** 과거 실행을 구독해 접속·재접속마다 실행당 최대 5,000줄 + 전체 잔차를 리플레이. 실행 중이거나 화면에 보이는 실행만 구독하고 선택 해제 시 `run.unsubscribe`.
- `web/src/state/applyEvent.ts:165` — `openSession`이 서버 `session.state`보다 먼저 `currentSessionId`를 바꿔 이전 세션의 턴 상태(오류·거부 카드 등)가 새 세션에 남음. 권위 있는 `state.session?.id`로 전환 감지.
- `web/src/state/applyEvent.ts:181` — `turn.messageId`가 첫 라운드 id로 고정돼(서버는 라운드마다 새 id 발급) 이후 라운드의 스트리밍 표시·모델 스탬프가 엉뚱한 메시지에 붙음.
- `web/src/assistant/Composer.tsx:117` — @멘션 팝업이 열려 있으면 한글 IME 조합 확정 Enter가 후보 선택으로 가로채짐. `isComposing` 검사를 `onKey` 최상단으로.

---

## 3. 검증이 안 끝난 발견 14건 (먼저 사실 확인 필요)

리뷰 워크플로의 검증자들이 사용량 한도에 걸려 **판정을 못 받은** 항목입니다. 반박된 것이 아니라 미확인이므로, 고치기 전에 코드로 사실부터 확인하세요. 상세는 `docs/review-2026-09-07.json`의 `refuted` 배열에서 `reasons`가 빈 항목들입니다.

**formats/datasets (6)**
- `datasets/service.ts:312` — 같은 경로 동시 `open()` 두 건이 모두 존재 검사를 통과해 파싱이 두 번 돌 가능성.
- `datasets/worker.ts:229` — 첫 작업에서 죽은 워커를 "기동 실패"로 오분류해 풀이 영구 인라인 모드로 전환.
- `formats/results.ts:73` — `listVolFields`가 디렉터리의 모든 파일을 동시에 열어 EMFILE을 "필드 없음"으로 삼킴.
- `formats/cartesian.ts:286` — 1셀 축의 패치를 무조건 `empty`로 표시(러스트는 `patches[]` 규칙에서 타입을 가져옴).
- `datasets/manifest.ts:113` — 정상 드라이버가 `TIME=0.0`을 쓰므로 VTU 시간축이 전부 0(PLAN §6.4는 이 경우 TIME을 쓰지 말라고 함).
- `datasets/service.ts:101` — 엔트리가 명시적 evict 외엔 안 지워져 실행마다 새 지문으로 `cellCenters`가 계속 쌓임.

**viewer (8)**
- `engine/ThreeSceneView.ts:153` — 클립 박스가 양쪽 백엔드에서 무동작일 가능성(three r185 WebGPU는 `material.clippingPlanes`를 안 읽고 `ClippingGroup`만 봄) → 사실이면 `ClippingGroup`으로 교체.
- `layers/StreamlineLayer.ts:47` — `Math.max(..., ...speeds)` 전개가 점 20만 개 이상에서 RangeError.
- `controller/ViewerController.ts:276` — 범위 밖 `timeIndex`로 로드 시 데이터셋만 교체되고 뷰는 옛 것으로 남음.
- `data/DatasetLoader.ts:85` — load가 FIFO 큐를 최대 10분 점유해 "load는 비차단"이라는 설계와 어긋남.
- `ui/TimeScrubber.tsx:10` — 재생/드래그가 `setTime`을 합치지 않아 큐가 무한히 쌓임.
- `controller/ViewerController.ts:623` — 로그 스케일에서 범례 범위와 실제 색 매핑 범위가 불일치.
- `engine/Engine.ts:278` — 그리드/그라운드/PMREM 타깃을 dispose 안 해 데이터셋 로드마다 GPU 버퍼 누수.
- `layers/edges.ts:8` — `a*2^32+b` 엣지 키가 정점 인덱스 2^21 초과 시 정밀도 손실.

**반박된 3건**(고칠 필요 없음): `tools/custom.ts:94`(argv 치환), `ws/client.ts:197`(ping/pong), `SearchPane.tsx:50`(검색 재실행).

---

## 4. 리뷰와 무관하게 남아 있는 계획 항목

1. **실제 GPU 기기 검증** — `README.md`의 체크리스트대로 Windows + CUDA에서 실제 바이너리 실행, `nvidia-smi` GPU 배지, 실제 API 키로 에이전트 루프.
2. **실제 브라우저 WebGPU** — 최신 Chrome/Edge에서 뷰어 배지가 `WebGPU`인지, PMREM·Line2·인스턴싱이 WGSL로 컴파일되는지.
3. **뷰어 미구현** — 축 트라이어드 클릭 정렬, GTAO/후처리, VDB 볼륨 렌더링, `onOpenResiduals` 버튼.
4. **모듈 보고서 누락분** — 에이전트/웹 셸 담당 에이전트가 보고서 JSON 반환에 실패해 산문 기록이 없음(코드와 테스트는 있음). 필요하면 해당 디렉터리를 훑어 API 문서를 보강.
5. **선택** — Tauri 데스크톱 셸, Source Control/Extensions 패널 고도화.

---

## 5. 재개 방법

```bash
cd gui
npm install                    # 이미 설치돼 있으면 생략
CFD_DEMO=1 npm run dev         # 데모 모드 (GPU·API 키 불필요)
npm run typecheck && npm test  # 회귀 확인
npm run e2e                    # 데모 모드 E2E 6종
```

작업 순서 제안: **2.1 → 2.2 → 2.3** 을 고치면서 각 항목마다 재현 테스트를 먼저 쓰고(대부분 vitest로 재현 가능), 그 다음 3장 미확인 항목을 확인 후 처리, 마지막으로 4장.

리뷰를 다시 돌리려면 워크플로 스크립트가 `PLAN.md` §12의 규칙과 `docs/review-2026-09-07.json`의 형식을 그대로 씁니다. 같은 방식(영역별 리뷰어 → 항목별 회의적 검증자 2명)을 재사용하면 됩니다.
