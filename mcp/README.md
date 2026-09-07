# MCP 사용법 — Iterations / ofgpu MCP server

AI 클라이언트가 이 저장소의 솔버를 **직접 실행**하게 하는 MCP 서버입니다.
Claude Code, Claude Desktop, 그 밖에 Model Context Protocol을 말하는 어떤
클라이언트든 붙일 수 있습니다. 붙이고 나면 "이 STL로 격자 만들고 k-omega로
2500번 돌려줘" 한마디에 실제 GPU에서 계산이 돕니다.

An MCP server that lets an AI client **actually run** the solvers in this
repository. Works with Claude Code, Claude Desktop, or any client that speaks the
Model Context Protocol. Once it is connected, "mesh this STL and solve it with
k-omega for 2,500 iterations" runs on the real GPU.

---

## 1. 준비물 / What you need

| | |
|---|---|
| Node.js | 18 이상 (의존성은 없습니다 — 파일 하나입니다) |
| GPU | CUDA 13을 지원하는 NVIDIA GPU |
| 솔버 바이너리 | `rust/target/release/ofgpu-*` 또는 설치 프로그램이 넣어 준 `bin/` |

바이너리가 없다면 저장소 루트에서:

```powershell
cd rust
cargo build --release
```

설치 프로그램(`iterations-*.exe`)으로 설치했다면 이미 있습니다 —
보통 `C:\Program Files\Iterations\bin`.

---

## 2. 붙이기 / Connecting

### Claude Code

프로젝트 루트에서 한 줄이면 됩니다.

```powershell
claude mcp add ofgpu -- node "C:\path\to\Iteration-CFD\mcp\server.mjs"
```

작업 폴더와 바이너리 위치를 명시하려면:

```powershell
claude mcp add ofgpu ^
  -e OFGPU_WORKSPACE=C:\path\to\Iteration-CFD ^
  -e OFGPU_BIN_DIR=C:\path\to\Iteration-CFD\rust\target\release ^
  -- node "C:\path\to\Iteration-CFD\mcp\server.mjs"
```

붙었는지 확인: `claude mcp list`

### Claude Desktop

`claude_desktop_config.json`에 다음을 넣습니다.
(Windows: `%APPDATA%\Claude\claude_desktop_config.json`,
macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "ofgpu": {
      "command": "node",
      "args": ["C:\\path\\to\\Iteration-CFD\\mcp\\server.mjs"],
      "env": {
        "OFGPU_WORKSPACE": "C:\\path\\to\\Iteration-CFD",
        "OFGPU_BIN_DIR": "C:\\path\\to\\Iteration-CFD\\rust\\target\\release"
      }
    }
  }
}
```

넣고 나서 Claude Desktop을 완전히 종료했다가 다시 켜야 반영됩니다.

### 그 밖의 클라이언트 / Other clients

stdio로 JSON-RPC를 주고받는 표준 MCP 서버입니다. 실행 명령은
`node <경로>/mcp/server.mjs` 하나뿐이고, 설정은 아래 환경 변수로만 합니다.

---

## 3. 환경 변수 / Environment

| 변수 | 뜻 | 기본값 |
|---|---|---|
| `OFGPU_WORKSPACE` | 도구가 읽고 쓸 수 있는 **유일한** 폴더 | 서버를 띄운 현재 폴더 |
| `OFGPU_BIN_DIR` | `ofgpu-*` 실행 파일이 있는 폴더 | `mcp/` 옆의 `rust/target/release` |
| `OFGPU_MCP_TIMEOUT_S` | 도구 하나가 돌 수 있는 최대 초 | `3600` |

---

## 4. 도구 목록 / The tools

| 도구 | 하는 일 | 읽기 전용 |
|---|---|---|
| `ofgpu_probe` | GPU를 확인하고 장치 결과가 호스트 결과와 비트 단위로 같은지 검사 | ○ |
| `ofgpu_list_cases` | 작업 폴더의 케이스 목록 — `*.jsonc`와 OpenFOAM 케이스 디렉터리, 격자·결과 유무까지 | ○ |
| `ofgpu_generate_mesh` | 바로 돌릴 수 있는 케이스 생성. `stl`을 주면 컷셀로 형상을 파냅니다 | |
| `ofgpu_solve` | 솔버 9종 중 하나 실행, 잔차 로그의 끝부분을 돌려줍니다 | |
| `ofgpu_validate` | 검증 스위트 실행 (인위해법·해석해·공개 벤치마크) | ○ |
| `ofgpu_read_case_file` | 케이스 파일 한 개 읽기 (최대 400줄) | ○ |

`ofgpu_solve`의 `solver` 값: `k-epsilon`, `k-omega`, `sa`, `lowmach`, `cht`,
`buoyant`, `plume`, `vof`, `datacentre`.

---

## 5. 해 보기 / A first conversation

동봉된 경주차 샘플([`cases/racecar.md`](../cases/racecar.md))로 시작하는 것이
가장 빠릅니다. 클라이언트에 이렇게 말하면 됩니다.

> GPU 먼저 확인하고, `cases/racecar.stl`을 `big` 프리셋 128 격자에 컷셀로 넣어서
> `cases/racecar_case`를 만들어 줘. 다 되면 k-omega로 2500번 돌리고 잔차가
> 어떻게 떨어졌는지 알려 줘.

클라이언트는 `ofgpu_probe` → `ofgpu_generate_mesh` → `ofgpu_solve`를 차례로
부릅니다. 격자 생성이 10–20분 걸린다는 것만 감안하십시오 — 형상 분류가
CPU에서 도는 구간입니다. 솔브는 40초 안팎입니다.

> Check the GPU, then cut `cases/racecar.stl` into a 128-cube `big` block as
> `cases/racecar_case`, and when that finishes solve it with k-omega for 2,500
> iterations and tell me how the residuals came down.

---

## 6. 무엇을 못 하게 되어 있는가 / What it will not do

이 서버는 AI가 부르는 것이므로, 부를 수 있는 범위를 좁게 못 박아 두었습니다.

- **셸이 없습니다.** 자식 프로세스는 언제나 인자 배열로 실행되고, 실행 파일은
  이 파일이 이름으로 찾아낸 `ofgpu-*` 하나뿐입니다. 호출자가 쓴 문자열이
  명령이 되는 경로가 없습니다.
- **작업 폴더 밖으로 나가지 못합니다.** 모든 경로는 `OFGPU_WORKSPACE` 기준으로
  해석한 뒤 그 안인지 검사하고, 아니면 거절합니다.
- **덮어쓰지 않습니다.** `ofgpu_generate_mesh`는 이미 있는 폴더를 대상으로
  하면 거절합니다.
- **파일을 지우지 않습니다.** 지우는 도구가 없습니다.

The server is called by a model, so what it can be made to do is pinned down:
no shell, argv arrays only, executables resolved by name from one directory,
every path resolved and checked against the workspace root, no overwrite, and no
delete tool at all.

---

## 7. 점검 / Checking it works

서버만 따로 두드려 볼 수 있습니다.

```powershell
node mcp\smoke.mjs
```

`initialize`, `tools/list`, 그리고 읽기 전용 도구 호출까지 왕복해 보고 결과를
출력합니다. GPU가 없는 기계에서도 프로토콜 부분은 통과해야 정상입니다.

---

## 라이선스 / Licence

이 폴더도 저장소와 같은 조건입니다 — Prosperity Public License 3.0.0과
라이선서 해석 조항. 개인·교육·정부·자선, 그리고 **공공을 목적으로 하는 연구**는
무료이고, 특정 제품이나 산업 이전을 목적으로 하는 기술개발은 정부출연연구기관에서
수행하더라도 30일 시험 후 유상 라이선스입니다. 자세한 것은
[`LICENSING.md`](../LICENSING.md).

문의: simul@msimul.com
