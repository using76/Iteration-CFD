// Headless drive client for the studio: connects to /ws the way the web app
// does, hands the assistant a prompt, and narrates to the console everything
// that comes back - streamed text, tool calls, run progress, the exit. With
// --ui it also stands in for the GUI, answering ui.command frames so
// gui_control works with no studio open. Run from gui/:
//   CFD_DEMO=1 npx tsx server/src/main.ts &
//   npx tsx server/scripts/ai-drive.ts --autopilot --ui --case cases/plume.jsonc
// The script never sees an API key: those are read by the server alone.
import WebSocket from 'ws'
import {
  CHAT_TIMEOUT_MAX_MS,
  ClientMsgSchema,
  ServerMsgSchema,
  type ChatResponse,
  type ClientMsg,
  type RunInfo,
  type ServerHello,
  type ServerMsg,
  type SessionState,
  type UiCommand,
  type UiMessage,
  type UiState,
  type Usage,
} from '@cfd/shared'

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

interface Options {
  url: string
  casePath: string
  prompt: string | null
  /** settings.set { autoApprove: 'all' }; otherwise reads + manual approvals. */
  autopilot: boolean
  timeoutSec: number
  /** Stand in for the GUI: answer ui.command, push ui.state. */
  ui: boolean
  /** Drive the conversation over POST /api/chat instead of the WebSocket. */
  http: boolean
}

const USAGE_LINE = 'usage: npx tsx server/scripts/ai-drive.ts [--url ws://127.0.0.1:$CFD_PORT/ws] [--case cases/plume.jsonc] [--prompt "<text>"] [--autopilot] [--timeout 900] [--ui] [--http]'

/** The server this drives is the one CFD_PORT names, so the two agree without a flag. */
function defaultUrl(): string {
  const port = Number(process.env.CFD_PORT)
  return `ws://127.0.0.1:${Number.isInteger(port) && port > 0 ? port : 8787}/ws`
}

function parseOptions(argv: string[]): Options {
  const o: Options = { url: defaultUrl(), casePath: 'cases/plume.jsonc', prompt: null, autopilot: false, timeoutSec: 900, ui: false, http: false }
  for (let i = 0; i < argv.length; i++) {
    const val = (): string => {
      const v = argv[++i]
      if (v === undefined) throw new Error(`${argv[i - 1]} needs a value`)
      return v
    }
    switch (argv[i]) {
      case '--url': o.url = val(); break
      case '--case': o.casePath = val(); break
      case '--prompt': o.prompt = val(); break
      case '--timeout': o.timeoutSec = Number(val()); break
      case '--autopilot': o.autopilot = true; break
      case '--ui': o.ui = true; break
      case '--http': o.http = true; break
      default: throw new Error(`unknown option ${argv[i]}\n${USAGE_LINE}`)
    }
  }
  if (!Number.isFinite(o.timeoutSec) || o.timeoutSec <= 0) throw new Error('--timeout wants a positive number of seconds')
  if (o.http && o.ui) throw new Error('--ui needs the WebSocket; drop --http or --ui')
  return o
}

// ---------------------------------------------------------------------------
// Console: streamed text goes out raw, everything else through say(), which
// first closes an open stream so lines never interleave mid-sentence.
// ---------------------------------------------------------------------------

const t0 = Date.now()

function stamp(): string {
  return `[${((Date.now() - t0) / 1000).toFixed(1).padStart(6)}s]`
}

let streamOpen: 'text' | 'thinking' | null = null

function beginStream(kind: 'text' | 'thinking'): void {
  if (streamOpen === kind) return
  if (streamOpen) process.stdout.write('\n')
  streamOpen = kind
  process.stdout.write(`${stamp()} ${kind === 'text' ? 'assistant' : 'thinking'}: `)
}

function endStream(): void {
  if (!streamOpen) return
  streamOpen = null
  process.stdout.write('\n')
}

function say(line: string): void {
  endStream()
  console.log(`${stamp()} ${line}`)
}

function emptyUsage(): Usage {
  return { inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0 }
}

function addUsage(total: Usage, u: Usage): void {
  total.inputTokens += u.inputTokens
  total.outputTokens += u.outputTokens
  total.cacheReadTokens += u.cacheReadTokens
  total.cacheWriteTokens += u.cacheWriteTokens
}

// ---------------------------------------------------------------------------
// The drive: connect, open a session, send the prompt, then print every frame
// that matters until the run exits and the assistant's turn ends.
// ---------------------------------------------------------------------------

interface Waiter {
  handle(m: ServerMsg): boolean
  fail(err: Error): void
}

async function drive(opts: Options): Promise<number> {
  const prompt = opts.prompt ?? `Run the case ${opts.casePath} with the default solver, wait for it to finish, then summarise the final residuals and whether it converged. Use gui_control to show the Velocity field when the run ends.`
  const promptWantsRun = /\brun\b|solver|솔버|실행/i.test(prompt)

  const state = {
    sessionId: null as string | null,
    /** Any run started since the prompt went out. */
    sawRun: false,
    lastRun: null as RunInfo | null,
    exitRun: null as RunInfo | null,
    lastRunPrintedAt: 0,
    lastAssistantText: '',
    usage: emptyUsage(),
    done: false,
  }

  const ws = new WebSocket(opts.url)
  const waiters = new Set<Waiter>()

  function waitFor<T>(what: string, timeoutMs: number, pick: (m: ServerMsg) => T | null): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const w: Waiter = {
        handle: (m) => {
          const v = pick(m)
          if (v === null) return false
          clearTimeout(timer)
          waiters.delete(w)
          resolve(v)
          return true
        },
        fail: (err: Error) => {
          clearTimeout(timer)
          waiters.delete(w)
          reject(err)
        },
      }
      const timer = setTimeout(() => w.fail(new Error(`timed out waiting for ${what}`)), timeoutMs)
      waiters.add(w)
    })
  }

  function send(msg: ClientMsg): void {
    const r = ClientMsgSchema.safeParse(msg)
    if (!r.success) {
      say(`!! refused to send invalid ${msg.t}: ${r.error.issues.map((i) => `${i.path.join('.')}: ${i.message}`).join('; ')}`)
      return
    }
    ws.send(JSON.stringify(r.data))
  }

  function verdict(): number {
    const answered = state.lastAssistantText.trim().length > 0
    // A prompt that asked for no run passes on the answer alone; asking for one and
    // never getting it is still a failure, so only a turn that ran nothing on purpose
    // is judged this way.
    if (!promptWantsRun && !state.sawRun) return answered ? 0 : 1
    const r = state.exitRun
    const runOk = r !== null && (r.status === 'done' || r.converged)
    return runOk && answered ? 0 : 1
  }

  let finishResolve: ((code: number) => void) | null = null
  const finished = new Promise<number>((resolve) => { finishResolve = resolve })

  function finish(reason: string): void {
    if (state.done || !finishResolve) return
    state.done = true
    say(`-- ${reason}`)
    const resolve = finishResolve
    finishResolve = null
    resolve(verdict())
  }

  function maybeFinish(): void {
    if (state.exitRun) finish('run exited and the assistant turn ended')
    else if (!state.sawRun && !promptWantsRun) finish('turn ended without any run')
  }

  function onFrame(raw: string): void {
    let parsed: unknown
    try {
      parsed = JSON.parse(raw)
    } catch {
      say('!! dropped a non-JSON frame')
      return
    }
    const r = ServerMsgSchema.safeParse(parsed)
    if (!r.success) {
      say(`!! dropped invalid frame: ${r.error.issues.map((i) => `${i.path.join('.')}: ${i.message}`).join('; ')}`)
      return
    }
    for (const w of [...waiters]) if (w.handle(r.data)) break
    handle(r.data)
  }

  ws.on('message', (data) => onFrame(String(data)))
  ws.on('close', () => {
    for (const w of [...waiters]) w.fail(new Error('connection closed'))
    finish('connection closed')
  })
  ws.on('error', () => { /* close follows and drives the finish */ })

  // -- ui stand-in -----------------------------------------------------------
  // A plausible projection of a studio screen; gui_control commands mutate it
  // and run frames keep its sim section moving, so gui_state reads something
  // true about the "screen" this script is hosting.
  const uiState: UiState = {
    activeTab: 'ai', activeStep: null, rightTab: null, tool: 'select', frame: null, projection: 'Perspective',
    showAxes: true, showColorBars: true, selection: null, runId: null, sim: null,
    case: null, tabs: [], run: null, viewer: null, problems: 0, connection: 'connected', locale: 'en',
  }
  let standInTab = 0
  let standInLayer = 0

  /** The dataset section of the stand-in screen, made the first time a result is opened. */
  function standInViewer(): NonNullable<UiState['viewer']> {
    uiState.viewer ??= { datasetId: null, field: null, time: null, colormap: null, range: null, representation: null, layers: [] }
    return uiState.viewer
  }

  /** The Geometry-tab section of the stand-in screen, made the first time a geometry command lands. */
  function standInGeometry(): NonNullable<UiState['geometry']> {
    uiState.geometry ??= { id: null, path: null, triangleCount: null, closed: null, openEdges: null, solids: [], selected: null, edits: 0, dirty: false }
    return uiState.geometry
  }

  function applyUiCommand(cmd: UiCommand): void {
    switch (cmd.type) {
      case 'select_tab': uiState.activeTab = cmd.tab; break
      case 'select_step': uiState.activeStep = cmd.step; break
      case 'open_panel': uiState.rightTab = cmd.panel; break
      case 'set_tool': uiState.tool = cmd.tool; break
      case 'set_projection': uiState.projection = cmd.projection; break
      case 'show_overlay':
        if (cmd.what === 'axes') uiState.showAxes = cmd.on
        else uiState.showColorBars = cmd.on
        break
      case 'show_field': standInViewer().field = cmd.field; break
      // The workspace commands: the stand-in screen has to *change*, or gui_state
      // reads back the same empty screen after every gui_control and the model
      // cannot tell a command that worked from one that did nothing.
      case 'open_case':
        uiState.case = { path: cmd.path, name: cmd.path.split(/[\\/]/).pop() ?? cmd.path, dirty: false }
        break
      case 'save_case':
        if (uiState.case) uiState.case = { ...uiState.case, dirty: false }
        break
      case 'open_tab': {
        standInTab += 1
        const id = `tab_${standInTab}`
        uiState.tabs = [...(uiState.tabs ?? []), { id, kind: cmd.kind, label: cmd.label ?? cmd.kind }]
        uiState.activeTab = id
        break
      }
      case 'close_tab':
        uiState.tabs = (uiState.tabs ?? []).filter((t) => t.id !== cmd.id)
        if (uiState.activeTab === cmd.id) uiState.activeTab = uiState.tabs.at(-1)?.id ?? 'ai'
        break
      case 'show_chart': uiState.activeTab = cmd.chart; break
      case 'open_result': {
        const v = standInViewer()
        v.datasetId = cmd.region ? `${cmd.path}#${cmd.region}` : cmd.path
        v.time = typeof cmd.timeIndex === 'number' ? cmd.timeIndex : v.time
        break
      }
      case 'post_warp': {
        const v = standInViewer()
        ;(v as { warp?: unknown }).warp = cmd.field ? { field: cmd.field, scale: cmd.scale ?? 1 } : null
        break
      }
      case 'set_post': {
        const v = standInViewer()
        if (cmd.colormap != null) v.colormap = cmd.colormap
        if (Array.isArray(cmd.range)) v.range = [cmd.range[0], cmd.range[1]]
        if (cmd.representation != null) v.representation = cmd.representation
        break
      }
      case 'add_layer': {
        standInLayer += 1
        const v = standInViewer()
        v.layers = [...(v.layers ?? []), { id: `${cmd.kind}_${standInLayer}`, type: cmd.kind, summary: JSON.stringify(cmd.args) }]
        break
      }
      case 'remove_layer': {
        const v = standInViewer()
        v.layers = (v.layers ?? []).filter((l) => l.id !== cmd.id)
        break
      }
      case 'set_locale': uiState.locale = cmd.locale; break
      // The Geometry tab: the stand-in answers with one plausible part so the
      // model can read back what it just asked the screen to do.
      case 'geometry_open':
      case 'geometry_import_step': {
        const g = standInGeometry()
        g.id = cmd.path
        g.path = cmd.path
        g.triangleCount = 0
        g.closed = true
        g.openEdges = 0
        g.solids = [{ name: 'body', triangles: 0, visible: true }]
        break
      }
      case 'geometry_part': {
        const g = standInGeometry()
        const solid = g.solids.find((s) => s.name === cmd.name) ?? g.solids[0]
        if (solid) {
          if (cmd.action === 'rename' && cmd.newName) solid.name = cmd.newName
          else if (cmd.action === 'select') { g.selected = cmd.name; solid.visible = true }
          else if (cmd.action === 'keep_only') for (const s of g.solids) s.visible = s === solid
          else if (cmd.action === 'drop') solid.visible = false
          else solid.visible = cmd.action === 'show'
        }
        break
      }
      case 'geometry_transform': {
        const g = standInGeometry()
        if (cmd.op === 'undo') g.edits = Math.max(0, g.edits - 1)
        else if (cmd.op === 'redo') g.edits += 1
        else if (cmd.op === 'reset') g.edits = 0
        else g.edits += 1
        g.dirty = true
        break
      }
      case 'geometry_boolean': {
        const g = standInGeometry()
        g.solids = [{ name: cmd.a, triangles: 0, visible: true }]
        break
      }
      case 'geometry_save': {
        const g = standInGeometry()
        g.path = cmd.path
        g.edits = 0
        g.dirty = false
        break
      }
      default: break
    }
  }

  function pushUiState(): void {
    if (!opts.ui) return
    send({ t: 'ui.state', state: { ...uiState } })
  }

  function syncUiRun(r: RunInfo): void {
    if (!opts.ui) return
    uiState.runId = r.id
    uiState.sim = { status: r.status, iteration: r.iter, maxIterations: r.targetIter }
    uiState.run = { id: r.id, status: r.status, iteration: r.iter, target: r.targetIter }
    pushUiState()
  }

  // -- frames that matter -----------------------------------------------------
  const blockKinds = new Map<string, 'text' | 'thinking'>()
  const printedStatus = new Map<string, string>()

  function handle(msg: ServerMsg): void {
    switch (msg.t) {
      case 'turn.start':
        say(`[turn] started (${msg.turnId})`)
        break
      case 'turn.done':
        addUsage(state.usage, msg.usage)
        say(`[turn] done model=${msg.model ?? '?'} usage in=${msg.usage.inputTokens} out=${msg.usage.outputTokens}`)
        maybeFinish()
        break
      case 'turn.error':
        say(`!! turn error: ${msg.message}`)
        if (!state.sawRun) finish('turn errored')
        break
      case 'turn.refusal':
        say(`!! turn refused (${msg.category ?? 'no category'})${msg.explanation ? `: ${msg.explanation}` : ''}`)
        if (!state.sawRun) finish('turn refused')
        break
      case 'turn.warning':
        say(`! ${msg.message}`)
        break
      case 'msg.user': {
        if (!msg.message.synthetic) break
        for (const b of msg.message.blocks) {
          if (b.kind === 'text') {
            say(`[notice] ${b.text.split('\n')[0]}`)
            break
          }
        }
        break
      }
      case 'msg.block_start':
        blockKinds.set(`${msg.messageId}:${msg.blockIndex}`, msg.kind)
        beginStream(msg.kind)
        break
      case 'msg.delta': {
        const kind = blockKinds.get(`${msg.messageId}:${msg.blockIndex}`)
        if (!kind) break
        if (streamOpen !== kind) beginStream(kind)
        process.stdout.write(msg.delta)
        break
      }
      case 'msg.done':
        endStream()
        if (msg.message.role === 'assistant') {
          state.lastAssistantText = msg.message.blocks.filter((b) => b.kind === 'text').map((b) => b.text).join('\n')
        }
        break
      case 'tool.update': {
        const c = msg.call
        if (printedStatus.get(c.toolUseId) === c.status) break
        printedStatus.set(c.toolUseId, c.status)
        say(`[tool] ${c.name} ${c.status}${c.summary ? ` — ${c.summary}` : ''}${c.error ? ` !! ${c.error}` : ''}`)
        break
      }
      case 'tool.approval_request': {
        const names = msg.approval.calls.map((c) => `${c.name}${c.preview ? ` (${c.preview.split('\n')[0]})` : ''}`).join(', ')
        say(`[approval] ${names}`)
        if (state.sessionId) send({ t: 'tool.approve', sessionId: state.sessionId, toolUseIds: msg.approval.calls.map((c) => c.toolUseId), remember: 'none' })
        break
      }
      case 'run.started':
        state.sawRun = true
        state.lastRun = msg.run
        state.exitRun = null
        send({ t: 'run.subscribe', runId: msg.run.id, fromSeq: 0 })
        say(`[run] ${msg.run.id} started: ${msg.run.binary} ${msg.run.casePath ?? ''}`)
        syncUiRun(msg.run)
        break
      case 'run.updated': {
        if (state.lastRun && msg.run.id !== state.lastRun.id) break
        state.lastRun = msg.run
        syncUiRun(msg.run)
        const now = Date.now()
        if (now - state.lastRunPrintedAt < 1000) break
        state.lastRunPrintedAt = now
        const r = msg.run
        say(`[run] ${r.id} iter ${r.iter}${r.targetIter !== null ? `/${r.targetIter}` : ''} ${r.status}${r.time !== null ? ` t=${r.time}s` : ''}`)
        break
      }
      case 'run.exit': {
        state.lastRun = msg.run
        state.exitRun = msg.run
        say(`[run] ${msg.run.id} exited: status=${msg.run.status} iterations=${msg.run.iter} converged=${msg.run.converged}${msg.run.error ? ` error="${msg.run.error}"` : ''}`)
        syncUiRun(msg.run)
        break
      }
      case 'ui.command':
        applyUiCommand(msg.cmd)
        say(`[ui] ${JSON.stringify(msg.cmd)} -> ok`)
        send({ t: 'ui.result', requestId: msg.requestId, ok: true, error: null })
        pushUiState()
        break
      case 'error':
        say(`!! server: ${msg.message}`)
        if (msg.fatal) finish('fatal server error')
        break
      default:
        break
    }
  }

  // -- handshake, prompt, wait ------------------------------------------------
  const opened = new Promise<void>((resolve, reject) => {
    ws.once('open', resolve)
    ws.once('error', reject)
  })
  say(`connecting to ${opts.url}`)
  await opened

  const hello = await waitFor<ServerHello>('hello', 10_000, (m) => (m.t === 'hello' ? m.hello : null))
  say(`server v${hello.version} mode=${hello.mode} llm=${hello.llm} model=${hello.model} gpu=${hello.gpu.state} workspace=${hello.workspaceRoot}`)

  send({ t: 'session.new' })
  const session = await waitFor('session.state', 10_000, (m) => (m.t === 'session.state' ? m.session : null))
  state.sessionId = session.id
  say(`session ${session.id} (autoApprove=${opts.autopilot ? 'all' : 'reads'})`)
  send({ t: 'settings.set', sessionId: session.id, patch: { autoApprove: opts.autopilot ? 'all' : 'reads' } })
  if (opts.ui) {
    say('standing in for the GUI: every ui.command is answered ok')
    pushUiState()
  }

  say(`prompt: ${prompt}`)
  send({
    t: 'user.message',
    sessionId: session.id,
    text: prompt,
    context: { activeFile: opts.casePath, activeRun: null, attachments: [opts.casePath], attachmentIds: [], selection: null },
  })

  const timeout = setTimeout(() => finish(`timeout after ${opts.timeoutSec}s`), opts.timeoutSec * 1000)
  const ping = setInterval(() => send({ t: 'ping', ts: Date.now() }), 20_000)
  const code = await finished
  clearTimeout(timeout)
  clearInterval(ping)
  ws.close()

  const r = state.exitRun
  const u = state.usage
  say(`VERDICT: ${r ? `run ${r.id} iterations=${r.iter} status=${r.status} converged=${r.converged}` : 'no run reached exit'} | final assistant text: ${state.lastAssistantText.trim() ? 'yes' : 'no'} | usage input=${u.inputTokens} output=${u.outputTokens} cacheRead=${u.cacheReadTokens} cacheWrite=${u.cacheWriteTokens}`)
  return code
}

// ---------------------------------------------------------------------------

/** ws://host:port/ws -> http://host:port, so --http needs no second flag. */
function httpBase(wsUrl: string): { base: string; token: string | null } {
  const u = new URL(wsUrl)
  const token = u.searchParams.get('token')
  u.protocol = u.protocol === 'wss:' ? 'https:' : 'http:'
  u.pathname = ''
  u.search = ''
  u.hash = ''
  return { base: u.toString().replace(/\/$/, ''), token }
}

/** The same conversation as drive(), over POST /api/chat. Returns the process exit code. */
async function driveHttp(opts: Options): Promise<number> {
  const prompt = opts.prompt ?? `Run the case ${opts.casePath} with the default solver, wait for it to finish, then summarise the final residuals and whether it converged. Use gui_control to show the Velocity field when the run ends.`
  const promptWantsRun = /\brun\b|solver|솔버|실행/i.test(prompt)
  const { base, token } = httpBase(opts.url)
  const auth: Record<string, string> = token ? { authorization: `Bearer ${token}` } : {}
  say(`connecting to ${base} (http; any token in the url is sent, never logged)`)

  const health = await fetch(`${base}/api/health`, { headers: auth })
  const healthBody = (await health.json().catch(() => ({}))) as { error?: string }
  if (!health.ok) {
    say(`!! /api/health ${health.status}: ${healthBody.error ?? 'no error in body'}`)
    return 2
  }
  const hello = await fetch(`${base}/api/hello`, { headers: auth })
  const helloBody = (await hello.json().catch(() => null)) as ServerHello | null
  if (!hello.ok || !helloBody) {
    say(`!! /api/hello ${hello.status}`)
    return 2
  }
  say(`server v${helloBody.version} mode=${helloBody.mode} llm=${helloBody.llm} model=${helloBody.model} gpu=${helloBody.gpu.state} workspace=${helloBody.workspaceRoot}`)

  say(`prompt: ${prompt}`)
  const started = Date.now()
  const res = await fetch(`${base}/api/chat`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', ...auth },
    body: JSON.stringify({
      text: prompt,
      attachments: [opts.casePath],
      activeFile: opts.casePath,
      autoApprove: opts.autopilot ? 'all' : 'reads',
      timeoutMs: Math.min(opts.timeoutSec * 1000, CHAT_TIMEOUT_MAX_MS),
    }),
  })
  const resBody = (await res.json().catch(() => null)) as (ChatResponse & { error?: string }) | null
  if (!res.ok || !resBody) {
    say(`!! POST /api/chat ${res.status}: ${resBody?.error ?? 'no error in body'}`)
    return 2
  }
  let body: ChatResponse = resBody

  let printed = 0
  let lastAssistantText = ''
  const printMessages = (messages: UiMessage[]): void => {
    for (const m of messages) {
      for (const b of m.blocks) {
        if (b.kind === 'text') {
          beginStream('text')
          process.stdout.write(b.text)
          endStream()
          if (m.role === 'assistant') lastAssistantText += b.text
        } else if (b.kind === 'thinking') say(b.text)
        else if (b.kind === 'tool') say(`[tool] ${b.call.name} ${b.call.status} — ${b.call.summary}`)
        else if (b.kind === 'notice') say(`[notice] ${b.text}`)
        else if (b.kind === 'image') say(`[image] ${b.mime} ${b.base64.length} bytes`)
        else if (b.kind === 'diff') say(`[diff] ${b.path} (${b.applied ? 'applied' : 'not applied'})`)
      }
      printed++
    }
  }
  printMessages(body.messages)

  // 'timeout' means the turn is still running: poll until it ends, as the brief's caller does.
  if (body.status === 'timeout') {
    const deadline = started + opts.timeoutSec * 1000
    let state: SessionState | null = null
    while (Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 2000))
      const st = await fetch(`${base}/api/sessions/${body.sessionId}`, { headers: auth })
      if (!st.ok) break
      const s = (await st.json()) as SessionState
      if (!s.turnActive) {
        state = s
        break
      }
    }
    if (state) {
      printMessages(state.messages.slice(printed))
      body = { ...body, status: 'done' }
    } else {
      say(`-- timeout after ${opts.timeoutSec}s with the turn still running`)
    }
  }

  let exitRun: RunInfo | null = null
  for (const id of body.runs) {
    const rr = await fetch(`${base}/api/runs/${id}`, { headers: auth })
    if (!rr.ok) continue
    const run = (await rr.json()) as RunInfo
    exitRun = run
    say(`[run] ${run.id} exited: status=${run.status} iterations=${run.iter} converged=${run.converged}${run.error ? ` error="${run.error}"` : ''}`)
  }

  const answered = lastAssistantText.trim().length > 0
  const sawRun = body.runs.length > 0
  let code: number
  if (!promptWantsRun && !sawRun) {
    // A prompt that asked for no run passes on the answer alone.
    code = answered ? 0 : 1
  } else {
    const runOk = exitRun !== null && (exitRun.status === 'done' || exitRun.converged)
    code = runOk && answered ? 0 : 1
  }
  const u = body.usage
  say(`VERDICT: ${exitRun ? `run ${exitRun.id} iterations=${exitRun.iter} status=${exitRun.status} converged=${exitRun.converged}` : 'no run reached exit'} | final assistant text: ${answered ? 'yes' : 'no'} | usage input=${u.inputTokens} output=${u.outputTokens} cacheRead=${u.cacheReadTokens} cacheWrite=${u.cacheWriteTokens}`)
  return code
}

async function main(): Promise<number> {
  const opts = parseOptions(process.argv.slice(2))
  return opts.http ? driveHttp(opts) : drive(opts)
}

main()
  .then((code) => process.exit(code))
  .catch((err: unknown) => {
    endStream()
    console.error(`ai-drive failed: ${err instanceof Error ? err.message : String(err)}`)
    process.exit(2)
  })
