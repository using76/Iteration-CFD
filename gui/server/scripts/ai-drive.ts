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
  ClientMsgSchema,
  ServerMsgSchema,
  type ClientMsg,
  type RunInfo,
  type ServerHello,
  type ServerMsg,
  type UiCommand,
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
}

const USAGE_LINE = 'usage: npx tsx server/scripts/ai-drive.ts [--url ws://127.0.0.1:$CFD_PORT/ws] [--case cases/plume.jsonc] [--prompt "<text>"] [--autopilot] [--timeout 900] [--ui]'

/** The server this drives is the one CFD_PORT names, so the two agree without a flag. */
function defaultUrl(): string {
  const port = Number(process.env.CFD_PORT)
  return `ws://127.0.0.1:${Number.isInteger(port) && port > 0 ? port : 8787}/ws`
}

function parseOptions(argv: string[]): Options {
  const o: Options = { url: defaultUrl(), casePath: 'cases/plume.jsonc', prompt: null, autopilot: false, timeoutSec: 900, ui: false }
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
      default: throw new Error(`unknown option ${argv[i]}\n${USAGE_LINE}`)
    }
  }
  if (!Number.isFinite(o.timeoutSec) || o.timeoutSec <= 0) throw new Error('--timeout wants a positive number of seconds')
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
  const uiState: UiState = { activeTab: 'ai', activeStep: null, rightTab: null, tool: 'select', frame: null, projection: 'Perspective', showAxes: true, showColorBars: true, selection: null, runId: null, sim: null }

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
    context: { activeFile: opts.casePath, activeRun: null, attachments: [opts.casePath], selection: null },
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

async function main(): Promise<number> {
  const opts = parseOptions(process.argv.slice(2))
  return drive(opts)
}

main()
  .then((code) => process.exit(code))
  .catch((err: unknown) => {
    endStream()
    console.error(`ai-drive failed: ${err instanceof Error ? err.message : String(err)}`)
    process.exit(2)
  })
