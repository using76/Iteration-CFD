// The WebSocket hub: one connection per browser tab. Validates every client
// frame with the shared zod schema, tracks which session/runs/viewer each
// client has, replays run logs on subscribe, batches log frames, brokers
// viewer commands to the most recently active viewer, and drops clients
// that fall silent.
import type { IncomingMessage } from 'node:http'
import { WebSocketServer, type WebSocket } from 'ws'
import {
  ClientMsgSchema,
  type ClientMsg,
  type LogLine,
  type RunInfo,
  type ServerHello,
  type ServerMsg,
  type SessionSummary,
  type UiCommand,
  type UiState,
  type ViewerCommand,
  type ViewerResult,
  type ViewerState,
} from '@cfd/shared'
import { silentLogger, type Logger } from '../log.js'
import type { RunManager } from '../runs/types.js'
import type { ClientConn, ClientMsgHandler, Hub, UiRequestResult } from './types.js'

export const HEARTBEAT_MS = 20_000
export const IDLE_CLOSE_MS = 60_000
export const LOG_FLUSH_MS = 50
export const REPLAY_BATCH = 500
export const NO_VIEWER_WAIT_MS = 5_000
export const VIEWER_TIMEOUT_MS = 30_000
export const UI_TIMEOUT_MS = 5_000

export interface HubDeps {
  hello(): ServerHello
  sessions(): SessionSummary[]
  runs: Pick<RunManager, 'get' | 'list' | 'log' | 'residuals' | 'metrics' | 'stop'>
  log?: Logger
  heartbeatMs?: number
  idleMs?: number
}

export interface HubHandle extends Hub {
  /** Adopt an upgraded socket. */
  accept(ws: WebSocket, req: IncomingMessage): ClientConn
  /** The ws server for `handleUpgrade`. */
  wss: WebSocketServer
  close(): void
}

interface PendingViewer {
  clientId: string
  resolve: (r: ViewerResult) => void
  timer: NodeJS.Timeout
}

interface PendingUi {
  clientId: string
  resolve: (r: UiRequestResult) => void
  timer: NodeJS.Timeout
}

interface Client extends ClientConn {
  ws: WebSocket
  lastActive: number
  lastViewerAt: number
  lastUiAt: number
  logBatches: Map<string, { lines: LogLine[]; timer: NodeJS.Timeout }>
}

function viewerError(code: ViewerResult['error'] extends infer E ? (E extends { code: infer C } ? C : never) : never, message: string): ViewerResult {
  return { ok: false, state: null, error: { code, message }, image: null }
}

function uiError(code: string, message: string): UiRequestResult {
  return { ok: false, state: null, error: { code, message } }
}

export function createHub(deps: HubDeps): HubHandle {
  const log = deps.log ?? silentLogger
  const clients = new Map<string, Client>()
  const msgHandlers = new Set<ClientMsgHandler>()
  const openHandlers = new Set<(c: ClientConn) => void>()
  const closeHandlers = new Set<(c: ClientConn) => void>()
  const pendingViewer = new Map<string, PendingViewer>()
  const viewerWaiters = new Set<(c: Client | null) => void>()
  const pendingUi = new Map<string, PendingUi>()
  let counter = 0
  const wss = new WebSocketServer({ noServer: true })

  const heartbeat = setInterval(() => {
    const now = Date.now()
    for (const c of clients.values()) {
      if (now - c.lastActive > (deps.idleMs ?? IDLE_CLOSE_MS)) {
        log.info(`client ${c.id} idle for ${Math.round((now - c.lastActive) / 1000)} s; closing`)
        c.ws.terminate()
        continue
      }
      if (c.ws.readyState === c.ws.OPEN) c.ws.ping()
    }
  }, deps.heartbeatMs ?? HEARTBEAT_MS)
  heartbeat.unref()

  function rawSend(c: Client, msg: ServerMsg) {
    if (c.ws.readyState !== c.ws.OPEN) return
    c.ws.send(JSON.stringify(msg))
  }

  function queueLog(c: Client, runId: string, lines: LogLine[]) {
    const b = c.logBatches.get(runId)
    if (b) {
      b.lines.push(...lines)
      return
    }
    const timer = setTimeout(() => {
      const batch = c.logBatches.get(runId)
      c.logBatches.delete(runId)
      if (batch && batch.lines.length) rawSend(c, { t: 'run.log', runId, lines: batch.lines })
    }, LOG_FLUSH_MS)
    c.logBatches.set(runId, { lines: [...lines], timer })
  }

  function send(c: Client, msg: ServerMsg) {
    if (msg.t === 'run.log') queueLog(c, msg.runId, msg.lines)
    else rawSend(c, msg)
  }

  function sendError(c: Client, message: string, fatal = false) {
    rawSend(c, { t: 'error', message, fatal })
  }

  function replayRun(c: Client, runId: string, fromSeq: number) {
    const info = deps.runs.get(runId)
    if (!info) {
      sendError(c, `no such run: ${runId}`)
      return
    }
    let seq = Math.max(1, fromSeq)
    for (;;) {
      const w = deps.runs.log(runId, seq, REPLAY_BATCH)
      if (w.lines.length) rawSend(c, { t: 'run.log', runId, lines: w.lines })
      if (w.lines.length < REPLAY_BATCH || w.nextSeq <= seq) break
      seq = w.nextSeq
    }
    for (const rec of deps.runs.residuals(runId)) rawSend(c, { t: 'run.residual', runId, rec })
    for (const rec of deps.runs.metrics(runId)) rawSend(c, { t: 'run.metric', runId, rec })
    rawSend(c, { t: 'run.updated', run: info })
  }

  function resolveViewer(requestId: string, result: ViewerResult) {
    const p = pendingViewer.get(requestId)
    if (!p) return
    pendingViewer.delete(requestId)
    clearTimeout(p.timer)
    p.resolve(result)
  }

  function noteViewer(c: Client, state: ViewerState) {
    c.viewerState = state
    c.lastViewerAt = Date.now()
    for (const w of viewerWaiters) w(c)
  }

  function resolveUi(requestId: string, result: UiRequestResult) {
    const p = pendingUi.get(requestId)
    if (!p) return
    pendingUi.delete(requestId)
    clearTimeout(p.timer)
    p.resolve(result)
  }

  function noteUi(c: Client, state: UiState) {
    c.uiState = state
    c.lastUiAt = Date.now()
  }

  async function dispatch(c: Client, msg: ClientMsg) {
    switch (msg.t) {
      case 'ping':
        rawSend(c, { t: 'pong', ts: msg.ts })
        return
      case 'run.subscribe':
        c.runs.add(msg.runId)
        replayRun(c, msg.runId, msg.fromSeq)
        return
      case 'run.unsubscribe':
        c.runs.delete(msg.runId)
        return
      case 'run.stop':
        try {
          await deps.runs.stop(msg.runId)
        } catch (err) {
          sendError(c, (err as Error).message)
        }
        return
      case 'viewer.state':
        noteViewer(c, msg.state)
        return
      case 'viewer.result':
        if (msg.result.state) noteViewer(c, msg.result.state)
        resolveViewer(msg.requestId, msg.result)
        return
      case 'ui.state':
        noteUi(c, msg.state)
        return
      case 'ui.result':
        // the frame's own state is the post-command screen; c.uiState is only the last
        // throttled report, which for a command inside the 250 ms window is the old one
        if (msg.state) noteUi(c, msg.state)
        resolveUi(msg.requestId, { ok: msg.ok, state: msg.state ?? c.uiState ?? null, error: msg.ok ? null : { code: 'UI_ERROR', message: msg.error ?? 'the UI rejected the command' } })
        return
      default:
        break
    }
    // session.open is the only frame that says which session this tab is
    // looking at. Every other frame carrying a sessionId names a target -
    // session.delete, session.rename, settings.set for a session in the history
    // menu - and re-pointing the connection at it silently cut the tab off from
    // the stream of the session actually on screen. session.new sets it from
    // the agent's reply.
    if (msg.t === 'session.open' && typeof msg.sessionId === 'string') c.sessionId = msg.sessionId
    if (msgHandlers.size === 0) {
      sendError(c, `no handler for ${msg.t}`)
      return
    }
    for (const h of msgHandlers) {
      try {
        await h(c, msg)
      } catch (err) {
        log.warn(`client message handler failed (${msg.t}): ${(err as Error).message}`)
        sendError(c, `${msg.t}: ${(err as Error).message}`)
      }
    }
  }

  function accept(ws: WebSocket, req: IncomingMessage): ClientConn {
    const id = `c_${Date.now().toString(36)}_${(++counter).toString(36)}`
    const client: Client = {
      id,
      sessionId: null,
      runs: new Set(),
      viewerState: null,
      uiState: null,
      ws,
      lastActive: Date.now(),
      lastViewerAt: 0,
      lastUiAt: 0,
      logBatches: new Map(),
      send: (msg) => send(client, msg),
      close: (code, reason) => ws.close(code, reason),
    }
    clients.set(id, client)
    log.debug(`client ${id} connected from ${req.socket.remoteAddress ?? '?'}`)

    ws.on('pong', () => {
      client.lastActive = Date.now()
    })
    ws.on('message', (data) => {
      client.lastActive = Date.now()
      let parsed: unknown
      try {
        parsed = JSON.parse(String(data))
      } catch {
        sendError(client, 'invalid JSON frame')
        return
      }
      const r = ClientMsgSchema.safeParse(parsed)
      if (!r.success) {
        const t = (parsed as { t?: unknown } | null)?.t
        sendError(client, `invalid message${typeof t === 'string' ? ` ${t}` : ''}: ${r.error.issues.map((i) => `${i.path.join('.')}: ${i.message}`).join('; ')}`)
        return
      }
      void dispatch(client, r.data)
    })
    ws.on('close', () => {
      clients.delete(id)
      for (const b of client.logBatches.values()) clearTimeout(b.timer)
      for (const [reqId, p] of pendingViewer) if (p.clientId === id) resolveViewer(reqId, viewerError('NO_VIEWER', 'the viewer client disconnected'))
      for (const [reqId, p] of pendingUi) if (p.clientId === id) resolveUi(reqId, uiError('NO_UI', 'the UI client disconnected'))
      for (const h of closeHandlers) {
        try {
          h(client)
        } catch (err) {
          log.warn(`close handler failed: ${(err as Error).message}`)
        }
      }
      log.debug(`client ${id} disconnected`)
    })
    ws.on('error', (err) => log.debug(`client ${id} socket error: ${err.message}`))

    try {
      rawSend(client, { t: 'hello', hello: deps.hello(), sessions: deps.sessions(), runs: deps.runs.list() as RunInfo[] })
    } catch (err) {
      log.warn(`hello failed: ${(err as Error).message}`)
    }
    for (const h of openHandlers) {
      try {
        h(client)
      } catch (err) {
        log.warn(`open handler failed: ${(err as Error).message}`)
      }
    }
    return client
  }

  /**
   * The clients a command for `sessionId` may land on. A session's commands go
   * only to the tabs that have that session open: falling back to "any open
   * client" let a turn whose own window had gone (a tab closed mid-turn, a
   * headless ai-drive, another operator's tab) drive a stranger's screen -
   * replacing their dataset and layers mid-sequence, which their own model then
   * read as its commands failing. Without a session (a test, a broadcast-style
   * caller) every open client is still a candidate.
   */
  function candidatesFor(sessionId: string | null | undefined): Client[] {
    const open = [...clients.values()].filter((c) => c.ws.readyState === c.ws.OPEN)
    return sessionId ? open.filter((c) => c.sessionId === sessionId) : open
  }

  function pickViewer(sessionId: string | null | undefined): Client | null {
    // Prefer a client that already reported a viewer state; the shell opens
    // the viewer tab on demand, so any client of the session can still host
    // the command.
    const candidates = candidatesFor(sessionId)
    if (!candidates.length) return null
    const withViewer = candidates.filter((c) => c.viewerState !== null)
    const pool = withViewer.length ? withViewer : candidates
    const score = (c: Client) => (c.viewerState ? 1e12 : 0) + Math.max(c.lastActive, c.lastViewerAt)
    pool.sort((a, b) => score(b) - score(a))
    return pool[0]
  }

  function waitForViewer(ms: number, sessionId: string | null | undefined): Promise<Client | null> {
    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        viewerWaiters.delete(waiter)
        resolve(pickViewer(sessionId))
      }, ms)
      const waiter = (c: Client | null) => {
        if (!c) return
        // a viewer that reported for another session is not this session's
        if (sessionId && c.sessionId !== sessionId) return
        viewerWaiters.delete(waiter)
        clearTimeout(timer)
        resolve(pickViewer(sessionId) ?? c)
      }
      viewerWaiters.add(waiter)
    })
  }

  // Every browser tab hosts the studio UI, so unlike pickViewer there is no
  // "has it mounted" tier — but a tab that has actually reported a uiState is
  // the one whose bridge demonstrably answers. A stale window loaded before a
  // code change never reports, never answers, and would otherwise win on
  // freshness alone and eat every screen command until someone reloads it.
  function pickUi(sessionId: string | null | undefined): Client | null {
    const candidates = candidatesFor(sessionId)
    if (!candidates.length) return null
    const withUi = candidates.filter((c) => c.uiState !== null)
    const pool = withUi.length ? withUi : candidates
    const score = (c: Client) => Math.max(c.lastActive, c.lastUiAt)
    pool.sort((a, b) => score(b) - score(a))
    return pool[0]
  }

  const hub: HubHandle = {
    wss,
    accept,
    broadcast(msg) {
      for (const c of clients.values()) send(c, msg)
    },
    sendToSession(sessionId, msg) {
      for (const c of clients.values()) if (c.sessionId === sessionId) send(c, msg)
    },
    sendToRun(runId, msg) {
      for (const c of clients.values()) if (c.runs.has(runId)) send(c, msg)
    },
    clients: () => [...clients.values()],
    hasViewerClient: () => [...clients.values()].some((c) => c.viewerState !== null),
    async requestViewer(cmd: ViewerCommand, opts = {}) {
      const timeoutMs = opts.timeoutMs ?? VIEWER_TIMEOUT_MS
      const client = pickViewer(opts.sessionId) ?? (await waitForViewer(Math.min(NO_VIEWER_WAIT_MS, timeoutMs), opts.sessionId))
      if (!client) return viewerError('NO_VIEWER', opts.sessionId ? 'no window with this session open has the 3D viewer; the operator may have closed it' : 'no connected client has the 3D viewer open')
      const requestId = `vc_${Date.now().toString(36)}_${(++counter).toString(36)}`
      return new Promise<ViewerResult>((resolve) => {
        const timer = setTimeout(() => {
          pendingViewer.delete(requestId)
          resolve(viewerError('TIMEOUT', `the viewer did not answer ${cmd.type} within ${timeoutMs} ms`))
        }, timeoutMs)
        pendingViewer.set(requestId, { clientId: client.id, resolve, timer })
        rawSend(client, { t: 'viewer.command', requestId, cmd })
      })
    },
    getUiState(sessionId) {
      const client = pickUi(sessionId)
      return client?.uiState ?? null
    },
    requestUi(cmd: UiCommand, opts = {}) {
      const timeoutMs = opts.timeoutMs ?? UI_TIMEOUT_MS
      const client = pickUi(opts.sessionId)
      if (!client) return Promise.resolve(uiError('NO_UI', opts.sessionId ? 'no window has this session open; the operator may have closed it, so the screen cannot be driven from here' : 'no connected client has the studio UI open'))
      const requestId = `ui_${Date.now().toString(36)}_${(++counter).toString(36)}`
      return new Promise<UiRequestResult>((resolve) => {
        const timer = setTimeout(() => {
          pendingUi.delete(requestId)
          resolve(uiError('TIMEOUT', `the UI did not answer ${cmd.type} within ${timeoutMs} ms`))
        }, timeoutMs)
        pendingUi.set(requestId, { clientId: client.id, resolve, timer })
        rawSend(client, { t: 'ui.command', requestId, cmd })
      })
    },
    onClientMessage(handler) {
      msgHandlers.add(handler)
      return () => msgHandlers.delete(handler)
    },
    onClientOpen(handler) {
      openHandlers.add(handler)
      return () => openHandlers.delete(handler)
    },
    onClientClose(handler) {
      closeHandlers.add(handler)
      return () => closeHandlers.delete(handler)
    },
    close() {
      clearInterval(heartbeat)
      for (const c of clients.values()) c.ws.close(1001, 'server shutting down')
      for (const [id, p] of pendingViewer) resolveViewer(id, viewerError('NO_VIEWER', 'server shutting down'))
      for (const [id, p] of pendingUi) resolveUi(id, uiError('NO_UI', 'server shutting down'))
      wss.close()
    },
  }
  return hub
}
