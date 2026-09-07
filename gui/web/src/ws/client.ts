// The single WebSocket to /ws. Frames are validated both ways, batched onto
// animation frames into the session store, and the reducer's effects are
// executed here. Reconnects with exponential backoff and replays runs.
import type { ClientMsg, ServerMsg } from '@cfd/shared'
import { api } from '../api/rest'
import { useEditorStore } from '../state/editorStore'
import { fsEvents } from '../state/fsEvents'
import { useSessionStore } from '../state/sessionStore'
import type { Effect } from '../state/types'
import { useUiStore } from '../state/uiStore'
import { backoffDelay, encodeClientFrame, parseServerFrame, wsUrlFor } from './frames'
import { createViewerBridge, type ViewerBridge } from './viewerBridge'

export const PING_INTERVAL_MS = 20_000
const SESSION_KEY = 'cfd-studio.session'

export interface WsClient {
  connect(): void
  close(): void
  send(msg: ClientMsg): boolean
  isOpen(): boolean
  readonly viewer: ViewerBridge
}

function readStoredSession(): string | null {
  try {
    return localStorage.getItem(SESSION_KEY)
  } catch {
    return null
  }
}

function storeSession(id: string | null): void {
  try {
    if (id) localStorage.setItem(SESSION_KEY, id)
    else localStorage.removeItem(SESSION_KEY)
  } catch {
    // storage unavailable: session simply is not remembered
  }
}

/** Longest a frame waits when requestAnimationFrame is not running (a hidden tab). */
const HIDDEN_FLUSH_MS = 50

/**
 * Frames the UI must not sit on. viewer.command has a server waiting on its
 * reply, tool.approval_request holds a turn until the user answers, and hello
 * is what opens the session.
 */
const FLUSH_NOW = new Set<ServerMsg['t']>(['hello', 'viewer.command', 'tool.approval_request', 'error'])

export function createWsClient(url: string = wsUrlFor(window.location)): WsClient {
  let ws: WebSocket | null = null
  let attempt = 0
  let closedByUser = false
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null
  let pingTimer: ReturnType<typeof setInterval> | null = null
  let queue: ServerMsg[] = []
  let flushScheduled = false
  let flushTimer: ReturnType<typeof setTimeout> | null = null
  const subscribed = new Set<string>()

  const session = useSessionStore
  const ui = useUiStore

  function send(msg: ClientMsg): boolean {
    if (!ws || ws.readyState !== WebSocket.OPEN) return false
    const text = encodeClientFrame(msg)
    if (text === null) return false
    ws.send(text)
    return true
  }

  const viewer = createViewerBridge(send, ui)

  function subscribeRun(runId: string) {
    const data = session.getState().runData[runId]
    subscribed.add(runId)
    send({ t: 'run.subscribe', runId, fromSeq: (data?.lastLogSeq ?? 0) + 1 })
  }

  /** Runs the UI is showing right now: the run panel, the terminal, the residuals tab. */
  function displayedRunIds(): Set<string> {
    const u = ui.getState()
    const out = new Set<string>()
    for (const id of [u.activeRunId, u.terminalRunId]) if (id) out.add(id)
    for (const tab of u.tabs) if (tab.kind === 'residuals' && tab.runId) out.add(tab.runId)
    return out
  }

  /**
   * The live runs and the displayed ones, and nothing else. Subscribing to a
   * finished run replays up to 5,000 log lines and its whole residual series;
   * the server restores every past run from gui/runs on startup, so doing that
   * for all of them on every connect and every reconnect was the cost of
   * opening a tab.
   */
  function wantedRunIds(): Set<string> {
    const want = displayedRunIds()
    for (const r of Object.values(session.getState().runs)) if (r.status === 'running' || r.status === 'queued') want.add(r.id)
    return want
  }

  function syncRunSubscriptions() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    const want = wantedRunIds()
    for (const id of want) if (!subscribed.has(id)) subscribeRun(id)
    for (const id of [...subscribed]) {
      if (want.has(id)) continue
      subscribed.delete(id)
      send({ t: 'run.unsubscribe', runId: id })
    }
  }

  function openBestSession() {
    const s = session.getState()
    const wanted = s.currentSessionId ?? readStoredSession()
    const sessions = s.sessions
    if (wanted && sessions.some((x) => x.id === wanted)) {
      send({ t: 'session.open', sessionId: wanted })
      return
    }
    if (sessions.length) {
      const latest = sessions.slice().sort((a, b) => (a.updatedAt < b.updatedAt ? 1 : -1))[0]
      send({ t: 'session.open', sessionId: latest.id })
      return
    }
    send({ t: 'session.new' })
  }

  async function reconcileRuns() {
    try {
      const list = await api.runs()
      session.getState().reconcileRuns(list)
      syncRunSubscriptions()
    } catch (err) {
      console.warn('ws: run reconcile failed', err)
    }
  }

  function onHello() {
    attempt = 0
    session.getState().setConnection('online')
    openBestSession()
    // A new socket carries none of the old socket's subscriptions.
    subscribed.clear()
    syncRunSubscriptions()
    void reconcileRuns()
    viewer.attach()
  }

  function runEffects(effects: Effect[]) {
    for (const e of effects) {
      switch (e.type) {
        case 'hello':
          onHello()
          break
        case 'run.subscribe':
          subscribeRun(e.runId)
          break
        case 'viewer.command':
          void viewer.handleCommand(e.requestId, e.cmd)
          break
        case 'viewer.open':
          void viewer.openPath(e.path, e.runId).then((r) => {
            if (!r.ok && r.error) session.getState().addNote('warning', `viewer: ${r.error.code}: ${r.error.message}`)
          })
          break
        case 'residuals.open':
          ui.getState().openResidualsTab(e.runId)
          break
        case 'fs.changed':
          useEditorStore.getState().onFsChanged(e.paths)
          fsEvents.emit(e.paths)
          break
        case 'session.deleted':
          storeSession(null)
          openBestSession()
          break
        case 'fatal':
          session.getState().addNote('error', e.message)
          break
      }
    }
  }

  function flush() {
    flushScheduled = false
    if (flushTimer) {
      clearTimeout(flushTimer)
      flushTimer = null
    }
    if (!queue.length) return
    const batch = queue
    queue = []
    const before = session.getState().currentSessionId
    let effects: Effect[] = []
    try {
      effects = session.getState().dispatch(batch)
    } catch (err) {
      console.error('ws: reducer failed on a batch of', batch.length, 'frames', err)
      return
    }
    const after = session.getState().currentSessionId
    if (after && after !== before) storeSession(after)
    runEffects(effects)
  }

  function enqueue(msg: ServerMsg) {
    queue.push(msg)
    if (FLUSH_NOW.has(msg.t)) {
      flush()
      return
    }
    if (flushScheduled) return
    flushScheduled = true
    // requestAnimationFrame does not fire in a hidden tab, and it was the only
    // thing draining this queue: a backgrounded studio processed nothing at all
    // and the queue grew without bound. The timer runs either way; rAF only
    // makes the visible case land on a paint, and whichever fires first clears
    // the other.
    flushTimer = setTimeout(flush, HIDDEN_FLUSH_MS)
    if (typeof requestAnimationFrame === 'function') requestAnimationFrame(() => flushScheduled && flush())
  }

  function stopTimers() {
    if (pingTimer) clearInterval(pingTimer)
    pingTimer = null
    if (reconnectTimer) clearTimeout(reconnectTimer)
    reconnectTimer = null
    if (flushTimer) clearTimeout(flushTimer)
    flushTimer = null
  }

  function scheduleReconnect() {
    if (closedByUser || reconnectTimer) return
    const delay = backoffDelay(attempt++)
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null
      open()
    }, delay)
  }

  function open() {
    if (closedByUser) return
    session.getState().setConnection(attempt === 0 ? 'connecting' : 'offline')
    let sock: WebSocket
    try {
      sock = new WebSocket(url)
    } catch (err) {
      console.warn('ws: cannot open socket', err)
      scheduleReconnect()
      return
    }
    ws = sock
    sock.onopen = () => {
      if (pingTimer) clearInterval(pingTimer)
      pingTimer = setInterval(() => send({ t: 'ping', ts: Date.now() }), PING_INTERVAL_MS)
    }
    sock.onmessage = (ev) => {
      const msg = parseServerFrame(ev.data)
      if (msg) enqueue(msg)
    }
    sock.onclose = () => {
      if (ws !== sock) return
      ws = null
      if (pingTimer) clearInterval(pingTimer)
      pingTimer = null
      session.getState().setConnection('offline')
      scheduleReconnect()
    }
    sock.onerror = () => {
      // the close event that follows drives the reconnect
    }
  }

  // Selecting a finished run in the UI is what subscribes to it, and moving off
  // it is what unsubscribes.
  const stopWatchingUi = ui.subscribe(syncRunSubscriptions)

  return {
    viewer,
    connect() {
      closedByUser = false
      if (ws) return
      open()
    },
    close() {
      closedByUser = true
      stopWatchingUi()
      subscribed.clear()
      stopTimers()
      viewer.detach()
      const sock = ws
      ws = null
      sock?.close()
      session.getState().setConnection('offline')
    },
    send,
    isOpen: () => ws?.readyState === WebSocket.OPEN,
  }
}

let client: WsClient | null = null

export function getWsClient(): WsClient {
  if (!client) client = createWsClient()
  return client
}
