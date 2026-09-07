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

const schedule: (fn: () => void) => void = typeof requestAnimationFrame === 'function' ? (fn) => requestAnimationFrame(() => fn()) : (fn) => setTimeout(fn, 16)

export function createWsClient(url: string = wsUrlFor(window.location)): WsClient {
  let ws: WebSocket | null = null
  let attempt = 0
  let closedByUser = false
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null
  let pingTimer: ReturnType<typeof setInterval> | null = null
  let queue: ServerMsg[] = []
  let flushScheduled = false

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
    send({ t: 'run.subscribe', runId, fromSeq: (data?.lastLogSeq ?? 0) + 1 })
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
      const known = session.getState().runs
      session.getState().reconcileRuns(list)
      for (const r of list) if (!known[r.id]) subscribeRun(r.id)
    } catch (err) {
      console.warn('ws: run reconcile failed', err)
    }
  }

  function onHello() {
    attempt = 0
    session.getState().setConnection('online')
    openBestSession()
    for (const runId of Object.keys(session.getState().runs)) subscribeRun(runId)
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
    if (flushScheduled) return
    flushScheduled = true
    schedule(flush)
  }

  function stopTimers() {
    if (pingTimer) clearInterval(pingTimer)
    pingTimer = null
    if (reconnectTimer) clearTimeout(reconnectTimer)
    reconnectTimer = null
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

  return {
    viewer,
    connect() {
      closedByUser = false
      if (ws) return
      open()
    },
    close() {
      closedByUser = true
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
