// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The ui watcher's lifetime (StrictMode runs connect; close; connect on one
// client) and the session.new locale, against a fake WebSocket and the real
// stores, in node.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const memory = new Map<string, string>()
if (typeof globalThis.localStorage === 'undefined') {
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    value: {
      getItem: (k: string) => (memory.has(k) ? (memory.get(k) as string) : null),
      setItem: (k: string, v: string) => void memory.set(k, String(v)),
      removeItem: (k: string) => void memory.delete(k),
      clear: () => void memory.clear(),
    },
  })
}
// zustand's default persist storage reads window.localStorage.
if (typeof globalThis.window === 'undefined') Object.defineProperty(globalThis, 'window', { configurable: true, value: globalThis })

const sockets: FakeWebSocket[] = []
class FakeWebSocket {
  static CONNECTING = 0
  static OPEN = 1
  static CLOSING = 2
  static CLOSED = 3
  readyState = FakeWebSocket.CONNECTING
  sent: string[] = []
  onopen: (() => void) | null = null
  onmessage: ((ev: { data: unknown }) => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  constructor(_url: string) {
    sockets.push(this)
  }
  send(text: string) {
    this.sent.push(text)
  }
  close() {
    this.readyState = FakeWebSocket.CLOSED
  }
}
Object.defineProperty(globalThis, 'WebSocket', { configurable: true, value: FakeWebSocket })

vi.mock('../api/rest', () => ({ api: { runs: async () => [] } }))
// editorStore pulls monaco in (browser-only); the client only calls onFsChanged.
vi.mock('../state/editorStore', () => ({ useEditorStore: { getState: () => ({ onFsChanged: () => {} }) } }))
vi.mock('./viewerBridge', () => ({
  createViewerBridge: () => ({
    attach: () => {},
    detach: () => {},
    handleCommand: async () => {},
    openPath: async () => ({ ok: true, state: null, error: null, image: null }),
  }),
}))
vi.mock('./uiBridge', () => ({ createUiBridge: () => ({ reportState: () => {}, handleCommand: async () => {} }) }))

// The localStorage shim above must exist before the stores evaluate.
const { createWsClient } = await import('./client')
const { useUiStore } = await import('../state/uiStore')
const { useSessionStore } = await import('../state/sessionStore')
const { FRAMES } = await import('../state/fixtures')

type Ws = ReturnType<typeof createWsClient>
const clients: Ws[] = []

function make(): Ws {
  const c = createWsClient('ws://test/ws')
  clients.push(c)
  return c
}

function openLatest(): FakeWebSocket {
  const sock = sockets.at(-1)
  if (!sock) throw new Error('no socket was created')
  sock.readyState = FakeWebSocket.OPEN
  sock.onopen?.()
  return sock
}

function frames(sock: FakeWebSocket): Array<Record<string, unknown>> {
  return sock.sent.map((line) => JSON.parse(line) as Record<string, unknown>)
}

function subscribes(): Array<Record<string, unknown>> {
  return sockets.flatMap((s) => frames(s)).filter((f) => f.t === 'run.subscribe')
}

beforeEach(() => {
  sockets.length = 0
  useUiStore.setState(useUiStore.getInitialState())
  useSessionStore.setState(useSessionStore.getInitialState())
})

afterEach(() => {
  for (const c of clients) c.close()
  clients.length = 0
})

describe('the ui watcher lives exactly while the client is connected', () => {
  it('StrictMode connect-close-connect still subscribes a selected run', () => {
    const c = make()
    c.connect()
    c.close()
    c.connect()
    openLatest()
    useUiStore.getState().setActiveRun('r_done')
    expect(subscribes()).toEqual([{ t: 'run.subscribe', runId: 'r_done', fromSeq: 1 }])
  })

  it('close without reconnect stops watching the ui', () => {
    const c = make()
    c.connect()
    c.close()
    useUiStore.getState().setActiveRun('r_x')
    expect(subscribes()).toEqual([])
  })

  it('a second connect without close does not watch twice', () => {
    const c = make()
    c.connect()
    c.connect()
    openLatest()
    useUiStore.getState().setActiveRun('r_y')
    expect(subscribes()).toEqual([{ t: 'run.subscribe', runId: 'r_y', fromSeq: 1 }])
    c.close()
    useUiStore.getState().setActiveRun('r_z')
    expect(subscribes()).toEqual([{ t: 'run.subscribe', runId: 'r_y', fromSeq: 1 }])
  })
})

describe('the session.new locale', () => {
  it('hello with no sessions opens a new session in the UI locale', () => {
    useUiStore.getState().setLocale('ko')
    useSessionStore.setState({ sessions: [], currentSessionId: null })
    const c = make()
    c.connect()
    const sock = openLatest()
    sock.onmessage?.({ data: JSON.stringify({ ...FRAMES.hello, sessions: [] }) })
    const news = frames(sock).filter((f) => f.t === 'session.new')
    expect(news).toEqual([{ t: 'session.new', locale: 'ko' }])
  })
})
