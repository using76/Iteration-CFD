import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import WebSocket from 'ws'
import { DEFAULT_VIEWER_STATE, type ClientMsg, type ServerMsg } from '@cfd/shared'
import { Router } from '../http/router.js'
import { createHttpServer, type HttpServerHandle } from '../http/server.js'
import { fakeRun, fakeRunManager, type FakeRuns } from '../http/test-fakes.js'
import { createHub, type HubHandle } from './hub.js'

let srv: HttpServerHandle
let hub: HubHandle
let runs: FakeRuns
let url: string

class Client {
  ws: WebSocket
  frames: ServerMsg[] = []
  private waiters: Array<{ pred: (m: ServerMsg) => boolean; resolve: (m: ServerMsg) => void }> = []
  constructor(u: string, headers: Record<string, string> = {}) {
    this.ws = new WebSocket(u, { headers })
    this.ws.on('message', (d) => {
      const m = JSON.parse(String(d)) as ServerMsg
      this.frames.push(m)
      this.waiters = this.waiters.filter((w) => {
        if (!w.pred(m)) return true
        w.resolve(m)
        return false
      })
    })
  }
  open() {
    return new Promise<void>((resolve, reject) => {
      this.ws.once('open', () => resolve())
      this.ws.once('error', reject)
    })
  }
  send(msg: ClientMsg | Record<string, unknown>) {
    this.ws.send(JSON.stringify(msg))
  }
  next<T extends ServerMsg['t']>(t: T, timeoutMs = 3000): Promise<Extract<ServerMsg, { t: T }>> {
    const existing = this.frames.find((m) => m.t === t)
    if (existing) {
      this.frames.splice(this.frames.indexOf(existing), 1)
      return Promise.resolve(existing as Extract<ServerMsg, { t: T }>)
    }
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(`no ${t} frame within ${timeoutMs} ms`)), timeoutMs)
      this.waiters.push({
        pred: (m) => m.t === t,
        resolve: (m) => {
          clearTimeout(timer)
          this.frames.splice(this.frames.indexOf(m), 1)
          resolve(m as Extract<ServerMsg, { t: T }>)
        },
      })
    })
  }
  close() {
    this.ws.close()
  }
}

beforeAll(async () => {
  runs = fakeRunManager()
  runs.runs.set('r_1', fakeRun('r_1', { logLines: 1200 }))
  runs.lines.set(
    'r_1',
    Array.from({ length: 1200 }, (_, i) => ({ seq: i + 1, stream: 'stdout' as const, text: `line ${i + 1}`, ts: i })),
  )
  runs.residualRecs.set('r_1', [1, 2].map((seq) => ({ seq, iter: seq * 25, time: null, wall: null, fields: { k: 1 / seq }, solverIters: null, raw: '' })))
  hub = createHub({
    hello: () => ({ version: 't', mode: 'demo', llm: 'mock', model: 'm', gpu: runs.gpu(), workspaceRoot: '/ws', availableBinaries: [], platform: 'linux' }),
    sessions: () => [],
    runs,
    heartbeatMs: 50,
    idleMs: 100_000,
  })
  srv = createHttpServer({ config: { host: '127.0.0.1', port: 0, allowRemote: false, authToken: null }, router: new Router(), hub, staticDir: null })
  url = `ws://127.0.0.1:${(await srv.listen()).port}/ws`
})
afterAll(async () => {
  hub.close()
  await srv.close()
})

describe('hub', () => {
  it('greets, answers ping, rejects invalid frames', async () => {
    const c = new Client(url)
    await c.open()
    const hello = await c.next('hello')
    expect(hello.hello.mode).toBe('demo')
    expect(hello.runs.map((r) => r.id)).toEqual(['r_1'])
    c.send({ t: 'ping', ts: 42 })
    expect((await c.next('pong')).ts).toBe(42)
    c.send({ t: 'nonsense' })
    expect((await c.next('error')).message).toContain('invalid message nonsense')
    c.ws.send('{not json')
    expect((await c.next('error')).message).toContain('invalid JSON')
    expect(hub.clients()).toHaveLength(1)
    c.close()
    await new Promise((r) => setTimeout(r, 50))
    expect(hub.clients()).toHaveLength(0)
  })

  it('replays logs in batches, residuals and the run on subscribe, then routes live frames', async () => {
    const c = new Client(url)
    await c.open()
    await c.next('hello')
    c.send({ t: 'run.subscribe', runId: 'r_1', fromSeq: 1 })
    const b1 = await c.next('run.log')
    const b2 = await c.next('run.log')
    const b3 = await c.next('run.log')
    expect(b1.lines).toHaveLength(500)
    expect(b2.lines[0].seq).toBe(501)
    expect(b3.lines).toHaveLength(200)
    expect(b3.lines.at(-1)?.seq).toBe(1200)
    const r1 = await c.next('run.residual')
    const r2 = await c.next('run.residual')
    expect([r1.rec.seq, r2.rec.seq]).toEqual([1, 2])
    expect((await c.next('run.updated')).run.id).toBe('r_1')
    hub.sendToRun('r_1', { t: 'run.written', runId: 'r_1', dir: 'cases/plume_jsonc/1' })
    expect((await c.next('run.written')).dir).toBe('cases/plume_jsonc/1')
    hub.sendToRun('r_1', { t: 'run.log', runId: 'r_1', lines: [{ seq: 1201, stream: 'stdout', text: 'a', ts: 0 }] })
    hub.sendToRun('r_1', { t: 'run.log', runId: 'r_1', lines: [{ seq: 1202, stream: 'stdout', text: 'b', ts: 0 }] })
    const batched = await c.next('run.log')
    expect(batched.lines.map((l) => l.seq)).toEqual([1201, 1202])
    c.send({ t: 'run.unsubscribe', runId: 'r_1' })
    await new Promise((r) => setTimeout(r, 20))
    hub.sendToRun('r_1', { t: 'run.written', runId: 'r_1', dir: 'x' })
    c.send({ t: 'run.subscribe', runId: 'r_9', fromSeq: 1 })
    expect((await c.next('error')).message).toContain('no such run')
    expect(c.frames.filter((f) => f.t === 'run.written')).toHaveLength(0)
    c.send({ t: 'run.stop', runId: 'r_1' })
    await new Promise((r) => setTimeout(r, 20))
    expect(runs.stopped).toContain('r_1')
    c.close()
  })

  it('brokers viewer commands to the most recent viewer and reports NO_VIEWER / TIMEOUT', async () => {
    const none = await hub.requestViewer({ type: 'clear' }, { timeoutMs: 100 })
    expect(none).toMatchObject({ ok: false, error: { code: 'NO_VIEWER' } })

    const c = new Client(url)
    await c.open()
    await c.next('hello')
    expect(hub.hasViewerClient()).toBe(false)
    c.send({ t: 'session.open', sessionId: 's_1' })
    c.send({ t: 'viewer.state', state: { ...DEFAULT_VIEWER_STATE, backend: 'webgl2' } })
    await new Promise((r) => setTimeout(r, 30))
    expect(hub.hasViewerClient()).toBe(true)
    expect(hub.clients()[0].sessionId).toBe('s_1')

    const p = hub.requestViewer({ type: 'setTime', index: 'last' }, { sessionId: 's_1', timeoutMs: 2000 })
    const cmd = await c.next('viewer.command')
    expect(cmd.cmd).toEqual({ type: 'setTime', index: 'last' })
    c.send({ t: 'viewer.result', requestId: cmd.requestId, result: { ok: true, state: { ...DEFAULT_VIEWER_STATE, backend: 'webgl2', field: 'U' }, error: null, image: null } })
    const res = await p
    expect(res.ok).toBe(true)
    expect(res.state?.field).toBe('U')
    expect(hub.clients()[0].viewerState?.field).toBe('U')

    const slow = await hub.requestViewer({ type: 'getState' }, { timeoutMs: 150 })
    expect(slow).toMatchObject({ ok: false, error: { code: 'TIMEOUT' } })

    const late = hub.requestViewer({ type: 'clear' }, { timeoutMs: 1000 })
    await c.next('viewer.command')
    c.close()
    expect(await late).toMatchObject({ ok: false, error: { code: 'NO_VIEWER' } })
  })

  it('routes other messages to the registered handler and tracks the session', async () => {
    const seen: string[] = []
    const off = hub.onClientMessage((client, msg) => {
      seen.push(`${msg.t}:${client.sessionId}`)
    })
    const c = new Client(url)
    await c.open()
    await c.next('hello')
    c.send({ t: 'user.message', sessionId: 's_7', text: 'hi', context: { activeFile: null, activeRun: null, attachments: [], selection: null } })
    await new Promise((r) => setTimeout(r, 30))
    expect(seen).toEqual(['user.message:s_7'])
    hub.sendToSession('s_7', { t: 'session.deleted', sessionId: 's_7' })
    expect((await c.next('session.deleted')).sessionId).toBe('s_7')
    hub.broadcast({ t: 'gpu', gpu: runs.gpu() })
    expect((await c.next('gpu')).gpu.state).toBe('demo')
    off()
    c.send({ t: 'session.list' })
    expect((await c.next('error')).message).toContain('no handler')
    c.close()
  })

  it('refuses upgrades from foreign origins but accepts loopback and origin-less clients', async () => {
    const bad = new Client(url, { origin: 'http://evil.example' })
    await expect(bad.open()).rejects.toThrow(/403/)
    const good = new Client(url, { origin: 'http://localhost:5173' })
    await good.open()
    await good.next('hello')
    good.close()
  })
})
