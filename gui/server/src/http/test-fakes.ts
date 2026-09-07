// In-memory stand-ins for the services the HTTP/WS layers delegate to.
import type { GpuState, LogLine, MetricRecord, ResidualRecord, RunInfo, SessionState, SessionSummary } from '@cfd/shared'
import type { AgentService } from '../agent/types.js'
import type { DatasetService } from '../datasets/types.js'
import type { RunEvent, RunManager } from '../runs/types.js'

export function fakeRun(id: string, over: Partial<RunInfo> = {}): RunInfo {
  return {
    id,
    binary: 'ofgpu-k-epsilon',
    argv: ['cases/plume.jsonc', '-iters', '200'],
    cwd: '',
    casePath: 'cases/plume.jsonc',
    outputRoot: 'cases/plume_jsonc',
    status: 'done',
    pid: null,
    startedAt: '2026-09-07T00:00:00.000Z',
    endedAt: '2026-09-07T00:01:00.000Z',
    exitCode: 0,
    signal: null,
    iter: 200,
    targetIter: 200,
    time: null,
    endTime: null,
    lastResidual: { k: 1e-6 },
    written: ['cases/plume_jsonc/1'],
    error: null,
    converged: true,
    device: 'demo',
    logLines: 3,
    mode: 'demo',
    label: null,
    ...over,
  }
}

export interface FakeRuns extends RunManager {
  runs: Map<string, RunInfo>
  lines: Map<string, LogLine[]>
  residualRecs: Map<string, ResidualRecord[]>
  stopped: string[]
  emit(ev: RunEvent): void
}

export function fakeRunManager(): FakeRuns {
  const runs = new Map<string, RunInfo>()
  const lines = new Map<string, LogLine[]>()
  const residualRecs = new Map<string, ResidualRecord[]>()
  const handlers = new Set<(ev: RunEvent) => void>()
  const gpu: GpuState = { state: 'demo', name: 'demo gpu', memUsedMB: 1, memTotalMB: 2, source: 'demo' }
  const need = (id: string) => {
    const r = runs.get(id)
    if (!r) throw Object.assign(new Error(`no such run: ${id}`), { status: 404 })
    return r
  }
  const fake: FakeRuns = {
    runs,
    lines,
    residualRecs,
    stopped: [],
    emit: (ev) => handlers.forEach((h) => h(ev)),
    async start(opts) {
      if (opts.binary === 'ofgpu-bad') throw Object.assign(new Error('unknown binary'), { status: 400 })
      const r = fakeRun(`r_${runs.size + 1}`, { binary: opts.binary, casePath: opts.casePath, status: 'running', label: opts.label })
      runs.set(r.id, r)
      return r
    },
    async stop(id) {
      const r = need(id)
      fake.stopped.push(id)
      return { ...r, status: 'killed' }
    },
    get: (id) => runs.get(id),
    list: () => [...runs.values()],
    log(id, fromSeq, max, grep) {
      need(id)
      const all = (lines.get(id) ?? []).filter((l) => l.seq >= fromSeq && (!grep || grep.test(l.text)))
      const out = all.slice(0, max)
      return { lines: out, total: (lines.get(id) ?? []).length, nextSeq: out.length ? out[out.length - 1].seq + 1 : fromSeq }
    },
    residuals: (id, fromSeq = 0) => (residualRecs.get(need(id).id) ?? []).filter((r) => r.seq >= fromSeq),
    metrics: (id): MetricRecord[] => {
      need(id)
      return []
    },
    residualsCsv: (id) => {
      need(id)
      return 'iter,time,k\n1,,0.5\n'
    },
    wait: async (id) => need(id),
    on: (h) => {
      handlers.add(h)
      return () => handlers.delete(h)
    },
    gpu: () => gpu,
    availableBinaries: () => ['ofgpu-k-epsilon'],
    shutdown: async () => {},
  }
  return fake
}

export function fakeAgent(): AgentService & { sessions: Map<string, SessionState>; handled: string[] } {
  const sessions = new Map<string, SessionState>()
  const summary = (s: SessionState): SessionSummary => ({ id: s.id, title: s.title, createdAt: s.createdAt, updatedAt: s.updatedAt, messageCount: s.messages.length })
  const svc = {
    sessions,
    handled: [] as string[],
    async handleClientMessage(_client: unknown, msg: { t: string }) {
      svc.handled.push(msg.t)
      return true
    },
    listSessions: () => [...sessions.values()].map(summary),
    getSessionState: (id: string) => sessions.get(id) ?? null,
    createSession() {
      const s: SessionState = {
        id: `s_${sessions.size + 1}`,
        title: 'new',
        createdAt: 'now',
        updatedAt: 'now',
        settings: { autoApprove: 'reads', effort: 'high', notifyOnRunEnd: true, locale: 'ko' },
        messages: [],
        pendingApprovals: [],
        runs: [],
        turnActive: false,
        customTools: [],
      }
      sessions.set(s.id, s)
      return s
    },
    deleteSession: (id: string) => sessions.delete(id),
    notifyRunEnded: () => {},
    shutdown: async () => {},
  }
  return svc as unknown as AgentService & { sessions: Map<string, SessionState>; handled: string[] }
}

export function fakeDatasets(): DatasetService & { opened: string[] } {
  const svc = {
    opened: [] as string[],
    async open(relPath: string) {
      svc.opened.push(relPath)
      return { datasetId: 'd_1', status: 'loading' as const, manifest: null, error: null }
    },
    get: () => undefined,
    blob: async (id: string, key: string) => (id === 'd_1' && key === 'f32' ? Buffer.from(new Float32Array([1, 2, 3]).buffer) : null),
    ensureField: async () => {
      throw new Error('not in the fake')
    },
    fieldStats: async () => {
      throw new Error('not in the fake')
    },
    discover: async (relRoot: string) => ({ root: relRoot, caseJsonc: null, hasPolyMesh: false, hasVtu: false, times: [], vtk: [], cellCount: null }),
    onProgress: () => () => {},
    evict: () => {},
    cacheBytes: () => 0,
  }
  return svc as unknown as DatasetService & { opened: string[] }
}
