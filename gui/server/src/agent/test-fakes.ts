// In-memory Hub / RunManager / DatasetService fakes for the agent and tool
// tests. The run fake finishes every run on its own a few ms after start,
// printing the log lines the tools parse (mesh cells, "written to").
import fsp from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { DEFAULT_VIEWER_STATE, type ClientMsg, type GpuState, type LogLine, type MetricRecord, type ResidualRecord, type RunInfo, type ServerMsg, type ViewerCommand, type ViewerResult } from '@cfd/shared'
import type { ServerConfig } from '../config.js'
import type { DatasetService } from '../datasets/types.js'
import type { RunEvent, RunManager, StartRunOptions, WaitOptions } from '../runs/types.js'
import type { ClientConn, Hub } from '../ws/types.js'

const here = path.dirname(fileURLToPath(import.meta.url))
export const GUI_DIR = path.resolve(here, '..', '..', '..')
export const REPO_ROOT = path.resolve(GUI_DIR, '..')

export interface TempWorkspace {
  tmp: string
  root: string
  config: ServerConfig
  cleanup(): Promise<void>
}

export async function makeWorkspace(opts: { plume?: boolean; spec?: boolean } = {}): Promise<TempWorkspace> {
  const tmp = await fsp.mkdtemp(path.join(os.tmpdir(), 'cfd-agent-'))
  const root = path.join(tmp, 'ws')
  await fsp.mkdir(path.join(root, 'cases'), { recursive: true })
  if (opts.plume !== false) await fsp.copyFile(path.join(REPO_ROOT, 'cases', 'plume.jsonc'), path.join(root, 'cases', 'plume.jsonc'))
  if (opts.spec) {
    await fsp.mkdir(path.join(root, 'rust'), { recursive: true })
    await fsp.writeFile(path.join(root, 'rust', 'SPEC-LIT.md'), '# Spec\n\n## 1. Intro\n\nHello.\n\n## 31. Periodic domains\n\nIntro of 31.\n\n### 31.1 Cyclic patch pairs from a case file\n\nCyclic pairs text with keyword translation invariants.\n\n### 31.2 Field output\n\nOther text.\n\n## 32. Next\n\nEnd.\n')
  }
  const config: ServerConfig = {
    version: '0.0.0-test',
    host: '127.0.0.1',
    port: 0,
    workspaceRoot: root,
    guiDir: GUI_DIR,
    sessionsDir: path.join(tmp, 'sessions'),
    runsDir: path.join(tmp, 'runs'),
    cacheDir: path.join(tmp, 'cache'),
    configDir: path.join(tmp, 'config'),
    demo: true,
    llm: 'mock',
    model: 'claude-opus-5',
    allowNoApiKey: true,
    binDir: null,
    mockSpeed: 100,
    allowRemote: false,
    authToken: null,
    maxConcurrentRuns: 4,
    logLevel: 'error',
  }
  return { tmp, root, config, cleanup: () => fsp.rm(tmp, { recursive: true, force: true }) }
}

// ---------------------------------------------------------------------------

export interface FakeHub extends Hub {
  sent: ServerMsg[]
  viewerResult: ((cmd: ViewerCommand) => ViewerResult) | null
  viewerCalls: ViewerCommand[]
  of<T extends ServerMsg['t']>(t: T): Array<Extract<ServerMsg, { t: T }>>
  clear(): void
}

export function fakeHub(): FakeHub {
  const sent: ServerMsg[] = []
  const hub: FakeHub = {
    sent,
    viewerCalls: [],
    viewerResult: (cmd) => ({ ok: true, state: { ...DEFAULT_VIEWER_STATE, datasetName: 'plume', cellCount: 82320, field: 'U', layers: [] }, error: null, image: cmd.type === 'screenshot' ? { base64: 'iVBORw0KGgo=', mime: 'image/png', width: 8, height: 8 } : null }),
    of: (t) => sent.filter((m) => m.t === t) as never,
    clear: () => sent.splice(0, sent.length),
    broadcast: (msg) => {
      sent.push(msg)
    },
    sendToSession: (_id, msg) => {
      sent.push(msg)
    },
    sendToRun: (_id, msg) => {
      sent.push(msg)
    },
    clients: () => [],
    hasViewerClient: () => hub.viewerResult !== null,
    async requestViewer(cmd) {
      hub.viewerCalls.push(cmd)
      if (!hub.viewerResult) return { ok: false, state: null, error: { code: 'NO_VIEWER', message: 'no viewer' }, image: null }
      return hub.viewerResult(cmd)
    },
    onClientMessage: () => () => {},
    onClientOpen: () => () => {},
    onClientClose: () => () => {},
  }
  return hub
}

export function fakeClient(sessionId: string | null = null): ClientConn & { sent: ServerMsg[]; of<T extends ServerMsg['t']>(t: T): Array<Extract<ServerMsg, { t: T }>> } {
  const sent: ServerMsg[] = []
  return {
    id: 'c1',
    sessionId,
    runs: new Set(),
    viewerState: null,
    sent,
    of: (t) => sent.filter((m) => m.t === t) as never,
    send: (m) => {
      sent.push(m)
    },
    close: () => {},
  }
}

export type ClientFrame = ClientMsg

// ---------------------------------------------------------------------------

export interface FakeRuns extends RunManager {
  runs: Map<string, RunInfo>
  logs: Map<string, LogLine[]>
  residualRecs: Map<string, ResidualRecord[]>
  started: StartRunOptions[]
  /** ms until a started run finishes (null = never on its own). */
  finishAfterMs: number | null
  /** Status the run ends with. */
  endStatus: RunInfo['status']
  finish(id: string, status?: RunInfo['status']): void
  addLine(id: string, text: string, stream?: LogLine['stream']): void
}

export function fakeRuns(opts: { finishAfterMs?: number | null; endStatus?: RunInfo['status'] } = {}): FakeRuns {
  const runs = new Map<string, RunInfo>()
  const logs = new Map<string, LogLine[]>()
  const residualRecs = new Map<string, ResidualRecord[]>()
  const handlers = new Set<(ev: RunEvent) => void>()
  const waiters = new Map<string, Set<() => void>>()
  const gpu: GpuState = { state: 'demo', name: 'NVIDIA GeForce RTX 4090 (demo)', memUsedMB: 12400, memTotalMB: 24576, source: 'demo' }
  let seq = 0
  const need = (id: string) => {
    const r = runs.get(id)
    if (!r) throw new Error(`no such run: ${id}`)
    return r
  }
  const emit = (ev: RunEvent) => handlers.forEach((h) => h(ev))
  const wake = (id: string) => {
    for (const w of waiters.get(id) ?? []) w()
  }
  const fake: FakeRuns = {
    runs,
    logs,
    residualRecs,
    started: [],
    finishAfterMs: opts.finishAfterMs === undefined ? 5 : opts.finishAfterMs,
    endStatus: opts.endStatus ?? 'done',
    addLine(id, text, stream = 'stdout') {
      const run = need(id)
      const line: LogLine = { seq: ++run.logLines, stream, text, ts: Date.now() }
      logs.get(id)!.push(line)
      emit({ type: 'log', runId: id, lines: [line] })
    },
    finish(id, status = fake.endStatus) {
      const run = need(id)
      if (run.status !== 'running' && run.status !== 'queued') return
      if (run.binary === 'ofgpu-generate-mesh') {
        fake.addLine(id, 'mesh: 24000 cells, 72400 faces')
        fake.addLine(id, `channel: 200 x 120 x 1 = 24000 cells -> ${run.outputRoot}`)
        fake.addLine(id, `written to ${run.outputRoot}`)
        run.written.push(run.outputRoot!)
      } else if (status === 'done') {
        run.iter = run.targetIter ?? 4000
        run.lastResidual = { epsilon: 3.6e-6, k: 2.1e-6, dk_k: 1.7e-4 }
        residualRecs.set(id, [
          { seq: ++seq, iter: 100, time: null, wall: null, fields: { epsilon: 3.6e-3, k: 2.1e-3 }, solverIters: null, raw: '' },
          { seq: ++seq, iter: run.iter, time: null, wall: null, fields: { epsilon: 3.6e-6, k: 2.1e-6 }, solverIters: null, raw: '' },
        ])
        fake.addLine(id, `${String(run.iter).padStart(7)}  epsilon res 3.600e-06 (3)  k res 2.100e-06 (2)  max dk/k 1.700e-04`)
        fake.addLine(id, 'converged: max relative change below 0.001')
        fake.addLine(id, `written to ${run.outputRoot}/1`)
        run.written.push(`${run.outputRoot}/1`)
        run.converged = true
      } else if (status === 'failed') {
        fake.addLine(id, 'error: this case asks for the kOmegaSST model; run it with ofgpu-k-omega', 'stderr')
        run.error = 'this case asks for the kOmegaSST model; run it with ofgpu-k-omega'
      }
      run.status = status
      run.endedAt = new Date().toISOString()
      run.exitCode = status === 'done' ? 0 : 1
      emit({ type: 'exit', run: { ...run } })
      wake(id)
    },
    async start(o) {
      if (o.binary === 'ofgpu-bad') throw new Error('unknown binary ofgpu-bad')
      fake.started.push(o)
      const id = `r_${runs.size + 1}`
      const isMesh = o.binary === 'ofgpu-generate-mesh'
      const outputRoot = isMesh ? (o.positionals[1] ?? null) : o.casePath ? (o.casePath.endsWith('.jsonc') ? o.casePath.replace(/\.jsonc$/, '_jsonc') : o.casePath) : null
      const iters = o.args.find((a) => a.flag === '-iters')
      const run: RunInfo = {
        id,
        binary: o.binary,
        argv: [...(o.casePath ? [o.casePath] : []), ...o.positionals, ...o.args.flatMap((a) => (a.value === true ? [a.flag] : [a.flag, String(a.value)]))],
        cwd: '',
        casePath: isMesh ? (o.positionals[1] ?? null) : o.casePath,
        outputRoot,
        status: 'running',
        pid: 4242,
        startedAt: new Date().toISOString(),
        endedAt: null,
        exitCode: null,
        signal: null,
        iter: 0,
        targetIter: iters ? Number(iters.value) : null,
        time: null,
        endTime: null,
        lastResidual: null,
        written: [],
        error: null,
        converged: false,
        device: 'demo',
        logLines: 0,
        mode: 'demo',
        label: o.label,
      }
      runs.set(id, run)
      logs.set(id, [])
      emit({ type: 'started', run: { ...run } })
      fake.addLine(id, `ofgpu ${o.binary.replace('ofgpu-', '')} | NVIDIA GeForce RTX 4090 (demo) sm_89 | 24564 MiB | precision double`)
      if (fake.finishAfterMs !== null) setTimeout(() => fake.finish(id), fake.finishAfterMs)
      return { ...run }
    },
    async stop(id) {
      fake.finish(id, 'killed')
      return { ...need(id) }
    },
    get: (id) => {
      const r = runs.get(id)
      return r ? { ...r, written: [...r.written] } : undefined
    },
    list: () => [...runs.values()].map((r) => ({ ...r })),
    log(id, fromSeq, max, grep) {
      need(id)
      const all = logs.get(id) ?? []
      const matching = all.filter((l) => l.seq >= fromSeq && (!grep || grep.test(l.text)))
      const out = matching.slice(0, max)
      return { lines: out, total: all.length, nextSeq: out.length ? out[out.length - 1].seq + 1 : fromSeq }
    },
    residuals: (id) => residualRecs.get(need(id).id) ?? [],
    metrics: (id): MetricRecord[] => {
      need(id)
      return []
    },
    residualsCsv: () => 'iter,k\n',
    wait(id, o: WaitOptions) {
      const run = need(id)
      const satisfied = () => {
        const r = need(id)
        if (o.untilStatus?.includes(r.status)) return true
        if (!o.untilStatus && r.status !== 'running' && r.status !== 'queued') return true
        if (o.untilIter !== undefined && r.iter >= o.untilIter) return true
        if (o.untilWritten && r.written.length) return true
        return false
      }
      if (satisfied()) return Promise.resolve({ ...run })
      return new Promise<RunInfo>((resolve) => {
        const set = waiters.get(id) ?? new Set()
        waiters.set(id, set)
        const timer = setTimeout(() => {
          set.delete(w)
          resolve({ ...need(id) })
        }, o.maxMs)
        const w = () => {
          if (!satisfied()) return
          clearTimeout(timer)
          set.delete(w)
          resolve({ ...need(id) })
        }
        set.add(w)
      })
    },
    on: (h) => {
      handlers.add(h)
      return () => handlers.delete(h)
    },
    gpu: () => gpu,
    availableBinaries: () => ['ofgpu-k-epsilon', 'ofgpu-generate-mesh'],
    shutdown: async () => {},
  }
  return fake
}

export function fakeDatasets(): DatasetService & { opened: string[] } {
  const svc = {
    opened: [] as string[],
    async open(relPath: string) {
      svc.opened.push(relPath)
      return { datasetId: 'd_1', status: 'loading' as const, manifest: null, error: null }
    },
    get: () => undefined,
    blob: async () => null,
    ensureField: async () => {
      throw new Error('not in the fake')
    },
    async fieldStats(root: string, time: string, field: string) {
      return { field, component: 'magnitude' as const, time, count: 3, min: 0, max: 2, mean: 1, rms: 1.2, histogram: { edges: [0, 1, 2], counts: [1, 2] }, argmin: 0, argmax: 2, root }
    },
    discover: async (relRoot: string) => ({ root: relRoot, caseJsonc: null, hasPolyMesh: false, hasVtu: false, times: [{ label: '1', value: 1, fields: ['U', 'p'] }], vtk: [], cellCount: 82320 }),
    onProgress: () => () => {},
    evict: () => {},
    cacheBytes: () => 0,
  }
  return svc as unknown as DatasetService & { opened: string[] }
}
