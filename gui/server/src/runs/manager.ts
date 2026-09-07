// The run manager: validate a request against the registry, queue it behind
// the GPU, spawn it, parse every line it prints, and tell everyone watching.
//
// Reconstructed along with the rest of this directory - see types.ts. Two of
// the review findings this directory carried are fixed as it is written rather
// than after: S1's missing stream error listeners (store.ts) and S2's phantom
// running run when launch throws (launch() below).
import path from 'node:path'
import readline from 'node:readline'
import { BINARY_NAMES, checkArgValue, getBinary, LogLineParser, RESIDUAL_SERIES_ORDER, type LogLine, type MetricRecord, type ResidualRecord, type RunInfo, type RunStatus } from '@cfd/shared'
import type { ServerConfig } from '../config.js'
import { createGpuMonitor, type GpuMonitor } from '../gpu/index.js'
import { silentLogger, type Logger } from '../log.js'
import { resolveInWorkspace } from '../workspace/paths.js'
import { availableBinaries as detectBinaries, buildArgv, dispatch, killTree } from './dispatch.js'
import { loadPastRuns, nullRunFiles, openRun, readPastLog, readPastSeries, type RunFiles } from './store.js'
import { isTerminal, RunRequestError, type LogWindow, type RunEvent, type RunManagerHandle, type StartRunOptions, type WaitOptions } from './types.js'

export type { RunManagerHandle } from './types.js'

/** Lines kept in memory per run. Everything is still on disk in log.txt. */
export const LOG_RING = 5000

interface LiveRun {
  info: RunInfo
  lines: LogLine[]
  /** seq of lines[0]; the ring drops the front, so index != seq. */
  firstSeq: number
  residuals: ResidualRecord[]
  metrics: MetricRecord[]
  files: RunFiles
  /** The request as given, kept apart from RunInfo so argv can be rebuilt on launch. */
  positionals: string[]
  args: StartRunOptions['args']
  parser: LogLineParser
  proc: ReturnType<typeof dispatch> | null
  waiters: Set<() => void>
  exited: (() => void) | null
  usesGpu: boolean
}

export interface RunManagerDeps {
  config: ServerConfig
  log?: Logger
  /** Injected in tests so the monitor never shells out to nvidia-smi. */
  gpuMonitor?: GpuMonitor
}

export async function createRunManager(deps: RunManagerDeps): Promise<RunManagerHandle> {
  const { config } = deps
  const log = deps.log ?? silentLogger
  const gpuMonitor = deps.gpuMonitor ?? createGpuMonitor({ demo: config.demo })

  const runs = new Map<string, LiveRun>()
  const handlers = new Set<(ev: RunEvent) => void>()
  const queue: string[] = []
  let gpuBusy = false
  let counter = 0
  let closing = false

  const emit = (ev: RunEvent) => {
    for (const h of handlers) {
      try {
        h(ev)
      } catch (err) {
        log.warn(`run event handler failed: ${(err as Error).message}`)
      }
    }
  }

  const need = (id: string): LiveRun => {
    const r = runs.get(id)
    if (!r) throw new RunRequestError(404, `no such run: ${id}`)
    return r
  }

  const snapshot = (r: LiveRun): RunInfo => ({ ...r.info, written: [...r.info.written], lastResidual: r.info.lastResidual ? { ...r.info.lastResidual } : null })

  const wake = (r: LiveRun) => {
    for (const w of [...r.waiters]) w()
  }

  // ---- restoring what a previous server left behind -------------------------

  for (const info of await loadPastRuns(config.runsDir, log)) {
    if (runs.has(info.id)) continue
    const { residuals, metrics } = await readPastSeries(config.runsDir, info.id)
    const lines = await readPastLog(config.runsDir, info.id)
    runs.set(info.id, {
      info,
      lines: lines.slice(-LOG_RING),
      firstSeq: Math.max(1, lines.length - LOG_RING + 1),
      residuals,
      metrics,
      files: nullRunFiles(path.join(config.runsDir, info.id)),
      positionals: [],
      args: [],
      parser: new LogLineParser('none'),
      proc: null,
      waiters: new Set(),
      exited: null,
      usesGpu: false,
    })
    const n = Number(/^r_(\d+)$/.exec(info.id)?.[1] ?? 0)
    if (n > counter) counter = n
  }

  // ---- validation -----------------------------------------------------------

  function validate(opts: StartRunOptions): { spec: NonNullable<ReturnType<typeof getBinary>>; cwdAbs: string; cwdRel: string; casePath: string | null; outputRoot: string | null } {
    const spec = getBinary(opts.binary)
    if (!spec) throw new RunRequestError(400, `unknown binary ${opts.binary}; available: ${BINARY_NAMES.join(', ')}`)

    for (const a of opts.args) {
      const flag = spec.flags.find((f) => f.name === a.flag)
      if (!flag) throw new RunRequestError(400, `${spec.name} has no option ${a.flag}`)
      const bad = checkArgValue(flag, a.value)
      if (bad) throw new RunRequestError(400, bad)
    }
    for (let i = 0; i < opts.positionals.length; i++) {
      const p = spec.positionals[i]
      if (!p) throw new RunRequestError(400, `${spec.name} takes ${spec.positionals.length} positional argument(s), got ${opts.positionals.length}`)
      const bad = checkArgValue(p, opts.positionals[i])
      if (bad) throw new RunRequestError(400, bad)
    }
    const required = spec.positionals.filter((p) => !p.optional).length
    if (opts.positionals.length + (opts.casePath ? 1 : 0) < required && !opts.casePath) {
      throw new RunRequestError(400, `${spec.name} needs ${required} positional argument(s)`)
    }

    // Every path-shaped input is confined to the workspace, here and nowhere else.
    let casePath: string | null = null
    if (opts.casePath) {
      const r = resolveInWorkspace(config.workspaceRoot, opts.casePath, { mustExist: true })
      casePath = r.rel
    }
    const positionalPaths = spec.positionals.map((p, i) => (p.type === 'path' && opts.positionals[i] ? resolveInWorkspace(config.workspaceRoot, opts.positionals[i]).rel : null))
    for (let i = 0; i < positionalPaths.length; i++) if (positionalPaths[i] !== null) opts.positionals[i] = positionalPaths[i]!

    const outputRoot = casePath ? (casePath.endsWith('.jsonc') ? casePath.replace(/\.jsonc$/, '_jsonc') : casePath) : (positionalPaths.find((p) => p !== null) ?? null)
    return { spec, cwdAbs: config.workspaceRoot, cwdRel: '', casePath, outputRoot }
  }

  // ---- the single-GPU queue -------------------------------------------------

  function pump(): void {
    if (closing) return
    while (queue.length) {
      const id = queue[0]
      const r = runs.get(id)
      if (!r || r.info.status !== 'queued') {
        queue.shift()
        continue
      }
      if (r.usesGpu && gpuBusy) return
      const running = [...runs.values()].filter((x) => x.info.status === 'running').length
      if (running >= Math.max(1, config.maxConcurrentRuns)) return
      queue.shift()
      void launch(r)
    }
  }

  // ---- launching ------------------------------------------------------------

  async function launch(r: LiveRun): Promise<void> {
    // Everything fallible happens before the run is called running. The
    // original left a phantom `running` run with no process when openRun or
    // spawn threw, which held the GPU slot for the life of the server and made
    // stop() wait forever.
    try {
      r.files = await openRun(config.runsDir, r.info.id, log)
      const spawned = dispatch({ config, binary: r.info.binary, argv: buildArgv({ casePath: r.info.casePath, positionals: r.positionals, args: r.args }), cwd: config.workspaceRoot })
      r.proc = spawned
      r.info.argv = spawned.argv
      r.info.mode = spawned.mode
      r.info.pid = spawned.child.pid ?? null
      r.info.status = 'running'
      if (r.usesGpu) {
        gpuBusy = true
        gpuMonitor.setActive(true)
      }
      attach(r, spawned)
      emit({ type: 'updated', run: snapshot(r) })
      r.files.saveInfo(snapshot(r))
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      log.warn(`run ${r.info.id} could not start: ${message}`)
      appendLines(r, [{ text: `error: ${message}`, stream: 'stderr' }])
      finish(r, { status: 'failed', exitCode: null, signal: null, error: message })
    }
  }

  function attach(r: LiveRun, spawned: NonNullable<LiveRun['proc']>): void {
    const { child } = spawned
    for (const which of ['stdout', 'stderr'] as const) {
      const stream = child[which]
      if (!stream) continue
      const rl = readline.createInterface({ input: stream, crlfDelay: Infinity })
      rl.on('line', (text) => onLine(r, text, which))
    }
    child.on('error', (err) => {
      appendLines(r, [{ text: `error: ${err.message}`, stream: 'stderr' }])
      finish(r, { status: 'failed', exitCode: null, signal: null, error: err.message })
    })
    child.on('close', (code, signal) => {
      if (isTerminal(r.info.status)) return
      const killed = signal !== null || r.info.status === 'killed'
      const status: RunStatus = killed ? 'killed' : code === 0 ? (r.info.error ? 'failed' : 'done') : 'failed'
      finish(r, { status, exitCode: code, signal: signal ?? null, error: r.info.error })
    })
  }

  // ---- the log --------------------------------------------------------------

  function appendLines(r: LiveRun, items: Array<{ text: string; stream: 'stdout' | 'stderr' }>): void {
    const now = Date.now()
    const lines: LogLine[] = items.map((i) => ({ seq: ++r.info.logLines, stream: i.stream, text: i.text, ts: now }))
    r.lines.push(...lines)
    if (r.lines.length > LOG_RING) {
      const dropped = r.lines.length - LOG_RING
      r.lines.splice(0, dropped)
      r.firstSeq += dropped
    }
    r.files.appendLog(lines)
    emit({ type: 'log', runId: r.info.id, lines })
  }

  function onLine(r: LiveRun, text: string, stream: 'stdout' | 'stderr'): void {
    appendLines(r, [{ text, stream }])
    let changed = false
    for (const ev of r.parser.feed(text)) {
      switch (ev.kind) {
        case 'residual': {
          const rec: ResidualRecord = { seq: r.residuals.length + 1, iter: ev.rec.iter, time: ev.rec.time, wall: ev.rec.wall, fields: ev.rec.fields, solverIters: ev.rec.solverIters, raw: ev.raw }
          r.residuals.push(rec)
          r.files.appendResidual(rec)
          if (rec.iter !== null && rec.iter > r.info.iter) r.info.iter = rec.iter
          if (rec.time !== null) r.info.time = rec.time
          r.info.lastResidual = rec.fields
          changed = true
          emit({ type: 'residual', runId: r.info.id, rec })
          break
        }
        case 'metric': {
          const rec: MetricRecord = { seq: r.metrics.length + 1, iter: ev.rec.iter, time: ev.rec.time, metrics: ev.rec.metrics, raw: ev.raw }
          r.metrics.push(rec)
          r.files.appendMetric(rec)
          emit({ type: 'metric', runId: r.info.id, rec })
          break
        }
        case 'written': {
          // Anchored in shared/residuals.ts so the transient drivers' "restart
          // checkpoint written to ..." line no longer lands here.
          const dir = toWorkspaceRelative(ev.dir)
          if (dir && !r.info.written.includes(dir)) {
            r.info.written.push(dir)
            changed = true
            emit({ type: 'written', runId: r.info.id, dir })
          }
          break
        }
        case 'converged':
          r.info.converged = true
          changed = true
          break
        case 'diverged':
          r.info.error = r.info.error ?? 'the solution diverged (NaN/Inf)'
          r.info.status = 'diverged'
          changed = true
          break
        case 'error':
          r.info.error = r.info.error ?? ev.message
          changed = true
          break
        case 'iterating':
          if (r.info.targetIter === null) r.info.targetIter = ev.total
          changed = true
          break
        case 'banner':
          r.info.device = ev.device
          changed = true
          break
        default:
          break
      }
    }
    if (changed) {
      emit({ type: 'updated', run: snapshot(r) })
      wake(r)
    }
  }

  /** Drivers print paths relative to their own cwd, which is the workspace root. */
  function toWorkspaceRelative(p: string): string | null {
    try {
      const abs = path.isAbsolute(p) ? p : path.resolve(config.workspaceRoot, p)
      return resolveInWorkspace(config.workspaceRoot, abs).rel || null
    } catch {
      return null
    }
  }

  function finish(r: LiveRun, end: { status: RunStatus; exitCode: number | null; signal: string | null; error: string | null }): void {
    if (isTerminal(r.info.status) && r.info.endedAt) return
    r.info.status = r.info.status === 'diverged' && end.status === 'failed' ? 'diverged' : end.status
    r.info.exitCode = end.exitCode
    r.info.signal = end.signal
    r.info.error = end.error ?? r.info.error
    r.info.endedAt = new Date().toISOString()
    r.info.pid = null
    if (r.usesGpu && gpuBusy) {
      gpuBusy = false
      gpuMonitor.setActive(false)
    }
    r.files.saveInfo(snapshot(r))
    void r.files.close()
    emit({ type: 'exit', run: snapshot(r) })
    r.exited?.()
    r.exited = null
    wake(r)
    pump()
  }

  // ---- the interface --------------------------------------------------------

  const manager: RunManagerHandle = {
    gpuMonitor,

    async start(opts) {
      if (closing) throw new RunRequestError(503, 'the server is shutting down')
      const { spec, casePath, outputRoot } = validate(opts)
      const id = `r_${++counter}`
      const iters = opts.args.find((a) => a.flag === '-iters')
      const info: RunInfo = {
        id,
        binary: spec.name,
        argv: [],
        cwd: '',
        casePath,
        outputRoot,
        status: 'queued',
        pid: null,
        startedAt: new Date().toISOString(),
        endedAt: null,
        exitCode: null,
        signal: null,
        iter: 0,
        targetIter: iters && typeof iters.value !== 'boolean' && iters.value !== null ? Number(iters.value) : null,
        time: null,
        endTime: null,
        lastResidual: null,
        written: [],
        error: null,
        converged: false,
        device: config.demo ? 'demo' : '',
        logLines: 0,
        mode: config.demo ? 'demo' : 'real',
        label: opts.label,
      }
      const r: LiveRun = {
        info,
        lines: [],
        firstSeq: 1,
        residuals: [],
        metrics: [],
        files: nullRunFiles(),
        positionals: [...opts.positionals],
        args: [...opts.args],
        parser: new LogLineParser(spec.residualStyle),
        proc: null,
        waiters: new Set(),
        exited: null,
        usesGpu: spec.gpu,
      }
      runs.set(id, r)
      emit({ type: 'started', run: snapshot(r) })
      queue.push(id)
      pump()
      return snapshot(r)
    },

    async stop(id) {
      const r = need(id)
      if (isTerminal(r.info.status)) return snapshot(r)
      if (r.info.status === 'queued') {
        finish(r, { status: 'killed', exitCode: null, signal: null, error: 'cancelled before it started' })
        return snapshot(r)
      }
      const done = new Promise<void>((resolve) => {
        r.exited = resolve
      })
      r.info.status = 'killed'
      if (r.proc) killTree(r.proc.child)
      // A driver that ignores SIGTERM still has to go.
      const hard = setTimeout(() => {
        if (r.proc && !isTerminal(r.info.status)) killTree(r.proc.child, 'SIGKILL')
      }, 4000)
      hard.unref?.()
      const guard = setTimeout(() => {
        if (!isTerminal(r.info.status) || !r.info.endedAt) finish(r, { status: 'killed', exitCode: null, signal: 'SIGKILL', error: r.info.error })
      }, 8000)
      guard.unref?.()
      await done
      clearTimeout(hard)
      clearTimeout(guard)
      return snapshot(r)
    },

    get: (id) => {
      const r = runs.get(id)
      return r ? snapshot(r) : undefined
    },

    list: () => [...runs.values()].map(snapshot).sort((a, b) => (a.startedAt < b.startedAt ? 1 : -1)),

    log(id, fromSeq, max, grep): LogWindow {
      const r = need(id)
      const matching = r.lines.filter((l) => l.seq >= fromSeq && (!grep || grep.test(l.text)))
      const out = matching.slice(0, Math.max(1, max))
      return { lines: out, total: r.info.logLines, nextSeq: out.length ? out[out.length - 1].seq + 1 : fromSeq }
    },

    residuals: (id, fromSeq = 0) => need(id).residuals.filter((r) => r.seq >= fromSeq),
    metrics: (id, fromSeq = 0) => need(id).metrics.filter((m) => m.seq >= fromSeq),

    residualsCsv(id) {
      const r = need(id)
      const names = RESIDUAL_SERIES_ORDER.filter((n) => r.residuals.some((rec) => rec.fields[n] !== undefined))
      const extra = [...new Set(r.residuals.flatMap((rec) => Object.keys(rec.fields)))].filter((n) => !names.includes(n as (typeof RESIDUAL_SERIES_ORDER)[number])).sort()
      const cols = [...names, ...extra]
      const head = ['iter', 'time', ...cols].join(',')
      const rows = r.residuals.map((rec) => [rec.iter ?? '', rec.time ?? '', ...cols.map((c) => (rec.fields[c] !== undefined ? rec.fields[c] : ''))].join(','))
      return `${head}\n${rows.join('\n')}${rows.length ? '\n' : ''}`
    },

    wait(id, opts: WaitOptions) {
      const r = need(id)
      const satisfied = (): boolean => {
        if (opts.untilStatus?.length) {
          if (opts.untilStatus.includes(r.info.status)) return true
        } else if (isTerminal(r.info.status)) return true
        if (opts.untilIter !== undefined && r.info.iter >= opts.untilIter) return true
        if (opts.untilWritten && r.info.written.length > 0) return true
        return false
      }
      if (satisfied()) return Promise.resolve(snapshot(r))
      return new Promise<RunInfo>((resolve) => {
        const timer = setTimeout(() => {
          r.waiters.delete(w)
          resolve(snapshot(r))
        }, Math.max(0, opts.maxMs))
        timer.unref?.()
        const w = () => {
          if (!satisfied()) return
          clearTimeout(timer)
          r.waiters.delete(w)
          resolve(snapshot(r))
        }
        r.waiters.add(w)
      })
    },

    on(handler) {
      handlers.add(handler)
      return () => handlers.delete(handler)
    },

    gpu: () => gpuMonitor.state(),

    availableBinaries: () => detectBinaries(config, BINARY_NAMES),

    async shutdown() {
      closing = true
      gpuMonitor.stop()
      const live = [...runs.values()].filter((r) => !isTerminal(r.info.status))
      await Promise.all(live.map((r) => manager.stop(r.info.id).catch(() => undefined)))
      await Promise.all([...runs.values()].map((r) => r.files.close().catch(() => undefined)))
      handlers.clear()
    },
  }

  return manager
}
