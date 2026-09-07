// Per-run files under <runsDir>/<runId>: the full log, the residual series,
// and a run.json snapshot so a restarted server can list what has already run.
//
// Reconstructed along with the rest of this directory - see types.ts.
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import type { LogLine, MetricRecord, ResidualRecord, RunInfo } from '@cfd/shared'
import type { Logger } from '../log.js'

export interface RunFiles {
  dir: string
  appendLog(lines: LogLine[]): void
  appendResidual(rec: ResidualRecord): void
  appendMetric(rec: MetricRecord): void
  saveInfo(run: RunInfo): void
  close(): Promise<void>
  /** True once a write failed; the manager stops feeding a broken sink. */
  readonly broken: boolean
}

/**
 * A dead sink. Used when the run directory cannot be created at all: the run
 * still streams to the client and to memory, it just leaves nothing on disk.
 */
export function nullRunFiles(dir = ''): RunFiles {
  return {
    dir,
    appendLog: () => {},
    appendResidual: () => {},
    appendMetric: () => {},
    saveInfo: () => {},
    close: async () => {},
    broken: true,
  }
}

export async function openRun(runsDir: string, runId: string, log: Logger): Promise<RunFiles> {
  const dir = path.join(runsDir, runId)
  await fsp.mkdir(dir, { recursive: true })

  const logStream = fs.createWriteStream(path.join(dir, 'log.txt'), { flags: 'a' })
  const resStream = fs.createWriteStream(path.join(dir, 'residuals.jsonl'), { flags: 'a' })
  let broken = false

  // Node's default for an unhandled 'error' on a stream is to throw out of the
  // event loop and take the process with it - and with it the solver that was
  // streaming into it, and the log that would have said why. A full disk or a
  // deleted run directory mid-run is enough. Mark the sink broken instead and
  // let the run finish in memory.
  const onError = (which: string) => (err: Error) => {
    if (broken) return
    broken = true
    log.warn(`run ${runId}: ${which} is no longer writable (${err.message}); the run continues without its files`)
  }
  logStream.on('error', onError('log.txt'))
  resStream.on('error', onError('residuals.jsonl'))

  const files: RunFiles = {
    dir,
    get broken() {
      return broken
    },
    appendLog(lines) {
      if (broken || !lines.length) return
      logStream.write(lines.map((l) => (l.stream === 'stderr' ? `! ${l.text}` : l.text)).join('\n') + '\n')
    },
    appendResidual(rec) {
      if (broken) return
      resStream.write(`${JSON.stringify({ kind: 'residual', ...rec })}\n`)
    },
    appendMetric(rec) {
      if (broken) return
      resStream.write(`${JSON.stringify({ kind: 'metric', ...rec })}\n`)
    },
    saveInfo(run) {
      if (broken) return
      // Synchronous and best-effort: this is a snapshot for the next server, and
      // a failure to write it must never interrupt a running solve.
      try {
        fs.writeFileSync(path.join(dir, 'run.json'), JSON.stringify(run))
      } catch (err) {
        log.warn(`run ${runId}: could not save run.json (${(err as Error).message})`)
      }
    },
    close() {
      return new Promise<void>((resolve) => {
        let left = 2
        const done = () => {
          if (--left === 0) resolve()
        }
        logStream.end(done)
        resStream.end(done)
      })
    },
  }
  return files
}

/** Runs recorded by a previous server, newest first. Anything unreadable is skipped. */
export async function loadPastRuns(runsDir: string, log: Logger): Promise<RunInfo[]> {
  let names: string[]
  try {
    names = await fsp.readdir(runsDir)
  } catch {
    return []
  }
  const out: RunInfo[] = []
  for (const name of names) {
    try {
      const raw = await fsp.readFile(path.join(runsDir, name, 'run.json'), 'utf8')
      const run = JSON.parse(raw) as RunInfo
      if (!run || typeof run.id !== 'string') continue
      // A run that was still going when the server died did not survive it.
      if (run.status === 'running' || run.status === 'queued') {
        run.status = 'killed'
        run.error = run.error ?? 'the server stopped while this run was going'
        run.endedAt = run.endedAt ?? new Date().toISOString()
      }
      out.push(run)
    } catch {
      // No run.json, or unreadable: nothing to restore for this directory.
    }
  }
  out.sort((a, b) => (a.startedAt < b.startedAt ? 1 : -1))
  if (out.length) log.debug?.(`restored ${out.length} past run(s) from ${runsDir}`)
  return out
}

/** The recorded log of a past run, for a client that subscribes to it. */
export async function readPastLog(runsDir: string, runId: string): Promise<LogLine[]> {
  try {
    const text = await fsp.readFile(path.join(runsDir, runId, 'log.txt'), 'utf8')
    const lines = text.split('\n')
    if (lines[lines.length - 1] === '') lines.pop()
    return lines.map((raw, i) => {
      const stderr = raw.startsWith('! ')
      return { seq: i + 1, stream: stderr ? ('stderr' as const) : ('stdout' as const), text: stderr ? raw.slice(2) : raw, ts: 0 }
    })
  } catch {
    return []
  }
}

/** The recorded residual and metric series of a past run. */
export async function readPastSeries(runsDir: string, runId: string): Promise<{ residuals: ResidualRecord[]; metrics: MetricRecord[] }> {
  const residuals: ResidualRecord[] = []
  const metrics: MetricRecord[] = []
  try {
    const text = await fsp.readFile(path.join(runsDir, runId, 'residuals.jsonl'), 'utf8')
    for (const line of text.split('\n')) {
      if (!line.trim()) continue
      try {
        const { kind, ...rec } = JSON.parse(line) as { kind?: string } & Record<string, unknown>
        if (kind === 'metric') metrics.push(rec as unknown as MetricRecord)
        else residuals.push(rec as unknown as ResidualRecord)
      } catch {
        // A torn last line from a server that was killed mid-write.
      }
    }
  } catch {
    // Nothing recorded.
  }
  return { residuals, metrics }
}
