// The contract between the run manager and everything that watches a run:
// the HTTP routes, the WS hub, the agent's tools, and the problems tracker.
//
// Reconstructed. The original file was never committed - gui/.gitignore's
// unanchored `runs/` matched this directory at every depth - and no copy
// survives in any ref, in the history, or on disk. Every name and signature
// here is pinned by an existing consumer: server/src/http/test-fakes.ts and
// server/src/agent/test-fakes.ts each implement RunManager in full, and
// main.ts, routes.ts, hub.ts, problems.ts, tools/run.ts and the agent loop
// call it. Where the two fakes disagreed, the real caller decided.
import type { GpuMonitor } from '../gpu/index.js'
import type { GpuState, LogLine, MetricRecord, ResidualRecord, RunInfo, RunStatus, StartRunRequest } from '@cfd/shared'

/** A run request plus the session that asked for it (null for the REST route). */
export interface StartRunOptions extends StartRunRequest {
  sessionId?: string | null
}

/**
 * What `wait` is waiting for. `maxMs` always applies; the rest are early exits,
 * and the call resolves with the run as it stands whichever arrives first - it
 * never rejects on timeout, so a caller reads the status to see what happened.
 */
export interface WaitOptions {
  maxMs: number
  untilIter?: number
  untilStatus?: RunStatus[]
  untilWritten?: boolean
}

/** A page of a run's log, with the cursor for the next page. */
export interface LogWindow {
  lines: LogLine[]
  total: number
  nextSeq: number
}

/**
 * Everything a run emits. `started`/`updated`/`exit` carry the whole RunInfo
 * because the UI replaces its copy; the streaming ones carry only the delta.
 */
export type RunEvent =
  | { type: 'started'; run: RunInfo }
  | { type: 'updated'; run: RunInfo }
  | { type: 'log'; runId: string; lines: LogLine[] }
  | { type: 'residual'; runId: string; rec: ResidualRecord }
  | { type: 'metric'; runId: string; rec: MetricRecord }
  | { type: 'written'; runId: string; dir: string }
  | { type: 'exit'; run: RunInfo }

export interface RunManager {
  /** Validate against the registry, queue, spawn. Rejects with a `status`-carrying error on bad input. */
  start(opts: StartRunOptions): Promise<RunInfo>
  /** Kill the process group; resolves once the run is terminal. */
  stop(id: string): Promise<RunInfo>
  get(id: string): RunInfo | undefined
  list(): RunInfo[]
  log(id: string, fromSeq: number, max: number, grep?: RegExp): LogWindow
  residuals(id: string, fromSeq?: number): ResidualRecord[]
  metrics(id: string, fromSeq?: number): MetricRecord[]
  residualsCsv(id: string): string
  wait(id: string, opts: WaitOptions): Promise<RunInfo>
  on(handler: (ev: RunEvent) => void): () => void
  gpu(): GpuState
  availableBinaries(): string[]
  shutdown(): Promise<void>
}

/** What createRunManager returns: the manager plus the GPU monitor main.ts drives. */
export interface RunManagerHandle extends RunManager {
  gpuMonitor: GpuMonitor
}

/** Thrown for anything the caller got wrong; `status` becomes the HTTP status. */
export class RunRequestError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message)
    this.name = 'RunRequestError'
  }
}

export const TERMINAL_STATUSES: readonly RunStatus[] = ['done', 'failed', 'killed', 'diverged']

export function isTerminal(status: RunStatus): boolean {
  return TERMINAL_STATUSES.includes(status)
}
