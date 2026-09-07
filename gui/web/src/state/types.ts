// Plain data the session reducer works on. No functions, no class instances,
// so the reducer can be tested with recorded fixtures.
import type {
  DatasetProgress,
  GpuState,
  LogLine,
  MetricRecord,
  Problem,
  ResidualRecord,
  RunInfo,
  ServerHello,
  SessionState,
  SessionSummary,
  ViewerCommand,
} from '@cfd/shared'

export const RESIDUAL_RING = 50_000
export const LOG_RING = 20_000
export const METRIC_RING = 50_000
export const OUTPUT_RING = 2_000

export interface RunData {
  residuals: ResidualRecord[]
  metrics: MetricRecord[]
  logs: LogLine[]
  lastLogSeq: number
  lastResidualSeq: number
  lastMetricSeq: number
}

export interface TurnState {
  active: boolean
  turnId: string | null
  messageId: string | null
  error: { message: string; retryable: boolean } | null
  refusal: { category: string | null; explanation: string | null } | null
  warnings: string[]
}

export interface OutputEntry {
  level: 'info' | 'warning' | 'error'
  text: string
  ts: number
  /** 'server' for `output` frames, 'client' for notes the UI adds itself. */
  origin: 'server' | 'client'
}

export interface SessionData {
  hello: ServerHello | null
  gpu: GpuState | null
  sessions: SessionSummary[]
  currentSessionId: string | null
  session: SessionState | null
  /** Streaming tool input JSON by toolUseId (cleared when the authoritative message arrives). */
  toolInputJson: Record<string, string>
  turn: TurnState
  runs: Record<string, RunInfo>
  runData: Record<string, RunData>
  /** Keyed by `${source}:${runId ?? path ?? '*'}`; each event replaces its key. */
  problems: Record<string, Problem[]>
  outputs: OutputEntry[]
  datasetProgress: Record<string, DatasetProgress>
  serverErrors: string[]
}

export type Effect =
  | { type: 'hello' }
  | { type: 'viewer.command'; requestId: string; cmd: ViewerCommand }
  | { type: 'viewer.open'; path: string; runId: string | null }
  | { type: 'residuals.open'; runId: string }
  | { type: 'run.subscribe'; runId: string }
  | { type: 'fs.changed'; paths: string[] }
  | { type: 'session.deleted'; sessionId: string }
  | { type: 'fatal'; message: string }

export const EMPTY_TURN: TurnState = { active: false, turnId: null, messageId: null, error: null, refusal: null, warnings: [] }

export function emptyRunData(): RunData {
  return { residuals: [], metrics: [], logs: [], lastLogSeq: 0, lastResidualSeq: 0, lastMetricSeq: 0 }
}

export function initialSessionData(): SessionData {
  return {
    hello: null,
    gpu: null,
    sessions: [],
    currentSessionId: null,
    session: null,
    toolInputJson: {},
    turn: EMPTY_TURN,
    runs: {},
    runData: {},
    problems: {},
    outputs: [],
    datasetProgress: {},
    serverErrors: [],
  }
}

export function problemKey(source: string, runId: string | null, path: string | null): string {
  return `${source}:${runId ?? path ?? '*'}`
}
