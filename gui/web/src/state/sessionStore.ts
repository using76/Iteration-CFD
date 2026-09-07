// Live mirror of everything the server tells us. All protocol frames go
// through `dispatch` (the pure reducer); the few client-only fields have
// their own setters.
import { create } from 'zustand'
import type { Problem, RunInfo, ServerMsg, SessionState } from '@cfd/shared'
import { applyEvents } from './applyEvent'
import { initialSessionData, type Effect, type OutputEntry, type RunData, type SessionData } from './types'

export type ConnectionStatus = 'connecting' | 'online' | 'offline'

export interface SessionStore extends SessionData {
  connection: ConnectionStatus
  /** Last text the user sent (for the error card's Retry). */
  lastUserText: string | null
  dispatch(msgs: ServerMsg[]): Effect[]
  setConnection(status: ConnectionStatus): void
  setLastUserText(text: string | null): void
  clearTurnMarkers(): void
  addNote(level: OutputEntry['level'], text: string): void
  clearOutputs(): void
  /** Replace the run list from a REST reconcile (GET /api/runs). */
  reconcileRuns(runs: RunInfo[]): void
  setCurrentSessionId(id: string | null): void
}

const EMPTY_RUN_DATA: RunData = { residuals: [], metrics: [], logs: [], lastLogSeq: 0, lastResidualSeq: 0, lastMetricSeq: 0 }
const NO_PROBLEMS: Problem[] = []

export const useSessionStore = create<SessionStore>()((set, get) => ({
  ...initialSessionData(),
  connection: 'connecting',
  lastUserText: null,
  dispatch(msgs) {
    const s = get()
    const { state, effects } = applyEvents(s, msgs)
    if (state !== s) set(state)
    return effects
  },
  setConnection(connection) {
    set({ connection })
  },
  setLastUserText(lastUserText) {
    set({ lastUserText })
  },
  clearTurnMarkers() {
    set((s) => ({ turn: { ...s.turn, error: null, refusal: null, warnings: [] } }))
  },
  addNote(level, text) {
    set((s) => ({ outputs: [...s.outputs.slice(-1999), { level, text, ts: Date.now(), origin: 'client' }] }))
  },
  clearOutputs() {
    set({ outputs: [] })
  },
  reconcileRuns(list) {
    set((s) => {
      const runs = { ...s.runs }
      for (const r of list) runs[r.id] = r
      return { runs }
    })
  },
  setCurrentSessionId(currentSessionId) {
    set({ currentSessionId })
  },
}))

// ---- selectors --------------------------------------------------------------

export const selectSession = (s: SessionStore): SessionState | null => s.session
export const selectRunData = (runId: string | null) => (s: SessionStore): RunData => (runId ? (s.runData[runId] ?? EMPTY_RUN_DATA) : EMPTY_RUN_DATA)
export const selectRun = (runId: string | null) => (s: SessionStore): RunInfo | null => (runId ? (s.runs[runId] ?? null) : null)

/** Newest first. Not a selector (returns a fresh array): call it inside useMemo on `s.runs`. */
export function sortRuns(runs: Record<string, RunInfo>): RunInfo[] {
  return Object.values(runs).sort((a, b) => (a.startedAt < b.startedAt ? 1 : a.startedAt > b.startedAt ? -1 : 0))
}

/** The run the UI should follow when the user has not picked one: the latest running, else the latest. */
export function pickActiveRun(runs: RunInfo[], preferred: string | null): RunInfo | null {
  if (preferred) {
    const p = runs.find((r) => r.id === preferred)
    if (p) return p
  }
  return runs.find((r) => r.status === 'running' || r.status === 'queued') ?? runs[0] ?? null
}

/** Not a selector (returns a fresh array): call it inside useMemo on `s.problems`. */
export function flattenProblems(problems: Record<string, Problem[]>): Problem[] {
  const lists = Object.values(problems)
  if (!lists.length) return NO_PROBLEMS
  return lists.flat()
}

export function isDemo(s: SessionStore): boolean {
  return s.hello?.mode === 'demo'
}
