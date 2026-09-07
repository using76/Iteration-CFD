// Problems panel feed: solver refusals, warnings and divergence from run
// logs (SOLVER_PROBLEM_PATTERNS), plus failed exits, broadcast per run.
import { SOLVER_PROBLEM_PATTERNS, type Problem, type RunInfo } from '@cfd/shared'
import type { RunManager } from './runs/types.js'
import type { Hub } from './ws/types.js'

export interface ProblemsTracker {
  problems(runId: string): Problem[]
  close(): void
}

const BROADCAST_DEBOUNCE_MS = 100

export function problemFromLine(run: RunInfo, seq: number, text: string): Problem | null {
  for (const p of SOLVER_PROBLEM_PATTERNS) {
    if (!p.re.test(text)) continue
    return {
      id: `${run.id}:${seq}`,
      severity: p.severity,
      message: text.trim(),
      source: 'solver',
      path: run.casePath,
      line: null,
      col: null,
      runId: run.id,
      logSeq: seq,
      hint: p.hint,
    }
  }
  return null
}

export function createProblemsTracker(deps: { runs: RunManager; hub: Pick<Hub, 'broadcast'> }): ProblemsTracker {
  const items = new Map<string, Problem[]>()
  const timers = new Map<string, NodeJS.Timeout>()

  function schedule(runId: string) {
    if (timers.has(runId)) return
    timers.set(
      runId,
      setTimeout(() => {
        timers.delete(runId)
        const run = deps.runs.get(runId)
        deps.hub.broadcast({ t: 'problems', source: 'solver', path: run?.casePath ?? null, runId, items: items.get(runId) ?? [] })
      }, BROADCAST_DEBOUNCE_MS),
    )
  }

  function add(runId: string, problem: Problem) {
    const list = items.get(runId) ?? []
    if (list.some((p) => p.id === problem.id)) return
    list.push(problem)
    items.set(runId, list)
    schedule(runId)
  }

  const off = deps.runs.on((ev) => {
    if (ev.type === 'log') {
      const run = deps.runs.get(ev.runId)
      if (!run) return
      for (const line of ev.lines) {
        if (line.stream === 'system') continue
        const p = problemFromLine(run, line.seq, line.text)
        if (p) add(ev.runId, p)
      }
      return
    }
    if (ev.type === 'exit') {
      const run = ev.run
      if (run.status === 'diverged') {
        add(run.id, { id: `${run.id}:diverged`, severity: 'error', message: `run diverged (${run.binary} on ${run.casePath ?? '?'})`, source: 'solver', path: run.casePath, line: null, col: null, runId: run.id, logSeq: null, hint: 'Lower the relaxation factors or the time step and rerun.' })
      } else if (run.status === 'failed') {
        const known = (items.get(run.id) ?? []).some((p) => p.severity === 'error')
        if (!known) add(run.id, { id: `${run.id}:exit`, severity: 'error', message: run.error ?? `${run.binary} exited with code ${run.exitCode ?? '?'}`, source: 'server', path: run.casePath, line: null, col: null, runId: run.id, logSeq: null, hint: null })
      }
      schedule(run.id)
    }
  })

  return {
    problems: (runId) => [...(items.get(runId) ?? [])],
    close() {
      off()
      for (const t of timers.values()) clearTimeout(t)
      timers.clear()
    },
  }
}
