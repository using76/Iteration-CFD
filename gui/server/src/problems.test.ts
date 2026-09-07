import { describe, expect, it } from 'vitest'
import type { RunInfo, ServerMsg } from '@cfd/shared'
import { createProblemsTracker, problemFromLine } from './problems.js'
import type { RunEvent, RunManager } from './runs/types.js'

const run: RunInfo = {
  id: 'r_1',
  binary: 'ofgpu-k-epsilon',
  argv: ['cases/plume.jsonc'],
  cwd: '',
  casePath: 'cases/plume.jsonc',
  outputRoot: 'cases/plume_jsonc',
  status: 'running',
  pid: 1,
  startedAt: new Date().toISOString(),
  endedAt: null,
  exitCode: null,
  signal: null,
  iter: 0,
  targetIter: null,
  time: null,
  endTime: null,
  lastResidual: null,
  written: [],
  error: null,
  converged: false,
  device: null,
  logLines: 0,
  mode: 'demo',
  label: null,
}

describe('problems', () => {
  it('classifies solver lines', () => {
    expect(problemFromLine(run, 3, 'error: no case directory given')).toMatchObject({ severity: 'error', source: 'solver', logSeq: 3, runId: 'r_1', path: 'cases/plume.jsonc' })
    expect(problemFromLine(run, 4, 'this case asks for the kOmegaSST model; run it with ofgpu-k-omega')).toMatchObject({ severity: 'error', hint: 'Run the named driver instead.' })
    expect(problemFromLine(run, 5, 'WARNING: relaxation factor 1.0')).toMatchObject({ severity: 'warning' })
    expect(problemFromLine(run, 6, '    400  epsilon res 3.612e-04 (14)')).toBeNull()
  })

  it('broadcasts per-run problem lists from log and exit events', async () => {
    let handler: ((ev: RunEvent) => void) | null = null
    const runs = {
      on: (h: (ev: RunEvent) => void) => {
        handler = h
        return () => {}
      },
      get: () => run,
    } as unknown as RunManager
    const frames: ServerMsg[] = []
    const tracker = createProblemsTracker({ runs, hub: { broadcast: (m) => frames.push(m) } })
    handler!({ type: 'log', runId: 'r_1', lines: [{ seq: 1, stream: 'stdout', text: 'mesh uploaded in 1 s', ts: 0 }, { seq: 2, stream: 'stderr', text: 'error: unknown option -x', ts: 0 }] })
    handler!({ type: 'exit', run: { ...run, status: 'failed', error: 'unknown option -x', exitCode: 1 } })
    await new Promise((r) => setTimeout(r, 150))
    expect(frames).toHaveLength(1)
    const f = frames[0]
    expect(f.t).toBe('problems')
    if (f.t !== 'problems') return
    expect(f.runId).toBe('r_1')
    expect(f.items).toHaveLength(1)
    expect(f.items[0]).toMatchObject({ id: 'r_1:2', severity: 'error', message: 'error: unknown option -x' })
    expect(tracker.problems('r_1')).toHaveLength(1)
    handler!({ type: 'exit', run: { ...run, id: 'r_2', status: 'diverged' } })
    await new Promise((r) => setTimeout(r, 150))
    expect(frames[1].t === 'problems' && frames[1].items[0].message).toContain('diverged')
    tracker.close()
  })
})
