import { spawn } from 'node:child_process'
import fsp from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { describe, expect, it } from 'vitest'
import type { RunEndWord, RunInfo, RunStatus } from '@cfd/shared'
import { closeStatus, createRunManager, type RunManagerDeps } from './manager.js'
import { makeTempWorkspace, type TempWorkspace } from './test-helpers.js'
import type { RunManagerHandle } from './types.js'

describe('run manager provenance', () => {
  it('start() stamps the case, the mesh and the machine into the record it returns', async () => {
    const ws: TempWorkspace = await makeTempWorkspace()
    try {
      // The mesh the plume case reads, with its summary: what mints a Mesh key.
      const poly = path.join(ws.root, 'cases', 'plume_jsonc', 'constant', 'polyMesh')
      await fsp.mkdir(poly, { recursive: true })
      await fsp.writeFile(path.join(poly, 'points'), '')
      await fsp.writeFile(path.join(poly, '.meshSummary.json'), JSON.stringify({ caseDir: 'cases/plume_jsonc' }))
      const runs = await createRunManager({ config: ws.config })
      try {
        const run = await runs.start({ binary: 'ofgpu-k-epsilon', casePath: 'cases/plume.jsonc', positionals: [], args: [], label: null })
        expect(run.caseId).toBe('cases/plume.jsonc')
        expect(run.meshId).toBe('cases/plume_jsonc/constant/polyMesh/.meshSummary.json#plume_jsonc')
        expect(run.machine?.hostname).toBe(os.hostname())
        // The monitor may or may not have polled; assert the type, never the value.
        expect(typeof run.machine?.gpu).toBe('string')
        expect('gitSha' in run && 'gitDirty' in run).toBe(true)
      } finally {
        await runs.shutdown()
      }
    } finally {
      await ws.cleanup()
    }
  })
})

type Scripted = Array<{ text: string; stream?: 'stdout' | 'stderr' }>
// The lines travel in the environment, never in the -e source, so no quoting
// survives into the script. The child prints each line, then hangs or exits.
const SCRIPT = `const L = JSON.parse(process.env.CFD_SCRIPTED_LINES); for (const l of L) (l.stream === 'stderr' ? process.stderr : process.stdout).write(l.text + '\\n'); if (process.env.CFD_SCRIPTED_HANG === '1') setInterval(() => {}, 1000); else process.exitCode = Number(process.env.CFD_SCRIPTED_CODE)`
function scripted(lines: Scripted, code: number, hang = false): NonNullable<RunManagerDeps['dispatch']> {
  return () => {
    const env = { ...process.env, CFD_SCRIPTED_LINES: JSON.stringify(lines), CFD_SCRIPTED_CODE: String(code), CFD_SCRIPTED_HANG: hang ? '1' : '0' }
    const child = spawn(process.execPath, ['-e', SCRIPT], { env, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true })
    return { child, argv: ['scripted'], mode: 'real' as const }
  }
}

/** Resolves with the run's `exit` event; rejects if none arrives in `ms`. */
function exitOf(runs: RunManagerHandle, id: string, ms = 5000): Promise<RunInfo> {
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => { off(); reject(new Error(`no exit event for ${id} within ${ms} ms`)) }, ms)
    const off = runs.on((ev) => {
      if (ev.type === 'exit' && ev.run.id === id) { clearTimeout(t); off(); resolve(ev.run) }
    })
  })
}

describe('how a run ends', () => {
  const REQUEST = { binary: 'ofgpu-k-epsilon', casePath: 'cases/plume.jsonc', positionals: [], args: [], label: null }

  it('a NaN line with no run ended line still ends the run: exit event, diverged, exit code 1', async () => {
    const ws: TempWorkspace = await makeTempWorkspace()
    const runs = await createRunManager({ config: ws.config, dispatch: scripted([{ text: '*** NaN/Inf ***' }], 1) })
    try {
      const run = await runs.start(REQUEST)
      const exited = exitOf(runs, run.id)
      const out = await exited
      expect(out.status).toBe('diverged')
      expect(out.exitCode).toBe(1)
      expect(typeof out.endedAt).toBe('string')
      expect(out.endWord).toBeUndefined()
    } finally {
      await runs.shutdown()
      await ws.cleanup()
    }
  }, 15_000)

  async function endsWith(lines: Scripted, code: number): Promise<RunInfo> {
    const ws: TempWorkspace = await makeTempWorkspace()
    const runs = await createRunManager({ config: ws.config, dispatch: scripted(lines, code) })
    try {
      const run = await runs.start(REQUEST)
      return await exitOf(runs, run.id)
    } finally {
      await runs.shutdown()
      await ws.cleanup()
    }
  }

  it('closeStatus is the ending table, row for row', () => {
    const rows: Array<{ code: number | null; signal: string | null; current: RunStatus; endWord: RunEndWord | undefined; error: string | null; status: RunStatus }> = [
      { code: 3, signal: null, current: 'running', endWord: 'refused', error: null, status: 'failed' },
      { code: 1, signal: null, current: 'running', endWord: 'error', error: null, status: 'failed' },
      { code: 2, signal: null, current: 'running', endWord: 'diverged', error: null, status: 'diverged' },
      { code: 0, signal: null, current: 'running', endWord: 'budget', error: null, status: 'done' },
      { code: 0, signal: null, current: 'running', endWord: undefined, error: null, status: 'done' },
      { code: 0, signal: null, current: 'running', endWord: undefined, error: 'x', status: 'failed' },
      { code: 1, signal: null, current: 'running', endWord: undefined, error: null, status: 'failed' },
      { code: 3, signal: null, current: 'running', endWord: undefined, error: null, status: 'failed' },
      { code: null, signal: 'SIGTERM', current: 'running', endWord: undefined, error: null, status: 'killed' },
      { code: 1, signal: null, current: 'killed', endWord: undefined, error: null, status: 'killed' },
      { code: 1, signal: null, current: 'diverged', endWord: undefined, error: null, status: 'diverged' },
    ]
    for (const { code, signal, current, endWord, error, status } of rows) {
      expect(closeStatus({ code, signal, current, endWord, error })).toBe(status)
    }
  }, 15_000)

  it('a refused run ends failed with its own word, apart from an errored one', async () => {
    const out = await endsWith([
      { text: 'error: -writeInterval: "10" is not supported by ofgpu', stream: 'stderr' },
      { text: 'run ended: refused | -writeInterval: "10" is not supported by ofgpu | exit code 3' },
    ], 3)
    expect(out.status).toBe('failed')
    expect(out.exitCode).toBe(3)
    expect(out.endWord).toBe('refused')
    expect(out.endDetail).toBe('-writeInterval: "10" is not supported by ofgpu')
    expect(out.error).toBe('-writeInterval: "10" is not supported by ofgpu')
  }, 15_000)

  it('an errored run ends failed with the error word', async () => {
    const out = await endsWith([{ text: 'run ended: error | boom | exit code 1' }], 1)
    expect(out.status).toBe('failed')
    expect(out.exitCode).toBe(1)
    expect(out.endWord).toBe('error')
    expect(out.error).toBe('boom')
  }, 15_000)

  it('a diverged run ends diverged', async () => {
    const out = await endsWith([{ text: 'run ended: diverged | diverged at outer iteration 12: a field went non-finite (NaN/Inf) | exit code 2' }], 2)
    expect(out.status).toBe('diverged')
    expect(out.exitCode).toBe(2)
    expect(out.endWord).toBe('diverged')
  }, 15_000)

  it('a budget run ends done with no error', async () => {
    const out = await endsWith([{ text: 'run ended: budget | 30 iterations reached | exit code 0' }], 0)
    expect(out.status).toBe('done')
    expect(out.exitCode).toBe(0)
    expect(out.endWord).toBe('budget')
    expect(out.error).toBeNull()
  }, 15_000)

  it('a plain finished run is done and its snapshot carries no end word key at all', async () => {
    const out = await endsWith([{ text: '    400  epsilon res 3.612e-04 (14)  k res 2.081e-04 (9)' }], 0)
    expect(out.status).toBe('done')
    expect(out.endWord).toBeUndefined()
    expect('endWord' in out).toBe(false)
  }, 15_000)

  it('a run stopped while it is still starting is never spawned', async () => {
    const ws: TempWorkspace = await makeTempWorkspace()
    let spawned = 0
    const inner = scripted([{ text: 'iterating 100 times' }], 0, true)
    const runs = await createRunManager({ config: ws.config, dispatch: (o) => { spawned++; return inner(o) } })
    try {
      const run = await runs.start(REQUEST)
      const stopped = await runs.stop(run.id)
      expect(stopped.status).toBe('killed')
      await new Promise((resolve) => setTimeout(resolve, 500))
      expect(spawned).toBe(0)
      expect(runs.get(run.id)?.status).toBe('killed')
    } finally {
      await runs.shutdown()
      await ws.cleanup()
    }
  }, 15_000)

  it('stop() resolves on the process close, not on the guard', async () => {
    const ws: TempWorkspace = await makeTempWorkspace()
    const runs = await createRunManager({ config: ws.config, dispatch: scripted([{ text: 'iterating 100 times' }], 0, true) })
    try {
      const run = await runs.start(REQUEST)
      await runs.wait(run.id, { maxMs: 5000, untilStatus: ['running'] })
      expect(runs.get(run.id)?.status).toBe('running')
      expect(runs.get(run.id)?.targetIter).toBe(100)
      const t0 = Date.now()
      const stopped = await runs.stop(run.id)
      const elapsed = Date.now() - t0
      expect(stopped.status).toBe('killed')
      expect(typeof stopped.endedAt).toBe('string')
      expect(elapsed).toBeLessThan(6000)
    } finally {
      await runs.shutdown()
      await ws.cleanup()
    }
  }, 15_000)
})
