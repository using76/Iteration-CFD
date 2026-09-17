import fsp from 'node:fs/promises'
import path from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import type { RunInfo } from '@cfd/shared'
import { silentLogger } from '../log.js'
import { makeTempWorkspace, type TempWorkspace } from './test-helpers.js'
import { loadPastRuns } from './store.js'

// The 24-key shape every legacy run.json carries (status 'done' so the
// in-memory repair for a dead server does not fire).
const legacy: RunInfo = {
  id: 'r_1',
  binary: 'ofgpu-k-epsilon',
  argv: ['ofgpu-k-epsilon', 'cases/plume.jsonc'],
  cwd: '',
  casePath: 'cases/plume.jsonc',
  outputRoot: 'cases/plume_jsonc',
  status: 'done',
  pid: null,
  startedAt: '2026-09-15T00:00:01.000Z',
  endedAt: '2026-09-15T00:01:01.000Z',
  exitCode: 0,
  signal: null,
  iter: 10,
  targetIter: 100,
  time: null,
  endTime: null,
  lastResidual: null,
  written: [],
  error: null,
  converged: false,
  device: '',
  logLines: 0,
  mode: 'demo',
  label: null,
}

describe('loadPastRuns', () => {
  let ws: TempWorkspace
  afterEach(async () => {
    if (ws) await ws.cleanup()
  })

  async function seedRuns(): Promise<string> {
    ws = await makeTempWorkspace({ plume: false })
    const runsDir = path.join(ws.tmp, 'runs')
    await fsp.mkdir(path.join(runsDir, 'r_1'), { recursive: true })
    await fsp.writeFile(path.join(runsDir, 'r_1', 'run.json'), JSON.stringify(legacy))
    const modern: RunInfo = {
      ...legacy,
      id: 'r_2',
      startedAt: '2026-09-15T00:00:02.000Z',
      gitSha: 'b'.repeat(40),
      gitDirty: false,
      caseId: 'cases/plume.jsonc',
      meshId: 'cases/box/constant/polyMesh/.meshSummary.json#box',
      machine: { hostname: 'H', gpu: 'X', platform: 'win32' },
    }
    await fsp.mkdir(path.join(runsDir, 'r_2'), { recursive: true })
    await fsp.writeFile(path.join(runsDir, 'r_2', 'run.json'), JSON.stringify(modern))
    return runsDir
  }

  it('a legacy run.json comes back with five explicit nulls', async () => {
    const runsDir = await seedRuns()
    const records = await loadPastRuns(runsDir, silentLogger)
    expect(records.map((r) => r.id)).toEqual(['r_2', 'r_1'])
    const old = records.find((r) => r.id === 'r_1')!
    expect(old.gitSha).toBeNull()
    expect(old.gitDirty).toBeNull()
    expect(old.caseId).toBeNull()
    expect(old.meshId).toBeNull()
    expect(old.machine).toBeNull()
    const fresh = records.find((r) => r.id === 'r_2')!
    expect(fresh.gitSha).toBe('b'.repeat(40))
    expect(fresh.gitDirty).toBe(false)
    expect(fresh.caseId).toBe('cases/plume.jsonc')
    expect(fresh.meshId).toBe('cases/box/constant/polyMesh/.meshSummary.json#box')
    expect(fresh.machine).toEqual({ hostname: 'H', gpu: 'X', platform: 'win32' })
  })

  it('reading a legacy record leaves the file byte-identical', async () => {
    const runsDir = await seedRuns()
    const runJson = path.join(runsDir, 'r_1', 'run.json')
    const before = await fsp.readFile(runJson, 'utf8')
    await loadPastRuns(runsDir, silentLogger)
    const after = await fsp.readFile(runJson, 'utf8')
    expect(after).toBe(before)
  })
})
