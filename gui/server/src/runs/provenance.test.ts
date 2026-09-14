import fsp from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import type { RunInfo } from '@cfd/shared'
import { makeTempWorkspace, type TempWorkspace } from './test-helpers.js'
import { caseAndMeshIds, collectRunProvenance, machineOf, normalizeProvenance } from './provenance.js'

// A polyMesh only needs an empty `points` file (that is all findPolyMeshDir
// probes); a mesh becomes a Mesh only once a .meshSummary.json sits beside it.
// The full shape is MeshSummaryRecord (../formats/meshSummary.ts) but only
// `caseDir` is load-bearing here.
const summary = (caseDir: string) => JSON.stringify({ caseDir })

async function writePolyMesh(root: string, caseRel: string, summaryText?: string) {
  const poly = path.join(root, caseRel, 'constant', 'polyMesh')
  await fsp.mkdir(poly, { recursive: true })
  await fsp.writeFile(path.join(poly, 'points'), '')
  if (summaryText !== undefined) await fsp.writeFile(path.join(poly, '.meshSummary.json'), summaryText)
  return poly
}

/** A minimal foam case: system/controlDict plus constant/polyMesh. */
async function writeFoamCase(root: string, caseRel: string, summaryText?: string) {
  await fsp.mkdir(path.join(root, caseRel, 'system'), { recursive: true })
  await fsp.writeFile(path.join(root, caseRel, 'system', 'controlDict'), '')
  await writePolyMesh(root, caseRel, summaryText)
}

describe('case and mesh ids', () => {
  let ws: TempWorkspace
  afterEach(async () => {
    if (ws) await ws.cleanup()
  })

  it('a jsonc case names itself and the mesh by the summary beside it', async () => {
    ws = await makeTempWorkspace()
    await writePolyMesh(ws.root, 'cases/plume_jsonc', summary('cases/plume_jsonc'))
    await expect(caseAndMeshIds(ws.root, 'cases/plume.jsonc', false)).resolves.toEqual({
      caseId: 'cases/plume.jsonc',
      meshId: 'cases/plume_jsonc/constant/polyMesh/.meshSummary.json#plume_jsonc',
    })
  })

  it('a foam case directory is its own case id', async () => {
    ws = await makeTempWorkspace({ plume: false })
    await writeFoamCase(ws.root, 'cases/box', summary('cases/box'))
    await expect(caseAndMeshIds(ws.root, 'cases/box', false)).resolves.toEqual({
      caseId: 'cases/box',
      meshId: 'cases/box/constant/polyMesh/.meshSummary.json#box',
    })
  })

  it('a polyMesh without a summary is not a Mesh yet', async () => {
    ws = await makeTempWorkspace({ plume: false })
    await writeFoamCase(ws.root, 'cases/box')
    await expect(caseAndMeshIds(ws.root, 'cases/box', false)).resolves.toEqual({ caseId: 'cases/box', meshId: null })
    await fsp.writeFile(path.join(ws.root, 'cases', 'box', 'constant', 'polyMesh', '.meshSummary.json'), '{ not json')
    await expect(caseAndMeshIds(ws.root, 'cases/box', false)).resolves.toEqual({ caseId: 'cases/box', meshId: null })
  })

  it('a summary with no caseDir falls back to the case directory name', async () => {
    ws = await makeTempWorkspace({ plume: false })
    await writeFoamCase(ws.root, 'cases/box', '{}')
    await expect(caseAndMeshIds(ws.root, 'cases/box', false)).resolves.toEqual({
      caseId: 'cases/box',
      meshId: 'cases/box/constant/polyMesh/.meshSummary.json#box',
    })
  })

  it('a mesh run never claims the mesh it is about to overwrite', async () => {
    ws = await makeTempWorkspace({ plume: false })
    await writeFoamCase(ws.root, 'cases/box', summary('cases/box'))
    await expect(caseAndMeshIds(ws.root, 'cases/box', true)).resolves.toEqual({ caseId: 'cases/box', meshId: null })
  })

  it('anything that is not a case yields two nulls', async () => {
    ws = await makeTempWorkspace({ plume: false })
    await fsp.writeFile(path.join(ws.root, 'cases', 'nope.step'), '')
    await expect(caseAndMeshIds(ws.root, null, false)).resolves.toEqual({ caseId: null, meshId: null })
    await expect(caseAndMeshIds(ws.root, 'cases/nope.step', false)).resolves.toEqual({ caseId: null, meshId: null })
    await expect(caseAndMeshIds(ws.root, 'cases/missing', false)).resolves.toEqual({ caseId: null, meshId: null })
  })
})

describe('machineOf', () => {
  it('an unknown gpu is the empty string, and hostname is the main field', () => {
    expect(machineOf(null)).toEqual({ hostname: os.hostname(), gpu: '', platform: process.platform })
    expect(machineOf('NVIDIA GeForce RTX 4090').gpu).toBe('NVIDIA GeForce RTX 4090')
    expect(Object.keys(machineOf(null))).toEqual(['hostname', 'gpu', 'platform'])
  })
})

describe('collectRunProvenance', () => {
  let ws: TempWorkspace
  afterEach(async () => {
    if (ws) await ws.cleanup()
  })

  it('returns five keys and never throws', async () => {
    ws = await makeTempWorkspace()
    const p = await collectRunProvenance({ workspaceRoot: ws.root, probeRel: 'cases/plume.jsonc', isMeshRun: false, gpuName: 'GPU-X' })
    expect(Object.keys(p).sort()).toEqual(['caseId', 'gitDirty', 'gitSha', 'machine', 'meshId'])
    expect(p.machine).toEqual({ hostname: os.hostname(), gpu: 'GPU-X', platform: process.platform })
    expect(p.caseId).toBe('cases/plume.jsonc')
    expect(p.gitSha).toBeNull()
    expect(p.gitDirty).toBeNull()
  })
})

describe('normalizeProvenance', () => {
  it('fills what is missing and preserves what is there, false included', () => {
    const legacy: RunInfo = {
      id: 'r_1',
      binary: 'ofgpu-k-epsilon',
      argv: ['ofgpu-k-epsilon', 'cases/plume.jsonc'],
      cwd: '',
      casePath: 'cases/plume.jsonc',
      outputRoot: 'cases/plume_jsonc',
      status: 'done',
      pid: null,
      startedAt: '2026-09-15T00:00:00.000Z',
      endedAt: null,
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
    const filled = normalizeProvenance(legacy)
    expect(filled.gitSha).toBeNull()
    expect(filled.gitDirty).toBeNull()
    expect(filled.caseId).toBeNull()
    expect(filled.meshId).toBeNull()
    expect(filled.machine).toBeNull()
    const partial: RunInfo = { ...legacy, gitDirty: false, caseId: 'cases/x.jsonc' }
    const kept = normalizeProvenance(partial)
    expect(kept.gitDirty).toBe(false)
    expect(kept.caseId).toBe('cases/x.jsonc')
    // The input object is unmodified and the function returns a new object.
    expect(partial.gitDirty).toBe(false)
    expect(partial.caseId).toBe('cases/x.jsonc')
    expect('gitSha' in partial).toBe(false)
    expect(kept).not.toBe(partial)
  })
})
