import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, test } from 'vitest'
import type { DatasetProgress } from '@cfd/shared'
import { loadConfig } from '../../config.js'
import { cartesianCellCenters } from '../../formats/cartesian.js'
import { analyticFields, makeFoamCase, makeJsoncCase, makeVtuSeries } from '../fixtures/makeCase.js'
import { createDatasetService, type DatasetServiceHandle } from '../service.js'

let workspace: string
let service: DatasetServiceHandle
let jsoncRel: string
let outRel: string

beforeAll(async () => {
  workspace = await fs.mkdtemp(path.join(os.tmpdir(), 'cfd-ws-'))
  const cases = path.join(workspace, 'cases')
  await makeJsoncCase(cases)
  await makeFoamCase(path.join(workspace, 'foamCase'))
  await makeVtuSeries(path.join(workspace, 'vtkCase'))
  const config = loadConfig({ CFD_WORKSPACE: workspace, CFD_GUI_DIR: path.join(workspace, 'gui'), CFD_DEMO: '1' })
  service = createDatasetService({ config, workers: 2 })
  jsoncRel = 'cases/fixture.jsonc'
  outRel = 'cases/fixture_jsonc'
})
afterAll(async () => {
  await service.close()
  await fs.rm(workspace, { recursive: true, force: true })
})

const f32 = (buf: Buffer): Float32Array => new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength))
const u32 = (buf: Buffer): Uint32Array => new Uint32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength))

describe('JSONC output tree', () => {
  test('open -> manifest with cartesian geometry, times, fields and time fallback', async () => {
    const progress: DatasetProgress[] = []
    const readyEvent = new Promise<void>((resolve) => {
      const off = service.onProgress((p) => {
        progress.push(p)
        if (p.stage === 'ready' || p.stage === 'error') {
          off()
          resolve()
        }
      })
    })
    const opened = await service.open(outRel)
    expect(opened.status).toBe('loading')
    const m = await service.whenReady(opened.datasetId)
    // whenReady resolves as soon as the manifest exists; the 'ready' event follows the initial field parse.
    await readyEvent
    expect(service.get(opened.datasetId)).toBe(m)
    expect(progress.map((p) => p.stage)).toEqual(['queued', 'scanning', 'geometry', 'geometry', 'fields', 'ready'])
    expect(progress[progress.length - 1].pct).toBe(100)

    expect(m.source).toBe('cartesian')
    expect(m.geometryFidelity).toBe('exact')
    expect(m.name).toBe('fixtureCase')
    expect(m.path).toBe(outRel)
    expect(m.cellCount).toBe(64)
    expect(m.up).toBe('y')
    expect(m.bounds).toEqual({ min: [0, 0, 0], max: [4, 2, 1] })
    expect(m.grid).not.toBeNull()
    expect(m.grid!.dims).toEqual([8, 4, 2])
    expect(m.grid!.uniform).toBe(false)
    expect(m.grid!.emptyAxis).toBeNull()
    expect(m.grid!.nodes.x).toEqual({ key: 'grid.x', dtype: 'f32', count: 9, components: 1, bytes: 36 })
    expect(m.surface.patches.map((p) => p.name)).toEqual(['inlet', 'outlet', 'vent', 'floor', 'ceiling', 'sideMin', 'sideMax'])
    expect(m.surface.patches[2].type).toBe('patch')
    expect(m.surface.patches[3].type).toBe('wall')
    expect(m.surface.triangleCount).toBe(2 * (8 + 8 + 16 + 16 + 32 + 32))
    expect(m.surface.vertexCount).toBe(4 * (8 + 8 + 16 + 16 + 32 + 32))
    expect(m.times.map((t) => t.label)).toEqual(['0', '50', '100'])
    expect(m.times.map((t) => t.value)).toEqual([0, 50, 100])
    expect(m.fields.map((f) => f.name)).toEqual(['U', 'epsilon', 'k', 'p'])
    const U = m.fields.find((f) => f.name === 'U')!
    expect(U.components).toBe(3)
    expect(U.unit).toBe('m/s')
    expect(U.perTime.map((p) => [p.timeIndex, p.sourceDir])).toEqual([[0, '0'], [1, '0'], [2, '0']])
    const p = m.fields.find((f) => f.name === 'p')!
    expect(p.perTime.map((p) => [p.timeIndex, p.sourceDir])).toEqual([[0, '0'], [1, '50'], [2, '100']])
    expect(p.perTime[2].blob).toEqual({ key: 'field.p.2', dtype: 'f32', count: 64, components: 1, bytes: 256 })
    // The initial field (U at the last time) was parsed during load and got a range.
    expect(U.perTime[2].range).not.toBeNull()
    expect(U.perTime[2].range!.max).toBeCloseTo(Math.hypot(3.75, 0.5 * 1.905), 1)
    expect(U.range).toEqual(U.perTime[2].range)
    expect(p.perTime[2].range).toBeNull()
    expect(m.meta.model).toBe('kEpsilon')
    expect(m.meta.caseJsonc).toBe(jsoncRel)
    // The surfaceScalarField phi in 100/ is excluded at discovery, silently.
    expect(m.fields.some((f) => f.name === 'phi')).toBe(false)
    expect(m.warnings).toEqual([])
  })

  test('blobs are served as little-endian typed arrays; field blobs parse on demand', async () => {
    const { datasetId } = await service.open(outRel)
    const m = await service.whenReady(datasetId)
    const gx = f32((await service.blob(datasetId, 'grid.x'))!)
    expect(Array.from(gx)).toEqual([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
    const idx = u32((await service.blob(datasetId, 'surface.indices'))!)
    expect(idx.length).toBe(3 * m.surface.triangleCount)
    const cot = u32((await service.blob(datasetId, 'surface.cellOfTri'))!)
    expect(cot.length).toBe(m.surface.triangleCount)
    expect(Math.max(...cot)).toBeLessThan(64)
    const pos = f32((await service.blob(datasetId, 'surface.positions'))!)
    expect(pos.length).toBe(3 * m.surface.vertexCount)
    expect(await service.blob(datasetId, 'surface.nope')).toBeNull()
    expect(await service.blob(datasetId, 'field.zeta.0')).toBeNull()

    const pInfo = await service.ensureField(datasetId, 'p', 1)
    expect(pInfo.sourceDir).toBe('50')
    expect(pInfo.range).toEqual({ min: -3.75, max: -0.25 })
    const pBlob = f32((await service.blob(datasetId, 'field.p.1'))!)
    const grid = (await import('../fixtures/makeCase.js')).FIXTURE_SPEC
    const centres = cartesianCellCenters((await import('../../formats/cartesian.js')).buildCartesianGrid(grid))
    for (let i = 0; i < 64; i++) expect(pBlob[i]).toBeCloseTo(-centres[3 * i], 6)
    const k2 = await service.ensureField(datasetId, 'k', 2)
    expect(k2.range!.min).toBeCloseTo(1.1, 6)
    expect(k2.range!.max).toBeCloseTo(1.1, 6)
    const kBlob = f32((await service.blob(datasetId, 'field.k.2'))!)
    expect(kBlob.length).toBe(64)
    expect(m.fields.find((f) => f.name === 'k')!.range!.min).toBeCloseTo(1.1, 6)
    await expect(service.ensureField(datasetId, 'p', 7)).rejects.toThrow(/time index 7/)
    await expect(service.ensureField(datasetId, 'zeta', 0)).rejects.toThrow(/no field "zeta"/)
    expect(service.cacheBytes()).toBeGreaterThan(0)
  })

  test('reopening the same path returns ready immediately with the same id; the jsonc and a time dir resolve to the same tree', async () => {
    const first = await service.open(outRel)
    expect(first.status).toBe('ready')
    const viaJsonc = await service.open(jsoncRel)
    const m1 = await service.whenReady(viaJsonc.datasetId)
    expect(m1.cellCount).toBe(64)
    expect(m1.times.length).toBe(3)
    expect(viaJsonc.datasetId).not.toBe(first.datasetId)
    const viaTime = await service.open('cases/fixture_jsonc/50')
    const m2 = await service.whenReady(viaTime.datasetId)
    expect(m2.times.map((t) => t.label)).toEqual(['0', '50', '100'])
    // preferred field at the opened time dir is being parsed: U falls back to 0/ (ensureField joins the in-flight parse)
    const U = m2.fields.find((f) => f.name === 'U')!
    expect(U.perTime[1].sourceDir).toBe('0')
    const parsed = await service.ensureField(viaTime.datasetId, 'U', 1)
    expect(parsed.range).not.toBeNull()
    expect(U.perTime[1].range).toEqual(parsed.range)
  })

  test('a second service instance restores the manifest from the disk cache', async () => {
    const config = loadConfig({ CFD_WORKSPACE: workspace, CFD_GUI_DIR: path.join(workspace, 'gui'), CFD_DEMO: '1' })
    const other = createDatasetService({ config, workers: 0 })
    try {
      const opened = await other.open(outRel)
      expect(opened.status).toBe('ready')
      const m = opened.manifest!
      expect(m.cellCount).toBe(64)
      expect(m.fields.find((f) => f.name === 'p')!.perTime[1].range).toEqual({ min: -3.75, max: -0.25 })
      const gx = f32((await other.blob(opened.datasetId, 'grid.x'))!)
      expect(gx.length).toBe(9)
      const stats = await other.fieldStats(outRel, '50', 'p', 'magnitude', { min: [0, 0, 0], max: [1, 2, 1] })
      expect(stats.count).toBe(16)
    } finally {
      await other.close()
    }
  })

  test('fieldStats: scalar, vector component, region restriction, histogram', async () => {
    const s = await service.fieldStats(outRel, '100', 'p')
    expect(s.component).toBe('scalar')
    expect(s.time).toBe('100')
    expect(s.count).toBe(64)
    expect(s.min).toBeCloseTo(-3.75, 6)
    expect(s.max).toBeCloseTo(-0.25, 6)
    expect(s.mean).toBeCloseTo(-2, 6)
    expect(s.rms).toBeCloseTo(Math.sqrt((0.25 ** 2 + 0.75 ** 2 + 1.25 ** 2 + 1.75 ** 2 + 2.25 ** 2 + 2.75 ** 2 + 3.25 ** 2 + 3.75 ** 2) / 8), 5)
    expect(s.argmin).toBe(7)
    expect(s.argmax).toBe(0)
    expect(s.histogram.edges.length).toBe(17)
    expect(s.histogram.counts.reduce((a, b) => a + b, 0)).toBe(64)

    const ux = await service.fieldStats(outRel, '100', 'U', 'x')
    expect(ux.component).toBe('x')
    expect(ux.min).toBeCloseTo(0.25, 6)
    expect(ux.max).toBeCloseTo(3.75, 6)
    const umag = await service.fieldStats(outRel, 'last', 'U', 'magnitude')
    expect(umag.max).toBeGreaterThan(3.75)

    const region = await service.fieldStats(outRel, '50', 'p', 'magnitude', { min: [0, 0, 0], max: [1, 2, 1] })
    expect(region.count).toBe(16)
    expect(region.min).toBeCloseTo(-0.75, 6)
    expect(region.max).toBeCloseTo(-0.25, 6)
    const eps = await service.fieldStats(outRel, '50', 'epsilon', 'magnitude', { min: [0, 0, 0.5], max: [4, 2, 1] })
    expect(eps.count).toBe(32)
    expect(eps.min).toBeCloseTo(0.75, 6)
    await expect(service.fieldStats(outRel, '75', 'p')).rejects.toThrow(/no time "75"/)
  })

  test('discover', async () => {
    const r = await service.discover(outRel)
    expect(r.root).toBe(outRel)
    expect(r.caseJsonc).toBe(jsoncRel)
    expect(r.hasPolyMesh).toBe(false)
    expect(r.hasVtu).toBe(false)
    expect(r.cellCount).toBe(64)
    expect(r.times.map((t) => [t.label, t.value, t.fields])).toEqual([
      ['0', 0, ['U', 'epsilon', 'k', 'p']],
      ['50', 50, ['epsilon', 'k', 'p']],
      ['100', 100, ['epsilon', 'k', 'p']],
    ])
    expect(r.vtk).toEqual([])
    const viaJsonc = await service.discover(jsoncRel)
    expect(viaJsonc.root).toBe(outRel)
  })
})

describe('OpenFOAM directory case', () => {
  test('polyMesh geometry with lattice detection, constant/g up axis, blockgen patch order', async () => {
    const opened = await service.open('foamCase', { preferField: 'p', timeIndex: 'last' })
    const m = await service.whenReady(opened.datasetId)
    await service.ensureField(opened.datasetId, 'p', 1)
    expect(m.source).toBe('polymesh')
    expect(m.geometryFidelity).toBe('exact')
    expect(m.cellCount).toBe(64)
    expect(m.up).toBe('y')
    expect(m.grid!.dims).toEqual([8, 4, 2])
    expect(m.grid!.uniform).toBe(false)
    expect(m.surface.patches.map((p) => [p.name, p.type])).toEqual([
      ['inlet', 'patch'],
      ['outlet', 'patch'],
      ['vent', 'patch'],
      ['floor', 'wall'],
      ['ceiling', 'wall'],
      ['sideMin', 'wall'],
      ['sideMax', 'wall'],
    ])
    expect(m.times.map((t) => t.label)).toEqual(['0', '200'])
    expect(m.fields.map((f) => f.name)).toEqual(['U', 'p'])
    const p = m.fields.find((f) => f.name === 'p')!
    expect(p.perTime[1].range).toEqual({ min: -3.75, max: -0.25 })
    const gy = f32((await service.blob(opened.datasetId, 'grid.y'))!)
    expect(gy.length).toBe(5)
    expect(gy[2]).toBeCloseTo(1, 6)
    const r = await service.discover('foamCase')
    expect(r.hasPolyMesh).toBe(true)
    expect(r.cellCount).toBe(64)
    const s = await service.fieldStats('foamCase', '200', 'U', 'y', { min: [0, 1, 0], max: [4, 2, 1] })
    expect(s.count).toBe(32)
    expect(s.min).toBeGreaterThan(0.5)
  })
})

describe('VTU series', () => {
  test('pvd -> proxy geometry, lattice from cell centres, cell data fields', async () => {
    const opened = await service.open('vtkCase/VTK/series.pvd')
    const m = await service.whenReady(opened.datasetId)
    expect(m.source).toBe('vtu')
    expect(m.geometryFidelity).toBe('proxy')
    expect(m.cellCount).toBe(64)
    expect(m.grid).not.toBeNull()
    expect(m.grid!.dims).toEqual([8, 4, 2])
    expect(m.warnings.some((w) => w.includes('proxy'))).toBe(true)
    expect(m.surface.patches.map((p) => p.name)).toEqual(['boundary'])
    expect(m.surface.triangleCount).toBe(2 * (8 + 8 + 16 + 16 + 32 + 32))
    expect(m.times.map((t) => [t.label, t.value])).toEqual([
      ['series_000000.vtu', 0],
      ['series_000010.vtu', 0.5],
    ])
    expect(m.fields.map((f) => f.name)).toEqual(['U', 'p'])
    const U = await service.ensureField(opened.datasetId, 'U', 1)
    expect(U.sourceDir).toBe('series_000010.vtu')
    expect(U.range!.min).toBeCloseTo(Math.hypot(0.25, 0.5 * 0.0951), 1)
    const p = await service.fieldStats('vtkCase/VTK/series.pvd', 'series_000010.vtu', 'p')
    expect(p.min).toBeCloseTo(-3.75, 5)
    const gx = f32((await service.blob(opened.datasetId, 'grid.x'))!)
    expect(Array.from(gx).map((v) => Math.round(v * 1e4) / 1e4)).toEqual([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
    const single = await service.open('vtkCase/VTK/series_000000.vtu')
    const ms = await service.whenReady(single.datasetId)
    expect(ms.times.length).toBe(1)
    expect(ms.times[0].value).toBe(0)
    const r = await service.discover('vtkCase/VTK/series.pvd')
    expect(r.hasVtu).toBe(true)
    expect(r.cellCount).toBe(64)
    expect(r.vtk.map((v) => v.kind)).toEqual(['pvd', 'vtu', 'vtu'])
  })
})

describe('errors', () => {
  test('a directory with nothing in it fails with a clear error', async () => {
    await fs.mkdir(path.join(workspace, 'nothing'), { recursive: true })
    const opened = await service.open('nothing')
    expect(opened.status).toBe('loading')
    await expect(service.whenReady(opened.datasetId)).rejects.toThrow(/no geometry/)
    const again = await service.open('nothing')
    expect(again.status).toBe('error')
    expect(again.error).toMatch(/no geometry/)
    service.evict(opened.datasetId)
    expect(service.get(opened.datasetId)).toBeUndefined()
    await expect(service.open('../outside')).rejects.toThrow()
  })
})
