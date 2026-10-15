// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// STU-14: a non-lattice polyMesh (the warped sphere-in-box) is contoured in the worker pool with
// polyiso, and the stored triangles are exactly polyiso's: the same cells, the same corners and
// normals rounded once to f32. The lattice sphere-in-box keeps its marching cubes and is refused.
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, test } from 'vitest'
import { loadConfig } from '../../config.js'
import { buildIsoMesh, isoSurface } from '../../formats/polyiso.js'
import { readPolyMesh } from '../../formats/polymesh.js'
import { makeSphereInBox } from '../fixtures/makeSphereInBox.js'
import { createDatasetService, type DatasetServiceHandle } from '../service.js'

let workspace: string
let service: DatasetServiceHandle
let warpedId: string
let latticeId: string

beforeAll(async () => {
  workspace = await fs.mkdtemp(path.join(os.tmpdir(), 'cfd-uiso-'))
  await makeSphereInBox(path.join(workspace, 'warped'), { warp: true })
  await makeSphereInBox(path.join(workspace, 'lattice'))
  const config = loadConfig({ CFD_WORKSPACE: workspace, CFD_GUI_DIR: path.join(workspace, 'gui'), CFD_DEMO: '1' })
  service = createDatasetService({ config, workers: 2 })
  warpedId = (await service.open('warped')).datasetId
  latticeId = (await service.open('lattice')).datasetId
  await service.whenReady(warpedId)
  await service.whenReady(latticeId)
})
afterAll(async () => {
  await service.close()
  await fs.rm(workspace, { recursive: true, force: true })
})

const f32 = (buf: Buffer): Float32Array => new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength))
const u32 = (buf: Buffer): Uint32Array => new Uint32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength))
const same = (a: Float32Array, b: Float32Array): boolean => Buffer.from(a.buffer).equals(Buffer.from(b.buffer))

describe('iso-surfaces of a non-lattice polyMesh', () => {
  test('p = 227.5 on the warped sphere-in-box: the stored triangles are exactly polyiso', async () => {
    const info = await service.iso(warpedId, { field: 'p', timeIndex: 0, values: [227.5] })
    const mesh = await readPolyMesh(path.join(workspace, 'warped', 'constant', 'polyMesh'))
    const p = new Float64Array(mesh.nCells)
    for (let c = 0; c < p.length; c++) p[c] = c
    const ref = isoSurface(buildIsoMesh(mesh), p, [227.5])
    console.log(`STU-14 warped sphere-in-box p = 227.5: ${info.triangleCount} triangles, area ${info.area.toFixed(4)}, fallback points ${info.fallbackPoints}, inverted tets ${info.invertedTets}`)
    expect(info.triangleCount).toBeGreaterThan(0)
    expect(info.triangleCount).toBe(ref.triCells.length)
    expect(info.valueCounts).toEqual(ref.valueCounts)
    expect(info.area).toBe(ref.area)
    expect(info.fallbackPoints).toBe(ref.fallbackPoints)
    expect(info.invertedTets).toBe(ref.invertedTets)
    expect(info.positions).toMatchObject({ dtype: 'f32', components: 3, count: 3 * info.triangleCount })
    expect(info.normals).toMatchObject({ dtype: 'f32', components: 3, count: 3 * info.triangleCount })
    expect(info.cells).toMatchObject({ dtype: 'u32', components: 1, count: info.triangleCount })
    expect([...u32((await service.blob(warpedId, info.cells.key))!)]).toEqual([...ref.triCells])
    expect(same(f32((await service.blob(warpedId, info.positions.key))!), Float32Array.from(ref.triPositions))).toBe(true)
    expect(same(f32((await service.blob(warpedId, info.normals.key))!), Float32Array.from(ref.triNormals))).toBe(true)
    expect(await service.iso(warpedId, { field: 'p', timeIndex: 0, values: [227.5] })).toBe(info)
    expect(info.key).toMatch(/^iso\.[0-9a-f]{16}$/)
  })

  test('a value outside the field stores an empty surface', async () => {
    const info = await service.iso(warpedId, { field: 'p', timeIndex: 0, values: [1e6] })
    expect(info.triangleCount).toBe(0)
    expect(info.valueCounts).toEqual([0])
    expect(info.area).toBe(0)
    expect((await service.blob(warpedId, info.positions.key))!.length).toBe(0)
  })

  test('the lattice keeps its marching cubes, and bad requests are refused by name', async () => {
    await expect(service.iso(latticeId, { field: 'p', timeIndex: 0, values: [1] })).rejects.toThrow(/^NO_POLY_ISO: /)
    await expect(service.iso(warpedId, { field: 'T', timeIndex: 0, values: [1] })).rejects.toThrow(/^INVALID: no field "T"/)
    await expect(service.iso(warpedId, { field: 'p', timeIndex: 0, values: [] })).rejects.toThrow(/^INVALID: /)
    await expect(service.iso(warpedId, { field: 'p', timeIndex: 0, values: [Number.NaN] })).rejects.toThrow(/^INVALID: /)
  })
})
