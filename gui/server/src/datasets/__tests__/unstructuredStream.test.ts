// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// STU-15: the streamlines of a non-lattice polyMesh (the warped sphere-in-box) are traced in the
// worker pool with polystream, and the stored lines are exactly polystream's: the same offsets,
// the points and speeds rounded once to f32. A seed outside is refused; all seeds outside, and the
// lattice sphere-in-box, are refused by name.
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, test } from 'vitest'
import { loadConfig } from '../../config.js'
import { writeFoamField } from '../../formats/foam.js'
import { readPolyMesh } from '../../formats/polymesh.js'
import { buildStreamMesh, polyStreamlines } from '../../formats/polystream.js'
import { makeSphereInBox } from '../fixtures/makeSphereInBox.js'
import { createDatasetService, type DatasetServiceHandle } from '../service.js'

let workspace: string
let service: DatasetServiceHandle
let warpedId: string
let latticeId: string
const U = [1, 0.5, 0.25]
const PATCHES = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax', 'sphere'].map((name) => ({ name, type: 'zeroGradient' }))

async function addU(dir: string): Promise<void> {
  const data = new Float64Array(3 * 456)
  for (let c = 0; c < 456; c++) data.set(U, 3 * c)
  await writeFoamField(path.join(dir, '0', 'U'), { name: 'U', class: 'volVectorField', dimensions: '[0 1 -1 0 0 0 0]', time: '0', data, components: 3, patches: PATCHES, collapseUniform: false })
}

beforeAll(async () => {
  workspace = await fs.mkdtemp(path.join(os.tmpdir(), 'cfd-ustream-'))
  await makeSphereInBox(path.join(workspace, 'warped'), { warp: true })
  await makeSphereInBox(path.join(workspace, 'lattice'))
  await addU(path.join(workspace, 'warped'))
  await addU(path.join(workspace, 'lattice'))
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

describe('streamlines of a non-lattice polyMesh', () => {
  test('uniform U on the warped sphere-in-box: the stored lines are exactly polystream, and an outside seed is refused', async () => {
    const seeds = [-0.9, 0.3, 0.2, 3, 0, 0, -0.9, -0.85, -0.8]
    const spec = { field: 'U', timeIndex: 0, seeds, direction: 'both' as const, maxLength: null }
    const info = await service.streamlines(warpedId, spec)
    const mesh = await readPolyMesh(path.join(workspace, 'warped', 'constant', 'polyMesh'))
    const cellU = new Float64Array(3 * mesh.nCells)
    for (let c = 0; c < mesh.nCells; c++) cellU.set(U, 3 * c)
    const ref = polyStreamlines(buildStreamMesh(mesh), cellU, seeds, { direction: 'both' })
    console.log(`STU-15 warped sphere-in-box, uniform U: ${info.lineCount} lines, ${info.pointCount} points, ends ${JSON.stringify(info.ends)}, refused ${JSON.stringify(info.refusedSeeds)}, fallback points ${info.fallbackPoints}`)
    expect(info.lineCount).toBe(3)
    expect(info.refusedSeeds).toEqual([1])
    expect(info.ends).toEqual(ref.ends)
    expect(info.ends.boundary).toBe(4)
    expect(info.pointCount).toBe(ref.speeds.length)
    expect(info.pointCount).toBeGreaterThan(10)
    expect(info.points).toMatchObject({ dtype: 'f32', components: 3, count: info.pointCount })
    expect(info.speeds).toMatchObject({ dtype: 'f32', components: 1, count: info.pointCount })
    expect(info.offsets).toMatchObject({ dtype: 'u32', components: 1, count: 4 })
    expect([...u32((await service.blob(warpedId, info.offsets.key))!)]).toEqual([...ref.offsets])
    expect(same(f32((await service.blob(warpedId, info.points.key))!), Float32Array.from(ref.points))).toBe(true)
    expect(same(f32((await service.blob(warpedId, info.speeds.key))!), Float32Array.from(ref.speeds))).toBe(true)
    expect(await service.streamlines(warpedId, spec)).toBe(info)
    expect(info.key).toMatch(/^stream\.[0-9a-f]{16}$/)
  })

  test('every seed outside, the lattice and bad requests are refused by name', async () => {
    const base = { field: 'U', timeIndex: 0, direction: 'both' as const, maxLength: null }
    await expect(service.streamlines(warpedId, { ...base, seeds: [3, 0, 0, 0, 0, 5] })).rejects.toThrow(/^OUTSIDE: every seed \(2\)/)
    await expect(service.streamlines(latticeId, { ...base, seeds: [0, 0, 0.9] })).rejects.toThrow(/^NO_POLY_STREAM: /)
    await expect(service.streamlines(warpedId, { ...base, field: 'p', seeds: [0, 0, 0.9] })).rejects.toThrow(/^INVALID: p has 1 components/)
    await expect(service.streamlines(warpedId, { ...base, seeds: [0, 0] })).rejects.toThrow(/^INVALID: /)
    await expect(service.streamlines(warpedId, { ...base, seeds: [0, Number.NaN, 0] })).rejects.toThrow(/^INVALID: /)
    await expect(service.streamlines(warpedId, { ...base, seeds: [0, 0, 0.9], maxLength: 0 })).rejects.toThrow(/^INVALID: maxLength/)
  })
})
