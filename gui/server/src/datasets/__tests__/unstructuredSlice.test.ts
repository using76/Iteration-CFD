// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// STU-12b: a non-lattice polyMesh (the warped sphere-in-box) is cut by a plane in the worker pool
// with polycut, and the stored triangles are exactly polycut's: the same cells, the same corners
// rounded once to f32. The unwarped sphere-in-box keeps its lattice and is never cut.
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, test } from 'vitest'
import type { ViewerDataset } from '@cfd/shared'
import { loadConfig } from '../../config.js'
import { cutPolyMesh } from '../../formats/polycut.js'
import { readPolyMesh } from '../../formats/polymesh.js'
import { makeSphereInBox } from '../fixtures/makeSphereInBox.js'
import { createDatasetService, type DatasetServiceHandle } from '../service.js'

let workspace: string
let service: DatasetServiceHandle
let warpedId: string
let latticeId: string
let warped: ViewerDataset
let lattice: ViewerDataset

beforeAll(async () => {
  workspace = await fs.mkdtemp(path.join(os.tmpdir(), 'cfd-ucut-'))
  await makeSphereInBox(path.join(workspace, 'warped'), { warp: true })
  await makeSphereInBox(path.join(workspace, 'lattice'))
  const config = loadConfig({ CFD_WORKSPACE: workspace, CFD_GUI_DIR: path.join(workspace, 'gui'), CFD_DEMO: '1' })
  service = createDatasetService({ config, workers: 2 })
  warpedId = (await service.open('warped')).datasetId
  latticeId = (await service.open('lattice')).datasetId
  warped = await service.whenReady(warpedId)
  lattice = await service.whenReady(latticeId)
})
afterAll(async () => {
  await service.close()
  await fs.rm(workspace, { recursive: true, force: true })
})

const f32 = (buf: Buffer): Float32Array => new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength))
const u32 = (buf: Buffer): Uint32Array => new Uint32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength))

describe('plane cuts of a non-lattice polyMesh', () => {
  test('the warped sphere-in-box has no lattice and is cut instead', () => {
    expect(warped.source).toBe('polymesh')
    expect(warped.cellCount).toBe(456)
    expect(warped.grid).toBeNull()
    expect(warped.planeCut).toBe(true)
    expect(warped.bounds).toEqual({ min: [-1, -1, -1], max: [1, 1, 1] })
  })

  test('slice x = 0: the stored triangles are exactly polycut', async () => {
    const info = await service.cut(warpedId, { origin: [0, 0, 0], normal: [2, 0, 0] })
    const mesh = await readPolyMesh(path.join(workspace, 'warped', 'constant', 'polyMesh'))
    const ref = cutPolyMesh(mesh, { origin: [0, 0, 0], normal: [1, 0, 0] })
    expect(info.normal).toEqual([1, 0, 0])
    expect(info.triangleCount).toBe(ref.triCells.length)
    expect(info.triangleCount).toBeGreaterThan(0)
    expect(info.positions).toMatchObject({ dtype: 'f32', components: 3, count: 3 * info.triangleCount })
    expect(info.cells).toMatchObject({ dtype: 'u32', components: 1, count: info.triangleCount })
    const cells = u32((await service.blob(warpedId, info.cells.key))!)
    const positions = f32((await service.blob(warpedId, info.positions.key))!)
    expect([...cells]).toEqual([...ref.triCells])
    const want = new Set(ref.triCells)
    expect(new Set(cells)).toEqual(want)
    expect(info.cellCount).toBe(want.size)
    expect(Buffer.from(positions.buffer).equals(Buffer.from(Float32Array.from(ref.triPositions).buffer))).toBe(true)
    expect(info.area).toBe(ref.area)
    expect(info.openCells).toBe(0)
    // The section is the box's 4 minus the body's hole (16 lattice cells, area 1, before the warp).
    expect(info.area).toBeGreaterThan(2.5)
    expect(info.area).toBeLessThan(3.5)
  })

  test('a second identical cut is the same stored cut', async () => {
    const a = await service.cut(warpedId, { origin: [0, 0.2, -0.1], normal: [0.3, 1, 0.2] })
    const b = await service.cut(warpedId, { origin: [0, 0.2, -0.1], normal: [0.6, 2, 0.4] })
    expect(b).toBe(a)
    expect(a.key).toMatch(/^cut\.[0-9a-f]{16}$/)
  })

  test('the lattice sphere-in-box keeps its grid and refuses a cut', async () => {
    expect(lattice.grid).not.toBeNull()
    expect(lattice.planeCut).toBe(false)
    await expect(service.cut(latticeId, { origin: [0, 0, 0], normal: [1, 0, 0] })).rejects.toThrow(/^NO_PLANE_CUT: /)
    await expect(service.cut(warpedId, { origin: [0, 0, 0], normal: [0, 0, 0] })).rejects.toThrow(/^INVALID: /)
  })
})
