// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The sphere-in-box polyMesh fixture reads back with seven patches, and walls-first picks the sphere
// alone while the Cartesian fixtures keep every patch.
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, test } from 'vitest'
import { defaultVisiblePatches, hiddenPatches, visibleBounds, type ViewerDataset } from '@cfd/shared'
import { loadConfig } from '../../config.js'
import { makeFoamCase, makeJsoncCase } from '../fixtures/makeCase.js'
import { makeSphereInBox } from '../fixtures/makeSphereInBox.js'
import { createDatasetService, type DatasetServiceHandle } from '../service.js'

let workspace: string
let service: DatasetServiceHandle
let sphereId: string
let foamId: string
let jsoncId: string
let sphere: ViewerDataset
let foam: ViewerDataset
let jsonc: ViewerDataset

beforeAll(async () => {
  workspace = await fs.mkdtemp(path.join(os.tmpdir(), 'cfd-sib-'))
  await makeJsoncCase(path.join(workspace, 'cases'))
  await makeFoamCase(path.join(workspace, 'foamCase'))
  await makeSphereInBox(path.join(workspace, 'sphereCase'))
  const config = loadConfig({ CFD_WORKSPACE: workspace, CFD_GUI_DIR: path.join(workspace, 'gui'), CFD_DEMO: '1' })
  service = createDatasetService({ config, workers: 2 })
  sphereId = (await service.open('sphereCase')).datasetId
  foamId = (await service.open('foamCase')).datasetId
  jsoncId = (await service.open('cases/fixture.jsonc')).datasetId
  sphere = await service.whenReady(sphereId)
  foam = await service.whenReady(foamId)
  jsonc = await service.whenReady(jsoncId)
})
afterAll(async () => {
  await service.close()
  await fs.rm(workspace, { recursive: true, force: true })
})

const f32 = (buf: Buffer): Float32Array => new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength))
const u32 = (buf: Buffer): Uint32Array => new Uint32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength))

describe('sphere-in-box fixture', () => {
  test('every boundary triangle belongs to a fluid cell', async () => {
    const cellOfTri = u32((await service.blob(sphereId, 'surface.cellOfTri'))!)
    for (const c of cellOfTri) expect(c).toBeLessThan(456)
    const distinct = (name: string): number => {
      const p = sphere.surface.patches.find((q) => q.name === name)!
      const set = new Set<number>()
      for (let t = p.triStart; t < p.triStart + p.triCount; t++) set.add(cellOfTri[t])
      return set.size
    }
    expect(distinct('xmin')).toBe(64)
    expect(distinct('sphere')).toBe(80)
  })

  test('the reader sees seven patches, the sphere a wall', () => {
    expect(sphere.source).toBe('polymesh')
    expect(sphere.cellCount).toBe(456)
    expect(sphere.bounds).toEqual({ min: [-1, -1, -1], max: [1, 1, 1] })
    expect(sphere.surface.patches.map((p) => [p.name, p.type])).toEqual([
      ['xmin', 'patch'],
      ['xmax', 'patch'],
      ['ymin', 'patch'],
      ['ymax', 'patch'],
      ['zmin', 'patch'],
      ['zmax', 'patch'],
      ['sphere', 'wall'],
    ])
    expect(sphere.surface.patches.map((p) => p.triCount)).toEqual([128, 128, 128, 128, 128, 128, 192])
    expect(sphere.surface.triangleCount).toBe(960)
  })

  test('walls first on the real manifest', async () => {
    const positions = f32((await service.blob(sphereId, 'surface.positions'))!)
    const indices = u32((await service.blob(sphereId, 'surface.indices'))!)
    expect(defaultVisiblePatches(sphere, positions, indices)).toEqual(['sphere'])
    const S = Math.sqrt(3) * 1e-6
    const b = visibleBounds(sphere, positions, indices, ['sphere'])
    const want = [-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]
    ;[...b.min, ...b.max].forEach((v, i) => expect(Math.abs(v - want[i])).toBeLessThanOrEqual(S))
    expect(hiddenPatches(sphere, ['sphere'])).toEqual(['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'])
  })

  test('Cartesian manifests keep every patch', async () => {
    const foamPos = f32((await service.blob(foamId, 'surface.positions'))!)
    const foamIdx = u32((await service.blob(foamId, 'surface.indices'))!)
    expect(defaultVisiblePatches(foam, foamPos, foamIdx)).toBe('all')
    const jsoncPos = f32((await service.blob(jsoncId, 'surface.positions'))!)
    const jsoncIdx = u32((await service.blob(jsoncId, 'surface.indices'))!)
    expect(defaultVisiblePatches(jsonc, jsoncPos, jsoncIdx)).toBe('all')
  })
})
