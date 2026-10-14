// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// docs/17 STU-13: the dataset service locates a world point in a polyMesh's cells in its worker
// pool. On the sphere-in-box lattice (456 convex hexes) every point lands in the hex whose centre
// is within half a cell on each axis, and a point in the sphere's hole is a miss. On the warped
// sphere-in-box (no lattice, no convex cell) every answer is labelled 'nearest-centre' and names
// the nearest centre among the cells whose box holds the point.
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, test } from 'vitest'
import { loadConfig } from '../../config.js'
import { buildCellLocator } from '../../formats/pointLocate.js'
import { polyMeshCellCenters, readPolyMesh, type PolyMesh } from '../../formats/polymesh.js'
import { makeSphereInBox } from '../fixtures/makeSphereInBox.js'
import { createDatasetService, type DatasetServiceHandle } from '../service.js'

type V3 = [number, number, number]

let workspace: string
let service: DatasetServiceHandle
let warpedId: string
let latticeId: string
let warpedMesh: PolyMesh
let latticeMesh: PolyMesh

beforeAll(async () => {
  workspace = await fs.mkdtemp(path.join(os.tmpdir(), 'cfd-locate-'))
  await makeSphereInBox(path.join(workspace, 'warped'), { warp: true })
  await makeSphereInBox(path.join(workspace, 'lattice'))
  const config = loadConfig({ CFD_WORKSPACE: workspace, CFD_GUI_DIR: path.join(workspace, 'gui'), CFD_DEMO: '1' })
  service = createDatasetService({ config, workers: 2 })
  warpedId = (await service.open('warped')).datasetId
  latticeId = (await service.open('lattice')).datasetId
  await service.whenReady(warpedId)
  await service.whenReady(latticeId)
  warpedMesh = await readPolyMesh(path.join(workspace, 'warped', 'constant', 'polyMesh'))
  latticeMesh = await readPolyMesh(path.join(workspace, 'lattice', 'constant', 'polyMesh'))
})
afterAll(async () => {
  await service.close()
  await fs.rm(workspace, { recursive: true, force: true })
})

/** Seeded uniform numbers in [0, 1) (mulberry32). */
function rng(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Per cell, the box of its points, computed here from the face lists. */
function cellBoxes(m: PolyMesh): Float64Array {
  const box = new Float64Array(6 * m.nCells)
  for (let c = 0; c < m.nCells; c++) box.set([Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity], 6 * c)
  for (let f = 0; f < m.nFaces; f++) {
    const cells = f < m.nInternalFaces ? [m.owner[f], m.neighbour[f]] : [m.owner[f]]
    for (let i = m.faceOffsets[f]; i < m.faceOffsets[f + 1]; i++) {
      const p = 3 * m.faceIndices[i]
      for (const c of cells) {
        for (let a = 0; a < 3; a++) {
          box[6 * c + a] = Math.min(box[6 * c + a], m.points[p + a])
          box[6 * c + 3 + a] = Math.max(box[6 * c + 3 + a], m.points[p + a])
        }
      }
    }
  }
  return box
}

describe('the dataset service locates a point in a polyMesh', () => {
  test('both sphere-in-box meshes say pointLocate; the lattice has 456 convex cells, the warped one none', async () => {
    expect((await service.whenReady(latticeId)).pointLocate).toBe(true)
    expect((await service.whenReady(warpedId)).pointLocate).toBe(true)
    expect(buildCellLocator(latticeMesh).convex.reduce((s, v) => s + v, 0)).toBe(456)
    expect(buildCellLocator(warpedMesh).convex.reduce((s, v) => s + v, 0)).toBe(0)
  })

  test('lattice: 300 random points land in the hex whose centre is within 0.125 per axis, and the hole is a miss', async () => {
    const centres = polyMeshCellCenters(latticeMesh)
    const r = rng(13)
    let hits = 0
    let holes = 0
    for (let n = 0; n < 300; n++) {
      const x: V3 = [2 * r() - 1, 2 * r() - 1, 2 * r() - 1]
      let want: number | null = null
      for (let c = 0; c < latticeMesh.nCells && want === null; c++) {
        if ([0, 1, 2].every((a) => Math.abs(centres[3 * c + a] - x[a]) < 0.125)) want = c
      }
      const got = await service.locate(latticeId, x)
      expect(got.point).toEqual(x)
      expect(got.cell).toBe(want)
      if (want === null) {
        holes++
        expect(got.method).toBeNull()
        expect(got.center).toBeNull()
      } else {
        hits++
        expect(got.method).toBe('contains')
        expect(got.center).toEqual([centres[3 * want], centres[3 * want + 1], centres[3 * want + 2]])
      }
    }
    expect(hits).toBeGreaterThan(0)
    expect(holes).toBeGreaterThan(0)
    expect((await service.locate(latticeId, [0.01, 0.02, 0.03])).cell).toBeNull()
    expect((await service.locate(latticeId, [1.5, 0, 0])).cell).toBeNull()
  })

  test('warped: every answer is labelled nearest-centre and names the nearest centre among the boxes holding the point', async () => {
    const centres = polyMeshCellCenters(warpedMesh)
    const box = cellBoxes(warpedMesh)
    const r = rng(29)
    let labelled = 0
    for (let n = 0; n < 300; n++) {
      const x: V3 = [2 * r() - 1, 2 * r() - 1, 2 * r() - 1]
      let want: number | null = null
      let best = Infinity
      for (let c = 0; c < warpedMesh.nCells; c++) {
        if (![0, 1, 2].every((a) => x[a] >= box[6 * c + a] && x[a] <= box[6 * c + 3 + a])) continue
        const d = (centres[3 * c] - x[0]) ** 2 + (centres[3 * c + 1] - x[1]) ** 2 + (centres[3 * c + 2] - x[2]) ** 2
        if (d < best) {
          best = d
          want = c
        }
      }
      const got = await service.locate(warpedId, x)
      expect(got.cell).toBe(want)
      expect(got.method).toBe(want === null ? null : 'nearest-centre')
      if (want !== null) labelled++
    }
    expect(labelled).toBeGreaterThan(250)
    expect((await service.locate(warpedId, [0.01, 0.02, 0.03])).cell).toBeNull()
  })

  test('refusals: a non-finite point is INVALID, an unknown id is unknown', async () => {
    await expect(service.locate(latticeId, [Number.NaN, 0, 0])).rejects.toThrow(/^INVALID/)
    await expect(service.locate('d_nope', [0, 0, 0])).rejects.toThrow(/unknown dataset/)
  })
})
