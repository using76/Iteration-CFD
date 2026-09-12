import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, test } from 'vitest'
import { caseInfoFromText } from './casejsonc.js'
import { resolveResultRoot } from './results.js'
import { discoverRegions, readRegionsManifest } from './regions.js'
import { buildCartesianGrid, cartesianToPolyMesh, type CartesianSpec } from './cartesian.js'
import { writeVtuPoints } from './vtu.js'

let t: string
beforeAll(async () => {
  t = await fs.mkdtemp(path.join(os.tmpdir(), 'cfd-regions-'))
})
afterAll(async () => {
  await fs.rm(t, { recursive: true, force: true })
})

const blockSpec: CartesianSpec = {
  bounds: { min: [0, 0, 0], max: [1, 1, 1] },
  cells: [1, 1, 1],
  grading: null,
  boundaries: { xmin: 'xMin', xmax: 'xMax', ymin: 'yMin', ymax: 'yMax', zmin: 'zMin', zmax: 'zMax' },
  regions: [],
  cyclic: [],
  patchTypes: {},
}

async function writePointsFile(dir: string): Promise<void> {
  await fs.mkdir(dir, { recursive: true })
  await fs.writeFile(path.join(dir, 'points'), 'FoamFile { class vectorField; object points; }\n0()\n')
}

describe('discoverRegions', () => {
  test('a regions.json layout', async () => {
    const layout = path.join(t, 'layout')
    await writePointsFile(path.join(layout, 'fluid', 'polyMesh'))
    await writePointsFile(path.join(layout, 'flap', 'polyMesh'))
    await fs.writeFile(
      path.join(layout, 'regions.json'),
      JSON.stringify({
        version: 1,
        regions: [
          { name: 'fluid', kind: 'fluid', polyMesh: 'fluid/polyMesh' },
          { name: 'flap', kind: 'solid', polyMesh: 'flap/polyMesh', material: 'steel' },
        ],
        interfaces: [],
      }),
    )
    const root = await resolveResultRoot(layout)
    const regions = await discoverRegions(root, null)
    expect(regions.map((r) => r.name)).toEqual(['fluid', 'flap'])
    expect(regions.map((r) => r.kind)).toEqual(['fluid', 'solid'])
    expect(regions.map((r) => r.material)).toEqual([null, 'steel'])
    expect(regions.map((r) => r.polyMeshDir)).toEqual([path.join(layout, 'fluid', 'polyMesh'), path.join(layout, 'flap', 'polyMesh')])
    expect(regions.map((r) => r.meshPath)).toEqual([path.join(layout, 'fluid'), path.join(layout, 'flap')])
    expect(regions.map((r) => r.path)).toEqual(regions.map((r) => r.meshPath))
    expect(regions.map((r) => r.source)).toEqual(['polyMesh', 'polyMesh'])
    expect(regions.map((r) => r.resultPath)).toEqual([null, null])
  })

  test('regions/<name>/ result directories', async () => {
    const out = path.join(t, 'out')
    await fs.mkdir(path.join(out, 'regions', 'a'), { recursive: true })
    await fs.mkdir(path.join(out, 'regions', 'b'), { recursive: true })
    const root = await resolveResultRoot(out)
    const regions = await discoverRegions(root, null)
    expect(regions.map((r) => r.name)).toEqual(['a', 'b'])
    expect(regions.map((r) => r.resultPath)).toEqual([path.join(out, 'regions', 'a'), path.join(out, 'regions', 'b')])
    expect(regions.map((r) => r.source)).toEqual(['dir', 'dir'])
    expect(regions.map((r) => r.path)).toEqual(regions.map((r) => r.resultPath))
  })

  test('a .cht.jsonc output root with VTU for two of four regions', async () => {
    const cases = path.join(t, 'cases')
    const text = `{
      "name": "stack",
      "regions": [
        { "name": "die", "kind": "solid", "mesh": { "bounds": { "min": [0,0,0], "max": [1,1,1] }, "cells": [1,1,1], "boundaries": { "xmin": "a", "xmax": "b", "ymin": "c", "ymax": "d", "zmin": "e", "zmax": "f" } } },
        { "name": "solder", "kind": "solid", "mesh": { "bounds": { "min": [0,0,1], "max": [1,1,2] }, "cells": [1,1,1], "boundaries": { "xmin": "a", "xmax": "b", "ymin": "c", "ymax": "d", "zmin": "e", "zmax": "f" } } },
        { "name": "spreader", "kind": "solid", "mesh": { "polyMesh": "mesh/spreader/polyMesh" } },
        { "name": "grease", "kind": "solid", "mesh": { "polyMesh": "mesh/grease/polyMesh" } }
      ],
      "initial": { "T": 300 },
      "run": { "steady": true }
    }`
    await fs.mkdir(path.join(cases, 'stack.cht_jsonc', 'VTK'), { recursive: true })
    await fs.writeFile(path.join(cases, 'stack.cht.jsonc'), text)
    const mesh = cartesianToPolyMesh(buildCartesianGrid(blockSpec), blockSpec)
    for (const name of ['die', 'solder']) {
      await writeVtuPoints(path.join(cases, 'stack.cht_jsonc', 'VTK', `${name}.vtu`), mesh, {
        time: 0,
        cellData: [{ name: 'T', components: 1, data: Float64Array.from([300]) }],
        pointData: [],
      })
    }
    const root = await resolveResultRoot(path.join(cases, 'stack.cht.jsonc'))
    const regions = await discoverRegions(root, caseInfoFromText(text, 'cases/stack.cht.jsonc'))
    expect(regions.map((r) => r.name)).toEqual(['die', 'solder', 'spreader', 'grease'])
    expect(regions.map((r) => r.source)).toEqual(['vtu', 'vtu', null, null])
    expect(regions.map((r) => r.cellCount)).toEqual([1, 1, null, null])
    expect(regions[0].path?.replace(/\\/g, '/')).toMatch(/VTK\/die\.vtu$/)
    expect(regions[1].path?.replace(/\\/g, '/')).toMatch(/VTK\/solder\.vtu$/)
    expect(regions[2].path).toBeNull()
    expect(regions[3].path).toBeNull()
    expect(regions.every((r) => r.kind === 'solid')).toBe(true)
  })

  test('readRegionsManifest refuses by name', async () => {
    const bad = path.join(t, 'bad-regions.json')
    await fs.writeFile(bad, '{"version":2,"regions":[]}')
    await expect(readRegionsManifest(bad)).rejects.toThrow(/version 2 is not 1/)
    await fs.writeFile(bad, '{"version":1}')
    await expect(readRegionsManifest(bad)).rejects.toThrow(/regions/)
  })
})
