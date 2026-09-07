import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, test } from 'vitest'
import { buildCartesianGrid, cartesianBoundarySurface, resolveWindows } from './cartesian.js'
import { caseInfoFromText, extractCartesianSpec, parseJsonc, readCaseJsonc } from './casejsonc.js'

const here = path.dirname(fileURLToPath(import.meta.url))
const repo = path.resolve(here, '..', '..', '..', '..')
const casesDir = path.join(repo, 'cases')

describe('parseJsonc', () => {
  test('comments and trailing commas are fine, syntax errors carry line/col', () => {
    const ok = parseJsonc('// c\n{ "a": [1, 2,], /* x */ "b": { "c": 1, }, }')
    expect(ok.errors).toEqual([])
    expect(ok.json).toEqual({ a: [1, 2], b: { c: 1 } })
    const bad = parseJsonc('{\n  "a": 1,\n  "b": }\n')
    expect(bad.errors.length).toBeGreaterThan(0)
    expect(bad.errors[0].line).toBe(3)
    expect(bad.errors[0].col).toBe(8)
    expect(bad.errors[0].message).toBe('ValueExpected')
  })
})

describe('plume.jsonc', () => {
  test('mesh, region, model, gravity, output dir', async () => {
    const info = await readCaseJsonc(path.join(casesDir, 'plume.jsonc'), 'cases/plume.jsonc')
    expect(info.errors).toEqual([])
    expect(info.name).toBe('plumeB')
    expect(info.model).toBe('kEpsilon')
    expect(info.turbulenceKind).toBe('RAS')
    expect(info.gravity).toEqual([0, 0, -9.81])
    expect(info.run).toEqual({ endTime: 1, deltaT: 1 })
    expect(info.output).toBeNull()
    expect(info.outputDir).toBe('cases/plume_jsonc')
    expect(info.initialFields).toEqual(['U', 'T', 'p', 'k', 'epsilon', 'omega', 'nut'])
    const mesh = info.mesh!
    expect(mesh.cells).toEqual([98, 42, 20])
    expect(mesh.bounds).toEqual({ min: [-8.32, -2.62, 0], max: [6.32, 3.62, 3] })
    expect(mesh.grading).toBeNull()
    expect(mesh.boundaries).toEqual({ xmin: 'wallXMin', xmax: 'outlet', ymin: 'wallYMin', ymax: 'wallYMax', zmin: 'floor', zmax: 'ceiling' })
    expect(mesh.regions).toEqual([{ name: 'inlet', on: 'zmin', shape: { kind: 'box', min: [-0.6, -0.6, -1], max: [0.6, 0.6, 1] } }])
    expect(mesh.cyclic).toEqual([])
    expect(mesh.patchTypes).toEqual({ wallXMin: 'wall', outlet: 'patch', wallYMin: 'wall', wallYMax: 'wall', floor: 'wall', ceiling: 'wall', inlet: 'patch' })

    // The window snaps to the same whole cells the Rust reader documents: i 52..60, j 14..22.
    const grid = buildCartesianGrid(mesh)
    expect(grid.emptyAxis).toBeNull()
    expect(resolveWindows(grid, mesh)).toEqual([{ slot: 4, name: 'inlet', a0: 52, a1: 60, b0: 14, b1: 22 }])
    const surface = cartesianBoundarySurface(grid, mesh)
    expect(surface.patches.map((p) => [p.name, p.type, p.triCount])).toEqual([
      ['wallXMin', 'wall', 2 * 42 * 20],
      ['outlet', 'patch', 2 * 42 * 20],
      ['wallYMin', 'wall', 2 * 98 * 20],
      ['wallYMax', 'wall', 2 * 98 * 20],
      ['inlet', 'patch', 2 * 64],
      ['floor', 'wall', 2 * (98 * 42 - 64)],
      ['ceiling', 'wall', 2 * 98 * 42],
    ])
  })
})

describe('cyclic and graded channel cases', () => {
  test('channelPeriodicWF.jsonc: cyclic pair, wall rules matched in order', async () => {
    const info = await readCaseJsonc(path.join(casesDir, 'channelPeriodicWF.jsonc'), 'cases/channelPeriodicWF.jsonc')
    expect(info.errors).toEqual([])
    expect(info.gravity).toEqual([0, 0, 0])
    const mesh = info.mesh!
    expect(mesh.cells).toEqual([8, 6, 4])
    expect(mesh.cyclic).toEqual([{ a: 'streamwiseMin', b: 'streamwiseMax' }])
    expect(mesh.patchTypes).toEqual({ streamwiseMin: 'cyclic', streamwiseMax: 'cyclic', wallHot0: 'wall', wallHot1: 'wall', wallSideMin: 'wall', wallSideMax: 'wall' })
  })

  test('channelPeriodicFluxLowRe.jsonc: two-sided grading', async () => {
    const info = await readCaseJsonc(path.join(casesDir, 'channelPeriodicFluxLowRe.jsonc'), 'cases/channelPeriodicFluxLowRe.jsonc')
    expect(info.errors).toEqual([])
    const mesh = info.mesh!
    expect(mesh.cells).toEqual([8, 50, 1])
    expect(mesh.grading).toEqual({ y: { expansion: 200, twoSided: true } })
    const grid = buildCartesianGrid(mesh)
    expect(grid.uniform).toBe(false)
    expect(grid.emptyAxis).toBe('z')
    const y = grid.nodes.y
    const wall = y[1] - y[0]
    const centre = y[25] - y[24]
    expect(centre / wall).toBeCloseTo(200, 8)
    expect(y[25]).toBeCloseTo((mesh.bounds.min[1] + mesh.bounds.max[1]) / 2, 12)
  })
})

describe('extractCartesianSpec', () => {
  test('rejects non-cartesian or malformed meshes, defaults missing patch rules to patch', () => {
    expect(extractCartesianSpec({ mesh: { kind: 'other' } })).toBeNull()
    expect(extractCartesianSpec({ mesh: { kind: 'cartesian', bounds: { min: [0, 0, 0] } } })).toBeNull()
    const minimal = extractCartesianSpec({
      mesh: { kind: 'cartesian', bounds: { min: [0, 0, 0], max: [1, 1, 1] }, cells: [2, 2, 2], boundaries: { xmin: 'a', xmax: 'b', ymin: 'c', ymax: 'd', zmin: 'e', zmax: 'f' } },
      patches: [{ match: 'a|b', kind: 'inlet' }, { match: 'c', kind: 'empty' }, { match: '(', kind: 'wall' }],
    })!
    expect(minimal.patchTypes).toEqual({ a: 'patch', b: 'patch', c: 'empty', d: 'patch', e: 'patch', f: 'patch' })
    expect(minimal.regions).toEqual([])
    expect(minimal.grading).toBeNull()
    const info = caseInfoFromText('{ "name": 3, "turbulence": { "model": "kOmegaSST" } }', 'x/y.jsonc')
    expect(info.name).toBeNull()
    expect(info.model).toBe('kOmegaSST')
    expect(info.mesh).toBeNull()
    expect(info.outputDir).toBe('x/y_jsonc')
    expect(info.initialFields).toEqual([])
  })
})
