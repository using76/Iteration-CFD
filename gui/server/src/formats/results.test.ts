import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, test } from 'vitest'
import { writeFoamField } from './foam.js'
import { fieldTimeFallback, listTimeDirs, parseTimeName, resolveResultRoot } from './results.js'

let root: string
let base: string

async function scalar(dir: string, name: string, cls: 'volScalarField' | 'surfaceScalarField' = 'volScalarField'): Promise<void> {
  if (cls === 'surfaceScalarField') {
    await fs.mkdir(dir, { recursive: true })
    await fs.writeFile(path.join(dir, name), `FoamFile\n{\n    class       surfaceScalarField;\n    object      ${name};\n}\ndimensions [0 3 -1 0 0 0 0];\ninternalField nonuniform List<scalar> 2(1 2);\n`)
    return
  }
  await writeFoamField(path.join(dir, name), { name, class: cls, dimensions: '[0 0 0 0 0 0 0]', time: path.basename(dir), data: Float32Array.from([1, 2, 3, 4]), components: 1, patches: [], collapseUniform: false })
}

beforeAll(async () => {
  base = await fs.mkdtemp(path.join(os.tmpdir(), 'cfd-results-'))
  root = path.join(base, 'case')
  await scalar(path.join(root, '0'), 'p')
  await writeFoamField(path.join(root, '0', 'U'), { name: 'U', class: 'volVectorField', dimensions: '[0 1 -1 0 0 0 0]', time: '0', data: new Float32Array(12), components: 3, patches: [], collapseUniform: false })
  await scalar(path.join(root, '0.5'), 'p')
  await scalar(path.join(root, '1e-05'), 'p')
  await scalar(path.join(root, '1e-05'), 'k')
  await scalar(path.join(root, 'results'), 'p')
  await scalar(path.join(root, 'results'), 'phi', 'surfaceScalarField')
  await fs.mkdir(path.join(root, 'constant', 'polyMesh'), { recursive: true })
  await fs.writeFile(path.join(root, 'constant', 'polyMesh', 'points'), 'FoamFile { class vectorField; object points; }\n0()\n')
  await fs.mkdir(path.join(root, 'system'), { recursive: true })
  await fs.writeFile(path.join(root, 'system', 'controlDict'), 'application x;\n')
  await fs.mkdir(path.join(root, 'VTK'), { recursive: true })
  await fs.writeFile(path.join(root, 'VTK', 'case.pvd'), '<VTKFile type="Collection"><Collection/></VTKFile>')
  await fs.writeFile(path.join(root, 'VTK', 'case_000001.vtu'), 'x')
  await fs.writeFile(path.join(root, 'VTK', 'parcels.vtp'), 'x')
  await fs.writeFile(path.join(root, 'notes.txt'), 'hello')
  await fs.mkdir(path.join(root, 'empty'), { recursive: true })
  // JSONC output tree
  await fs.writeFile(path.join(base, 'plume.jsonc'), '{ "name": "plume" }')
  await scalar(path.join(base, 'plume_jsonc', '0'), 'p')
  await scalar(path.join(base, 'plume_jsonc', '100'), 'p')
})
afterAll(async () => {
  await fs.rm(base, { recursive: true, force: true })
})

describe('parseTimeName', () => {
  test('numeric names only', () => {
    expect(parseTimeName('0')).toBe(0)
    expect(parseTimeName('0.5')).toBe(0.5)
    expect(parseTimeName('1e-05')).toBe(1e-5)
    expect(parseTimeName('4000')).toBe(4000)
    expect(parseTimeName('-2')).toBe(-2)
    expect(parseTimeName('results')).toBeNull()
    expect(parseTimeName('0.orig')).toBeNull()
    expect(parseTimeName('')).toBeNull()
    expect(parseTimeName('1e')).toBeNull()
    expect(parseTimeName('Infinity')).toBeNull()
  })
})

describe('listTimeDirs / fieldTimeFallback', () => {
  test('numeric dirs by value, then named dirs; phi and non-time dirs excluded', async () => {
    const times = await listTimeDirs(root)
    expect(times.map((t) => t.name)).toEqual(['0', '1e-05', '0.5', 'results'])
    expect(times.map((t) => t.value)).toEqual([0, 1e-5, 0.5, null])
    expect(times[0].fields).toEqual(['U', 'p'])
    expect(times[0].fieldClasses).toEqual({ U: 'volVectorField', p: 'volScalarField' })
    expect(times[1].fields).toEqual(['k', 'p'])
    expect(times[3].fields).toEqual(['p'])
    expect(fieldTimeFallback(times, 'U', 3)?.name).toBe('0')
    expect(fieldTimeFallback(times, 'k', 3)?.name).toBe('1e-05')
    expect(fieldTimeFallback(times, 'k', 0)).toBeNull()
    expect(fieldTimeFallback(times, 'p', 2)?.name).toBe('0.5')
    expect(fieldTimeFallback(times, 'p', 99)?.name).toBe('results')
    expect(fieldTimeFallback(times, 'zeta', 3)).toBeNull()
  })
})

describe('resolveResultRoot', () => {
  test('foam case directory', async () => {
    const r = await resolveResultRoot(root)
    expect(r.kind).toBe('foamCase')
    expect(r.rootAbs).toBe(root)
    expect(r.hasPolyMesh).toBe(true)
    expect(r.caseJsoncAbs).toBeNull()
    expect(r.vtk.map((v) => [path.basename(v.abs), v.kind])).toEqual([
      ['case.pvd', 'pvd'],
      ['case_000001.vtu', 'vtu'],
      ['parcels.vtp', 'vtp'],
    ])
  })

  test('time directory resolves to its parent', async () => {
    const r = await resolveResultRoot(path.join(root, '0.5'))
    expect(r.kind).toBe('timeDir')
    expect(r.timeDir).toBe('0.5')
    expect(r.rootAbs).toBe(root)
    expect(r.hasPolyMesh).toBe(true)
  })

  test('jsonc case and its output directory', async () => {
    const jsonc = path.join(base, 'plume.jsonc')
    const r = await resolveResultRoot(jsonc)
    expect(r.kind).toBe('jsoncCase')
    expect(r.rootAbs).toBe(path.join(base, 'plume_jsonc'))
    expect(r.caseJsoncAbs).toBe(jsonc)
    expect(r.hasPolyMesh).toBe(false)
    const out = await resolveResultRoot(path.join(base, 'plume_jsonc'))
    expect(out.kind).toBe('outputDir')
    expect(out.caseJsoncAbs).toBe(jsonc)
    const t = await resolveResultRoot(path.join(base, 'plume_jsonc', '100'))
    expect(t.kind).toBe('timeDir')
    expect(t.caseJsoncAbs).toBe(jsonc)
    expect(t.rootAbs).toBe(path.join(base, 'plume_jsonc'))
    const missing = await resolveResultRoot(path.join(base, 'plume.jsonc'))
    expect(missing.rootAbs).toBe(path.join(base, 'plume_jsonc'))
  })

  test('vtk files', async () => {
    const r = await resolveResultRoot(path.join(root, 'VTK', 'case_000001.vtu'))
    expect(r.kind).toBe('vtu')
    expect(r.rootAbs).toBe(root)
    expect(r.vtkFile).toBe(path.join(root, 'VTK', 'case_000001.vtu'))
    expect(r.vtk.map((v) => v.kind)).toEqual(['pvd', 'vtu', 'vtp'])
    const p = await resolveResultRoot(path.join(root, 'VTK', 'case.pvd'))
    expect(p.kind).toBe('pvd')
    expect(p.vtkFile).toBe(path.join(root, 'VTK', 'case.pvd'))
    expect(p.hasPolyMesh).toBe(true)
    const loose = await resolveResultRoot(path.join(base, 'plume_jsonc', '100', 'p'))
      .then(() => 'ok')
      .catch(() => 'rejected')
    expect(loose).toBe('rejected')
  })

  test('plain directories and unknown files', async () => {
    const r = await resolveResultRoot(path.join(root, 'empty'))
    expect(r.kind).toBe('outputDir')
    await expect(resolveResultRoot(path.join(root, 'notes.txt'))).rejects.toThrow(/not a case/)
  })
})
