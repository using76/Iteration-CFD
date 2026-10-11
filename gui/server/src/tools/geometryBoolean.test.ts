// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// Tests for the STL boolean tool: cube-fixture volumes, twenty seeded random
// rigid pairs, the refusal codes, the default out-naming and the licence scan.
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { measureSolid, readSurface, writeStl } from '../formats/stl.js'
import type { ToolContext } from './context.js'
import { BooleanError, StlBooleanSchema, booleanMeshes, defaultBooleanOut, edgeDefects, geometryStlBoolean, stlBoolean, type BooleanOp, type TriMesh } from './geometryBoolean.js'

// The STU-16 fixture: the unit cube's corners and its 12 outward triangles;
// cubeStl emits them as ASCII (mode 'open' drops the last triangle,
// 'nonmanifold' writes the first triangle twice).
const CUBE: Array<[number, number, number]> = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
const TRI: number[][] = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
const rel = (x: number, e: number): number => Math.abs(x - e) / Math.abs(e)

const cubeMesh = (dx = 0): TriMesh => {
  const positions = new Float32Array(24)
  for (let i = 0; i < 8; i++) { positions[i * 3] = CUBE[i][0] + dx; positions[i * 3 + 1] = CUBE[i][1]; positions[i * 3 + 2] = CUBE[i][2] }
  return { positions, indices: Uint32Array.from(TRI.flat()) }
}

const cubeStl = (name: string, dx: number, mode: 'full' | 'open' | 'nonmanifold'): string => {
  const tris = mode === 'open' ? TRI.slice(0, -1) : TRI
  const lines = [`solid ${name}`]
  const emit = (t: number[]): void => {
    lines.push('facet normal 0 0 0', 'outer loop')
    for (const i of t) lines.push(`vertex ${CUBE[i][0] + dx} ${CUBE[i][1]} ${CUBE[i][2]}`)
    lines.push('endloop', 'endfacet')
  }
  emit(tris[0])
  if (mode === 'nonmanifold') emit(tris[0])
  for (let k = 1; k < tris.length; k++) emit(tris[k])
  lines.push(`endsolid ${name}`)
  return `${lines.join('\n')}\n`
}

const TWO_STL = `${cubeStl('a', 0, 'full')}${cubeStl('b', 0.5, 'full')}`
const OPEN_STL = `${cubeStl('a', 0, 'open')}${cubeStl('b', 0.5, 'full')}`
const NM_STL = `${cubeStl('a', 0, 'nonmanifold')}${cubeStl('b', 0.5, 'full')}`

const mulberry32 = (seed: number): (() => number) => {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Shoemake's uniform unit quaternion from three uniforms, as a row-major 3x3 rotation matrix. */
const rotation = (rnd: () => number): number[] => {
  const u1 = rnd(), u2 = rnd(), u3 = rnd()
  const q = [Math.sqrt(1 - u1) * Math.sin(2 * Math.PI * u2), Math.sqrt(1 - u1) * Math.cos(2 * Math.PI * u2), Math.sqrt(u1) * Math.sin(2 * Math.PI * u3), Math.sqrt(u1) * Math.cos(2 * Math.PI * u3)]
  const [x, y, z, w] = q
  return [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
}

/** The centred unit cube through R and then t, computed in f64 and stored f32. */
const rotatedCube = (R: number[], t: number[] | null): TriMesh => {
  const positions = new Float32Array(24)
  for (let i = 0; i < 8; i++) {
    const c = [CUBE[i][0] - 0.5, CUBE[i][1] - 0.5, CUBE[i][2] - 0.5]
    positions[i * 3] = R[0] * c[0] + R[1] * c[1] + R[2] * c[2] + (t ? t[0] : 0)
    positions[i * 3 + 1] = R[3] * c[0] + R[4] * c[1] + R[5] * c[2] + (t ? t[1] : 0)
    positions[i * 3 + 2] = R[6] * c[0] + R[7] * c[1] + R[8] * c[2] + (t ? t[2] : 0)
  }
  return { positions, indices: Uint32Array.from(TRI.flat()) }
}

let dir: string
let wsRoot: string
beforeAll(async () => {
  dir = await fsp.mkdtemp(path.join(os.tmpdir(), 'stlbool-'))
  wsRoot = await fsp.mkdtemp(path.join(os.tmpdir(), 'stlbool-ws-'))
  await fsp.mkdir(path.join(wsRoot, 'cases'), { recursive: true })
  await fsp.writeFile(path.join(wsRoot, 'cases', 'two.stl'), TWO_STL, 'utf8')
  await fsp.writeFile(path.join(wsRoot, 'cases', 'open.stl'), OPEN_STL, 'utf8')
  for (const [n, text] of [['two.stl', TWO_STL], ['open.stl', OPEN_STL], ['nm.stl', NM_STL]] as const) await fsp.writeFile(path.join(dir, n), text, 'utf8')
})
afterAll(async () => {
  await fsp.rm(dir, { recursive: true, force: true })
  await fsp.rm(wsRoot, { recursive: true, force: true })
})

describe('stl booleans', () => {
  it('B1 booleanMeshes volumes on the two overlapping cubes', async () => {
    const A = cubeMesh()
    const B = cubeMesh(0.5)
    const fuse = await booleanMeshes('fuse', A, B)
    const cut = await booleanMeshes('cut', A, B)
    const common = await booleanMeshes('common', A, B)
    expect(rel(fuse.volume, 1.5)).toBeLessThanOrEqual(1e-12)
    expect(rel(cut.volume, 0.5)).toBeLessThanOrEqual(1e-12)
    expect(rel(common.volume, 0.5)).toBeLessThanOrEqual(1e-12)
    for (const r of [fuse, cut, common]) {
      expect(rel(r.volumeA, 1)).toBeLessThanOrEqual(1e-12)
      expect(rel(r.volumeB, 1)).toBeLessThanOrEqual(1e-12)
    }
  })

  it('B2 twenty seeded random rigid pairs keep the inclusion-exclusion identity and stay closed', async () => {
    const rnd = mulberry32(17)
    let worst = 0
    const results: Array<{ op: BooleanOp; mesh: TriMesh; k: number }> = []
    for (let k = 0; k < 20; k++) {
      const R1 = rotation(rnd)
      const R2 = rotation(rnd)
      const t = [rnd() - 0.5, rnd() - 0.5, rnd() - 0.5]
      const A = rotatedCube(R1, null)
      const B = rotatedCube(R2, t)
      const per = { fuse: await booleanMeshes('fuse', A, B), cut: await booleanMeshes('cut', A, B), common: await booleanMeshes('common', A, B) }
      const resid = Math.abs(per.fuse.volume + per.common.volume - per.fuse.volumeA - per.fuse.volumeB)
      const sum = per.fuse.volumeA + per.fuse.volumeB
      worst = Math.max(worst, resid / sum)
      expect(resid).toBeLessThanOrEqual(1e-9 * sum)
      for (const op of ['fuse', 'cut', 'common'] as const) results.push({ op, mesh: per[op].mesh, k })
    }
    for (const { op, mesh, k } of results) {
      const p = path.join(dir, `b2-${k}-${op}.stl`)
      await writeStl(p, { positions: mesh.positions, indices: mesh.indices, solids: [] }, { binary: true })
      const s = await readSurface(p)
      expect(edgeDefects(s.indices, 0, s.indices.length / 3)).toEqual({ open: 0, nonManifold: 0 })
      expect(measureSolid(s.positions, s.indices, 0, s.indices.length / 3).openEdges).toBe(0)
    }
    expect(worst).toBeLessThanOrEqual(1e-9)
  })

  it('B3 stlBoolean writes closed fuse, cut and common files beside the input', async () => {
    const src = path.join(dir, 'two.stl')
    for (const [op, vol] of [['fuse', 1.5], ['cut', 0.5], ['common', 0.5]] as Array<[BooleanOp, number]>) {
      const out = path.join(dir, `b3-${op}.stl`)
      const r = await stlBoolean({ srcAbs: src, outAbs: out, op, a: 'a', b: 'b' })
      expect(rel(r.volume, vol)).toBeLessThanOrEqual(1e-12)
      expect(r.openEdges).toBe(0)
      expect(r.nonManifoldEdges).toBe(0)
      const s = await readSurface(out)
      expect(s.indices.length / 3).toBe(r.triangles)
      expect(s.positions.length / 3).toBe(r.vertices)
    }
  })

  it('B4 an open part is refused as STL-OPEN and nothing is written', async () => {
    const out = path.join(dir, 'b4-out.stl')
    const err = await stlBoolean({ srcAbs: path.join(dir, 'open.stl'), outAbs: out, op: 'fuse', a: 'a', b: 'b' }).then(() => null, (e: unknown) => e)
    expect(err).toBeInstanceOf(BooleanError)
    const be = err as BooleanError
    expect(be.code).toBe('STL-OPEN')
    expect(be.message).toContain("part 'a'")
    expect(be.message).toContain('3 open edge(s)')
    expect(be.message).toContain('0 non-manifold edge(s)')
    expect(be.message).toContain('stl_repair')
    expect(fs.existsSync(out)).toBe(false)
  })

  it('B5 a non-manifold part is refused as STL-OPEN too', async () => {
    const err = await stlBoolean({ srcAbs: path.join(dir, 'nm.stl'), outAbs: path.join(dir, 'b5-out.stl'), op: 'cut', a: 'a', b: 'b' }).then(() => null, (e: unknown) => e)
    expect(err).toBeInstanceOf(BooleanError)
    expect((err as BooleanError).code).toBe('STL-OPEN')
    expect((err as BooleanError).message).toContain('0 open edge(s)')
    expect((err as BooleanError).message).toContain('3 non-manifold edge(s)')
  })

  it('B6 edgeDefects matches measureSolid on both broken fixtures', async () => {
    for (const f of ['open.stl', 'nm.stl']) {
      const s = await readSurface(path.join(dir, f))
      const a = s.solids.find((x) => x.name === 'a')
      if (!a) throw new Error(`no solid 'a' in ${f}`)
      const d = edgeDefects(s.indices, a.first, a.count)
      expect(d.open + d.nonManifold).toBe(measureSolid(s.positions, s.indices, a.first, a.count).openEdges)
    }
  })

  it('B7 unknown part and a === b refusals', async () => {
    const bad = { srcAbs: path.join(dir, 'two.stl'), outAbs: path.join(dir, 'b7.stl'), op: 'fuse' as const }
    const e1 = await stlBoolean({ ...bad, a: 'a', b: 'zzz' }).then(() => null, (e: unknown) => e)
    expect((e1 as BooleanError).code).toBe('NO_PART')
    expect((e1 as BooleanError).message).toContain('parts: a, b')
    const e2 = await stlBoolean({ ...bad, a: 'a', b: 'a' }).then(() => null, (e: unknown) => e)
    expect((e2 as BooleanError).code).toBe('INVALID')
  })

  it('B8 defaultBooleanOut naming', () => {
    expect(defaultBooleanOut('cases/two.stl', 'cut')).toBe('cases/two-cut.stl')
    expect(defaultBooleanOut('two.stl', 'fuse')).toBe('two-fuse.stl')
  })

  it('B9 the ToolDef resolves, refuses and overwrites inside a workspace', async () => {
    const ctx = { workspaceRoot: wsRoot } as unknown as ToolContext
    const call = (over: Record<string, unknown> = {}) => geometryStlBoolean.run(StlBooleanSchema.parse({ path: 'cases/two.stl', op: 'common', a: 'a', b: 'b', ...over }), ctx)
    const first = await call()
    expect(first.ok).toBe(true)
    expect((first.data as { path: string }).path).toBe('cases/two-common.stl')
    expect(rel((first.data as { volume: number }).volume, 0.5)).toBeLessThanOrEqual(1e-12)
    const again = await call()
    expect(again.ok).toBe(false)
    expect(again.error?.code).toBe('EXISTS')
    expect((await call({ overwrite: true })).ok).toBe(true)
    expect((await call({ path: '../x.stl' })).error?.code).toBe('OUTSIDE_WORKSPACE')
    expect((await call({ out: 'cases/x.obj', overwrite: true })).error?.code).toBe('INVALID')
    const opened = await geometryStlBoolean.run(StlBooleanSchema.parse({ path: 'cases/open.stl', op: 'fuse', a: 'a', b: 'b' }), ctx)
    expect(opened.error?.code).toBe('STL-OPEN')
  })

  it('B10 every dependency licence is declared and none is GPL', async () => {
    const lockPath = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../package-lock.json')
    const lock = JSON.parse(await fsp.readFile(lockPath, 'utf8')) as { packages: Record<string, { license?: unknown; version?: string; link?: boolean }> }
    const bad: string[] = []
    for (const [k, v] of Object.entries(lock.packages)) {
      if (!k.includes('node_modules/') || v.link === true) continue
      const text = typeof v.license === 'string' ? v.license : JSON.stringify(v.license)
      if (!text || /GPL/i.test(text)) bad.push(k)
    }
    expect(bad).toEqual([])
    expect(lock.packages['node_modules/manifold-3d']?.version).toBe('3.2.1')
    expect(lock.packages['node_modules/manifold-3d']?.license).toBe('Apache-2.0')
  })
})
