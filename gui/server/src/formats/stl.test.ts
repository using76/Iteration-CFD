import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { applyTransform, connectedComponents, measureSolid, readSurface, sniffStlBinary, writeStl } from './stl.js'

let dir: string
beforeAll(async () => {
  dir = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'cfd-stl-'))
})
afterAll(async () => {
  await fs.promises.rm(dir, { recursive: true, force: true })
})

const TET_FACETS = [
  [[0, 0, 0], [0, 1, 0], [1, 0, 0]],
  [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
  [[0, 0, 0], [1, 0, 0], [0, 0, 1]],
  [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
]
const tetAscii = (name: string, ox = 0, facets: number[][][] = TET_FACETS): string =>
  `solid ${name}\n` + facets.map((f) => `facet normal 0 0 0\nouter loop\n${f.map(([x, y, z]) => `vertex ${x + ox} ${y} ${z}\n`).join('')}endloop\nendfacet\n`).join('') + `endsolid ${name}\n`
const TET = tetAscii('tet')
const OBJ = 'v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 0 0 1\nv 1 0 1\nv 1 1 1\nv 0 1 1\no cube\nf 1 4 3 2\nf 5 6 7 8\nf 1 2 6 5\nf 3 4 8 7\nf 1 5 8 4\nf 2 3 7 6\n'
const cubeTris = (mn: number, mx: number): number[] => {
  const v = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]].map(([a, b, c]) => [a ? mx : mn, b ? mx : mn, c ? mx : mn])
  const f = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [0, 4, 7, 3], [1, 2, 6, 5]]
  const out: number[] = []
  for (const [a, b, c, d] of f) for (const [p, q, r] of [[a, b, c], [a, c, d]]) out.push(...v[p], ...v[q], ...v[r])
  return out
}
const writeBinary = (rel: string, tris: number[]): string => {
  const n = tris.length / 9
  const buf = Buffer.alloc(84 + 50 * n)
  buf.writeUInt32LE(n, 80)
  for (let t = 0; t < n; t++) for (let v = 0; v < 9; v++) buf.writeFloatLE(tris[t * 9 + v], 84 + t * 50 + 12 + v * 4)
  const p = path.join(dir, rel)
  fs.writeFileSync(p, buf)
  return p
}
const cmp = (a: Float32Array | Uint32Array, b: Float32Array | Uint32Array): number =>
  Buffer.compare(Buffer.from(a.buffer, a.byteOffset, a.byteLength), Buffer.from(b.buffer, b.byteOffset, b.byteLength))

describe('stl', () => {
  it('reads_an_ascii_tetrahedron', async () => {
    fs.writeFileSync(path.join(dir, 'tet.stl'), TET)
    const g = await readSurface(path.join(dir, 'tet.stl'))
    expect(g.format).toBe('stl-ascii')
    expect(g.indices.length / 3).toBe(4)
    expect(g.positions.length / 3).toBe(4)
    expect(g.solids.map((s) => s.name)).toEqual(['tet'])
    const m = measureSolid(g.positions, g.indices, 0, 4)
    expect(m.closed).toBe(true)
    expect(m.openEdges).toBe(0)
    expect(m.bounds).toEqual({ min: [0, 0, 0], max: [1, 1, 1] })
    expect(Math.abs(m.volume - 1 / 6)).toBeLessThan(1e-9)
    expect(Math.abs(m.area - (1.5 + Math.sqrt(3) / 2))).toBeLessThan(1e-9)
  })

  it('binary_round_trip_is_identical', async () => {
    const g1 = await readSurface(path.join(dir, 'tet.stl'))
    const bin = path.join(dir, 'tet_bin.stl')
    await writeStl(bin, g1, { binary: true })
    const g2 = await readSurface(bin)
    expect(g2.format).toBe('stl-binary')
    expect(cmp(g1.positions, g2.positions)).toBe(0)
    expect(cmp(g1.indices, g2.indices)).toBe(0)
    expect(g2.indices.length / 3).toBe(4)
  })

  it('binary_starting_with_solid_is_still_binary', async () => {
    const raw = fs.readFileSync(path.join(dir, 'tet_bin.stl'))
    raw.write('solid fake', 0, 'ascii')
    expect(sniffStlBinary(raw)).toBe(true)
    const p = path.join(dir, 'tet_fake.stl')
    fs.writeFileSync(p, raw)
    const g = await readSurface(p)
    expect(g.format).toBe('stl-binary')
    expect(g.indices.length / 3).toBe(4)
  })

  it('a_hole_is_reported', async () => {
    const p = path.join(dir, 'hole.stl')
    fs.writeFileSync(p, tetAscii('tet', 0, TET_FACETS.slice(1)))
    const g = await readSurface(p)
    expect(g.indices.length / 3).toBe(3)
    const m = measureSolid(g.positions, g.indices, 0, 3)
    expect(m.closed).toBe(false)
    expect(m.openEdges).toBe(3)
  })

  it('two_boxes_split_into_two_solids', async () => {
    // One component reached through a triangle whose last two corners are already joined: merging a stale root once cycled here.
    expect(connectedComponents(new Uint32Array([9, 10, 11, 9, 11, 12, 0, 1, 2, 9, 1, 2]), 13, 64).solids.length).toBe(1)
    const g = await readSurface(writeBinary('boxes.stl', [...cubeTris(0, 1), ...cubeTris(2, 4)]))
    expect(g.solids.length).toBe(2)
    expect(g.solids.map((s) => s.count)).toEqual([12, 12])
    const volumes = [1, 8]
    g.solids.forEach((s, i) => {
      const m = measureSolid(g.positions, g.indices, s.first, s.count)
      expect(m.closed).toBe(true)
      expect(Math.abs(m.volume - volumes[i])).toBeLessThan(1e-6)
    })
  })

  it('obj_cube_reads', async () => {
    const p = path.join(dir, 'cube.obj')
    fs.writeFileSync(p, OBJ)
    const g = await readSurface(p)
    expect(g.format).toBe('obj')
    expect(g.solids.map((s) => s.name)).toEqual(['cube'])
    expect(g.indices.length / 3).toBe(12)
    const m = measureSolid(g.positions, g.indices, 0, 12)
    expect(m.closed).toBe(true)
    expect(Math.abs(m.volume - 1)).toBeLessThan(1e-6)
    fs.writeFileSync(path.join(dir, 'two.step'), 'ISO-10303-21;\n')
    await expect(readSurface(path.join(dir, 'two.step'))).rejects.toThrow(/not an STL or OBJ file/)
  })

  it('ascii_names_survive_write_and_rename', async () => {
    const src = path.join(dir, 'two_tets.stl')
    fs.writeFileSync(src, tetAscii('a') + tetAscii('b', 2))
    const g0 = await readSurface(src)
    expect(g0.solids.map((s) => s.name)).toEqual(['a', 'b'])
    const out = path.join(dir, 'renamed.stl')
    await writeStl(out, g0, { binary: false, names: { a: 'left' } })
    const g = await readSurface(out)
    expect(g.solids.map((s) => s.name)).toEqual(['left', 'b'])
  })

  it('transform_last_row_is_checked', async () => {
    const g = await readSurface(path.join(dir, 'tet.stl'))
    expect(() => applyTransform(g.positions, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2])).toThrow(/\[0,0,0,1\]/)
    const moved = applyTransform(g.positions, [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    let minX = Infinity
    for (let i = 0; i < moved.length; i += 3) minX = Math.min(minX, moved[i])
    expect(minX).toBe(1)
  })
})
