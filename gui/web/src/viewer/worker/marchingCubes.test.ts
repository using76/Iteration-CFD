import { describe, expect, it } from 'vitest'
import { StructuredGrid, uniformNodes } from '../data/StructuredGrid'
import { MC_LOOPS, marchingCubes } from './marchingCubes'

function sphereField(n: number, r: number) {
  const grid = new StructuredGrid({ dims: [n, n, n], x: uniformNodes(-1, 1, n), y: uniformNodes(-1, 1, n), z: uniformNodes(-1, 1, n) })
  const f = new Float32Array(grid.cellCount)
  const c = [0, 0, 0]
  for (let cell = 0; cell < grid.cellCount; cell++) {
    grid.centerOf(cell, c)
    f[cell] = Math.hypot(c[0], c[1], c[2]) - r
  }
  return { grid, f }
}

describe('marchingCubes', () => {
  it('generates a table with 254 non-empty configurations and closed loops', () => {
    expect(MC_LOOPS.length).toBe(256)
    expect(MC_LOOPS[0]).toEqual([])
    expect(MC_LOOPS[255]).toEqual([])
    let nonEmpty = 0
    for (const loops of MC_LOOPS) {
      if (loops.length) nonEmpty++
      for (const loop of loops) expect(loop.length).toBeGreaterThanOrEqual(3)
    }
    expect(nonEmpty).toBe(254)
    // one positive corner -> one triangle on the three edges of vertex 0
    expect(MC_LOOPS[1].length).toBe(1)
    expect([...MC_LOOPS[1][0]].sort()).toEqual([0, 3, 8])
    // two opposite corners -> two separate triangles
    expect(MC_LOOPS[1 | (1 << 6)].length).toBe(2)
  })

  it('extracts a watertight sphere whose vertices sit within one cell of the radius', () => {
    const n = 40
    const r = 0.6
    const { grid, f } = sphereField(n, r)
    const cell = 2 / n
    const iso = marchingCubes(grid, f, 0)
    expect(iso.triangleCount).toBeGreaterThan(1000)
    expect(iso.positions.length).toBe(iso.triangleCount * 9)
    expect(iso.normals.length).toBe(iso.positions.length)

    let area = 0
    const edges = new Map<string, number>()
    const key = (o: number) => `${iso.positions[o].toFixed(6)},${iso.positions[o + 1].toFixed(6)},${iso.positions[o + 2].toFixed(6)}`
    for (let t = 0; t < iso.triangleCount; t++) {
      const o = t * 9
      const keys = [key(o), key(o + 3), key(o + 6)]
      for (let v = 0; v < 3; v++) {
        const rad = Math.hypot(iso.positions[o + v * 3], iso.positions[o + v * 3 + 1], iso.positions[o + v * 3 + 2])
        expect(Math.abs(rad - r)).toBeLessThan(cell)
        // normals point outward (toward increasing f)
        const dot = iso.positions[o + v * 3] * iso.normals[o + v * 3] + iso.positions[o + v * 3 + 1] * iso.normals[o + v * 3 + 1] + iso.positions[o + v * 3 + 2] * iso.normals[o + v * 3 + 2]
        expect(dot).toBeGreaterThan(0)
      }
      for (let e = 0; e < 3; e++) {
        const a = keys[e]
        const b = keys[(e + 1) % 3]
        const k = a < b ? `${a}|${b}` : `${b}|${a}`
        edges.set(k, (edges.get(k) ?? 0) + 1)
      }
      const ax = iso.positions[o + 3] - iso.positions[o]
      const ay = iso.positions[o + 4] - iso.positions[o + 1]
      const az = iso.positions[o + 5] - iso.positions[o + 2]
      const bx = iso.positions[o + 6] - iso.positions[o]
      const by = iso.positions[o + 7] - iso.positions[o + 1]
      const bz = iso.positions[o + 8] - iso.positions[o + 2]
      area += 0.5 * Math.hypot(ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)
    }
    for (const count of edges.values()) expect(count).toBe(2)
    expect(area / (4 * Math.PI * r * r)).toBeGreaterThan(0.97)
    expect(area / (4 * Math.PI * r * r)).toBeLessThan(1.03)
  })

  it('returns nothing for 2-D lattices', () => {
    const grid = new StructuredGrid({ dims: [4, 4, 1], x: uniformNodes(0, 1, 4), y: uniformNodes(0, 1, 4), z: uniformNodes(0, 1, 1) })
    const iso = marchingCubes(grid, new Float32Array(16).fill(1), 0.5)
    expect(iso.triangleCount).toBe(0)
  })
})
