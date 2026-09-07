import { describe, expect, it } from 'vitest'
import { StructuredGrid, uniformNodes } from '../data/StructuredGrid'
import { integrateStreamlines, lineSeeds, planeSeeds } from './streamlines'

function rotationField(n: number) {
  const grid = new StructuredGrid({ dims: [n, n, 1], x: uniformNodes(-2, 2, n), y: uniformNodes(-2, 2, n), z: uniformNodes(0, 1, 1) })
  const U = new Float32Array(grid.cellCount * 3)
  const c = [0, 0, 0]
  for (let cell = 0; cell < grid.cellCount; cell++) {
    grid.centerOf(cell, c)
    U[cell * 3] = -c[1]
    U[cell * 3 + 1] = c[0]
    U[cell * 3 + 2] = 0
  }
  return { grid, U }
}

describe('streamlines', () => {
  it('keeps a solid-body rotation streamline on its circle', () => {
    const { grid, U } = rotationField(80)
    const seeds = new Float32Array([1, 0, 0.5])
    const lines = integrateStreamlines(grid, U, seeds, { direction: 'forward', maxLength: 2 * Math.PI * 1.05, planarAxis: 2 })
    expect(lines.offsets.length).toBe(2)
    const count = lines.offsets[1]
    expect(count).toBeGreaterThan(50)
    let maxErr = 0
    for (let i = 0; i < count; i++) {
      const rad = Math.hypot(lines.points[i * 3], lines.points[i * 3 + 1])
      maxErr = Math.max(maxErr, Math.abs(rad - 1))
      expect(lines.points[i * 3 + 2]).toBeCloseTo(0.5, 6)
      expect(lines.speeds[i]).toBeCloseTo(1, 1)
    }
    expect(maxErr).toBeLessThan(0.02)
    // closed the loop: the last point is near the seed again
    const last = (count - 1) * 3
    expect(Math.hypot(lines.points[last] - 1, lines.points[last + 1])).toBeLessThan(0.4)
  })

  it('integrates both directions and stops at the domain boundary', () => {
    const n = 20
    const grid = new StructuredGrid({ dims: [n, 4, 4], x: uniformNodes(0, 10, n), y: uniformNodes(0, 1, 4), z: uniformNodes(0, 1, 4) })
    const U = new Float32Array(grid.cellCount * 3)
    for (let cell = 0; cell < grid.cellCount; cell++) U[cell * 3] = 2
    const lines = integrateStreamlines(grid, U, [5, 0.5, 0.5], { direction: 'both' })
    const count = lines.offsets[1]
    expect(lines.points[0]).toBeLessThan(0.5)
    expect(lines.points[(count - 1) * 3]).toBeGreaterThan(9.5)
    // monotone in x
    for (let i = 1; i < count; i++) expect(lines.points[i * 3]).toBeGreaterThan(lines.points[(i - 1) * 3])
    // a stagnant field yields an empty line
    const still = integrateStreamlines(grid, new Float32Array(grid.cellCount * 3), [5, 0.5, 0.5], {})
    expect(still.offsets).toEqual(Uint32Array.from([0, 0]))
  })

  it('builds line and plane seeds', () => {
    const l = lineSeeds([0, 0, 0], [1, 2, 3], 3)
    expect(Array.from(l)).toEqual([0, 0, 0, 0.5, 1, 1.5, 1, 2, 3])
    const p = planeSeeds([0, 0, 0], [2, 4, 6], 0, 1, 2, 3)
    expect(p.length).toBe(18)
    for (let i = 0; i < 6; i++) expect(p[i * 3]).toBe(1)
    expect(p[1]).toBe(1)
    expect(p[2]).toBe(1)
  })
})
