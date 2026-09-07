import { describe, expect, it } from 'vitest'
import { StructuredGrid, gradedNodes, locateInterval, uniformNodes } from './StructuredGrid'

function gradedGrid() {
  return new StructuredGrid({ dims: [4, 3, 2], x: gradedNodes(0, 1, 4, 4), y: new Float32Array([0, 0.1, 0.5, 1]), z: uniformNodes(-1, 1, 2) })
}

describe('StructuredGrid', () => {
  it('locates intervals by binary search on graded nodes', () => {
    const nodes = new Float32Array([0, 0.125, 0.5, 1])
    expect(locateInterval(nodes, 0)).toBe(0)
    expect(locateInterval(nodes, 0.05)).toBe(0)
    expect(locateInterval(nodes, 0.125)).toBe(1)
    expect(locateInterval(nodes, 0.49)).toBe(1)
    expect(locateInterval(nodes, 0.75)).toBe(2)
    expect(locateInterval(nodes, 1)).toBe(2)
    expect(locateInterval(nodes, 1.01)).toBe(-1)
    expect(locateInterval(nodes, -0.01)).toBe(-1)
    expect(locateInterval(nodes, Number.NaN)).toBe(-1)
  })

  it('produces graded nodes with the requested expansion ratio', () => {
    const n = gradedNodes(0, 1, 4, 4)
    expect(n.length).toBe(5)
    expect(n[0]).toBe(0)
    expect(n[4]).toBe(1)
    const first = n[1] - n[0]
    const last = n[4] - n[3]
    expect(last / first).toBeCloseTo(4, 3)
  })

  it('maps world to cell index and back through the centres', () => {
    const g = gradedGrid()
    expect(g.cellCount).toBe(24)
    expect(g.cellIndex(1, 2, 1)).toBe(1 + 4 * (2 + 3 * 1))
    const cell = g.locate(0.05, 0.75, 0.5)
    expect(cell).toBe(g.cellIndex(0, 2, 1))
    const c = g.centerOf(cell)
    expect(c[1]).toBeCloseTo(0.75)
    expect(c[2]).toBeCloseTo(0.5)
    expect(g.locate(2, 0.5, 0.5)).toBe(-1)
    expect(g.clampAxis(0, 5)).toBe(3)
    expect(g.clampAxis(0, -5)).toBe(0)
    expect(g.minSpacingAt(0.99, 0.99, 0.5)).toBeCloseTo(g.nodes[0][4] - g.nodes[0][3], 6)
    expect(g.minSpacingAt(0.01, 0.99, 0.5)).toBeCloseTo(g.nodes[0][1] - g.nodes[0][0], 6)
  })

  it('interpolates cell-centred data trilinearly and clamps at the walls', () => {
    const g = new StructuredGrid({ dims: [4, 4, 1], x: uniformNodes(0, 4, 4), y: uniformNodes(0, 4, 4), z: uniformNodes(0, 1, 1) })
    // f = x + 2y evaluated at cell centres
    const f = new Float32Array(g.cellCount)
    const c = [0, 0, 0]
    for (let cell = 0; cell < g.cellCount; cell++) {
      g.centerOf(cell, c)
      f[cell] = c[0] + 2 * c[1]
    }
    const out = [0]
    expect(g.sampleTrilinear(f, 1, 1.7, 2.2, 0.5, out)).toBe(true)
    expect(out[0]).toBeCloseTo(1.7 + 4.4, 5)
    // beyond the last centre the value is clamped to the wall cell
    expect(g.sampleTrilinear(f, 1, 3.9, 0.1, 0.5, out)).toBe(true)
    expect(out[0]).toBeCloseTo(3.5 + 1, 5)
    expect(g.sampleTrilinear(f, 1, 4.1, 0.1, 0.5, out)).toBe(false)
    expect(g.sampleNearest(f, 1, 1.7, 2.2, 0.5, out)).toBe(true)
    expect(out[0]).toBeCloseTo(1.5 + 5, 5)
  })
})
