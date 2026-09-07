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

// A cut-cell mesh is the block minus the cells the body occupies. The lattice
// still exists; some of its sites just have no cell behind them, and a site
// index is no longer a cell index.
describe('a lattice with holes', () => {
  /** 4x4x1, with the two middle sites of the bottom row carved out. */
  function holed() {
    const sites = 4 * 4 * 1
    const index = new Int32Array(sites)
    let cell = 0
    for (let s = 0; s < sites; s++) index[s] = s === 5 || s === 6 ? -1 : cell++
    return new StructuredGrid({ dims: [4, 4, 1], x: uniformNodes(0, 4, 4), y: uniformNodes(0, 4, 4), z: uniformNodes(0, 1, 1), index })
  }

  it('maps a site to its cell, and says which sites have none', () => {
    const g = holed()
    expect(g.cellIndex(0, 0, 0)).toBe(0)
    expect(g.cellIndex(1, 1, 0)).toBe(-1)
    expect(g.cellIndex(2, 1, 0)).toBe(-1)
    // Sites after the holes shift down by two, which is exactly what a cell
    // array written by the mesher does.
    expect(g.cellIndex(3, 1, 0)).toBe(5)
    expect(g.blocked(1, 1, 0)).toBe(true)
    expect(g.blocked(0, 0, 0)).toBe(false)
  })

  it('refuses an index that does not cover the lattice', () => {
    expect(() => new StructuredGrid({ dims: [2, 2, 1], x: uniformNodes(0, 2, 2), y: uniformNodes(0, 2, 2), z: uniformNodes(0, 1, 1), index: new Int32Array(3) })).toThrow(/3 entries for 4 sites/)
  })

  it('interpolates from the corners that exist and refuses where none do', () => {
    const g = holed()
    // 14 cells; give every cell the value 1 so any correct interpolation is 1.
    const field = new Float32Array(14).fill(1)
    const out = [0]
    // Beside the hole: some corners are missing, the answer is still 1 rather
    // than a fraction of it - the weight is renormalised, not padded with zero.
    expect(g.sampleTrilinear(field, 1, 1.0, 1.5, 0.5, out)).toBe(true)
    expect(out[0]).toBeCloseTo(1, 12)
    // Dead centre of the two holes, far from any live corner.
    expect(g.sampleTrilinear(field, 1, 2.0, 1.5, 0.5, out)).toBe(false)
  })

  it('gives a site its own centre without going through a cell index', () => {
    const g = holed()
    expect(Array.from(g.centerOfSite(3, 1, 0) as number[])).toEqual([3.5, 1.5, 0.5])
  })
})
