import { describe, expect, test } from 'vitest'
import {
  buildCartesianGrid,
  cartesianBoundarySurface,
  cartesianCellCenters,
  cartesianToPolyMesh,
  cellRangeFromBounds,
  fillGraded,
  gradedNodes,
  locateCell,
  type CartesianSpec,
} from './cartesian.js'

const near = (a: ArrayLike<number>, b: number[], tol = 1e-12): void => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < b.length; i++) expect(Math.abs(a[i] - b[i])).toBeLessThanOrEqual(tol)
}

describe('grading', () => {
  test('two-sided n=8 expansion 20 matches the hand formula and is symmetric', () => {
    const v = gradedNodes(0, 1, 8, 20, true)
    const r = Math.cbrt(20)
    const sum = 2 * (1 + r + r * r + 20)
    near(v, [0, 1 / sum, (1 + r) / sum, (1 + r + r * r) / sum, 0.5, 1 - (1 + r + r * r) / sum, 1 - (1 + r) / sum, 1 - 1 / sum, 1])
    expect(v[1]).toBeCloseTo(0.016086, 6)
    expect(v[2]).toBeCloseTo(0.059751, 6)
    const biggest = v[4] - v[3]
    const smallest = v[1] - v[0]
    expect(biggest / smallest).toBeCloseTo(20, 10)
    for (let i = 0; i <= 8; i++) expect(v[i] + v[8 - i]).toBeCloseTo(1, 14)
    expect(v[0]).toBe(0)
    expect(v[8]).toBe(1)
  })

  test('two-sided odd n straddles the centre', () => {
    const v = gradedNodes(0, 1, 3, 20, true)
    near(v, [0, 1 / 22, 21 / 22, 1])
  })

  test('one-sided n=5 expansion 20 is a geometric progression with last/first = 20', () => {
    const v = gradedNodes(0, 1, 5, 20, false)
    const r = Math.pow(20, 1 / 4)
    const den = Math.pow(r, 5) - 1
    near(v, [0, (r - 1) / den, (r * r - 1) / den, (r ** 3 - 1) / den, (r ** 4 - 1) / den, 1])
    expect(v[1]).toBeCloseTo(0.026995, 6)
    expect((v[5] - v[4]) / (v[1] - v[0])).toBeCloseTo(20, 10)
    expect(gradedNodes(2, 3, 5, 20, false)[0]).toBe(2)
    expect(gradedNodes(2, 3, 5, 20, false)[5]).toBe(3)
  })

  test('uniform fallbacks exactly as the Rust branches', () => {
    near(gradedNodes(0, 1, 2, 20, true), [0, 0.5, 1])
    near(gradedNodes(0, 1, 1, 20, false), [0, 1])
    near(gradedNodes(0, 1, 4, 0, false), [0, 0.25, 0.5, 0.75, 1])
    near(gradedNodes(0, 1, 4, -3, false), [0, 0.25, 0.5, 0.75, 1])
    near(gradedNodes(0, 1, 4, NaN, false), [0, 0.25, 0.5, 0.75, 1])
    near(gradedNodes(0, 1, 4, 1 + 1e-12, false), [0, 0.25, 0.5, 0.75, 1])
    near(gradedNodes(0, 1, 4, 0, true), [0, 0.25, 0.5, 0.75, 1])
    expect(Array.from(gradedNodes(5, 9, 0, 20, true))).toEqual([5])
    near(fillGraded(0, 2, 4, 1), [0, 0.5, 1, 1.5, 2])
    const g = fillGraded(0, 1, 3, 4)
    expect((g[3] - g[2]) / (g[1] - g[0])).toBeCloseTo(4, 12)
  })
})

const spec: CartesianSpec = {
  bounds: { min: [-1, 0, 0], max: [2, 1, 0.5] },
  cells: [3, 2, 1],
  grading: null,
  boundaries: { xmin: 'left', xmax: 'right', ymin: 'bottom', ymax: 'top', zmin: 'front', zmax: 'back' },
  regions: [{ name: 'window', on: 'ymin', shape: { kind: 'box', min: [0, -1, -1], max: [2, 1, 1] } }],
  cyclic: [{ a: 'left', b: 'right' }],
  patchTypes: { left: 'cyclic', right: 'cyclic', bottom: 'wall', top: 'wall', front: 'empty', back: 'empty', window: 'patch' },
}

describe('grid', () => {
  test('nodes, centres, lattice flags and locateCell', () => {
    const g = buildCartesianGrid(spec)
    expect(g.dims).toEqual([3, 2, 1])
    near(g.nodes.x, [-1, 0, 1, 2])
    near(g.nodes.y, [0, 0.5, 1])
    near(g.nodes.z, [0, 0.5])
    expect(g.uniform).toBe(true)
    expect(g.emptyAxis).toBe('z')
    const c = cartesianCellCenters(g)
    expect(c.length).toBe(18)
    expect(Array.from(c.subarray(0, 3))).toEqual([-0.5, 0.25, 0.25])
    expect(Array.from(c.subarray(3 * 4, 3 * 5))).toEqual([0.5, 0.75, 0.25])
    expect(locateCell(g, [0.5, 0.75, 0.1])).toBe(4)
    expect(locateCell(g, [-1, 0, 0])).toBe(0)
    expect(locateCell(g, [2, 1, 0.5])).toBe(5)
    expect(locateCell(g, [2.1, 0.5, 0.1])).toBe(-1)
    expect(locateCell(g, [0, 0.5, NaN])).toBe(-1)
    const graded = buildCartesianGrid({ ...spec, grading: { y: { expansion: 20, twoSided: true } }, cells: [3, 8, 1] })
    expect(graded.uniform).toBe(false)
    expect(graded.nodes.y[4]).toBeCloseTo(0.5, 14)
    const degenerate = buildCartesianGrid({ ...spec, grading: { y: { expansion: 20, twoSided: true } } })
    expect(degenerate.uniform).toBe(true)
  })

  test('cellRangeFromBounds keeps cells whose centre is inside, nearest cell when too thin', () => {
    const nodes = Float64Array.from([0, 1, 2, 3, 4])
    expect(cellRangeFromBounds(nodes, 0.9, 3.1)).toEqual([1, 3])
    expect(cellRangeFromBounds(nodes, 1.4, 1.6)).toEqual([1, 2])
    expect(cellRangeFromBounds(nodes, -5, 5)).toEqual([0, 4])
  })
})

describe('boundary surface and polyMesh', () => {
  test('patches follow blockgen order with the window before its host', () => {
    const g = buildCartesianGrid(spec)
    const s = cartesianBoundarySurface(g, spec)
    expect(s.patches.map((p) => p.name)).toEqual(['left', 'right', 'window', 'bottom', 'top', 'front', 'back'])
    expect(s.patches.map((p) => p.type)).toEqual(['cyclic', 'cyclic', 'patch', 'wall', 'wall', 'empty', 'empty'])
    expect(s.patches.map((p) => p.triCount)).toEqual([4, 4, 4, 2, 6, 12, 12])
    expect(s.indices.length).toBe(3 * 44)
    expect(s.positions.length).toBe(3 * 4 * 22)
    expect(s.bounds).toEqual({ min: [-1, 0, 0], max: [2, 1, 0.5] })
    // window on ymin claims cells with x centre in [0,2]: i = 1,2 -> owner cells 1 and 2
    const w = s.patches[2]
    expect(Array.from(s.cellOfTri.subarray(w.triStart, w.triStart + w.triCount))).toEqual([1, 1, 2, 2])
    // every triangle normal points outward: (centroid - cell centre) . n > 0
    const centres = cartesianCellCenters(g)
    for (let t = 0; t < s.indices.length / 3; t++) {
      const a = s.indices[3 * t]
      const b = s.indices[3 * t + 1]
      const c = s.indices[3 * t + 2]
      const cx = (s.positions[3 * a] + s.positions[3 * b] + s.positions[3 * c]) / 3
      const cy = (s.positions[3 * a + 1] + s.positions[3 * b + 1] + s.positions[3 * c + 1]) / 3
      const cz = (s.positions[3 * a + 2] + s.positions[3 * b + 2] + s.positions[3 * c + 2]) / 3
      const cell = s.cellOfTri[t]
      const dot = s.normals[3 * a] * (cx - centres[3 * cell]) + s.normals[3 * a + 1] * (cy - centres[3 * cell + 1]) + s.normals[3 * a + 2] * (cz - centres[3 * cell + 2])
      expect(dot).toBeGreaterThan(0)
      // winding agrees with the stored normal
      const ux = s.positions[3 * b] - s.positions[3 * a]
      const uy = s.positions[3 * b + 1] - s.positions[3 * a + 1]
      const uz = s.positions[3 * b + 2] - s.positions[3 * a + 2]
      const vx = s.positions[3 * c] - s.positions[3 * a]
      const vy = s.positions[3 * c + 1] - s.positions[3 * a + 1]
      const vz = s.positions[3 * c + 2] - s.positions[3 * a + 2]
      const wn = (uy * vz - uz * vy) * s.normals[3 * a] + (uz * vx - ux * vz) * s.normals[3 * a + 1] + (ux * vy - uy * vx) * s.normals[3 * a + 2]
      expect(wn).toBeGreaterThan(0)
    }
  })

  test('cartesianToPolyMesh counts, numbering and cyclic pairing', () => {
    const g = buildCartesianGrid(spec)
    const m = cartesianToPolyMesh(g, spec)
    expect(m.nCells).toBe(6)
    expect(m.nPoints).toBe(24)
    expect(m.nInternalFaces).toBe(7)
    expect(m.nFaces).toBe(29)
    // blockgen: cell 0's +x face first (owner 0, neighbour 1), then its +y face (owner 0, neighbour 3)
    expect(m.owner[0]).toBe(0)
    expect(m.neighbour[0]).toBe(1)
    expect(m.owner[1]).toBe(0)
    expect(m.neighbour[1]).toBe(3)
    for (let f = 0; f < m.nInternalFaces; f++) expect(m.owner[f]).toBeLessThan(m.neighbour[f])
    for (let f = 1; f < m.nInternalFaces; f++) expect(m.owner[f] * 100 + m.neighbour[f]).toBeGreaterThan(m.owner[f - 1] * 100 + m.neighbour[f - 1])
    expect(m.boundary.map((p) => [p.name, p.type, p.nFaces, p.startFace])).toEqual([
      ['left', 'cyclic', 2, 7],
      ['right', 'cyclic', 2, 9],
      ['window', 'patch', 2, 11],
      ['bottom', 'wall', 1, 13],
      ['top', 'wall', 3, 14],
      ['front', 'empty', 6, 17],
      ['back', 'empty', 6, 23],
    ])
    expect(m.boundary[0].extra).toEqual({ neighbourPatch: 'right' })
    expect(m.boundary[1].extra).toEqual({ neighbourPatch: 'left' })
    expect(m.boundary[2].extra).toEqual({})
    // point numbering is i fastest: point 1 is (0,0,0), point 4 is (-1,0.5,0)
    expect(Array.from(m.points.subarray(3, 6))).toEqual([0, 0, 0])
    expect(Array.from(m.points.subarray(12, 15))).toEqual([-1, 0.5, 0])
  })
})
