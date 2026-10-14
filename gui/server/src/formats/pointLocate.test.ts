// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// docs/17 STU-13 gates of point location on a polyMesh: on the mixed unit cube every one of
// 10,000 random interior points lands in the cell a brute-force half-space oracle names,
// points outside the cube are MISSes, and a warped or non-convex cell answers labelled
// 'nearest-centre'.
import { describe, expect, it } from 'vitest'
import { MIX_CELLS, buildMesh, mixedCube, newell, type V3 } from './fixtures/mixedCube.js'
import { buildCellLocator, locatePoint } from './pointLocate.js'
import type { PolyMesh } from './polymesh.js'

/** Seeded uniform numbers in [0, 1) (mulberry32). */
function rng(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const P = (m: PolyMesh, i: number): V3 => [m.points[3 * i], m.points[3 * i + 1], m.points[3 * i + 2]]

/** The oracle: every cell whose faces' planes (Newell normal through the first vertex) all have x on or behind them. */
function oracle(m: PolyMesh, x: V3): number[] {
  const pts = Array.from({ length: m.nPoints }, (_, i) => P(m, i))
  const out: number[] = []
  for (let c = 0; c < m.nCells; c++) {
    let worst = -Infinity
    for (let f = 0; f < m.nFaces; f++) {
      const own = m.owner[f] === c
      const nb = f < m.nInternalFaces && m.neighbour[f] === c
      if (!own && !nb) continue
      const face = Array.from(m.faceIndices.subarray(m.faceOffsets[f], m.faceOffsets[f + 1]))
      if (nb) face.reverse()
      const n = newell(pts, face)
      const l = Math.hypot(n[0], n[1], n[2])
      const v = pts[face[0]]
      worst = Math.max(worst, ((x[0] - v[0]) * n[0] + (x[1] - v[1]) * n[1] + (x[2] - v[2]) * n[2]) / l)
    }
    if (worst <= 1e-9) out.push(c)
  }
  return out
}

/** An n^3 lattice of unit-spaced hexes over [0, n]^3; cell (i, j, k) is i + n (j + n k). */
function hexLattice(n: number): PolyMesh {
  const pts: V3[] = []
  const id = (i: number, j: number, k: number): number => i + (n + 1) * (j + (n + 1) * k)
  for (let k = 0; k <= n; k++) for (let j = 0; j <= n; j++) for (let i = 0; i <= n; i++) pts.push([i, j, k])
  const cells: number[][][] = []
  for (let k = 0; k < n; k++) {
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const v = (a: number, b: number, c: number): number => id(i + a, j + b, k + c)
        cells.push([
          [v(0, 0, 0), v(0, 1, 0), v(0, 1, 1), v(0, 0, 1)], [v(1, 0, 0), v(1, 1, 0), v(1, 1, 1), v(1, 0, 1)],
          [v(0, 0, 0), v(1, 0, 0), v(1, 0, 1), v(0, 0, 1)], [v(0, 1, 0), v(1, 1, 0), v(1, 1, 1), v(0, 1, 1)],
          [v(0, 0, 0), v(1, 0, 0), v(1, 1, 0), v(0, 1, 0)], [v(0, 0, 1), v(1, 0, 1), v(1, 1, 1), v(0, 1, 1)],
        ])
      }
    }
  }
  return buildMesh(pts, cells)
}

/** One cell: the unit cube with corner (1, 1, 1) lifted to `top`, so its three faces there are not planar. */
function warpedHex(top: V3): PolyMesh {
  const pts: V3[] = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], top, [0, 1, 1]]
  return buildMesh(pts, [[[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]])
}

/** One cell with planar faces that is not convex: the L of (0,0) (2,0) (2,1) (1,1) (1,2) (0,2) extruded over z in [0, 1]. */
function lPrism(): PolyMesh {
  const ring: Array<[number, number]> = [[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]]
  const pts: V3[] = [...ring.map(([x, y]): V3 => [x, y, 0]), ...ring.map(([x, y]): V3 => [x, y, 1])]
  const faces: number[][] = [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]
  const dirs: V3[] = [[0, 0, -1], [0, 0, 1]]
  for (let k = 0; k < 6; k++) {
    const a = ring[k]
    const b = ring[(k + 1) % 6]
    faces.push([k, (k + 1) % 6, 6 + ((k + 1) % 6), 6 + k])
    // The outward normal of the edge a -> b of a counter-clockwise ring is (dy, -dx).
    dirs.push([b[1] - a[1], -(b[0] - a[0]), 0])
  }
  return buildMesh(pts, [faces], [dirs])
}

describe('point location on the mixed unit cube', () => {
  const mesh = mixedCube()
  const loc = buildCellLocator(mesh)

  it('every one of its nine cells is convex with planar faces', () => {
    expect(mesh.nCells).toBe(MIX_CELLS.length)
    expect([...loc.convex]).toEqual([1, 1, 1, 1, 1, 1, 1, 1, 1])
  })

  it('10,000 random interior points: the located cell is the brute-force oracle cell for every one', () => {
    const r = rng(1301)
    const hits = new Array<number>(mesh.nCells).fill(0)
    let agree = 0
    for (let n = 0; n < 10_000; n++) {
      const x: V3 = [r(), r(), r()]
      const want = oracle(mesh, x)
      expect(want.length).toBeGreaterThan(0)
      const got = locatePoint(loc, x)
      expect(got?.method).toBe('contains')
      if (got?.cell === want[0]) agree++
      hits[got!.cell]++
    }
    expect(agree).toBe(10_000)
    // Every cell, the smallest tets included, is hit.
    expect(hits.every((h) => h > 0)).toBe(true)
  })

  it('points outside the cube are MISSes, also 1e-6 outside a wall; 1e-6 inside is found', () => {
    const r = rng(7)
    let tried = 0
    while (tried < 2_000) {
      const x: V3 = [3 * r() - 1, 3 * r() - 1, 3 * r() - 1]
      if (x.every((v) => v >= 0 && v <= 1)) continue
      tried++
      expect(locatePoint(loc, x)).toBeNull()
    }
    for (let a = 0; a < 3; a++) {
      for (const side of [0, 1]) {
        const out: V3 = [0.37, 0.61, 0.43]
        out[a] = side ? 1 + 1e-6 : -1e-6
        expect(locatePoint(loc, out)).toBeNull()
        const inside: V3 = [0.37, 0.61, 0.43]
        inside[a] = side ? 1 - 1e-6 : 1e-6
        expect(locatePoint(loc, inside)?.method).toBe('contains')
      }
    }
    expect(locatePoint(loc, [Number.NaN, 0.5, 0.5])).toBeNull()
  })

  it('a point on the face G and H share answers the lower id, G', () => {
    expect(locatePoint(loc, [0.5, 0.5, 0.2])).toMatchObject({ cell: 0, method: 'contains' })
    expect(locatePoint(loc, [0.5, 0.5, 0.25])).toMatchObject({ cell: 1, method: 'contains' })
  })
})

describe('point location with many buckets', () => {
  it('a 20^3 hex lattice: 20 buckets per axis, and 2,000 random points land in their lattice cell', () => {
    const n = 20
    const loc = buildCellLocator(hexLattice(n))
    expect(loc.dims).toEqual([20, 20, 20])
    expect(loc.convex.every((v) => v === 1)).toBe(true)
    const r = rng(42)
    for (let t = 0; t < 2_000; t++) {
      const x: V3 = [n * r(), n * r(), n * r()]
      const hit = locatePoint(loc, x)
      expect(hit).toMatchObject({ cell: Math.floor(x[0]) + n * (Math.floor(x[1]) + n * Math.floor(x[2])), method: 'contains' })
      expect(hit!.center.map((v, a) => v - Math.floor(x[a]))).toEqual([0.5, 0.5, 0.5])
    }
    expect(locatePoint(loc, [10.5, 10.5, 20.5])).toBeNull()
  })
})

describe('warped and non-convex cells answer labelled nearest-centre', () => {
  it('a hex with a lifted corner is not convex; a point in it is labelled nearest-centre', () => {
    const loc = buildCellLocator(warpedHex([1, 1, 1.2]))
    expect([...loc.convex]).toEqual([0])
    expect(locatePoint(loc, [0.5, 0.5, 0.5])).toMatchObject({ cell: 0, method: 'nearest-centre' })
    expect(locatePoint(loc, [0.5, 0.5, 1.3])).toBeNull()
    // The same cell with the corner in place is a convex cube.
    const flat = buildCellLocator(warpedHex([1, 1, 1]))
    expect([...flat.convex]).toEqual([1])
    expect(locatePoint(flat, [0.5, 0.5, 0.5])).toMatchObject({ cell: 0, method: 'contains' })
  })

  it('an L prism has planar faces but is not convex: inside it, and in its notch, the answer says nearest-centre', () => {
    const loc = buildCellLocator(lPrism())
    expect([...loc.convex]).toEqual([0])
    expect(locatePoint(loc, [0.5, 0.5, 0.5])).toMatchObject({ cell: 0, method: 'nearest-centre' })
    // The notch is outside the cell but inside its box: no exact test exists, so the label carries the doubt.
    expect(locatePoint(loc, [1.5, 1.5, 0.5])).toMatchObject({ cell: 0, method: 'nearest-centre' })
    expect(locatePoint(loc, [2.5, 0.5, 0.5])).toBeNull()
  })
})
