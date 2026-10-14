// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// STU-15 gates (SPEC-LIT §148.3): a uniform velocity on the mixed unit cube gives a straight line
// ending on the box to 1e-12; the rotation U = (-y, x, 0) on a tetrahedral [-1,1]^3 (20 hexes per
// axis, each split into six tets) keeps its radius to 1e-3 over one revolution at step h/10; a
// seed outside the domain is refused.
import { describe, expect, it } from 'vitest'
import { buildMesh, mixedCube, type V3 } from './fixtures/mixedCube.js'
import { buildStreamMesh, polyStreamlines } from './polystream.js'

/** n^3 hexes on [-1,1]^3, each split into the six Kuhn tetrahedra about its 000-111 diagonal. */
function kuhnCube(n: number) {
  const pts: V3[] = []
  const id = (i: number, j: number, k: number): number => i + (n + 1) * (j + (n + 1) * k)
  for (let k = 0; k <= n; k++) for (let j = 0; j <= n; j++) for (let i = 0; i <= n; i++) pts.push([-1 + (2 * i) / n, -1 + (2 * j) / n, -1 + (2 * k) / n])
  const perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
  const cells: number[][][] = []
  for (let k = 0; k < n; k++) {
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        for (const perm of perms) {
          const v = [0, 0, 0]
          const corners = [id(i, j, k)]
          for (const axis of perm) {
            v[axis] = 1
            corners.push(id(i + v[0], j + v[1], k + v[2]))
          }
          const [a, b, c, d] = corners
          cells.push([[a, b, c], [a, b, d], [a, c, d], [b, c, d]])
        }
      }
    }
  }
  return buildMesh(pts, cells)
}

/** Distance from x to the line s + t u. */
function lineDistance(x: number[], s: number[], u: number[]): number {
  const d = [x[0] - s[0], x[1] - s[1], x[2] - s[2]]
  const uu = Math.hypot(u[0], u[1], u[2])
  const c = [d[1] * u[2] - d[2] * u[1], d[2] * u[0] - d[0] * u[2], d[0] * u[1] - d[1] * u[0]]
  return Math.hypot(c[0], c[1], c[2]) / uu
}

describe('streamlines on a polyhedral polyMesh (SPEC-LIT 148.3)', () => {
  it('uniform U on the mixed cube: the line is straight and ends on the box to 1e-12', () => {
    const mesh = mixedCube()
    const sm = buildStreamMesh(mesh)
    const u = [0.8, 0.5, 0.3]
    const U = new Float64Array(3 * mesh.nCells)
    for (let c = 0; c < mesh.nCells; c++) U.set(u, 3 * c)
    const seed = [0.3, 0.4, 0.5]
    const r = polyStreamlines(sm, U, seed, { step: 0.02 })
    expect(r.refusedSeeds).toEqual([])
    expect(r.fallbackPoints).toBe(0)
    expect(r.ends).toEqual({ boundary: 2, length: 0, steps: 0, stall: 0, lost: 0 })
    const n = r.offsets[1]
    let worst = 0
    for (let i = 0; i < n; i++) worst = Math.max(worst, lineDistance([r.points[3 * i], r.points[3 * i + 1], r.points[3 * i + 2]], seed, u))
    // The exact ends: the ray s + t u leaves [0,1]^3 at t = min_a (1 - s_a) / u_a forward, min_a s_a / u_a backward.
    const tf = Math.min(...[0, 1, 2].map((a) => (1 - seed[a]) / u[a]))
    const tb = Math.min(...[0, 1, 2].map((a) => seed[a] / u[a]))
    const fwd = [0, 1, 2].map((a) => seed[a] + tf * u[a])
    const back = [0, 1, 2].map((a) => seed[a] - tb * u[a])
    const end = [r.points[3 * (n - 1)], r.points[3 * (n - 1) + 1], r.points[3 * (n - 1) + 2]]
    const start = [r.points[0], r.points[1], r.points[2]]
    const eF = Math.hypot(end[0] - fwd[0], end[1] - fwd[1], end[2] - fwd[2])
    const eB = Math.hypot(start[0] - back[0], start[1] - back[1], start[2] - back[2])
    console.log(`STU-15 uniform U, mixed cube: ${n} points, worst distance to the line ${worst.toExponential(2)}, end errors ${eF.toExponential(2)} forward, ${eB.toExponential(2)} backward`)
    expect(worst).toBeLessThanOrEqual(1e-12)
    expect(eF).toBeLessThanOrEqual(1e-12)
    expect(eB).toBeLessThanOrEqual(1e-12)
    for (let i = 0; i < n; i++) expect(Math.abs(r.speeds[i] - Math.hypot(0.8, 0.5, 0.3))).toBeLessThanOrEqual(1e-12)
  })

  it('U = (-y, x, 0) on the 20^3 tet cube: one revolution at step h/10 keeps the radius to 1e-3', () => {
    const mesh = kuhnCube(20)
    const sm = buildStreamMesh(mesh)
    const C = sm.iso.centroids
    const U = new Float64Array(3 * mesh.nCells)
    for (let c = 0; c < mesh.nCells; c++) U.set([-C[3 * c + 1], C[3 * c], 0], 3 * c)
    const r0 = 0.5
    const z0 = 0.05
    const h = 2 / 20
    const r = polyStreamlines(sm, U, [r0, 0, z0], { direction: 'forward', step: h / 10, maxLength: 2 * Math.PI * r0 })
    expect(r.refusedSeeds).toEqual([])
    expect(r.ends).toEqual({ boundary: 0, length: 1, steps: 0, stall: 0, lost: 0 })
    const n = r.offsets[1]
    let drift = 0
    let dz = 0
    for (let i = 0; i < n; i++) {
      drift = Math.max(drift, Math.abs(Math.hypot(r.points[3 * i], r.points[3 * i + 1]) - r0) / r0)
      dz = Math.max(dz, Math.abs(r.points[3 * i + 2] - z0))
    }
    const close = Math.hypot(r.points[3 * (n - 1)] - r0, r.points[3 * (n - 1) + 1], r.points[3 * (n - 1) + 2] - z0)
    console.log(`STU-15 rotation, ${mesh.nCells} tets: ${n} points, radius drift ${drift.toExponential(2)} relative, |dz| ${dz.toExponential(2)}, end-to-seed ${close.toExponential(2)}, fallback points ${r.fallbackPoints}`)
    expect(drift).toBeLessThanOrEqual(1e-3)
    expect(dz).toBeLessThanOrEqual(1e-12)
  })

  it('a seed outside the domain is refused and the others are traced', () => {
    const mesh = mixedCube()
    const sm = buildStreamMesh(mesh)
    const U = new Float64Array(3 * mesh.nCells)
    for (let c = 0; c < mesh.nCells; c++) U.set([1, 0, 0], 3 * c)
    const r = polyStreamlines(sm, U, [1.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -1e-6, 0.5], { step: 0.05 })
    expect(r.refusedSeeds).toEqual([0, 2])
    expect(r.offsets[1]).toBe(0)
    expect(r.offsets[2]).toBeGreaterThan(2)
    expect(r.offsets[3]).toBe(r.offsets[2])
    const tets = kuhnCube(4)
    const t = polyStreamlines(buildStreamMesh(tets), new Float64Array(3 * tets.nCells).fill(1), [2, 0, 0])
    expect(t.refusedSeeds).toEqual([0])
    expect([...t.offsets]).toEqual([0, 0])
  })
})
