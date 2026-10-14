// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// STU-14 (SPEC-LIT §148.2): on the mixed unit cube, a linear cell field is reconstructed at the
// points exactly, and its iso-surface is the exact plane section of the cube: every vertex on the
// plane and the area of the independent plane ∩ cube polygon.
import { describe, expect, it } from 'vitest'
import { buildIsoMesh, isoSurface, reconstructPointValues } from './polyiso.js'
import { buildMesh, mixedCube, type V3 } from './fixtures/mixedCube.js'

const G: V3 = [2, -1, 0.5]
const f = (x: number, y: number, z: number): number => G[0] * x + G[1] * y + G[2] * z

/** The area of {x in [0,1]^3 : G . x = level}: the plane's crossings of the 12 edges, sorted by angle, shoelace. Shares no code with polyiso. */
function planeCubeArea(level: number): number {
  const corners: V3[] = []
  for (let i = 0; i < 8; i++) corners.push([i & 1, (i >> 1) & 1, (i >> 2) & 1])
  const pts: V3[] = []
  for (let i = 0; i < 8; i++) {
    for (const bit of [1, 2, 4]) {
      if (i & bit) continue
      const a = corners[i]
      const b = corners[i | bit]
      const da = f(...a) - level
      const db = f(...b) - level
      if ((da < 0 && db > 0) || (da > 0 && db < 0)) {
        const t = da / (da - db)
        pts.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])])
      }
    }
  }
  const n = Math.hypot(...G)
  const u: V3 = [G[0] / n, G[1] / n, G[2] / n]
  const m: V3 = [0, 0, 0]
  for (const p of pts) for (let k = 0; k < 3; k++) m[k] += p[k] / pts.length
  // e1 = u x (0, 0, 1), e2 = u x e1: an in-plane frame.
  const e1: V3 = [u[1], -u[0], 0]
  const l1 = Math.hypot(...e1)
  for (let k = 0; k < 3; k++) e1[k] /= l1
  const e2: V3 = [u[1] * e1[2] - u[2] * e1[1], u[2] * e1[0] - u[0] * e1[2], u[0] * e1[1] - u[1] * e1[0]]
  const uv = pts.map((p) => {
    const d: V3 = [p[0] - m[0], p[1] - m[1], p[2] - m[2]]
    return [d[0] * e1[0] + d[1] * e1[1] + d[2] * e1[2], d[0] * e2[0] + d[1] * e2[1] + d[2] * e2[2]]
  })
  uv.sort((a, b) => Math.atan2(a[1], a[0]) - Math.atan2(b[1], b[0]))
  let s = 0
  for (let k = 0; k < uv.length; k++) {
    const [x0, y0] = uv[k]
    const [x1, y1] = uv[(k + 1) % uv.length]
    s += x0 * y1 - x1 * y0
  }
  return Math.abs(s) / 2
}

describe('iso-surfaces of a cell field on a polyMesh', () => {
  const mesh = mixedCube()
  const iso = buildIsoMesh(mesh)
  const cellValues = new Float64Array(mesh.nCells)
  for (let c = 0; c < mesh.nCells; c++) cellValues[c] = f(iso.centroids[3 * c], iso.centroids[3 * c + 1], iso.centroids[3 * c + 2])

  it('the centroids are the cells volume centroids: the volume-weighted mean of the centroids is the cube centre', () => {
    // sum_c V_c C_c = the cube's first moment (1/2, 1/2, 1/2) when the cells tile it.
    expect(iso.invertedTets).toBe(0)
    let vol = 0
    const m = [0, 0, 0]
    for (let c = 0; c < mesh.nCells; c++) {
      const v = cellVolume(c)
      vol += v
      for (let k = 0; k < 3; k++) m[k] += v * iso.centroids[3 * c + k]
    }
    expect(Math.abs(vol - 1)).toBeLessThan(1e-14)
    for (let k = 0; k < 3; k++) expect(Math.abs(m[k] - 0.5)).toBeLessThan(1e-14)
  })

  it('f = 2x - y + 0.5z: the reconstructed point values equal f to 1e-12', () => {
    const pv = reconstructPointValues(iso, cellValues)
    expect(pv.fallbackPoints).toBe(0)
    let worst = 0
    for (let p = 0; p < mesh.nPoints; p++) {
      const want = f(mesh.points[3 * p], mesh.points[3 * p + 1], mesh.points[3 * p + 2])
      worst = Math.max(worst, Math.abs(pv.values[p] - want))
    }
    console.log(`STU-14 point reconstruction: worst |phi_p - f(x_p)| = ${worst.toExponential(2)} over ${mesh.nPoints} points`)
    expect(worst).toBeLessThanOrEqual(1e-12)
  })

  it('the iso f = 0.3 lies on the plane to 1e-12, has the plane ∩ cube area to 1e-9 and faces up the gradient', () => {
    const r = isoSurface(iso, cellValues, [0.3])
    const nTri = r.triCells.length
    expect(nTri).toBeGreaterThan(0)
    expect(r.valueCounts).toEqual([nTri])
    const gl = Math.hypot(...G)
    let worst = 0
    for (let k = 0; k < 3 * nTri; k++) {
      const d = Math.abs(f(r.triPositions[3 * k], r.triPositions[3 * k + 1], r.triPositions[3 * k + 2]) - 0.3) / gl
      worst = Math.max(worst, d)
    }
    const want = planeCubeArea(0.3)
    const rel = Math.abs(r.area - want) / want
    console.log(`STU-14 iso f = 0.3: ${nTri} triangles in ${new Set(r.triCells).size} cells, worst distance ${worst.toExponential(2)}, area ${r.area} vs ${want} (rel ${rel.toExponential(2)})`)
    expect(worst).toBeLessThanOrEqual(1e-12)
    expect(rel).toBeLessThanOrEqual(1e-9)
    expect(r.areas).toEqual([r.area])
    for (let k = 0; k < 3 * nTri; k++) {
      const dot = r.triNormals[3 * k] * G[0] + r.triNormals[3 * k + 1] * G[1] + r.triNormals[3 * k + 2] * G[2]
      expect(dot).toBeGreaterThan(0)
    }
  })

  it('several values come in order, and a value outside the field gives no triangle', () => {
    const r = isoSurface(iso, cellValues, [5, 0.3, -0.3, -2])
    expect(r.valueCounts[0]).toBe(0)
    expect(r.valueCounts[3]).toBe(0)
    expect(r.valueCounts[1]).toBeGreaterThan(0)
    expect(r.valueCounts[2]).toBeGreaterThan(0)
    expect(r.triCells.length).toBe(r.valueCounts.reduce((s, n) => s + n, 0))
    expect(Math.abs(r.areas[2] - planeCubeArea(-0.3)) / planeCubeArea(-0.3)).toBeLessThanOrEqual(1e-9)
  })

  it('a stretched hex block (cells 1000:1, and graded 1.5 from 1e-5) is reconstructed exactly with no fallback', () => {
    // Columns scaled to unit length: anisotropy alone does not read as a flat stencil.
    for (const [dz, grade] of [[1e-3 / 6, 1], [1e-5, 1.5]]) {
      const m = hexBlock(6, dz, grade)
      const im = buildIsoMesh(m)
      const h = (x: number, y: number, z: number): number => 2 * x - y + 500 * z
      const v = new Float64Array(m.nCells)
      for (let c = 0; c < m.nCells; c++) v[c] = h(im.centroids[3 * c], im.centroids[3 * c + 1], im.centroids[3 * c + 2])
      const pv = reconstructPointValues(im, v)
      expect(pv.fallbackPoints).toBe(0)
      let worst = 0
      for (let p = 0; p < m.nPoints; p++) worst = Math.max(worst, Math.abs(pv.values[p] - h(m.points[3 * p], m.points[3 * p + 1], m.points[3 * p + 2])))
      expect(worst).toBeLessThanOrEqual(1e-12)
    }
  })

  /** Cell volume by the divergence theorem over the fan-triangulated faces (independent of polyiso's tets). */
  function cellVolume(c: number): number {
    let v = 0
    for (let k = iso.cellFaceOffsets[c]; k < iso.cellFaceOffsets[c + 1]; k++) {
      const fc = iso.cellFaces[k]
      const sign = mesh.owner[fc] === c ? 1 : -1
      const s = mesh.faceOffsets[fc]
      const n = mesh.faceOffsets[fc + 1] - s
      const F = [iso.faceCentres[3 * fc], iso.faceCentres[3 * fc + 1], iso.faceCentres[3 * fc + 2]]
      for (let j = 0; j < n; j++) {
        const a = 3 * mesh.faceIndices[s + j]
        const b = 3 * mesh.faceIndices[s + ((j + 1) % n)]
        const A = [mesh.points[a], mesh.points[a + 1], mesh.points[a + 2]]
        const B = [mesh.points[b], mesh.points[b + 1], mesh.points[b + 2]]
        // F . (A x B) / 6 is the signed volume of (origin, F, A, B).
        v += (sign * (F[0] * (A[1] * B[2] - A[2] * B[1]) + F[1] * (A[2] * B[0] - A[0] * B[2]) + F[2] * (A[0] * B[1] - A[1] * B[0]))) / 6
      }
    }
    return v
  }
})

/** n x n x n hexes over [0,1]^2 x [0, z_n], the first layer dz thick and each next one `grade` times the last. */
function hexBlock(n: number, dz: number, grade: number) {
  const pts: V3[] = []
  const zs = [0]
  for (let k = 0, d = dz; k < n; k++, d *= grade) zs.push(zs[k] + d)
  const id = (i: number, j: number, k: number): number => i + (n + 1) * (j + (n + 1) * k)
  for (let k = 0; k <= n; k++) for (let j = 0; j <= n; j++) for (let i = 0; i <= n; i++) pts.push([i / n, j / n, zs[k]])
  const cells: number[][][] = []
  for (let k = 0; k < n; k++) {
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const v = (a: number, b: number, c: number): number => id(i + a, j + b, k + c)
        cells.push([
          [v(0, 0, 0), v(1, 0, 0), v(1, 1, 0), v(0, 1, 0)],
          [v(0, 0, 1), v(1, 0, 1), v(1, 1, 1), v(0, 1, 1)],
          [v(0, 0, 0), v(1, 0, 0), v(1, 0, 1), v(0, 0, 1)],
          [v(0, 1, 0), v(1, 1, 0), v(1, 1, 1), v(0, 1, 1)],
          [v(0, 0, 0), v(0, 1, 0), v(0, 1, 1), v(0, 0, 1)],
          [v(1, 0, 0), v(1, 1, 0), v(1, 1, 1), v(1, 0, 1)],
        ])
      }
    }
  }
  return buildMesh(pts, cells)
}
