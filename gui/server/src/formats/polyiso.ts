// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// Iso-surfaces of a cell field on a polyhedral polyMesh (SPEC-LIT §148.2): the cell values are
// reconstructed at the mesh points by a weighted least-squares fit that is exact for linear
// fields, every cell is split into tetrahedra (cell centroid, face centre, one face edge), and
// each tetrahedron is contoured by marching tetrahedra.
import type { PolyMesh } from './polymesh.js'

/** The fit grows its stencil by point rings until it has full rank, at most this many rings... */
export const ISO_MAX_RINGS = 6
/** ...and stops growing once the stencil holds this many cells. */
export const ISO_MAX_STENCIL = 256
/** A fit has full rank when its smallest |R_kk| exceeds this fraction of its largest. */
export const ISO_RANK_FLOOR = 0.1

/** The topology and geometry of a polyMesh that the iso-surface needs, built once per mesh. */
export interface IsoMesh {
  mesh: PolyMesh
  /** Vertex mean of each face, 3 per face. */
  faceCentres: Float64Array
  /** Volume centroid of each cell's face-fan triangulation, 3 per cell. */
  centroids: Float64Array
  /** Faces of cell c: cellFaces[cellFaceOffsets[c] .. cellFaceOffsets[c + 1]). */
  cellFaceOffsets: Uint32Array
  cellFaces: Uint32Array
  /** Distinct points of cell c, in the same CSR form. */
  cellPointOffsets: Uint32Array
  cellPoints: Uint32Array
  /** Cells that have point p as a vertex, in the same CSR form. */
  pointCellOffsets: Uint32Array
  pointCells: Uint32Array
  /** Tetrahedra (centroid, face centre, a, b) whose signed volume is not positive: a cell not star-shaped about its centroid. */
  invertedTets: number
}

export interface PointValues {
  /** One value per mesh point; NaN at a point of no cell. */
  values: Float64Array
  /** Points whose stencil never reached full rank: their value is the stencil's weighted mean, not exact for linear fields. */
  fallbackPoints: number
}

export interface PolyIso {
  /** Three corners (9 numbers) per triangle. */
  triPositions: Float64Array
  /** The triangle's unit normal at each corner (9 numbers per triangle), pointing to increasing values. */
  triNormals: Float64Array
  /** Cell id per triangle. */
  triCells: Int32Array
  /** Triangles of each iso value, in the order given (the triangles are in that order too). */
  valueCounts: number[]
  /** Area of each iso value's surface. */
  areas: number[]
  /** Total area. */
  area: number
  fallbackPoints: number
  invertedTets: number
}

/** Builds the iso-surface topology of `mesh`: face centres, cell centroids and the cell/point adjacency. */
export function buildIsoMesh(mesh: PolyMesh): IsoMesh {
  const { points, faceOffsets, faceIndices, owner, neighbour, nPoints, nCells, nFaces, nInternalFaces } = mesh
  const faceCentres = new Float64Array(3 * nFaces)
  for (let f = 0; f < nFaces; f++) {
    const s = faceOffsets[f]
    const e = faceOffsets[f + 1]
    let x = 0
    let y = 0
    let z = 0
    for (let k = s; k < e; k++) {
      const p = 3 * faceIndices[k]
      x += points[p]
      y += points[p + 1]
      z += points[p + 2]
    }
    const n = e - s
    faceCentres[3 * f] = x / n
    faceCentres[3 * f + 1] = y / n
    faceCentres[3 * f + 2] = z / n
  }

  const cellFaceOffsets = new Uint32Array(nCells + 1)
  for (let f = 0; f < nFaces; f++) {
    cellFaceOffsets[owner[f] + 1]++
    if (f < nInternalFaces) cellFaceOffsets[neighbour[f] + 1]++
  }
  for (let c = 0; c < nCells; c++) cellFaceOffsets[c + 1] += cellFaceOffsets[c]
  const cellFaces = new Uint32Array(cellFaceOffsets[nCells])
  const fill = cellFaceOffsets.slice(0, nCells)
  for (let f = 0; f < nFaces; f++) {
    cellFaces[fill[owner[f]]++] = f
    if (f < nInternalFaces) cellFaces[fill[neighbour[f]]++] = f
  }

  // Distinct points per cell, then the inverse map.
  const stamp = new Int32Array(nPoints).fill(-1)
  const cellPointOffsets = new Uint32Array(nCells + 1)
  const cp: number[] = []
  for (let c = 0; c < nCells; c++) {
    for (let k = cellFaceOffsets[c]; k < cellFaceOffsets[c + 1]; k++) {
      const f = cellFaces[k]
      for (let j = faceOffsets[f]; j < faceOffsets[f + 1]; j++) {
        const p = faceIndices[j]
        if (stamp[p] !== c) {
          stamp[p] = c
          cp.push(p)
        }
      }
    }
    cellPointOffsets[c + 1] = cp.length
  }
  const cellPoints = Uint32Array.from(cp)
  const pointCellOffsets = new Uint32Array(nPoints + 1)
  for (let k = 0; k < cellPoints.length; k++) pointCellOffsets[cellPoints[k] + 1]++
  for (let p = 0; p < nPoints; p++) pointCellOffsets[p + 1] += pointCellOffsets[p]
  const pointCells = new Uint32Array(cellPoints.length)
  const pfill = pointCellOffsets.slice(0, nPoints)
  for (let c = 0; c < nCells; c++) {
    for (let k = cellPointOffsets[c]; k < cellPointOffsets[c + 1]; k++) pointCells[pfill[cellPoints[k]]++] = c
  }

  // Volume centroids (148.4) about the mean of the face centres, then the tets about the centroid.
  const centroids = new Float64Array(3 * nCells)
  let invertedTets = 0
  for (let c = 0; c < nCells; c++) {
    const fs = cellFaceOffsets[c]
    const fe = cellFaceOffsets[c + 1]
    let rx = 0
    let ry = 0
    let rz = 0
    for (let k = fs; k < fe; k++) {
      const f = 3 * cellFaces[k]
      rx += faceCentres[f]
      ry += faceCentres[f + 1]
      rz += faceCentres[f + 2]
    }
    const nf = fe - fs
    rx /= nf
    ry /= nf
    rz /= nf
    let vs = 0
    let sx = 0
    let sy = 0
    let sz = 0
    forEachTet(mesh, faceCentres, cellFaces, fs, fe, c, (fx, fy, fz, a, b, sign) => {
      const v = sign * tetVolume6(rx, ry, rz, fx, fy, fz, points, a, b)
      vs += v
      sx += v * (rx + fx + points[a] + points[b])
      sy += v * (ry + fy + points[a + 1] + points[b + 1])
      sz += v * (rz + fz + points[a + 2] + points[b + 2])
    })
    let cx = rx
    let cy = ry
    let cz = rz
    if (vs > 0) {
      cx = sx / (4 * vs)
      cy = sy / (4 * vs)
      cz = sz / (4 * vs)
    }
    centroids[3 * c] = cx
    centroids[3 * c + 1] = cy
    centroids[3 * c + 2] = cz
    forEachTet(mesh, faceCentres, cellFaces, fs, fe, c, (fx, fy, fz, a, b, sign) => {
      if (!(sign * tetVolume6(cx, cy, cz, fx, fy, fz, points, a, b) > 0)) invertedTets++
    })
  }

  return { mesh, faceCentres, centroids, cellFaceOffsets, cellFaces, cellPointOffsets, cellPoints, pointCellOffsets, pointCells, invertedTets }
}

/**
 * Calls `visit` once per tetrahedron of cell c's face fans: the face centre (fx, fy, fz), the
 * offsets a and b (into mesh.points) of one face edge in the face's stored order, and sign +1 when
 * c owns the face (stored order is outward from c), -1 when c is its neighbour.
 */
function forEachTet(
  mesh: PolyMesh,
  faceCentres: Float64Array,
  cellFaces: Uint32Array,
  fs: number,
  fe: number,
  c: number,
  visit: (fx: number, fy: number, fz: number, a: number, b: number, sign: number) => void,
): void {
  const { faceOffsets, faceIndices, owner } = mesh
  for (let k = fs; k < fe; k++) {
    const f = cellFaces[k]
    const sign = owner[f] === c ? 1 : -1
    const s = faceOffsets[f]
    const n = faceOffsets[f + 1] - s
    const fx = faceCentres[3 * f]
    const fy = faceCentres[3 * f + 1]
    const fz = faceCentres[3 * f + 2]
    for (let j = 0; j < n; j++) visit(fx, fy, fz, 3 * faceIndices[s + j], 3 * faceIndices[s + ((j + 1) % n)], sign)
  }
}

/** 6 V of the tetrahedron (r, F, a, b): ((a - F) x (b - F)) . (F - r), positive when (F, a, b) faces away from r. */
function tetVolume6(rx: number, ry: number, rz: number, fx: number, fy: number, fz: number, pts: Float64Array, a: number, b: number): number {
  const ax = pts[a] - fx
  const ay = pts[a + 1] - fy
  const az = pts[a + 2] - fz
  const bx = pts[b] - fx
  const by = pts[b + 1] - fy
  const bz = pts[b + 2] - fz
  return (ay * bz - az * by) * (fx - rx) + (az * bx - ax * bz) * (fy - ry) + (ax * by - ay * bx) * (fz - rz)
}

/**
 * The point values of a cell field (148.5): at each point p, the weighted least-squares fit
 * phi_c ~ phi_p + g . (x_c - x_p) over the stencil's cell centroids, with weights L / |x_c - x_p|,
 * solved by Householder QR. Exact for a linear field wherever the stencil has full rank.
 */
export function reconstructPointValues(iso: IsoMesh, cellValues: ArrayLike<number>): PointValues {
  const { mesh, centroids, cellPointOffsets, cellPoints, pointCellOffsets, pointCells } = iso
  const { points, nPoints, nCells } = mesh
  if (cellValues.length < nCells) throw new Error(`reconstructPointValues: ${cellValues.length} cell values for ${nCells} cells`)
  const values = new Float64Array(nPoints)
  const mark = new Int32Array(nCells).fill(-1)
  const stencil: number[] = []
  let A = new Float64Array(4 * 64)
  let b = new Float64Array(64)
  let fallbackPoints = 0
  for (let p = 0; p < nPoints; p++) {
    // A point of no cell (an orphan of the points file) has no value and is no face's vertex.
    if (pointCellOffsets[p] === pointCellOffsets[p + 1]) {
      values[p] = NaN
      continue
    }
    stencil.length = 0
    for (let k = pointCellOffsets[p]; k < pointCellOffsets[p + 1]; k++) {
      const c = pointCells[k]
      mark[c] = p
      stencil.push(c)
    }
    const px = points[3 * p]
    const py = points[3 * p + 1]
    const pz = points[3 * p + 2]
    let frontier = 0
    let solved = false
    for (let ring = 1; ; ring++) {
      // Rows of the finite cells: [L/d, dx/d, dy/d, dz/d] = [phi_c L/d].
      let L = 0
      for (const c of stencil) if (Number.isFinite(cellValues[c])) L = Math.max(L, Math.hypot(centroids[3 * c] - px, centroids[3 * c + 1] - py, centroids[3 * c + 2] - pz))
      let n = 0
      if (A.length < 4 * stencil.length) {
        A = new Float64Array(8 * stencil.length)
        b = new Float64Array(2 * stencil.length)
      }
      for (const c of stencil) {
        const v = cellValues[c]
        if (!Number.isFinite(v)) continue
        const dx = centroids[3 * c] - px
        const dy = centroids[3 * c + 1] - py
        const dz = centroids[3 * c + 2] - pz
        const d = Math.hypot(dx, dy, dz)
        const o = 4 * n
        if (d > 0) {
          A[o] = L / d
          A[o + 1] = dx / d
          A[o + 2] = dy / d
          A[o + 3] = dz / d
          b[n] = (v * L) / d
        } else {
          A[o] = 1
          A[o + 1] = 0
          A[o + 2] = 0
          A[o + 3] = 0
          b[n] = v
        }
        n++
      }
      const x0 = n >= 4 ? lsqPointValue(A, b, n) : null
      if (x0 !== null) {
        values[p] = x0
        solved = true
        break
      }
      if (ring >= ISO_MAX_RINGS || stencil.length >= ISO_MAX_STENCIL) break
      const before = stencil.length
      for (let s = frontier; s < before; s++) {
        const c = stencil[s]
        for (let k = cellPointOffsets[c]; k < cellPointOffsets[c + 1]; k++) {
          const q = cellPoints[k]
          for (let j = pointCellOffsets[q]; j < pointCellOffsets[q + 1]; j++) {
            const nc = pointCells[j]
            if (mark[nc] !== p) {
              mark[nc] = p
              stencil.push(nc)
            }
          }
        }
      }
      frontier = before
      if (stencil.length === before) break
    }
    if (solved) continue
    fallbackPoints++
    let ws = 0
    let vs = 0
    for (const c of stencil) {
      const v = cellValues[c]
      if (!Number.isFinite(v)) continue
      const d = Math.hypot(centroids[3 * c] - px, centroids[3 * c + 1] - py, centroids[3 * c + 2] - pz)
      if (d === 0) {
        ws = 1
        vs = v
        break
      }
      ws += 1 / d
      vs += v / d
    }
    values[p] = ws > 0 ? vs / ws : NaN
  }
  return { values, fallbackPoints }
}

/**
 * Householder QR of the n x 4 row-major system A x = b (A and b are overwritten), its columns
 * first scaled to unit length so that a stretched stencil is not mistaken for a flat one. Returns
 * x[0], or null when the smallest |R_kk| is not above ISO_RANK_FLOOR times the largest.
 */
function lsqPointValue(A: Float64Array, b: Float64Array, n: number): number | null {
  const scale = [0, 0, 0, 0]
  for (let j = 0; j < 4; j++) {
    let s = 0
    for (let i = 0; i < n; i++) s += A[4 * i + j] * A[4 * i + j]
    s = Math.sqrt(s)
    if (!(s > 0)) return null
    scale[j] = s
    for (let i = 0; i < n; i++) A[4 * i + j] /= s
  }
  const diag = [0, 0, 0, 0]
  for (let k = 0; k < 4; k++) {
    let norm = 0
    for (let i = k; i < n; i++) norm += A[4 * i + k] * A[4 * i + k]
    norm = Math.sqrt(norm)
    if (!(norm > 0)) return null
    const alpha = A[4 * k + k] > 0 ? -norm : norm
    A[4 * k + k] -= alpha
    let vv = 0
    for (let i = k; i < n; i++) vv += A[4 * i + k] * A[4 * i + k]
    for (let j = k + 1; j < 4; j++) {
      let s = 0
      for (let i = k; i < n; i++) s += A[4 * i + k] * A[4 * i + j]
      const f = (2 * s) / vv
      for (let i = k; i < n; i++) A[4 * i + j] -= f * A[4 * i + k]
    }
    let s = 0
    for (let i = k; i < n; i++) s += A[4 * i + k] * b[i]
    const f = (2 * s) / vv
    for (let i = k; i < n; i++) b[i] -= f * A[4 * i + k]
    diag[k] = alpha
  }
  let hi = 0
  let lo = Infinity
  for (const r of diag) {
    hi = Math.max(hi, Math.abs(r))
    lo = Math.min(lo, Math.abs(r))
  }
  if (!(lo > ISO_RANK_FLOOR * hi)) return null
  const x = [0, 0, 0, 0]
  for (let k = 3; k >= 0; k--) {
    let s = b[k]
    for (let j = k + 1; j < 4; j++) s -= A[4 * k + j] * x[j]
    x[k] = s / diag[k]
  }
  return x[0] / scale[0]
}

/**
 * The iso-surfaces of a cell field at each of `isoValues` (148.6): marching tetrahedra on the
 * tets (centroid, face centre, a, b), with the cell value at the centroid, the mean of the face's
 * point values at its centre and the reconstructed point values at a and b. A value >= the iso
 * value is on the + side. `pointValues` defaults to reconstructPointValues(iso, cellValues).
 */
export function isoSurface(iso: IsoMesh, cellValues: ArrayLike<number>, isoValues: readonly number[], pointValues?: PointValues): PolyIso {
  const { mesh, faceCentres, centroids, cellFaceOffsets, cellFaces } = iso
  const { points, faceOffsets, faceIndices, nCells, nFaces } = mesh
  const pv = pointValues ?? reconstructPointValues(iso, cellValues)
  const phi = pv.values
  const faceValues = new Float64Array(nFaces)
  for (let f = 0; f < nFaces; f++) {
    let s = 0
    for (let k = faceOffsets[f]; k < faceOffsets[f + 1]; k++) s += phi[faceIndices[k]]
    faceValues[f] = s / (faceOffsets[f + 1] - faceOffsets[f])
  }

  const pos: number[] = []
  const nrm: number[] = []
  const cells: number[] = []
  const valueCounts: number[] = []
  const areas: number[] = []
  // The tet's four corners and values; q holds the crossing points of one triangle.
  const X = new Float64Array(12)
  const V = new Float64Array(4)
  const q = new Float64Array(12)
  let areaNow = 0
  let cellNow = 0
  let dirx = 0
  let diry = 0
  let dirz = 0

  const crossing = (u: number, m: number, slot: number, level: number): void => {
    // From the + end u to the - end m (148.7).
    const t = (V[u] - level) / (V[u] - V[m])
    q[3 * slot] = X[3 * u] + t * (X[3 * m] - X[3 * u])
    q[3 * slot + 1] = X[3 * u + 1] + t * (X[3 * m + 1] - X[3 * u + 1])
    q[3 * slot + 2] = X[3 * u + 2] + t * (X[3 * m + 2] - X[3 * u + 2])
  }
  const emit = (i: number, j: number, k: number): void => {
    const ax = q[3 * j] - q[3 * i]
    const ay = q[3 * j + 1] - q[3 * i + 1]
    const az = q[3 * j + 2] - q[3 * i + 2]
    const bx = q[3 * k] - q[3 * i]
    const by = q[3 * k + 1] - q[3 * i + 1]
    const bz = q[3 * k + 2] - q[3 * i + 2]
    let nx = ay * bz - az * by
    let ny = az * bx - ax * bz
    let nz = ax * by - ay * bx
    const len = Math.hypot(nx, ny, nz)
    if (!(len > 0)) return
    let second = j
    let third = k
    if (nx * dirx + ny * diry + nz * dirz < 0) {
      second = k
      third = j
      nx = -nx
      ny = -ny
      nz = -nz
    }
    for (const s of [i, second, third]) {
      pos.push(q[3 * s], q[3 * s + 1], q[3 * s + 2])
      nrm.push(nx / len, ny / len, nz / len)
    }
    cells.push(cellNow)
    areaNow += 0.5 * len
  }

  for (const level of isoValues) {
    const start = cells.length
    areaNow = 0
    for (let c = 0; c < nCells; c++) {
      const vc = cellValues[c]
      if (!Number.isFinite(vc)) continue
      cellNow = c
      X[0] = centroids[3 * c]
      X[1] = centroids[3 * c + 1]
      X[2] = centroids[3 * c + 2]
      V[0] = vc
      for (let k = cellFaceOffsets[c]; k < cellFaceOffsets[c + 1]; k++) {
        const f = cellFaces[k]
        X[3] = faceCentres[3 * f]
        X[4] = faceCentres[3 * f + 1]
        X[5] = faceCentres[3 * f + 2]
        V[1] = faceValues[f]
        if (!Number.isFinite(V[1])) continue
        const s = faceOffsets[f]
        const n = faceOffsets[f + 1] - s
        for (let j = 0; j < n; j++) {
          const a = faceIndices[s + j]
          const bb = faceIndices[s + ((j + 1) % n)]
          V[2] = phi[a]
          V[3] = phi[bb]
          if (!Number.isFinite(V[2]) || !Number.isFinite(V[3])) continue
          let plus = 0
          for (let i = 0; i < 4; i++) if (V[i] >= level) plus++
          if (plus === 0 || plus === 4) continue
          X[6] = points[3 * a]
          X[7] = points[3 * a + 1]
          X[8] = points[3 * a + 2]
          X[9] = points[3 * bb]
          X[10] = points[3 * bb + 1]
          X[11] = points[3 * bb + 2]
          contourTet(level, plus)
        }
      }
    }
    valueCounts.push(cells.length - start)
    areas.push(areaNow)
  }

  function contourTet(level: number, plus: number): void {
    const P: number[] = []
    const M: number[] = []
    let hi = -1
    let lo = -1
    for (let i = 0; i < 4; i++) {
      if (V[i] >= level) {
        P.push(i)
        if (hi < 0 || V[i] > V[hi]) hi = i
      } else {
        M.push(i)
        if (lo < 0 || V[i] < V[lo]) lo = i
      }
    }
    // Every triangle faces from the lowest - corner to the highest + corner.
    dirx = X[3 * hi] - X[3 * lo]
    diry = X[3 * hi + 1] - X[3 * lo + 1]
    dirz = X[3 * hi + 2] - X[3 * lo + 2]
    if (plus === 1) {
      crossing(P[0], M[0], 0, level)
      crossing(P[0], M[1], 1, level)
      crossing(P[0], M[2], 2, level)
      emit(0, 1, 2)
    } else if (plus === 3) {
      crossing(P[0], M[0], 0, level)
      crossing(P[1], M[0], 1, level)
      crossing(P[2], M[0], 2, level)
      emit(0, 1, 2)
    } else {
      // The quad a0b0, a0b1, a1b1, a1b0: consecutive crossings share a corner.
      crossing(P[0], M[0], 0, level)
      crossing(P[0], M[1], 1, level)
      crossing(P[1], M[1], 2, level)
      crossing(P[1], M[0], 3, level)
      emit(0, 1, 2)
      emit(0, 2, 3)
    }
  }

  let area = 0
  for (const a of areas) area += a
  return {
    triPositions: Float64Array.from(pos),
    triNormals: Float64Array.from(nrm),
    triCells: Int32Array.from(cells),
    valueCounts,
    areas,
    area,
    fallbackPoints: pv.fallbackPoints,
    invertedTets: iso.invertedTets,
  }
}
