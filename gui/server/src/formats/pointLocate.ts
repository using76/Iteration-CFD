// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The cell of a polyMesh that contains a world point (docs/17 STU-13). Candidates
// come from a uniform bucket grid over the cells' bounding boxes; a convex cell
// with planar faces is tested exactly as the intersection of its faces' inner
// half-spaces, each face's normal pointing out of the cell by OpenFOAM's owner
// rule (the sign convention of SPEC-LIT §148.1). A warped or non-convex cell has
// no such test: when only such cells can hold the point, the answer is the one
// whose centre is nearest, labelled 'nearest-centre' so it is never taken for
// an exact location.
import { faceGeometry, polyMeshCellCenters, type PolyMesh } from './polymesh.js'

/** Distances are compared to this fraction of the cell's bounding-box diagonal. */
export const LOCATE_REL_TOL = 1e-9
/** Buckets per axis never exceed this (the bucket count of a big mesh is 64^3). */
const MAX_BUCKETS_PER_AXIS = 64

export interface CellLocator {
  nCells: number
  /** CSR cell -> faces: entry `f` is a face the cell owns, `~f` (negative) one it neighbours. */
  cellFaceStarts: Int32Array
  cellFaces: Int32Array
  /** Per cell [minx, miny, minz, maxx, maxy, maxz]. */
  bbox: Float64Array
  /** 1 when the cell is convex with planar faces, so the half-space test is exact. */
  convex: Uint8Array
  /** Cell centres (face-centroid, area-weighted), as the viewer and line_sample use them. */
  centres: Float32Array
  dims: [number, number, number]
  min: [number, number, number]
  /** Bucket width per axis; 0 on an axis of zero span. */
  widths: [number, number, number]
  /** The mesh bounding box. */
  lo: [number, number, number]
  hi: [number, number, number]
  /** CSR bucket -> cells, each list in increasing cell id. */
  bucketStarts: Int32Array
  bucketCells: Int32Array
  mesh: PolyMesh
}

export interface LocateHit {
  cell: number
  /** 'contains': the cell is convex and the point lies in it. 'nearest-centre': the nearest centre among the warped or non-convex cells whose box holds the point. */
  method: 'contains' | 'nearest-centre'
  center: [number, number, number]
}

function diag(bbox: Float64Array, c: number): number {
  const o = 6 * c
  return Math.hypot(bbox[o + 3] - bbox[o], bbox[o + 4] - bbox[o + 1], bbox[o + 5] - bbox[o + 2])
}

/** Build the cell -> face lists, the boxes, the convexity flags and the bucket grid of a mesh. */
export function buildCellLocator(mesh: PolyMesh): CellLocator {
  const { nCells, nFaces, nInternalFaces, owner, neighbour, points, faceOffsets, faceIndices } = mesh
  const cellFaceStarts = new Int32Array(nCells + 1)
  for (let f = 0; f < nFaces; f++) {
    cellFaceStarts[owner[f] + 1]++
    if (f < nInternalFaces) cellFaceStarts[neighbour[f] + 1]++
  }
  for (let c = 0; c < nCells; c++) cellFaceStarts[c + 1] += cellFaceStarts[c]
  const cellFaces = new Int32Array(cellFaceStarts[nCells])
  const cursor = cellFaceStarts.slice(0, nCells)
  for (let f = 0; f < nFaces; f++) {
    cellFaces[cursor[owner[f]]++] = f
    if (f < nInternalFaces) cellFaces[cursor[neighbour[f]]++] = ~f
  }

  const bbox = new Float64Array(6 * nCells)
  for (let c = 0; c < nCells; c++) {
    bbox.set([Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity], 6 * c)
    for (let k = cellFaceStarts[c]; k < cellFaceStarts[c + 1]; k++) {
      const f = cellFaces[k] < 0 ? ~cellFaces[k] : cellFaces[k]
      for (let i = faceOffsets[f]; i < faceOffsets[f + 1]; i++) {
        const p = 3 * faceIndices[i]
        for (let a = 0; a < 3; a++) {
          const v = points[p + a]
          if (v < bbox[6 * c + a]) bbox[6 * c + a] = v
          if (v > bbox[6 * c + 3 + a]) bbox[6 * c + 3 + a] = v
        }
      }
    }
  }

  // Convex with planar faces: every vertex of the cell lies on or behind every face's plane,
  // and the face's own vertices lie on it.
  const convex = new Uint8Array(nCells)
  const g = new Float64Array(6)
  for (let c = 0; c < nCells; c++) {
    const tol = LOCATE_REL_TOL * diag(bbox, c)
    let ok = cellFaceStarts[c + 1] - cellFaceStarts[c] >= 4
    for (let k = cellFaceStarts[c]; ok && k < cellFaceStarts[c + 1]; k++) {
      const signed = cellFaces[k]
      const f = signed < 0 ? ~signed : signed
      faceGeometry(mesh, f, g, 0)
      const len = Math.hypot(g[3], g[4], g[5])
      if (!(len > 0)) {
        ok = false
        break
      }
      const s = (signed < 0 ? -1 : 1) / len
      const nx = s * g[3]
      const ny = s * g[4]
      const nz = s * g[5]
      for (let j = cellFaceStarts[c]; ok && j < cellFaceStarts[c + 1]; j++) {
        const h = cellFaces[j] < 0 ? ~cellFaces[j] : cellFaces[j]
        for (let i = faceOffsets[h]; i < faceOffsets[h + 1]; i++) {
          const p = 3 * faceIndices[i]
          const d = (points[p] - g[0]) * nx + (points[p + 1] - g[1]) * ny + (points[p + 2] - g[2]) * nz
          if (d > tol || (h === f && d < -tol)) {
            ok = false
            break
          }
        }
      }
    }
    convex[c] = ok ? 1 : 0
  }

  const lo: [number, number, number] = [Infinity, Infinity, Infinity]
  const hi: [number, number, number] = [-Infinity, -Infinity, -Infinity]
  for (let c = 0; c < nCells; c++) {
    for (let a = 0; a < 3; a++) {
      if (bbox[6 * c + a] < lo[a]) lo[a] = bbox[6 * c + a]
      if (bbox[6 * c + 3 + a] > hi[a]) hi[a] = bbox[6 * c + 3 + a]
    }
  }
  const dims: [number, number, number] = [1, 1, 1]
  const widths: [number, number, number] = [0, 0, 0]
  const per = Math.min(MAX_BUCKETS_PER_AXIS, Math.max(1, Math.ceil(Math.cbrt(nCells))))
  for (let a = 0; a < 3; a++) {
    const span = hi[a] - lo[a]
    if (span > 0) {
      dims[a] = per
      widths[a] = span / per
    }
  }
  const min: [number, number, number] = [lo[0], lo[1], lo[2]]
  const axisBucket = (v: number, a: number): number => {
    if (dims[a] === 1) return 0
    const b = Math.floor((v - min[a]) / widths[a])
    return b < 0 ? 0 : b >= dims[a] ? dims[a] - 1 : b
  }
  // Each cell goes into every bucket its box, grown by the tolerance, overlaps.
  const range = (c: number, a: number): [number, number] => {
    const t = LOCATE_REL_TOL * diag(bbox, c)
    return [axisBucket(bbox[6 * c + a] - t, a), axisBucket(bbox[6 * c + 3 + a] + t, a)]
  }
  const nBuckets = dims[0] * dims[1] * dims[2]
  const bucketStarts = new Int32Array(nBuckets + 1)
  const visit = (c: number, fn: (b: number) => void): void => {
    const [x0, x1] = range(c, 0)
    const [y0, y1] = range(c, 1)
    const [z0, z1] = range(c, 2)
    for (let z = z0; z <= z1; z++) for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) fn(x + dims[0] * (y + dims[1] * z))
  }
  for (let c = 0; c < nCells; c++) visit(c, (b) => bucketStarts[b + 1]++)
  for (let b = 0; b < nBuckets; b++) bucketStarts[b + 1] += bucketStarts[b]
  const fill = bucketStarts.slice(0, nBuckets)
  const bucketCells = new Int32Array(bucketStarts[nBuckets])
  for (let c = 0; c < nCells; c++) visit(c, (b) => (bucketCells[fill[b]++] = c))

  const centres = polyMeshCellCenters(mesh)
  return { nCells, cellFaceStarts, cellFaces, bbox, convex, centres, dims, min, widths, lo, hi, bucketStarts, bucketCells, mesh }
}

/** True when p lies in convex cell c: on or behind every face's plane, within the tolerance. */
function inConvexCell(loc: CellLocator, c: number, p: readonly [number, number, number], g: Float64Array): boolean {
  const tol = LOCATE_REL_TOL * diag(loc.bbox, c)
  for (let k = loc.cellFaceStarts[c]; k < loc.cellFaceStarts[c + 1]; k++) {
    const signed = loc.cellFaces[k]
    faceGeometry(loc.mesh, signed < 0 ? ~signed : signed, g, 0)
    const s = (signed < 0 ? -1 : 1) / Math.hypot(g[3], g[4], g[5])
    const d = s * ((p[0] - g[0]) * g[3] + (p[1] - g[1]) * g[4] + (p[2] - g[2]) * g[5])
    if (d > tol) return false
  }
  return true
}

/**
 * The cell that contains p, or null when no cell does (the point is outside the mesh, or in a hole).
 * Several convex cells hold a point on their shared face; the lowest id answers.
 */
export function locatePoint(loc: CellLocator, p: readonly [number, number, number]): LocateHit | null {
  if (!p.every(Number.isFinite) || loc.nCells === 0) return null
  const b: [number, number, number] = [0, 0, 0]
  for (let a = 0; a < 3; a++) {
    const t = LOCATE_REL_TOL * (loc.hi[a] - loc.lo[a] + Math.abs(loc.lo[a]) + Math.abs(loc.hi[a]))
    if (p[a] < loc.lo[a] - t || p[a] > loc.hi[a] + t) return null
    if (loc.dims[a] === 1) continue
    const v = Math.floor((p[a] - loc.min[a]) / loc.widths[a])
    b[a] = v < 0 ? 0 : v >= loc.dims[a] ? loc.dims[a] - 1 : v
  }
  const bucket = b[0] + loc.dims[0] * (b[1] + loc.dims[1] * b[2])
  const g = new Float64Array(6)
  let nearest = -1
  let nearestD2 = Infinity
  for (let k = loc.bucketStarts[bucket]; k < loc.bucketStarts[bucket + 1]; k++) {
    const c = loc.bucketCells[k]
    const o = 6 * c
    const t = LOCATE_REL_TOL * diag(loc.bbox, c)
    if (p[0] < loc.bbox[o] - t || p[1] < loc.bbox[o + 1] - t || p[2] < loc.bbox[o + 2] - t) continue
    if (p[0] > loc.bbox[o + 3] + t || p[1] > loc.bbox[o + 4] + t || p[2] > loc.bbox[o + 5] + t) continue
    if (loc.convex[c]) {
      if (inConvexCell(loc, c, p, g)) return { cell: c, method: 'contains', center: centreOf(loc, c) }
      continue
    }
    const dx = loc.centres[3 * c] - p[0]
    const dy = loc.centres[3 * c + 1] - p[1]
    const dz = loc.centres[3 * c + 2] - p[2]
    const d2 = dx * dx + dy * dy + dz * dz
    if (d2 < nearestD2) {
      nearestD2 = d2
      nearest = c
    }
  }
  return nearest >= 0 ? { cell: nearest, method: 'nearest-centre', center: centreOf(loc, nearest) } : null
}

function centreOf(loc: CellLocator, c: number): [number, number, number] {
  return [loc.centres[3 * c], loc.centres[3 * c + 1], loc.centres[3 * c + 2]]
}
