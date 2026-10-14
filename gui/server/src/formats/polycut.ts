// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// Exact plane cut of a polyhedral polyMesh (SPEC-LIT §148.1): per-cell section
// polygons, triangulated, with their cell ids. One pass over the faces; each
// cell's cap segments are chained into loops by their directed-edge keys.
import type { PolyMesh } from './polymesh.js'

export interface CutPlane {
  /** Any point of the plane. */
  origin: readonly [number, number, number]
  /** Plane normal (any length > 0). The section polygons are counter-clockwise about it. */
  normal: readonly [number, number, number]
}

export interface PolyCut {
  /** Cell id per section polygon. */
  polyCells: Int32Array
  /** Polygon k has the points polyPoints[3*polyOffsets[k] .. 3*polyOffsets[k+1]). */
  polyOffsets: Uint32Array
  polyPoints: Float64Array
  /** Three corners (9 numbers) per triangle, counter-clockwise about the unit normal. */
  triPositions: Float64Array
  /** Cell id per triangle. */
  triCells: Int32Array
  /** Sum of the triangles' areas (= the polygons' areas). */
  area: number
  /** Cells whose cap segments did not close into loops (an open cell surface); 0 on a valid polyMesh. */
  openCells: number
}

/**
 * Cuts every cell of `mesh` by `plane`. A point with signed distance d >= 0 is on
 * the + side (a zero counts as +), so a face lying in the plane is reported once, by
 * the cell on its - side, and a contact at a vertex or an edge gives no polygon.
 */
export function cutPolyMesh(mesh: PolyMesh, plane: CutPlane): PolyCut {
  const [ox, oy, oz] = plane.origin
  let [nx, ny, nz] = plane.normal
  const len = Math.hypot(nx, ny, nz)
  if (!(len > 0) || !Number.isFinite(len)) throw new Error('cutPolyMesh: the plane normal must be finite and non-zero')
  nx /= len
  ny /= len
  nz /= len
  const { points, faceOffsets, faceIndices, owner, neighbour, nPoints, nFaces, nInternalFaces } = mesh
  const d = new Float64Array(nPoints)
  for (let i = 0; i < nPoints; i++) {
    d[i] = nx * (points[3 * i] - ox) + ny * (points[3 * i + 1] - oy) + nz * (points[3 * i + 2] - oz)
  }

  // Crossing points, one per directed edge (+ end, - end), computed from the + end.
  const keyToPoint = new Map<number, number>()
  const xyz: number[] = []
  const posKey: number[] = []
  const crossing = (plus: number, minus: number): number => {
    const key = plus * nPoints + minus
    let id = keyToPoint.get(key)
    if (id !== undefined) return id
    id = posKey.length
    keyToPoint.set(key, id)
    const dp = d[plus]
    const t = dp / (dp - d[minus])
    const p = 3 * plus
    const m = 3 * minus
    xyz.push(
      points[p] + t * (points[m] - points[p]),
      points[p + 1] + t * (points[m + 1] - points[p + 1]),
      points[p + 2] + t * (points[m + 2] - points[p + 2]),
    )
    // t = 0 puts the point on the + vertex: the same position as every other edge leaving it.
    posKey.push(dp === 0 ? plus : nPoints + id)
    return id
  }

  const segCell: number[] = []
  const segStart: number[] = []
  const segEnd: number[] = []
  const cross: number[] = []
  const pairs: number[] = []
  const order4: number[] = []
  // A non-convex face crossed 4+ times: its cap intervals are the crossings sorted
  // along the cut line, X_in then X_out, all in one direction sigma. sigma is the
  // sign of sum(s_out - s_in), which no pairing changes. A warped face whose sorted
  // crossings do not alternate keeps the Sutherland-Hodgman pairs.
  const pairAlongLine = (a: number, n: number): void => {
    let fx = 0
    let fy = 0
    let fz = 0
    for (let k = 0; k < n; k++) {
      const p = 3 * faceIndices[a + k]
      const q = 3 * faceIndices[a + ((k + 1) % n)]
      fx += (points[p + 1] - points[q + 1]) * (points[p + 2] + points[q + 2])
      fy += (points[p + 2] - points[q + 2]) * (points[p] + points[q])
      fz += (points[p] - points[q]) * (points[p + 1] + points[q + 1])
    }
    const wx = ny * fz - nz * fy
    const wy = nz * fx - nx * fz
    const wz = nx * fy - ny * fx
    const s = (id: number): number => wx * xyz[3 * id] + wy * xyz[3 * id + 1] + wz * xyz[3 * id + 2]
    let total = 0
    for (let j = 0; j < pairs.length; j += 2) total += s(pairs[j + 1]) - s(pairs[j])
    if (total === 0) return
    const sigma = total > 0 ? 1 : -1
    order4.length = 0
    for (let j = 0; j < cross.length; j++) order4.push(j)
    const key = (j: number): number => sigma * s(cross[j] >= 0 ? cross[j] : -1 - cross[j])
    const alternates = (): boolean => order4.every((j, r) => (cross[j] >= 0) === (r % 2 === 0))
    // Coincident crossings (a + vertex with d = 0): X_in first, else X_out first.
    order4.sort((i, j) => key(i) - key(j) || (cross[i] >= 0 ? -1 : 1) - (cross[j] >= 0 ? -1 : 1))
    if (!alternates()) {
      order4.sort((i, j) => key(i) - key(j) || (cross[i] >= 0 ? 1 : -1) - (cross[j] >= 0 ? 1 : -1))
      if (!alternates()) return
    }
    pairs.length = 0
    for (let r = 0; r < order4.length; r += 2) pairs.push(cross[order4[r]], -1 - cross[order4[r + 1]])
  }
  for (let f = 0; f < nFaces; f++) {
    const a = faceOffsets[f]
    const n = faceOffsets[f + 1] - a
    let pos = false
    let neg = false
    for (let k = 0; k < n; k++) {
      if (d[faceIndices[a + k]] >= 0) pos = true
      else neg = true
    }
    if (!pos || !neg) continue
    // Crossings in the owner's (outward) walk: +id for X_in (+ to -), -(id+1) for X_out (- to +).
    cross.length = 0
    for (let k = 0; k < n; k++) {
      const u = faceIndices[a + k]
      const v = faceIndices[a + ((k + 1) % n)]
      const su = d[u] >= 0
      const sv = d[v] >= 0
      if (su && !sv) cross.push(crossing(u, v))
      else if (!su && sv) cross.push(-1 - crossing(v, u))
    }
    const m = cross.length
    // Sutherland-Hodgman: the owner's cap runs from each X_in back to the X_out before it.
    pairs.length = 0
    for (let j = 0; j < m; j++) if (cross[j] >= 0) pairs.push(cross[j], -1 - cross[(j + m - 1) % m])
    if (m > 2) pairAlongLine(a, n)
    for (let j = 0; j < pairs.length; j += 2) {
      const xin = pairs[j]
      const xout = pairs[j + 1]
      segCell.push(owner[f])
      segStart.push(xin)
      segEnd.push(xout)
      if (f < nInternalFaces) {
        segCell.push(neighbour[f])
        segStart.push(xout)
        segEnd.push(xin)
      }
    }
  }

  const nSeg = segCell.length
  const order = Array.from({ length: nSeg }, (_, i) => i).sort((i, j) => segCell[i] - segCell[j] || i - j)
  const polyCells: number[] = []
  const polyOffsets: number[] = [0]
  const polyPoints: number[] = []
  const triPositions: number[] = []
  const triCells: number[] = []
  let area = 0
  let openCells = 0
  const byStart = new Map<number, number[]>()
  const used = new Uint8Array(nSeg)
  const loop: number[] = []
  for (let s0 = 0; s0 < nSeg; ) {
    const cell = segCell[order[s0]]
    let s1 = s0
    while (s1 < nSeg && segCell[order[s1]] === cell) s1++
    byStart.clear()
    for (let s = s0; s < s1; s++) {
      const g = order[s]
      const list = byStart.get(segStart[g])
      if (list) list.push(g)
      else byStart.set(segStart[g], [g])
    }
    let open = false
    for (let s = s0; s < s1; s++) {
      const g0 = order[s]
      if (used[g0]) continue
      used[g0] = 1
      loop.length = 0
      loop.push(segStart[g0])
      let cur = segEnd[g0]
      let closed = true
      while (cur !== segStart[g0]) {
        const next = (byStart.get(cur) ?? []).find((g) => !used[g])
        if (next === undefined) {
          closed = false
          break
        }
        used[next] = 1
        loop.push(cur)
        cur = segEnd[next]
      }
      if (!closed) {
        open = true
        continue
      }
      // Drop consecutive coincident points (crossings at a + vertex with d = 0).
      const pts: number[] = []
      for (const id of loop) {
        if (pts.length === 0 || posKey[pts[pts.length - 1]] !== posKey[id]) pts.push(id)
      }
      while (pts.length > 1 && posKey[pts[0]] === posKey[pts[pts.length - 1]]) pts.pop()
      if (pts.length < 3) continue
      polyCells.push(cell)
      for (const id of pts) polyPoints.push(xyz[3 * id], xyz[3 * id + 1], xyz[3 * id + 2])
      polyOffsets.push(polyPoints.length / 3)
      for (const [i0, i1, i2] of triangulate(pts, xyz, nx, ny, nz)) {
        const ax = xyz[3 * i0]
        const ay = xyz[3 * i0 + 1]
        const az = xyz[3 * i0 + 2]
        const bx = xyz[3 * i1]
        const by = xyz[3 * i1 + 1]
        const bz = xyz[3 * i1 + 2]
        const cx = xyz[3 * i2]
        const cy = xyz[3 * i2 + 1]
        const cz = xyz[3 * i2 + 2]
        triPositions.push(ax, ay, az, bx, by, bz, cx, cy, cz)
        triCells.push(cell)
        area += triArea(ax, ay, az, bx, by, bz, cx, cy, cz, nx, ny, nz)
      }
    }
    if (open) openCells++
    s0 = s1
  }
  return {
    polyCells: Int32Array.from(polyCells),
    polyOffsets: Uint32Array.from(polyOffsets),
    polyPoints: Float64Array.from(polyPoints),
    triPositions: Float64Array.from(triPositions),
    triCells: Int32Array.from(triCells),
    area,
    openCells,
  }
}

/** Signed area of triangle abc about the unit normal n: 1/2 ((b - a) x (c - a)) . n. */
function triArea(
  ax: number, ay: number, az: number, bx: number, by: number, bz: number,
  cx: number, cy: number, cz: number, nx: number, ny: number, nz: number,
): number {
  const ux = bx - ax
  const uy = by - ay
  const uz = bz - az
  const vx = cx - ax
  const vy = cy - ay
  const vz = cz - az
  return 0.5 * (nx * (uy * vz - uz * vy) + ny * (uz * vx - ux * vz) + nz * (ux * vy - uy * vx))
}

/**
 * Triangles of a planar loop (point ids into xyz), keeping its orientation: a fan
 * from the first point when the loop is convex, ear clipping otherwise.
 */
function triangulate(pts: number[], xyz: number[], nx: number, ny: number, nz: number): Array<[number, number, number]> {
  const k = pts.length
  if (k === 3) return [[pts[0], pts[1], pts[2]]]
  // In-plane basis (u, v) with u x v = n.
  let ux: number, uy: number, uz: number
  if (Math.abs(nx) <= Math.abs(ny) && Math.abs(nx) <= Math.abs(nz)) [ux, uy, uz] = [0, nz, -ny]
  else if (Math.abs(ny) <= Math.abs(nz)) [ux, uy, uz] = [-nz, 0, nx]
  else [ux, uy, uz] = [ny, -nx, 0]
  const ul = Math.hypot(ux, uy, uz)
  ux /= ul
  uy /= ul
  uz /= ul
  const vx = ny * uz - nz * uy
  const vy = nz * ux - nx * uz
  const vz = nx * uy - ny * ux
  const x = new Float64Array(k)
  const y = new Float64Array(k)
  for (let i = 0; i < k; i++) {
    const p = 3 * pts[i]
    x[i] = xyz[p] * ux + xyz[p + 1] * uy + xyz[p + 2] * uz
    y[i] = xyz[p] * vx + xyz[p + 1] * vy + xyz[p + 2] * vz
  }
  let signed = 0
  for (let i = 0; i < k; i++) {
    const j = (i + 1) % k
    signed += x[i] * y[j] - x[j] * y[i]
  }
  const sgn = signed < 0 ? -1 : 1
  const turn = (i: number, j: number, l: number): number =>
    sgn * ((x[j] - x[i]) * (y[l] - y[i]) - (y[j] - y[i]) * (x[l] - x[i]))
  // A turn below -1e-12 of the squared extent is reflex; collinear points (split
  // coplanar faces) and round-off leave a loop convex.
  let ext = 0
  for (let i = 1; i < k; i++) ext = Math.max(ext, Math.abs(x[i] - x[0]), Math.abs(y[i] - y[0]))
  const tol = 1e-12 * ext * ext
  let convex = true
  for (let i = 0; i < k && convex; i++) {
    if (turn((i + k - 1) % k, i, (i + 1) % k) < -tol) convex = false
  }
  const out: Array<[number, number, number]> = []
  if (convex) {
    for (let i = 1; i + 1 < k; i++) out.push([pts[0], pts[i], pts[i + 1]])
    return out
  }
  const idx = Array.from({ length: k }, (_, i) => i)
  while (idx.length > 3) {
    const m = idx.length
    let ear = -1
    for (let j = 0; j < m && ear < 0; j++) {
      const i0 = idx[(j + m - 1) % m]
      const i1 = idx[j]
      const i2 = idx[(j + 1) % m]
      if (turn(i0, i1, i2) <= 0) continue
      let blocked = false
      for (const r of idx) {
        if (r === i0 || r === i1 || r === i2) continue
        if ((x[r] === x[i0] && y[r] === y[i0]) || (x[r] === x[i2] && y[r] === y[i2])) continue
        if (turn(i0, i1, r) >= 0 && turn(i1, i2, r) >= 0 && turn(i2, i0, r) >= 0) {
          blocked = true
          break
        }
      }
      if (!blocked) ear = j
    }
    if (ear < 0) break
    out.push([pts[idx[(ear + m - 1) % m]], pts[idx[ear]], pts[idx[(ear + 1) % m]]])
    idx.splice(ear, 1)
  }
  for (let i = 1; i + 1 < idx.length; i++) out.push([pts[idx[0]], pts[idx[i]], pts[idx[i + 1]]])
  return out
}
