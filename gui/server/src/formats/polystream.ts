// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// Streamlines of a cell velocity on a polyhedral polyMesh (SPEC-LIT §148.3): the velocity is
// reconstructed at the mesh points as in §148.2, interpolated linearly in the tetrahedra
// (centroid, face centre, one face edge) of §148.6, and integrated by classical RK4 in arc length.
// The tracer walks from tetrahedron to tetrahedron along each straight segment, crossing polyMesh
// faces into the neighbour cell, and a line ends where its segment leaves through a boundary face.
import { buildCellLocator, type CellLocator } from './pointLocate.js'
import { buildIsoMesh, reconstructPointValues, type IsoMesh } from './polyiso.js'
import type { PolyMesh } from './polymesh.js'

/** A point is in a tetrahedron when each barycentric coordinate is at least -STREAM_BARY_TOL. */
export const STREAM_BARY_TOL = 1e-12
/** A segment walk that crosses more tetrahedra than this gives up and ends the line as 'lost'. */
export const STREAM_MAX_WALK = 100_000

export type StreamDirection = 'forward' | 'backward' | 'both'

export interface PolyStreamOptions {
  direction: StreamDirection
  /** Maximum line length (arc length) per direction. */
  maxLength: number
  maxSteps: number
  /** A fixed step (a distance); null: stepFactor times the current cell's length cbrt(V). */
  step: number | null
  stepFactor: number
  /** A line stops where the speed falls to this fraction of the largest cell speed. */
  minSpeedFraction: number
}

export const DEFAULT_POLY_STREAM_OPTIONS: PolyStreamOptions = {
  direction: 'both',
  maxLength: Number.POSITIVE_INFINITY,
  maxSteps: 4000,
  step: null,
  stepFactor: 0.5,
  minSpeedFraction: 1e-6,
}

/** Why a line (one direction of one seed) ended. */
export type StreamEnd = 'boundary' | 'length' | 'steps' | 'stall' | 'lost'

/** The topology a streamline tracer needs, built once per mesh. */
export interface StreamMesh {
  iso: IsoMesh
  locator: CellLocator
  /** cbrt of each cell's face-fan volume (148.4): the default step's length scale. */
  lengths: Float64Array
}

export interface PolyStreamlines {
  /** xyz of every point of every line, concatenated (one line per seed: upstream -> seed -> downstream). */
  points: Float64Array
  /** |U| at every point. */
  speeds: Float64Array
  /** Line i spans points [offsets[i], offsets[i+1]); length = seed count + 1. */
  offsets: Uint32Array
  /** Seeds outside the domain (no tetrahedron holds them): refused, their lines are empty. */
  refusedSeeds: number[]
  /** How each traced direction ended. */
  ends: Record<StreamEnd, number>
  fallbackPoints: number
  invertedTets: number
}

/** A tetrahedron (C, F, a, b) of cell `cell`: slot `slot` of iso.cellFaces, edge `edge` of that face. */
interface Tet {
  cell: number
  slot: number
  edge: number
}

/** Builds the tracer's topology: the iso topology of §148.2, the cell locator's buckets and the cell lengths. */
export function buildStreamMesh(mesh: PolyMesh, iso: IsoMesh = buildIsoMesh(mesh), locator: CellLocator = buildCellLocator(mesh)): StreamMesh {
  const { points, faceOffsets, faceIndices, owner } = mesh
  const { faceCentres, centroids, cellFaceOffsets, cellFaces } = iso
  const lengths = new Float64Array(mesh.nCells)
  for (let c = 0; c < mesh.nCells; c++) {
    let v6 = 0
    const cx = centroids[3 * c]
    const cy = centroids[3 * c + 1]
    const cz = centroids[3 * c + 2]
    for (let k = cellFaceOffsets[c]; k < cellFaceOffsets[c + 1]; k++) {
      const f = cellFaces[k]
      const sign = owner[f] === c ? 1 : -1
      const s = faceOffsets[f]
      const n = faceOffsets[f + 1] - s
      const fx = faceCentres[3 * f] - cx
      const fy = faceCentres[3 * f + 1] - cy
      const fz = faceCentres[3 * f + 2] - cz
      for (let j = 0; j < n; j++) {
        const a = 3 * faceIndices[s + j]
        const b = 3 * faceIndices[s + ((j + 1) % n)]
        const ax = points[a] - cx
        const ay = points[a + 1] - cy
        const az = points[a + 2] - cz
        const bx = points[b] - cx
        const by = points[b + 1] - cy
        const bz = points[b + 2] - cz
        v6 += sign * (fx * (ay * bz - az * by) + fy * (az * bx - ax * bz) + fz * (ax * by - ay * bx))
      }
    }
    lengths[c] = Math.cbrt(Math.abs(v6) / 6)
  }
  return { iso, locator, lengths }
}

/**
 * The tracer over one velocity field: the corner values of 148.6 for each component and the
 * tetrahedron geometry. All arithmetic is f64.
 */
class Tracer {
  private readonly mesh: PolyMesh
  private readonly iso: IsoMesh
  /** Point velocities, 3 per point. */
  private readonly pu: Float64Array
  /** Face-centre velocities (the mean of the face's point velocities), 3 per face. */
  private readonly fu: Float64Array
  /** Corners (C, F, a, b) of the current tetrahedron, 12 numbers, and scratch barycentrics. */
  private readonly X = new Float64Array(12)
  private readonly lp = new Float64Array(4)
  private readonly lx = new Float64Array(4)
  readonly fallbackPoints: number

  constructor(
    private readonly sm: StreamMesh,
    private readonly cu: ArrayLike<number>,
    readonly minSpeed: number,
  ) {
    this.iso = sm.iso
    this.mesh = sm.iso.mesh
    const { nPoints, nCells, nFaces, faceOffsets, faceIndices } = this.mesh
    this.pu = new Float64Array(3 * nPoints)
    let fallback = 0
    const comp = new Float64Array(nCells)
    for (let a = 0; a < 3; a++) {
      for (let c = 0; c < nCells; c++) comp[c] = cu[3 * c + a]
      const pv = reconstructPointValues(this.iso, comp)
      fallback = Math.max(fallback, pv.fallbackPoints)
      for (let p = 0; p < nPoints; p++) this.pu[3 * p + a] = pv.values[p]
    }
    this.fallbackPoints = fallback
    this.fu = new Float64Array(3 * nFaces)
    for (let f = 0; f < nFaces; f++) {
      const s = faceOffsets[f]
      const n = faceOffsets[f + 1] - s
      for (let a = 0; a < 3; a++) {
        let sum = 0
        for (let k = s; k < s + n; k++) sum += this.pu[3 * faceIndices[k] + a]
        this.fu[3 * f + a] = sum / n
      }
    }
  }

  /** The cell's length scale cbrt(V), for the default step. */
  lengthOf(c: number): number {
    return this.sm.lengths[c]
  }

  private face(t: Tet): number {
    return this.iso.cellFaces[t.slot]
  }

  /** Point offsets (into mesh.points / 3) of the tet's edge corners a and b. */
  private edgePoints(t: Tet): [number, number] {
    const { faceOffsets, faceIndices } = this.mesh
    const f = this.face(t)
    const s = faceOffsets[f]
    const n = faceOffsets[f + 1] - s
    return [faceIndices[s + t.edge], faceIndices[s + ((t.edge + 1) % n)]]
  }

  /** Loads the tet's corners into X. */
  private load(t: Tet): void {
    const { X } = this
    const { points } = this.mesh
    const { centroids, faceCentres } = this.iso
    const f = this.face(t)
    const [a, b] = this.edgePoints(t)
    for (let i = 0; i < 3; i++) {
      X[i] = centroids[3 * t.cell + i]
      X[3 + i] = faceCentres[3 * f + i]
      X[6 + i] = points[3 * a + i]
      X[9 + i] = points[3 * b + i]
    }
  }

  /**
   * Barycentric coordinates of x in the loaded tet (148.8): lambda_i is the signed volume of the tet
   * with corner i replaced by x, over the tet's own. False when the tet has no volume.
   */
  private bary(x: ArrayLike<number>, out: Float64Array): boolean {
    const X = this.X
    const det = (i: number, j: number, k: number, o: ArrayLike<number>, oo: number): number => {
      // det[P_i - o, P_j - o, P_k - o], with o = X[oo..] when o is X, else the point o.
      const ox = o[oo]
      const oy = o[oo + 1]
      const oz = o[oo + 2]
      const ax = X[3 * i] - ox
      const ay = X[3 * i + 1] - oy
      const az = X[3 * i + 2] - oz
      const bx = X[3 * j] - ox
      const by = X[3 * j + 1] - oy
      const bz = X[3 * j + 2] - oz
      const cx = X[3 * k] - ox
      const cy = X[3 * k + 1] - oy
      const cz = X[3 * k + 2] - oz
      return ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx)
    }
    const d = det(1, 2, 3, X, 0)
    if (!(d !== 0) || !Number.isFinite(d)) return false
    // Corner i replaced by x: an even permutation of (x, P_j, P_k, P_l) about the corner order.
    out[0] = det(1, 2, 3, x, 0) / d
    out[1] = -det(0, 2, 3, x, 0) / d
    out[2] = det(0, 1, 3, x, 0) / d
    out[3] = -det(0, 1, 2, x, 0) / d
    return true
  }

  /** The velocity at x in tet t (x already in t), written to u; returns the speed, or -1 when the tet has no volume. */
  velocity(t: Tet, x: ArrayLike<number>, u: Float64Array): number {
    this.load(t)
    if (!this.bary(x, this.lx)) return -1
    const l = this.lx
    const f = this.face(t)
    const [a, b] = this.edgePoints(t)
    for (let i = 0; i < 3; i++) u[i] = l[0] * this.cu[3 * t.cell + i] + l[1] * this.fu[3 * f + i] + l[2] * this.pu[3 * a + i] + l[3] * this.pu[3 * b + i]
    return Math.hypot(u[0], u[1], u[2])
  }

  /** True when x lies in tet t to STREAM_BARY_TOL. */
  contains(t: Tet, x: ArrayLike<number>): boolean {
    this.load(t)
    if (!this.bary(x, this.lx)) return false
    const l = this.lx
    return l[0] >= -STREAM_BARY_TOL && l[1] >= -STREAM_BARY_TOL && l[2] >= -STREAM_BARY_TOL && l[3] >= -STREAM_BARY_TOL
  }

  /** The first tet of cell c that holds x, or null. */
  tetInCell(c: number, x: ArrayLike<number>): Tet | null {
    const { cellFaceOffsets, cellFaces } = this.iso
    const { faceOffsets } = this.mesh
    for (let k = cellFaceOffsets[c]; k < cellFaceOffsets[c + 1]; k++) {
      const f = cellFaces[k]
      const n = faceOffsets[f + 1] - faceOffsets[f]
      for (let j = 0; j < n; j++) {
        const t = { cell: c, slot: k, edge: j }
        if (this.contains(t, x)) return t
      }
    }
    return null
  }

  /** The tet holding x: candidate cells from the locator's bucket and box, then the exact tet test. */
  locate(x: readonly [number, number, number]): Tet | null {
    const loc = this.sm.locator
    if (!x.every(Number.isFinite) || loc.nCells === 0) return null
    const b: [number, number, number] = [0, 0, 0]
    for (let a = 0; a < 3; a++) {
      const tol = 1e-9 * (loc.hi[a] - loc.lo[a] + Math.abs(loc.lo[a]) + Math.abs(loc.hi[a]))
      if (x[a] < loc.lo[a] - tol || x[a] > loc.hi[a] + tol) return null
      if (loc.dims[a] === 1) continue
      const v = Math.floor((x[a] - loc.min[a]) / loc.widths[a])
      b[a] = v < 0 ? 0 : v >= loc.dims[a] ? loc.dims[a] - 1 : v
    }
    const bucket = b[0] + loc.dims[0] * (b[1] + loc.dims[1] * b[2])
    for (let k = loc.bucketStarts[bucket]; k < loc.bucketStarts[bucket + 1]; k++) {
      const c = loc.bucketCells[k]
      const o = 6 * c
      const bb = loc.bbox
      const tol = 1e-9 * Math.hypot(bb[o + 3] - bb[o], bb[o + 4] - bb[o + 1], bb[o + 5] - bb[o + 2])
      if (x[0] < bb[o] - tol || x[1] < bb[o + 1] - tol || x[2] < bb[o + 2] - tol) continue
      if (x[0] > bb[o + 3] + tol || x[1] > bb[o + 4] + tol || x[2] > bb[o + 5] + tol) continue
      const t = this.tetInCell(c, x)
      if (t) return t
    }
    return null
  }

  /** The tet across face `i` of tet t (the face opposite corner i: 0 C, 1 F, 2 a, 3 b); null at a boundary face. */
  private across(t: Tet, i: number): Tet | null {
    const { faceOffsets, faceIndices, owner, neighbour, nInternalFaces } = this.mesh
    const { cellFaceOffsets, cellFaces } = this.iso
    const f = this.face(t)
    const n = faceOffsets[f + 1] - faceOffsets[f]
    if (i === 0) {
      // (F, a, b) is part of polyMesh face f: the neighbour cell's tet on the same face edge.
      if (f >= nInternalFaces) return null
      const other = owner[f] === t.cell ? neighbour[f] : owner[f]
      for (let k = cellFaceOffsets[other]; k < cellFaceOffsets[other + 1]; k++) if (cellFaces[k] === f) return { cell: other, slot: k, edge: t.edge }
      return null
    }
    if (i === 2) return { cell: t.cell, slot: t.slot, edge: (t.edge + 1) % n } // (C, F, b): the next edge starts at b
    if (i === 3) return { cell: t.cell, slot: t.slot, edge: (t.edge + n - 1) % n } // (C, F, a): the previous edge ends at a
    // i === 1, (C, a, b): the cell's other face with edge {a, b}.
    const [a, b] = this.edgePoints(t)
    for (let k = cellFaceOffsets[t.cell]; k < cellFaceOffsets[t.cell + 1]; k++) {
      if (k === t.slot) continue
      const g = cellFaces[k]
      const s = faceOffsets[g]
      const m = faceOffsets[g + 1] - s
      for (let j = 0; j < m; j++) {
        const p = faceIndices[s + j]
        const q = faceIndices[s + ((j + 1) % m)]
        if ((p === a && q === b) || (p === b && q === a)) return { cell: t.cell, slot: k, edge: j }
      }
    }
    return null
  }

  /**
   * Walks the segment p -> x from tet t, which holds p (148.10). Returns the tet holding x, or the
   * point where the segment leaves the domain through a boundary face, or 'lost'.
   */
  walk(t: Tet, p: ArrayLike<number>, x: ArrayLike<number>): { inside: true; tet: Tet } | { inside: false; tet: Tet; exit: Float64Array } | 'lost' {
    const { lp, lx } = this
    let cur = t
    let entry = -1
    let start: ArrayLike<number> = p
    for (let step = 0; step < STREAM_MAX_WALK; step++) {
      this.load(cur)
      if (!this.bary(x, lx) || !this.bary(start, lp)) return 'lost'
      let exitFace = -1
      let tExit = Infinity
      for (let i = 0; i < 4; i++) {
        if (i === entry || !(lx[i] < -STREAM_BARY_TOL)) continue
        const a = Math.max(lp[i], 0)
        const ti = a / (a - lx[i])
        if (ti < tExit) {
          tExit = ti
          exitFace = i
        }
      }
      if (exitFace < 0) return { inside: true, tet: cur }
      const q = new Float64Array(3)
      for (let i = 0; i < 3; i++) q[i] = start[i] + tExit * (x[i] - start[i])
      const next = this.across(cur, exitFace)
      if (!next) {
        if (exitFace === 0) return { inside: false, tet: cur, exit: q }
        return 'lost'
      }
      // The face we entered through is the one of `next` that leads back to `cur`.
      entry = -1
      for (let i = 0; i < 4; i++) {
        const back = this.across(next, i)
        if (back && back.cell === cur.cell && back.slot === cur.slot && back.edge === cur.edge) {
          entry = i
          break
        }
      }
      cur = next
      start = q
    }
    return 'lost'
  }
}

/** An empty count of line ends. */
function noEnds(): Record<StreamEnd, number> {
  return { boundary: 0, length: 0, steps: 0, stall: 0, lost: 0 }
}

/**
 * Streamlines of the cell velocity `cellU` (3 per cell) from `seeds` (3 per seed), SPEC-LIT §148.3.
 * A seed that no tetrahedron holds is refused: its line is empty and its index is in refusedSeeds.
 */
export function polyStreamlines(sm: StreamMesh, cellU: ArrayLike<number>, seeds: ArrayLike<number>, options: Partial<PolyStreamOptions> = {}): PolyStreamlines {
  const opts: PolyStreamOptions = { ...DEFAULT_POLY_STREAM_OPTIONS, ...options }
  const nCells = sm.iso.mesh.nCells
  if (cellU.length !== 3 * nCells) throw new Error(`polyStreamlines: ${cellU.length} velocity numbers for ${nCells} cells`)
  let umax = 0
  for (let c = 0; c < nCells; c++) umax = Math.max(umax, Math.hypot(cellU[3 * c], cellU[3 * c + 1], cellU[3 * c + 2]))
  const tracer = new Tracer(sm, cellU, opts.minSpeedFraction * umax)
  const ends = noEnds()
  const points: number[] = []
  const speeds: number[] = []
  const offsets: number[] = [0]
  const refusedSeeds: number[] = []
  for (let s = 0; 3 * s + 2 < seeds.length; s++) {
    const seed: [number, number, number] = [seeds[3 * s], seeds[3 * s + 1], seeds[3 * s + 2]]
    const tet = tracer.locate(seed)
    if (!tet) {
      refusedSeeds.push(s)
      offsets.push(offsets[offsets.length - 1])
      continue
    }
    const back = opts.direction === 'forward' ? [] : trace(tracer, tet, seed, -1, opts, ends)
    const fwd = opts.direction === 'backward' ? [] : trace(tracer, tet, seed, 1, opts, ends)
    // Each trace starts at the seed: the line is reversed(back) without its seed, then fwd.
    const line: number[] = []
    for (let i = back.length - 4; i >= 4; i -= 4) line.push(back[i], back[i + 1], back[i + 2], back[i + 3])
    if (fwd.length > 0) for (let i = 0; i < fwd.length; i++) line.push(fwd[i])
    else if (back.length >= 4) line.push(back[0], back[1], back[2], back[3])
    if (line.length < 8) {
      offsets.push(offsets[offsets.length - 1])
      continue
    }
    for (let i = 0; i < line.length; i += 4) {
      points.push(line[i], line[i + 1], line[i + 2])
      speeds.push(line[i + 3])
    }
    offsets.push(points.length / 3)
  }
  return {
    points: Float64Array.from(points),
    speeds: Float64Array.from(speeds),
    offsets: Uint32Array.from(offsets),
    refusedSeeds,
    ends,
    fallbackPoints: tracer.fallbackPoints,
    invertedTets: sm.iso.invertedTets,
  }
}

/** One direction of one seed: [x, y, z, speed] per point, the seed first (148.9). */
function trace(tracer: Tracer, seedTet: Tet, seed: readonly [number, number, number], sign: 1 | -1, opts: PolyStreamOptions, ends: Record<StreamEnd, number>): number[] {
  const out: number[] = []
  const u = new Float64Array(3)
  const k1 = new Float64Array(3)
  const k2 = new Float64Array(3)
  const k3 = new Float64Array(3)
  const k4 = new Float64Array(3)
  const y = new Float64Array(3)
  const p = Float64Array.from(seed)
  let tet = seedTet
  let length = 0
  // The unit direction at x in tet t, into k; the speed, or -1 when stalled or degenerate.
  const dir = (t: Tet, x: ArrayLike<number>, k: Float64Array): number => {
    const speed = tracer.velocity(t, x, u)
    if (!(speed > tracer.minSpeed) || speed === 0) return -1
    for (let i = 0; i < 3; i++) k[i] = (sign * u[i]) / speed
    return speed
  }
  // The direction at the stage point p + c k, walking to it from the step's tet; 'out' when it is outside.
  const stage = (c: number, k: Float64Array, into: Float64Array): number | 'out' | 'lost' => {
    for (let i = 0; i < 3; i++) y[i] = p[i] + c * k[i]
    const w = tracer.walk(tet, p, y)
    if (w === 'lost') return 'lost'
    if (!w.inside) return 'out'
    return dir(w.tet, y, into)
  }
  const speed0 = dir(tet, p, k1)
  out.push(p[0], p[1], p[2], speed0 < 0 ? Math.max(0, tracer.velocity(tet, p, u)) : speed0)
  if (speed0 < 0) {
    ends.stall++
    return out
  }
  for (let step = 0; step < opts.maxSteps; step++) {
    const speed = step === 0 ? speed0 : dir(tet, p, k1)
    if (speed < 0) {
      ends[tracer.velocity(tet, p, u) < 0 ? 'lost' : 'stall']++
      return out
    }
    let h = opts.step ?? opts.stepFactor * tracer.lengthOf(tet.cell)
    let last = false
    if (length + h >= opts.maxLength) {
      h = opts.maxLength - length
      last = true
    }
    if (!(h > 0)) {
      ends.length++
      return out
    }
    const s2 = stage(0.5 * h, k1, k2)
    const s3 = typeof s2 === 'number' && s2 >= 0 ? stage(0.5 * h, k2, k3) : s2
    const s4 = typeof s3 === 'number' && s3 >= 0 ? stage(h, k3, k4) : s3
    let target: Float64Array
    if (s4 === 'lost') {
      ends.lost++
      return out
    }
    if (s4 === 'out') {
      // A stage point is outside: the forward-Euler step, which ends at the boundary if it leaves.
      target = new Float64Array(3)
      for (let i = 0; i < 3; i++) target[i] = p[i] + h * k1[i]
    } else if (s4 < 0) {
      ends.stall++
      return out
    } else {
      target = new Float64Array(3)
      for (let i = 0; i < 3; i++) target[i] = p[i] + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
    }
    const w = tracer.walk(tet, p, target)
    if (w === 'lost') {
      ends.lost++
      return out
    }
    const q = w.inside ? target : w.exit
    const sq = tracer.velocity(w.tet, q, u)
    out.push(q[0], q[1], q[2], sq < 0 ? 0 : sq)
    length += Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2])
    p.set(q)
    tet = w.tet
    if (!w.inside) {
      ends.boundary++
      return out
    }
    if (last) {
      ends.length++
      return out
    }
  }
  ends.steps++
  return out
}
