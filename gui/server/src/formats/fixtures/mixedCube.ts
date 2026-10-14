// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The mixed unit cube of the polyMesh post-processing tests (moved verbatim out of
// polycut.test.ts by docs/17 STU-13 so pointLocate.test.ts shares it): a polyMesh built
// from cells given as face lists, and the nine cells that fill [0,1]^3.
import type { PolyMesh } from '../polymesh.js'

export type V3 = [number, number, number]

const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
const dot = (a: V3, b: V3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

/** Newell normal of a polygon (area vector times 2). */
export function newell(pts: V3[], face: number[]): V3 {
  const n: V3 = [0, 0, 0]
  for (let k = 0; k < face.length; k++) {
    const p = pts[face[k]]
    const q = pts[face[(k + 1) % face.length]]
    n[0] += (p[1] - q[1]) * (p[2] + q[2])
    n[1] += (p[2] - q[2]) * (p[0] + q[0])
    n[2] += (p[0] - q[0]) * (p[1] + q[1])
  }
  return n
}

/**
 * A polyMesh from cells given as face lists. A face is oriented outward by the
 * cell's vertex mean (convex cells) unless `outward` gives its direction; a face
 * met twice becomes an internal face of the lower cell, the higher as neighbour.
 */
export function buildMesh(pts: V3[], cells: number[][][], outward?: V3[][]): PolyMesh {
  const faces = new Map<string, { verts: number[]; owner: number; neighbour: number }>()
  cells.forEach((cellFaces, c) => {
    const ids = [...new Set(cellFaces.flat())]
    const cc: V3 = [0, 0, 0]
    for (const i of ids) for (let a = 0; a < 3; a++) cc[a] += pts[i][a] / ids.length
    cellFaces.forEach((face, k) => {
      const fc: V3 = [0, 0, 0]
      for (const i of face) for (let a = 0; a < 3; a++) fc[a] += pts[i][a] / face.length
      const dir = outward ? outward[c][k] : sub(fc, cc)
      const verts = dot(newell(pts, face), dir) < 0 ? [...face].reverse() : [...face]
      const key = [...face].sort((x, y) => x - y).join(',')
      const seen = faces.get(key)
      if (!seen) faces.set(key, { verts, owner: c, neighbour: -1 })
      else {
        if (seen.neighbour >= 0) throw new Error(`face ${key} is in three cells`)
        if (dot(newell(pts, seen.verts), newell(pts, verts)) >= 0) throw new Error(`face ${key} has one orientation in both cells`)
        seen.neighbour = c
      }
    })
  })
  const all = [...faces.values()]
  const internal = all.filter((f) => f.neighbour >= 0).sort((x, y) => x.owner - y.owner || x.neighbour - y.neighbour)
  const boundary = all.filter((f) => f.neighbour < 0)
  const ordered = [...internal, ...boundary]
  const faceOffsets = new Uint32Array(ordered.length + 1)
  ordered.forEach((f, i) => (faceOffsets[i + 1] = faceOffsets[i] + f.verts.length))
  return {
    nPoints: pts.length,
    nCells: cells.length,
    nFaces: ordered.length,
    nInternalFaces: internal.length,
    points: Float64Array.from(pts.flat()),
    faceOffsets,
    faceIndices: Uint32Array.from(ordered.flatMap((f) => f.verts)),
    owner: Int32Array.from(ordered.map((f) => f.owner)),
    neighbour: Int32Array.from(internal.map((f) => f.neighbour)),
    boundary: [{ name: 'walls', type: 'wall', nFaces: boundary.length, startFace: internal.length, extra: {} }],
  }
}

// The mixed unit cube: G (two stacked hexes merged, middle ring at uneven heights),
// H (hex), P (prism along y), Py (pyramid on P's hypotenuse, apex p on x = 1), and
// five tets filling the rest of the prism above P's hypotenuse.
export const MIX_PTS: V3[] = [
  [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
  [0, 0, 0.1], [1, 0, 0.13], [1, 1, 0.08], [0, 1, 0.11],
  [0, 0, 0.2], [1, 0, 0.2], [1, 1, 0.2], [0, 1, 0.2],
  [0, 0, 0.3], [1, 0, 0.3], [1, 1, 0.3], [0, 1, 0.3], // e a a' e'
  [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1], // b b' c c'
  [1, 0.45, 0.6], [1, 0, 0.62], // p q
]
const [A, A1, B, B1, C, C1, PP, Q] = [13, 14, 16, 17, 18, 19, 20, 21]
export const MIX_CELLS: number[][][] = [
  [[0, 1, 2, 3], [8, 9, 10, 11], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [4, 5, 9, 8], [5, 6, 10, 9], [6, 7, 11, 10], [7, 4, 8, 11]],
  [[8, 9, 10, 11], [12, 13, 14, 15], [8, 9, 13, 12], [9, 10, 14, 13], [10, 11, 15, 14], [11, 8, 12, 15]],
  [[12, 13, 14, 15], [12, B, B1, 15], [12, A, B], [15, A1, B1], [A, B, B1, A1]],
  [[A, B, B1, A1], [A, B, PP], [A1, B1, PP], [B, B1, PP], [A, A1, PP]],
  [[A, B, Q], [A, Q, PP], [B, Q, PP], [A, B, PP]],
  [[Q, B, C], [Q, C, PP], [B, C, PP], [B, Q, PP]],
  [[A1, B1, C1], [A1, C1, PP], [B1, C1, PP], [A1, B1, PP]],
  [[B, C, C1], [B, C, PP], [C, C1, PP], [B, C1, PP]],
  [[B, C1, B1], [B, C1, PP], [C1, B1, PP], [B, B1, PP]],
]

/** The mixed unit cube as a polyMesh. */
export function mixedCube(): PolyMesh {
  return buildMesh(MIX_PTS, MIX_CELLS)
}
