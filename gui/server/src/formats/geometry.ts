// Geometry types shared by the format readers and the dataset service.
import type { PatchInfo } from '@cfd/shared'

export interface SurfaceGeometry {
  /** xyz per vertex; vertices are not shared across patches. */
  positions: Float32Array
  normals: Float32Array
  /** Triangle vertex indices (3 per triangle). */
  indices: Uint32Array
  /** Owner cell of each triangle. */
  cellOfTri: Uint32Array
  patches: PatchInfo[]
  bounds: { min: [number, number, number]; max: [number, number, number] }
}

export type Bounds = SurfaceGeometry['bounds']

/** Stable, distinguishable patch colours (0..1 RGB), by patch index. */
export function patchColor(index: number): [number, number, number] {
  const hue = (index * 137.508) % 360
  const s = 0.45
  const l = 0.62
  const c = (1 - Math.abs(2 * l - 1)) * s
  const x = c * (1 - Math.abs(((hue / 60) % 2) - 1))
  const m = l - c / 2
  let r = 0
  let g = 0
  let b = 0
  if (hue < 60) [r, g, b] = [c, x, 0]
  else if (hue < 120) [r, g, b] = [x, c, 0]
  else if (hue < 180) [r, g, b] = [0, c, x]
  else if (hue < 240) [r, g, b] = [0, x, c]
  else if (hue < 300) [r, g, b] = [x, 0, c]
  else [r, g, b] = [c, 0, x]
  return [r + m, g + m, b + m]
}

export function emptyBounds(): Bounds {
  return { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] }
}

export function extendBounds(b: Bounds, x: number, y: number, z: number): void {
  if (x < b.min[0]) b.min[0] = x
  if (y < b.min[1]) b.min[1] = y
  if (z < b.min[2]) b.min[2] = z
  if (x > b.max[0]) b.max[0] = x
  if (y > b.max[1]) b.max[1] = y
  if (z > b.max[2]) b.max[2] = z
}

/** Flat (per-triangle) normals written per vertex, for an unindexed-style surface where vertices are not shared across triangles of different patches. */
export function computeFlatNormals(positions: Float32Array, indices: Uint32Array): Float32Array {
  const normals = new Float32Array(positions.length)
  for (let t = 0; t < indices.length; t += 3) {
    const a = indices[t] * 3
    const b = indices[t + 1] * 3
    const c = indices[t + 2] * 3
    const abx = positions[b] - positions[a]
    const aby = positions[b + 1] - positions[a + 1]
    const abz = positions[b + 2] - positions[a + 2]
    const acx = positions[c] - positions[a]
    const acy = positions[c + 1] - positions[a + 1]
    const acz = positions[c + 2] - positions[a + 2]
    let nx = aby * acz - abz * acy
    let ny = abz * acx - abx * acz
    let nz = abx * acy - aby * acx
    const len = Math.hypot(nx, ny, nz) || 1
    nx /= len
    ny /= len
    nz /= len
    for (const v of [a, b, c]) {
      normals[v] += nx
      normals[v + 1] += ny
      normals[v + 2] += nz
    }
  }
  for (let i = 0; i < normals.length; i += 3) {
    const len = Math.hypot(normals[i], normals[i + 1], normals[i + 2]) || 1
    normals[i] /= len
    normals[i + 1] /= len
    normals[i + 2] /= len
  }
  return normals
}
