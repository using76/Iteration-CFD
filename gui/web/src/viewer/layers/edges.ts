// Line geometries derived from the surface triangles: cell edges (triangle
// edges minus the diagonals inside one cell face) restricted to visible
// patches.
import type { PatchInfo } from '@cfd/shared'
import { BufferGeometry, Float32BufferAttribute } from 'three/webgpu'

function edgeKey(a: number, b: number): number {
  return a < b ? a * 4294967296 + b : b * 4294967296 + a
}

/**
 * Edges of the triangles in the given patches, excluding edges shared by two
 * triangles of the same owner cell (the diagonals of a split quad / fan).
 */
export function cellEdgeGeometry(positions: Float32Array, indices: Uint32Array, cellOfTri: Uint32Array, patches: PatchInfo[]): BufferGeometry {
  const cells = new Map<number, number>()
  const conflict = new Set<number>()
  for (const p of patches) {
    for (let t = p.triStart; t < p.triStart + p.triCount; t++) {
      const cell = cellOfTri[t]
      for (let e = 0; e < 3; e++) {
        const a = indices[t * 3 + e]
        const b = indices[t * 3 + ((e + 1) % 3)]
        const key = edgeKey(a, b)
        const prev = cells.get(key)
        if (prev === undefined) cells.set(key, cell)
        else if (prev === cell) conflict.add(key)
      }
    }
  }
  const out: number[] = []
  const seen = new Set<number>()
  for (const p of patches) {
    for (let t = p.triStart; t < p.triStart + p.triCount; t++) {
      for (let e = 0; e < 3; e++) {
        const a = indices[t * 3 + e]
        const b = indices[t * 3 + ((e + 1) % 3)]
        const key = edgeKey(a, b)
        if (conflict.has(key) || seen.has(key)) continue
        seen.add(key)
        out.push(positions[a * 3], positions[a * 3 + 1], positions[a * 3 + 2], positions[b * 3], positions[b * 3 + 1], positions[b * 3 + 2])
      }
    }
  }
  const g = new BufferGeometry()
  g.setAttribute('position', new Float32BufferAttribute(out, 3))
  return g
}
