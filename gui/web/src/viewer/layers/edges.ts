// Line geometries derived from the surface triangles: cell edges (triangle
// edges minus the diagonals inside one cell face) restricted to visible
// patches.
import type { PatchInfo } from '@cfd/shared'
import { BufferGeometry, Float32BufferAttribute } from 'three/webgpu'

/**
 * Edges of the triangles in the given patches, excluding edges shared by two
 * triangles of the same owner cell (the diagonals of a split quad / fan).
 *
 * An edge is the unordered vertex pair. It is dropped when a later occurrence
 * has the same cell as its first occurrence; otherwise it is emitted once, in
 * order of first occurrence. The pairs are grouped by two stable counting sorts
 * of the half-edges (by high index, then by low index), so the grouping is
 * exact for every Uint32 index: a float key lo * 2^32 + hi stops being exact
 * at lo = 2^21, and a Map stops growing at 2^24 entries.
 */
export function cellEdgeGeometry(positions: Float32Array, indices: Uint32Array, cellOfTri: Uint32Array, patches: PatchInfo[]): BufferGeometry {
  let nHalf = 0
  for (const p of patches) nHalf += 3 * p.triCount
  const lo = new Uint32Array(nHalf)
  const hi = new Uint32Array(nHalf)
  const cell = new Uint32Array(nHalf)
  let h = 0
  let range = 0
  for (const p of patches) {
    for (let t = p.triStart; t < p.triStart + p.triCount; t++) {
      for (let e = 0; e < 3; e++) {
        const a = indices[t * 3 + e]
        const b = indices[t * 3 + ((e + 1) % 3)]
        lo[h] = a < b ? a : b
        hi[h] = a < b ? b : a
        cell[h] = cellOfTri[t]
        if (hi[h] >= range) range = hi[h] + 1
        h++
      }
    }
  }
  // Stable counting sort by hi, then by lo: equal pairs end up adjacent, in half-edge order.
  const byHi = countingSort(hi, range, null)
  const order = countingSort(lo, range, byHi)
  const emit = new Uint8Array(nHalf)
  let nEdges = 0
  for (let s = 0; s < nHalf; ) {
    const first = order[s]
    let end = s + 1
    let conflict = false
    while (end < nHalf && lo[order[end]] === lo[first] && hi[order[end]] === hi[first]) {
      if (cell[order[end]] === cell[first]) conflict = true
      end++
    }
    if (!conflict) {
      emit[first] = 1
      nEdges++
    }
    s = end
  }
  const out = new Float32Array(nEdges * 6)
  let o = 0
  h = 0
  for (const p of patches) {
    for (let t = p.triStart; t < p.triStart + p.triCount; t++) {
      for (let e = 0; e < 3; e++, h++) {
        if (!emit[h]) continue
        const a = indices[t * 3 + e]
        const b = indices[t * 3 + ((e + 1) % 3)]
        out[o++] = positions[a * 3]
        out[o++] = positions[a * 3 + 1]
        out[o++] = positions[a * 3 + 2]
        out[o++] = positions[b * 3]
        out[o++] = positions[b * 3 + 1]
        out[o++] = positions[b * 3 + 2]
      }
    }
  }
  const g = new BufferGeometry()
  g.setAttribute('position', new Float32BufferAttribute(out, 3))
  return g
}

/** The permutation that stably sorts `items` (default 0..n-1) by key[item], keys in [0, range). */
function countingSort(key: Uint32Array, range: number, items: Uint32Array | null): Uint32Array {
  const n = key.length
  const start = new Uint32Array(range + 1)
  for (let i = 0; i < n; i++) start[key[i] + 1]++
  for (let k = 0; k < range; k++) start[k + 1] += start[k]
  const out = new Uint32Array(n)
  for (let i = 0; i < n; i++) {
    const item = items ? items[i] : i
    out[start[key[item]]++] = item
  }
  return out
}
