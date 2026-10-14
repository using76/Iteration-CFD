// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// STU-11: the cell-edge key is exact for every Uint32 vertex index (today's lo * 2^32 + hi
// stops being exact at lo = 2^21), and a 4 M-vertex strip gives its analytic edge count.
import type { PatchInfo } from '@cfd/shared'
import { describe, expect, it } from 'vitest'
import { cellEdgeGeometry } from './edges'

function patch(triCount: number, triStart = 0): PatchInfo {
  return { name: 'p', type: 'wall', triStart, triCount, color: [0, 0, 0] }
}

function segments(positions: Float32Array, indices: Uint32Array, cells: Uint32Array, patches: PatchInfo[]): Float32Array {
  return cellEdgeGeometry(positions, indices, cells, patches).getAttribute('position').array as Float32Array
}

/** The pre-STU-11 algorithm with an exact string key: the reference for order and bytes. */
function reference(positions: Float32Array, indices: Uint32Array, cellOfTri: Uint32Array, patches: PatchInfo[]): Float32Array {
  const key = (a: number, b: number): string => (a < b ? `${a},${b}` : `${b},${a}`)
  const cells = new Map<string, number>()
  const conflict = new Set<string>()
  for (const p of patches)
    for (let t = p.triStart; t < p.triStart + p.triCount; t++)
      for (let e = 0; e < 3; e++) {
        const k = key(indices[t * 3 + e], indices[t * 3 + ((e + 1) % 3)])
        const prev = cells.get(k)
        if (prev === undefined) cells.set(k, cellOfTri[t])
        else if (prev === cellOfTri[t]) conflict.add(k)
      }
  const out: number[] = []
  const seen = new Set<string>()
  for (const p of patches)
    for (let t = p.triStart; t < p.triStart + p.triCount; t++)
      for (let e = 0; e < 3; e++) {
        const a = indices[t * 3 + e]
        const b = indices[t * 3 + ((e + 1) % 3)]
        const k = key(a, b)
        if (conflict.has(k) || seen.has(k)) continue
        seen.add(k)
        out.push(positions[a * 3], positions[a * 3 + 1], positions[a * 3 + 2], positions[b * 3], positions[b * 3 + 1], positions[b * 3 + 2])
      }
  return new Float32Array(out)
}

describe('cellEdgeGeometry (STU-11)', () => {
  const a = 2 ** 21 + 1
  const b = 2 ** 22
  const c = 2 ** 22 + 1
  const positions = new Float32Array((c + 1) * 3)
  for (const v of [0, 1, a, b, c]) positions.set([v, v % 7, 1], v * 3)

  it('keeps two distinct edges whose low index is above 2^21 apart', () => {
    // Under lo * 2^32 + hi the doubles near 2^53 are 2 apart, so (a, b) and (a, c) share a key.
    expect(a * 4294967296 + b).toBe(a * 4294967296 + c)
    const indices = new Uint32Array([a, b, 0, a, c, 1])
    expect(segments(positions, indices, new Uint32Array([0, 1]), [patch(2)]).length / 6).toBe(6)
    // Same cell: today the collision reads as an in-cell diagonal and drops both.
    expect(segments(positions, indices, new Uint32Array([0, 0]), [patch(2)]).length / 6).toBe(6)
  })

  it('gives the analytic 3M - 2 edges on a 4 M-vertex strip, with no diagonal', () => {
    const M = 2_000_000
    const pos = new Float32Array(2 * M * 3)
    // Column k holds vertex 2k at (k, 0) and 2k + 1 at (k, 1), so indices reach 4 M - 1.
    for (let k = 0; k < M; k++) {
      pos[2 * k * 3] = k
      pos[(2 * k + 1) * 3] = k
      pos[(2 * k + 1) * 3 + 1] = 1
    }
    const idx = new Uint32Array((M - 1) * 6)
    const cells = new Uint32Array((M - 1) * 2)
    for (let k = 0; k < M - 1; k++) {
      idx.set([2 * k, 2 * k + 2, 2 * k + 1, 2 * k + 2, 2 * k + 3, 2 * k + 1], k * 6)
      cells[2 * k] = k
      cells[2 * k + 1] = k
    }
    const seg = segments(pos, idx, cells, [patch((M - 1) * 2)])
    expect(seg.length / 6).toBe(3 * M - 2)
    let bad = 0
    for (let s = 0; s < seg.length; s += 6) {
      const dx = Math.abs(seg[s + 3] - seg[s])
      const dy = Math.abs(seg[s + 4] - seg[s + 1])
      if (dx + dy !== 1 || seg[s + 2] !== 0 || seg[s + 5] !== 0) bad++
    }
    expect(bad).toBe(0)
  }, 120_000)

  it('is byte-equal to the reference on a small mixed mesh with several patches', () => {
    let seed = 12345
    const rnd = (n: number): number => {
      seed = (seed * 1103515245 + 12345) % 2147483648
      return seed % n
    }
    const nv = 40
    const pos = new Float32Array(nv * 3)
    for (let i = 0; i < pos.length; i++) pos[i] = rnd(1000) / 7
    const nt = 300
    const idx = new Uint32Array(nt * 3)
    for (let i = 0; i < idx.length; i++) idx[i] = rnd(nv)
    const cells = new Uint32Array(nt)
    for (let t = 0; t < nt; t++) cells[t] = rnd(25)
    const patches = [patch(120, 0), patch(100, 200), patch(80, 120)]
    const got = segments(pos, idx, cells, patches)
    const want = reference(pos, idx, cells, patches)
    expect(got.length).toBeGreaterThan(0)
    expect(Array.from(got)).toEqual(Array.from(want))
  })
})
