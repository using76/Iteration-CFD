// The block a cut-cell mesh was carved out of. Without this the viewer refuses
// every slice, streamline and glyph on exactly the meshes that most need them.
import { describe, expect, it } from 'vitest'
import { emptyBounds, extendBounds } from '../../formats/geometry.js'
import { blockLatticeFromCellCenters } from '../manifest.js'

/**
 * A uniform n^3 block over [0,1]^3 with a solid region removed and the cells
 * touching it given displaced centroids — which is what a cut cell has.
 */
function carvedBlock(n: number, solid: (i: number, j: number, k: number) => boolean, cut: (i: number, j: number, k: number) => boolean) {
  const h = 1 / n
  const centres: number[] = []
  const sites: number[] = []
  for (let k = 0; k < n; k++) {
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        if (solid(i, j, k)) continue
        // A cut cell's centroid is its own, pulled off the lattice site.
        const d = cut(i, j, k) ? 0.31 * h : 0
        centres.push((i + 0.5) * h + d, (j + 0.5) * h - d * 0.6, (k + 0.5) * h + d * 0.4)
        sites.push(i + n * (j + n * k))
      }
    }
  }
  const bounds = emptyBounds()
  extendBounds(bounds, 0, 0, 0)
  extendBounds(bounds, 1, 1, 1)
  return { centers: Float32Array.from(centres), bounds, sites }
}

describe('blockLatticeFromCellCenters', () => {
  it('recovers the block, the holes and which cell sits at each site', () => {
    const n = 24
    const solid = (i: number, j: number, k: number) => i >= 10 && i < 14 && j >= 10 && j < 14 && k >= 10 && k < 14
    const cut = (i: number, j: number, k: number) => i >= 9 && i < 15 && j >= 9 && j < 15 && k >= 9 && k < 15
    const { centers, bounds, sites } = carvedBlock(n, solid, cut)

    const r = blockLatticeFromCellCenters(centers, bounds)
    expect(r).not.toBeNull()
    expect(r!.grid.dims).toEqual([n, n, n])
    expect(r!.grid.uniform).toBe(true)
    expect(r!.holes).toBe(4 * 4 * 4)

    // Every cell landed on the site it actually came from, cut cells included.
    for (let c = 0; c < sites.length; c++) expect(r!.index[sites[c]]).toBe(c)
    // And the solid region is holes, not cells.
    expect(r!.index[12 + n * (12 + n * 12)]).toBe(-1)
  })

  it('leaves an unbroken block to the exact detector', () => {
    const { centers, bounds } = carvedBlock(12, () => false, () => false)
    expect(blockLatticeFromCellCenters(centers, bounds)).toBeNull()
  })

  it('refuses a mesh that is mostly missing', () => {
    // A quarter of the block: whatever this is, it is not that block.
    const { centers, bounds } = carvedBlock(16, (i) => i >= 4, () => false)
    expect(blockLatticeFromCellCenters(centers, bounds)).toBeNull()
  })
})
