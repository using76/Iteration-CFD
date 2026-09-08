// The block a cut-cell mesh was carved out of. Without this the viewer refuses
// every slice, streamline and glyph on exactly the meshes that most need them.
import { describe, expect, it } from 'vitest'
import { emptyBounds, extendBounds } from '../../formats/geometry.js'
import { blockLatticeFromCellCenters, upAxisFromGroundPatch } from '../manifest.js'

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

  it('is not fooled by the cut centroids outnumbering the sites', () => {
    // What the real race-car mesh does: 128 sites per axis, and thousands of
    // cut cells whose centroids land off-lattice at a handful of recurring
    // fractions. That fills far more distinct bins than there are sites, so any
    // rule that takes the middle of the occupied bins takes the noise and reads
    // 128 sites as 158. The floor has to come off the fullest bin instead.
    const n = 128
    const h = 1 / n
    const centres: number[] = []
    let cut = 0
    for (let k = 0; k < n; k++) {
      for (let j = 0; j < n; j++) {
        for (let i = 0; i < n; i++) {
          const solid = i >= 52 && i < 76 && j >= 52 && j < 76 && k >= 52 && k < 76
          if (solid) continue
          const onSkin = i >= 50 && i < 78 && j >= 50 && j < 78 && k >= 50 && k < 78
          // Cut fractions repeat, so each lands in a bin with company -- which
          // is what defeats a count-of-2 floor.
          const d = onSkin ? h * (0.06 + 0.01 * (cut++ % 40)) : 0
          centres.push((i + 0.5) * h + d, (j + 0.5) * h + d * 0.7, (k + 0.5) * h - d * 0.5)
        }
      }
    }
    const bounds = emptyBounds()
    extendBounds(bounds, 0, 0, 0)
    extendBounds(bounds, 1, 1, 1)
    const r = blockLatticeFromCellCenters(Float32Array.from(centres), bounds)
    expect(r).not.toBeNull()
    expect(r!.grid.dims).toEqual([n, n, n])
    expect(r!.holes).toBe(24 * 24 * 24)
  })
})

describe('upAxisFromGroundPatch', () => {
  /** A patch of `n` triangles whose every vertex normal is `n0`. */
  function patchSurface(name: string, normal: [number, number, number], tris = 4) {
    const normals = new Float32Array(tris * 9)
    const indices = new Uint32Array(tris * 3)
    for (let t = 0; t < tris; t++) {
      for (let k = 0; k < 3; k++) {
        const v = 3 * t + k
        indices[3 * t + k] = v
        normals[3 * v] = normal[0]
        normals[3 * v + 1] = normal[1]
        normals[3 * v + 2] = normal[2]
      }
    }
    return {
      patches: [{ name, type: 'wall', triStart: 0, triCount: tris, color: [0, 0, 0] as [number, number, number] }],
      normals,
      indices,
    }
  }

  it('reads the up axis off a floor that faces y', () => {
    // What ofgpu-generate-mesh writes: a tunnel with bottomWall at y = 0.
    expect(upAxisFromGroundPatch(patchSurface('bottomWall', [0, 1, 0]))).toBe('y')
    expect(upAxisFromGroundPatch(patchSurface('floor', [0, -1, 0]))).toBe('y')
    expect(upAxisFromGroundPatch(patchSurface('ground', [0, 0, 1]))).toBe('z')
  })

  it('says nothing when there is nothing to read', () => {
    expect(upAxisFromGroundPatch(patchSurface('inlet', [1, 0, 0]))).toBeNull()
    // A "bottomWall" that is not flat in any one axis is not a floor.
    expect(upAxisFromGroundPatch(patchSurface('bottomWall', [0, 0.6, 0.8]))).toBeNull()
    expect(upAxisFromGroundPatch({ patches: [], normals: new Float32Array(0), indices: new Uint32Array(0) })).toBeNull()
  })
})
