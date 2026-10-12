// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// A wall off the domain box is a body wall and opens visible; the camera fits what is shown.
import { describe, expect, it } from 'vitest'
import { bodyWallPatches, defaultVisiblePatches, hiddenPatches, visibleBounds } from './viewerPatches'
import type { BlobRef, PatchInfo, StructuredGridInfo, ViewerDataset } from './viewerDataset'

type Vec = [number, number, number]
interface Tri {
  name: string
  type: string
  xyz: Vec[]
}

function ref(key: string, count: number, components: number): BlobRef {
  return { key, dtype: 'f32', count, components, bytes: count * components * 4 }
}

/** One triangle per patch on the unit box, decoded arrays alongside the manifest. */
function build(patches: Tri[], grid: StructuredGridInfo | null = null): { manifest: ViewerDataset; positions: Float32Array; indices: Uint32Array } {
  const positions: number[] = []
  const indices: number[] = []
  const infos: PatchInfo[] = []
  let triStart = 0
  for (const p of patches) {
    const base = positions.length / 3
    for (const v of p.xyz) positions.push(v[0], v[1], v[2])
    indices.push(base, base + 1, base + 2)
    infos.push({ name: p.name, type: p.type, triStart, triCount: 1, color: [0.5, 0.5, 0.5] })
    triStart += 1
  }
  const vertexCount = positions.length / 3
  const triangleCount = indices.length / 3
  const manifest: ViewerDataset = {
    id: 't',
    name: 't',
    path: 'demo://t',
    source: 'polymesh',
    geometryFidelity: 'exact',
    units: { length: 'm' },
    bounds: { min: [0, 0, 0], max: [1, 1, 1] },
    up: 'y',
    cellCount: 1,
    grid,
    surface: {
      patches: infos,
      positions: ref('surface.positions', vertexCount, 3),
      normals: ref('surface.normals', vertexCount, 3),
      indices: { key: 'surface.indices', dtype: 'u32', count: triangleCount, components: 3, bytes: triangleCount * 12 },
      cellOfTri: { key: 'surface.cellOfTri', dtype: 'u32', count: triangleCount, components: 1, bytes: triangleCount * 4 },
      triangleCount,
      vertexCount,
    },
    fields: [],
    times: [],
    meta: {},
    warnings: [],
  }
  return { manifest, positions: Float32Array.from(positions), indices: Uint32Array.from(indices) }
}

const FLOOR: Tri = { name: 'floor', type: 'wall', xyz: [[0, 0, 0], [1, 0, 0], [0, 0, 1]] }
const INNER: Tri = { name: 'inner', type: 'wall', xyz: [[0.5, 0.5, 0.5], [0.6, 0.5, 0.5], [0.5, 0.6, 0.5]] }
const PROBE: Tri = { name: 'probe', type: 'patch', xyz: [[0.4, 0.4, 0.4], [0.45, 0.4, 0.4], [0.4, 0.45, 0.4]] }

function gridZ(emptyAxis: 'z' | null): StructuredGridInfo {
  return { dims: [2, 2, 1], nodes: { x: ref('grid.x', 3, 1), y: ref('grid.y', 3, 1), z: ref('grid.z', 2, 1) }, uniform: true, emptyAxis }
}

describe('viewerPatches', () => {
  it('a wall lying on the box is not a body wall', () => {
    const { manifest, positions, indices } = build([FLOOR, INNER, PROBE])
    expect(bodyWallPatches(manifest, positions, indices)).toEqual(['inner'])
  })

  it('the box tolerance is 1e-6 of the diagonal', () => {
    const near = build([{ name: 'flap', type: 'wall', xyz: [[0.5, 1e-9, 0.5], [0.6, 0, 0.5], [0.5, 0, 0.6]] }])
    expect(bodyWallPatches(near.manifest, near.positions, near.indices)).toEqual([])
    const far = build([{ name: 'flap', type: 'wall', xyz: [[0.5, 1e-3, 0.5], [0.6, 0, 0.5], [0.5, 0, 0.6]] }])
    expect(bodyWallPatches(far.manifest, far.positions, far.indices)).toEqual(['flap'])
  })

  it('the default set', () => {
    const a = build([FLOOR, INNER, PROBE])
    expect(defaultVisiblePatches(a.manifest, a.positions, a.indices)).toEqual(['inner'])
    const onlyFloor = build([FLOOR])
    expect(defaultVisiblePatches(onlyFloor.manifest, onlyFloor.positions, onlyFloor.indices)).toBe('all')
    const twoD = build(
      [
        { name: 'front', type: 'empty', xyz: INNER.xyz },
        { name: 'back', type: 'empty', xyz: INNER.xyz },
        INNER,
      ],
      gridZ('z'),
    )
    expect(defaultVisiblePatches(twoD.manifest, twoD.positions, twoD.indices)).toEqual(['front', 'back'])
    const noEmpty = build([INNER], gridZ('z'))
    expect(defaultVisiblePatches(noEmpty.manifest, noEmpty.positions, noEmpty.indices)).toBe('all')
  })

  it('visible bounds', () => {
    const { manifest, positions, indices } = build([FLOOR, INNER, PROBE])
    const inner = visibleBounds(manifest, positions, indices, ['inner'])
    expect(inner.min[0]).toBeCloseTo(0.5, 6)
    expect(inner.min[1]).toBeCloseTo(0.5, 6)
    expect(inner.min[2]).toBeCloseTo(0.5, 6)
    expect(inner.max[0]).toBeCloseTo(0.6, 6)
    expect(inner.max[1]).toBeCloseTo(0.6, 6)
    expect(inner.max[2]).toBeCloseTo(0.5, 6)
    const all = visibleBounds(manifest, positions, indices, 'all')
    expect(all).toEqual({ min: [0, 0, 0], max: [1, 1, 1] })
    all.min[0] = 5
    expect(manifest.bounds.min[0]).toBe(0)
    expect(visibleBounds(manifest, positions, indices, [])).toEqual({ min: [0, 0, 0], max: [1, 1, 1] })
  })

  it('hidden patches', () => {
    const { manifest } = build([FLOOR, INNER, PROBE])
    expect(hiddenPatches(manifest, 'all')).toEqual([])
    expect(hiddenPatches(manifest, ['inner'])).toEqual(['floor', 'probe'])
  })
})

