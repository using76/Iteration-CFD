// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// An in-memory sphere-in-a-box demo dataset (the 8x8x8 box minus the 56 cells the body occupies):
// the 3-D case that opens on its `sphere` wall instead of the six far-field box patches. The same
// geometry the server fixture fixtures/makeSphereInBox writes as a polyMesh.
import type { BlobRef, FieldInfo, PatchInfo, ViewerDataset } from '@cfd/shared'
import type { TypedArray } from './BlobCache'
import type { SyntheticDataset } from './transport'

export const SYNTHETIC_SPHERE_IN_BOX_PATH = 'demo://sphere-in-box'

const H = 0.25
const N = 8
const R2 = 0.6 * 0.6

function f32Ref(key: string, count: number, components: number): BlobRef {
  return { key, dtype: 'f32', count, components, bytes: count * components * 4 }
}

function u32Ref(key: string, count: number, components: number): BlobRef {
  return { key, dtype: 'u32', count, components, bytes: count * components * 4 }
}

function node(n: number): number {
  return -1 + H * n
}

type Vec = [number, number, number]

interface CellFace {
  corners: Vec[]
  normal: Vec
}

/** The 4 corner coordinates and outward normal of one hex-cell face, wound for the right-hand rule. */
function cellFace(cell: Vec, axis: 0 | 1 | 2, side: 0 | 1): CellFace {
  const u = ((axis + 1) % 3) as 0 | 1 | 2
  const v = ((axis + 2) % 3) as 0 | 1 | 2
  const normal: Vec = [0, 0, 0]
  normal[axis] = side === 0 ? -1 : 1
  const f = node(cell[axis] + side)
  const at = (uc: number, vc: number): Vec => {
    const c: Vec = [0, 0, 0]
    c[axis] = f
    c[u] = uc
    c[v] = vc
    return c
  }
  const u0 = node(cell[u])
  const u1 = node(cell[u] + 1)
  const v0 = node(cell[v])
  const v1 = node(cell[v] + 1)
  const corners = side === 1 ? [at(u0, v0), at(u1, v0), at(u1, v1), at(u0, v1)] : [at(u0, v0), at(u0, v1), at(u1, v1), at(u1, v0)]
  return { corners, normal }
}

interface FaceOut extends CellFace {
  owner: number
}

export function buildSyntheticSphereInBox(): SyntheticDataset {
  // Cells in increasing i + 8j + 64k order; the sites whose centre lies inside the sphere are holes.
  const removed: boolean[] = new Array(N * N * N).fill(false)
  const cellOfSite: number[] = new Array(N * N * N).fill(-1)
  let cellCount = 0
  for (let k = 0; k < N; k++) {
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < N; i++) {
        const site = i + N * j + N * N * k
        const x = node(i) + H / 2
        const y = node(j) + H / 2
        const z = node(k) + H / 2
        if (x * x + y * y + z * z < R2) removed[site] = true
        else cellOfSite[site] = cellCount++
      }
    }
  }
  const positions: number[] = []
  const normals: number[] = []
  const indices: number[] = []
  const cellOfTri: number[] = []
  const patches: PatchInfo[] = []
  let triStart = 0
  const emit = (name: string, type: string, color: [number, number, number], faces: FaceOut[]): void => {
    for (const face of faces) {
      const base = positions.length / 3
      for (const c of face.corners) positions.push(c[0], c[1], c[2])
      for (let i = 0; i < 4; i++) normals.push(face.normal[0], face.normal[1], face.normal[2])
      indices.push(base, base + 1, base + 2, base, base + 2, base + 3)
      cellOfTri.push(face.owner, face.owner)
    }
    patches.push({ name, type, triStart, triCount: faces.length * 2, color })
    triStart += faces.length * 2
  }
  const site = (i: number, j: number, k: number): number => i + N * j + N * N * k
  // The six box faces: every boundary cell keeps its outer face (a boundary cell is never removed).
  const faceSpec: [0 | 1 | 2, 0 | 1, string][] = [
    [0, 0, 'xmin'],
    [0, 1, 'xmax'],
    [1, 0, 'ymin'],
    [1, 1, 'ymax'],
    [2, 0, 'zmin'],
    [2, 1, 'zmax'],
  ]
  for (const [axis, side, name] of faceSpec) {
    const faces: FaceOut[] = []
    for (let k = 0; k < N; k++) {
      for (let j = 0; j < N; j++) {
        for (let i = 0; i < N; i++) {
          if ([i, j, k][axis] !== (side === 0 ? 0 : N - 1)) continue
          faces.push({ ...cellFace([i, j, k], axis, side), owner: cellOfSite[site(i, j, k)] })
        }
      }
    }
    emit(name, 'patch', [0.7, 0.7, 0.72], faces)
  }
  // The body wall: every face between a kept cell and a removed cell, normal out of the fluid.
  const dirs: [0 | 1 | 2, 0 | 1][] = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
  const sphereFaces: FaceOut[] = []
  for (let k = 0; k < N; k++) {
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < N; i++) {
        if (removed[site(i, j, k)]) continue
        for (const [axis, side] of dirs) {
          const nb = [i, j, k]
          nb[axis] += side === 0 ? -1 : 1
          if (nb[axis] < 0 || nb[axis] >= N || !removed[site(nb[0], nb[1], nb[2])]) continue
          sphereFaces.push({ ...cellFace([i, j, k], axis, side), owner: cellOfSite[site(i, j, k)] })
        }
      }
    }
  }
  emit('sphere', 'wall', [0.85, 0.45, 0.3], sphereFaces)

  const vertexCount = positions.length / 3
  const triangleCount = indices.length / 3
  const p = new Float32Array(cellCount)
  for (let c = 0; c < cellCount; c++) p[c] = c
  const range = { min: 0, max: cellCount - 1 }
  const blobs = new Map<string, TypedArray>([
    ['surface.positions', Float32Array.from(positions)],
    ['surface.normals', Float32Array.from(normals)],
    ['surface.indices', Uint32Array.from(indices)],
    ['surface.cellOfTri', Uint32Array.from(cellOfTri)],
    ['field.p.0', p],
  ])
  const field: FieldInfo = {
    name: 'p',
    components: 1,
    location: 'cell',
    range,
    unit: null,
    perTime: [{ timeIndex: 0, blob: f32Ref('field.p.0', cellCount, 1), range, sourceDir: '0' }],
  }
  const manifest: ViewerDataset = {
    id: 'synthetic-sphere-in-box',
    name: 'sphere in box (demo)',
    path: SYNTHETIC_SPHERE_IN_BOX_PATH,
    source: 'polymesh',
    geometryFidelity: 'exact',
    units: { length: 'm' },
    bounds: { min: [-1, -1, -1], max: [1, 1, 1] },
    up: 'y',
    cellCount,
    grid: null,
    pointCount: null,
    surface: {
      patches,
      positions: f32Ref('surface.positions', vertexCount, 3),
      normals: f32Ref('surface.normals', vertexCount, 3),
      indices: u32Ref('surface.indices', triangleCount, 3),
      cellOfTri: u32Ref('surface.cellOfTri', triangleCount, 1),
      triangleCount,
      vertexCount,
    },
    fields: [field],
    times: [{ index: 0, value: 0, label: '0' }],
    meta: {},
    warnings: [],
  }
  return { manifest, blobs }
}
