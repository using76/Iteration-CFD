// An in-memory channel-with-jet dataset (default 60x30x20 cells) in exactly
// the manifest + blob shape the server produces. Used by `?demoDataset=1`,
// the browser probe and the headless tests.
import type { BlobRef, FieldInfo, PatchInfo, ViewerDataset } from '@cfd/shared'
import type { TypedArray } from './BlobCache'
import { StructuredGrid, gradedNodes, uniformNodes } from './StructuredGrid'
import type { SyntheticDataset } from './transport'

export const SYNTHETIC_CHANNEL_PATH = 'demo://channel'
export const SYNTHETIC_CHANNEL_2D_PATH = 'demo://channel2d'

export interface SyntheticOptions {
  id?: string
  path?: string
  name?: string
  dims?: [number, number, number]
  bounds?: { min: [number, number, number]; max: [number, number, number] }
  /** Number of time steps (the jet strengthens with each). */
  times?: number
}

function f32Ref(key: string, count: number, components: number): BlobRef {
  return { key, dtype: 'f32', count, components, bytes: count * components * 4 }
}

function u32Ref(key: string, count: number, components: number): BlobRef {
  return { key, dtype: 'u32', count, components, bytes: count * components * 4 }
}

/** Two-sided grading: cells shrink toward both ends (expansion ratio `ratio` from the wall to the centre). */
function twoSidedNodes(a: number, b: number, n: number, ratio: number): Float32Array {
  const half = Math.floor(n / 2)
  if (half < 1 || n % 2 !== 0) return uniformNodes(a, b, n)
  const mid = 0.5 * (a + b)
  const lower = gradedNodes(a, mid, half, ratio)
  const out = new Float32Array(n + 1)
  for (let i = 0; i <= half; i++) out[i] = lower[i]
  for (let i = 1; i <= half; i++) out[half + i] = mid + (mid - lower[half - i])
  out[n] = b
  return out
}

interface PatchBuild {
  info: PatchInfo
  positions: number[]
  normals: number[]
  indices: number[]
  cellOfTri: number[]
}

/** Boundary quads of one box face as two triangles each, vertices shared within the patch only. */
function buildFace(grid: StructuredGrid, axis: 0 | 1 | 2, side: 0 | 1, name: string, type: string, color: [number, number, number], vertexBase: number, triBase: number): PatchBuild {
  const ua = ((axis + 1) % 3) as 0 | 1 | 2
  const va = ((axis + 2) % 3) as 0 | 1 | 2
  const nu = grid.dims[ua]
  const nv = grid.dims[va]
  const fixedNode = side === 0 ? grid.min[axis] : grid.max[axis]
  const fixedCell = side === 0 ? 0 : grid.dims[axis] - 1
  const positions: number[] = []
  const normals: number[] = []
  const normal = [0, 0, 0]
  normal[axis] = side === 0 ? -1 : 1
  for (let j = 0; j <= nv; j++) {
    for (let i = 0; i <= nu; i++) {
      const p = [0, 0, 0]
      p[axis] = fixedNode
      p[ua] = grid.nodes[ua][i]
      p[va] = grid.nodes[va][j]
      positions.push(p[0], p[1], p[2])
      normals.push(normal[0], normal[1], normal[2])
    }
  }
  const indices: number[] = []
  const cellOfTri: number[] = []
  const ijk = [0, 0, 0]
  for (let j = 0; j < nv; j++) {
    for (let i = 0; i < nu; i++) {
      const a = vertexBase + i + (nu + 1) * j
      const b = a + 1
      const c = a + (nu + 1)
      const d = c + 1
      // Wind so the triangle normal matches the outward face normal.
      if (side === 1) indices.push(a, b, d, a, d, c)
      else indices.push(a, d, b, a, c, d)
      ijk[axis] = fixedCell
      ijk[ua] = i
      ijk[va] = j
      const cell = grid.cellIndex(ijk[0], ijk[1], ijk[2])
      cellOfTri.push(cell, cell)
    }
  }
  const info: PatchInfo = { name, type, triStart: triBase, triCount: indices.length / 3, color }
  return { info, positions, normals, indices, cellOfTri }
}

export function buildSyntheticChannel(opts: SyntheticOptions = {}): SyntheticDataset {
  const dims = opts.dims ?? [60, 30, 20]
  const bounds = opts.bounds ?? { min: [0, 0, 0], max: [3, 1.5, 1] }
  const twoD = dims[2] === 1
  const id = opts.id ?? (twoD ? 'synthetic-channel-2d' : 'synthetic-channel')
  const path = opts.path ?? (twoD ? SYNTHETIC_CHANNEL_2D_PATH : SYNTHETIC_CHANNEL_PATH)
  const timeCount = opts.times ?? 3
  const x = uniformNodes(bounds.min[0], bounds.max[0], dims[0])
  const y = twoSidedNodes(bounds.min[1], bounds.max[1], dims[1], 2.5)
  const z = uniformNodes(bounds.min[2], bounds.max[2], dims[2])
  const grid = new StructuredGrid({ dims, x, y, z })
  const blobs = new Map<string, TypedArray>()
  blobs.set('grid.x', x)
  blobs.set('grid.y', y)
  blobs.set('grid.z', z)

  const faces: PatchBuild[] = []
  let vertexBase = 0
  let triBase = 0
  const spec: [0 | 1 | 2, 0 | 1, string, string, [number, number, number]][] = [
    [0, 0, 'inlet', 'patch', [0.35, 0.65, 0.95]],
    [0, 1, 'outlet', 'patch', [0.95, 0.55, 0.3]],
    [1, 0, 'bottomWall', 'wall', [0.6, 0.6, 0.62]],
    [1, 1, 'topWall', 'wall', [0.6, 0.6, 0.62]],
    [2, 0, 'front', twoD ? 'empty' : 'wall', [0.7, 0.7, 0.72]],
    [2, 1, 'back', twoD ? 'empty' : 'wall', [0.7, 0.7, 0.72]],
  ]
  for (const [axis, side, name, type, color] of spec) {
    const f = buildFace(grid, axis, side, name, type, color, vertexBase, triBase)
    faces.push(f)
    vertexBase += f.positions.length / 3
    triBase += f.info.triCount
  }
  const positions = Float32Array.from(faces.flatMap((f) => f.positions))
  const normals = Float32Array.from(faces.flatMap((f) => f.normals))
  const indices = Uint32Array.from(faces.flatMap((f) => f.indices))
  const cellOfTri = Uint32Array.from(faces.flatMap((f) => f.cellOfTri))
  blobs.set('surface.positions', positions)
  blobs.set('surface.normals', normals)
  blobs.set('surface.indices', indices)
  blobs.set('surface.cellOfTri', cellOfTri)
  const vertexCount = positions.length / 3
  const triangleCount = indices.length / 3

  const times = Array.from({ length: timeCount }, (_, i) => {
    const iter = timeCount === 1 ? 4000 : Math.round((4000 * (i + 1)) / timeCount)
    return { index: i, value: iter, label: String(iter) }
  })
  const fieldRanges: Record<string, { min: number; max: number }> = {}
  const fieldDefs: { name: string; components: 1 | 3; unit: string }[] = [
    { name: 'U', components: 3, unit: 'm/s' },
    { name: 'p', components: 1, unit: 'm^2/s^2' },
    { name: 'k', components: 1, unit: 'm^2/s^2' },
  ]
  const fields: FieldInfo[] = fieldDefs.map((f) => ({ name: f.name, components: f.components, location: 'cell', range: null, unit: f.unit, perTime: [] }))
  for (const t of times) {
    const strength = timeCount === 1 ? 1 : 0.5 + (0.5 * (t.index + 1)) / timeCount
    const { U, p, k } = analyticFields(grid, bounds, strength)
    const perField: Record<string, Float32Array> = { U, p, k }
    for (const f of fields) {
      const data = perField[f.name]
      const key = `field.${f.name}.${t.index}`
      blobs.set(key, data)
      const range = magnitudeRange(data, f.components)
      f.perTime.push({ timeIndex: t.index, blob: f32Ref(key, grid.cellCount, f.components), range, sourceDir: t.label })
      const acc = fieldRanges[f.name] ?? { min: Number.POSITIVE_INFINITY, max: Number.NEGATIVE_INFINITY }
      fieldRanges[f.name] = { min: Math.min(acc.min, range.min), max: Math.max(acc.max, range.max) }
    }
  }
  for (const f of fields) f.range = fieldRanges[f.name]

  const manifest: ViewerDataset = {
    id,
    name: opts.name ?? (twoD ? 'channel-2d (demo)' : 'channel (demo)'),
    path,
    source: 'cartesian',
    geometryFidelity: 'exact',
    units: { length: 'm' },
    bounds: { min: [...bounds.min], max: [...bounds.max] },
    up: 'y',
    cellCount: grid.cellCount,
    grid: {
      dims: [...dims],
      nodes: { x: f32Ref('grid.x', dims[0] + 1, 1), y: f32Ref('grid.y', dims[1] + 1, 1), z: f32Ref('grid.z', dims[2] + 1, 1) },
      uniform: false,
      emptyAxis: twoD ? 'z' : null,
    },
    surface: {
      patches: faces.map((f) => f.info),
      positions: f32Ref('surface.positions', vertexCount, 3),
      normals: f32Ref('surface.normals', vertexCount, 3),
      indices: u32Ref('surface.indices', triangleCount, 3),
      cellOfTri: u32Ref('surface.cellOfTri', triangleCount, 1),
      triangleCount,
      vertexCount,
    },
    fields,
    times,
    meta: { caseName: 'channel', model: 'kEpsilon', turbulence: 'RAS', cellCount: grid.cellCount, rootKind: 'demo', seriesKind: 'synthetic' },
    warnings: [],
  }
  return { manifest, blobs }
}

function magnitudeRange(data: Float32Array, components: number): { min: number; max: number } {
  let min = Number.POSITIVE_INFINITY
  let max = Number.NEGATIVE_INFINITY
  const n = data.length / components
  for (let i = 0; i < n; i++) {
    const v = components === 1 ? data[i] : Math.hypot(data[i * 3], data[i * 3 + 1], data[i * 3 + 2])
    if (v < min) min = v
    if (v > max) max = v
  }
  return { min, max }
}

/** Power-law channel profile + a decaying Gaussian jet + a gentle swirl, linear pressure drop, k from the jet. */
function analyticFields(grid: StructuredGrid, bounds: { min: [number, number, number]; max: [number, number, number] }, strength: number) {
  const n = grid.cellCount
  const U = new Float32Array(n * 3)
  const p = new Float32Array(n)
  const k = new Float32Array(n)
  const L = bounds.max[0] - bounds.min[0]
  const H = bounds.max[1] - bounds.min[1]
  const W = bounds.max[2] - bounds.min[2]
  const yc = bounds.min[1] + 0.5 * H
  const zc = bounds.min[2] + 0.5 * W
  const c = [0, 0, 0]
  for (let cell = 0; cell < n; cell++) {
    grid.centerOf(cell, c)
    const xr = (c[0] - bounds.min[0]) / L
    const eta = Math.min(1, Math.abs((c[1] - yc) / (0.5 * H)))
    const zeta = W > 0 ? Math.min(1, Math.abs((c[2] - zc) / (0.5 * W))) : 0
    const bulk = 0.9 * Math.pow(Math.max(0, 1 - eta), 1 / 7) * Math.pow(Math.max(0, 1 - zeta * zeta), 0.15)
    const r2 = ((c[1] - yc) / (0.12 * H)) ** 2 + (W > 0 ? ((c[2] - zc) / (0.12 * W)) ** 2 : 0)
    const jet = 1.1 * strength * Math.exp(-r2) * Math.exp(-1.5 * xr)
    const swirl = 0.08 * strength * Math.sin(2 * Math.PI * xr)
    U[cell * 3] = bulk + jet
    U[cell * 3 + 1] = swirl * (c[2] - zc)
    U[cell * 3 + 2] = -swirl * (c[1] - yc)
    p[cell] = 0.6 * (1 - xr) + 0.15 * jet
    k[cell] = 0.01 + 0.12 * jet * jet + 0.02 * eta * eta
  }
  return { U, p, k }
}
