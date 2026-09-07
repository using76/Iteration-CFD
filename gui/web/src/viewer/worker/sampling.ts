// Field-to-pixel and field-to-vertex sampling used by the worker (and inline
// in tests / headless mode). No DOM, no three.js.
import type { FieldComponent } from '@cfd/shared'
import type { Axis, StructuredGrid } from '../data/StructuredGrid'
import { planeBasis, planeBoxPolygon, type V3 } from './clipPolygon'

export type ScalarTransform = { kind: 'linear' } | { kind: 'log10'; floor: number }

/** Colour mapping of a (possibly transformed) scalar through a 256-entry RGBA8 table. */
export interface MapperSpec {
  lut: Uint8Array
  min: number
  max: number
  transform: ScalarTransform
}

export function applyTransform(v: number, transform: ScalarTransform): number {
  if (transform.kind === 'linear') return v
  return Math.log10(v > transform.floor ? v : transform.floor)
}

/** Scalar view of a field: the magnitude or one component of a vector, or the scalar itself. */
export function scalarOf(field: Float32Array, components: 1 | 3, component: FieldComponent | null): Float32Array {
  if (components === 1) return field
  const n = field.length / 3
  const out = new Float32Array(n)
  if (component === 'x' || component === 'y' || component === 'z') {
    const c = component === 'x' ? 0 : component === 'y' ? 1 : 2
    for (let i = 0; i < n; i++) out[i] = field[i * 3 + c]
    return out
  }
  for (let i = 0; i < n; i++) out[i] = Math.hypot(field[i * 3], field[i * 3 + 1], field[i * 3 + 2])
  return out
}

export function transformScalars(s: Float32Array, transform: ScalarTransform): Float32Array {
  if (transform.kind === 'linear') return s
  const out = new Float32Array(s.length)
  for (let i = 0; i < s.length; i++) out[i] = applyTransform(s[i], transform)
  return out
}

/** [min, max] over finite values; [0, 1] when there are none. */
export function scalarRange(s: ArrayLike<number>): [number, number] {
  let min = Number.POSITIVE_INFINITY
  let max = Number.NEGATIVE_INFINITY
  for (let i = 0; i < s.length; i++) {
    const v = s[i]
    if (!Number.isFinite(v)) continue
    if (v < min) min = v
    if (v > max) max = v
  }
  if (min > max) return [0, 1]
  return [min, max]
}

/** Smallest positive finite value (log-scale floor); 1e-12 when none. */
export function minPositive(s: ArrayLike<number>): number {
  let m = Number.POSITIVE_INFINITY
  for (let i = 0; i < s.length; i++) {
    const v = s[i]
    if (v > 0 && v < m) m = v
  }
  return Number.isFinite(m) ? m : 1e-12
}

/** Per-vertex scalar: mean of the owner-cell values of the triangles touching the vertex. */
export function surfaceScalars(cellOfTri: Uint32Array, indices: Uint32Array, vertexCount: number, cellScalar: Float32Array): Float32Array {
  const sum = new Float64Array(vertexCount)
  const cnt = new Uint32Array(vertexCount)
  const triCount = cellOfTri.length
  for (let t = 0; t < triCount; t++) {
    const v = cellScalar[cellOfTri[t]]
    for (let c = 0; c < 3; c++) {
      const vi = indices[t * 3 + c]
      sum[vi] += v
      cnt[vi]++
    }
  }
  const out = new Float32Array(vertexCount)
  for (let i = 0; i < vertexCount; i++) out[i] = cnt[i] ? sum[i] / cnt[i] : Number.NaN
  return out
}

function writePixel(rgba: Uint8Array, px: number, value: number, mapper: MapperSpec): void {
  if (!Number.isFinite(value)) {
    rgba[px + 3] = 0
    return
  }
  const t = applyTransform(value, mapper.transform)
  let f = mapper.max > mapper.min ? (t - mapper.min) / (mapper.max - mapper.min) : 0.5
  f = f < 0 ? 0 : f > 1 ? 1 : f
  const i = Math.round(f * 255) * 4
  rgba[px] = mapper.lut[i]
  rgba[px + 1] = mapper.lut[i + 1]
  rgba[px + 2] = mapper.lut[i + 2]
  rgba[px + 3] = 255
}

export interface SampledImage {
  width: number
  height: number
  rgba: Uint8Array
  /** World-space quad corners (u0,v0) (u1,v0) (u1,v1) (u0,v1); row 0 of the image is at v0. */
  corners: Float32Array
  /** Outline polygon of the sampled region in world space (xyz per point). */
  outline: Float32Array
}

export function resolutionFor(cells: number, maxRes: number): number {
  return Math.max(16, Math.min(maxRes, Math.ceil(cells * 2)))
}

/** Axis-aligned slice sampled uniformly in world space at ~2 pixels per cell. */
export function sampleSlice(grid: StructuredGrid, scalar: Float32Array, axis: Axis, position: number, interpolate: boolean, mapper: MapperSpec, maxRes = 1024): SampledImage {
  const ua = ((axis + 1) % 3) as Axis
  const va = ((axis + 2) % 3) as Axis
  const dims = grid.dims
  const width = resolutionFor(dims[ua], maxRes)
  const height = resolutionFor(dims[va], maxRes)
  const u0 = grid.min[ua]
  const u1 = grid.max[ua]
  const v0 = grid.min[va]
  const v1 = grid.max[va]
  const pos = Math.min(grid.max[axis], Math.max(grid.min[axis], position))
  const rgba = new Uint8Array(width * height * 4)
  const p = [0, 0, 0]
  const out = [0]
  p[axis] = pos
  for (let row = 0; row < height; row++) {
    p[va] = v0 + ((v1 - v0) * (row + 0.5)) / height
    for (let col = 0; col < width; col++) {
      p[ua] = u0 + ((u1 - u0) * (col + 0.5)) / width
      const ok = interpolate ? grid.sampleTrilinear(scalar, 1, p[0], p[1], p[2], out) : grid.sampleNearest(scalar, 1, p[0], p[1], p[2], out)
      writePixel(rgba, (row * width + col) * 4, ok ? out[0] : Number.NaN, mapper)
    }
  }
  const corner = (u: number, v: number): V3 => {
    const c: V3 = [0, 0, 0]
    c[axis] = pos
    c[ua] = u
    c[va] = v
    return c
  }
  const corners = [corner(u0, v0), corner(u1, v0), corner(u1, v1), corner(u0, v1)]
  return { width, height, rgba, corners: Float32Array.from(corners.flat()), outline: Float32Array.from(corners.flat()) }
}

/** Arbitrary plane: the box-clipped polygon is parameterised in (u,v) and sampled on its bounding rectangle. */
export function samplePlane(grid: StructuredGrid, scalar: Float32Array, origin: V3, normal: V3, interpolate: boolean, mapper: MapperSpec, maxRes = 1024): SampledImage | null {
  const polygon = planeBoxPolygon(origin, normal, grid.min, grid.max)
  if (polygon.length < 3) return null
  const { u, v } = planeBasis(normal)
  let umin = Number.POSITIVE_INFINITY
  let umax = Number.NEGATIVE_INFINITY
  let vmin = Number.POSITIVE_INFINITY
  let vmax = Number.NEGATIVE_INFINITY
  for (const p of polygon) {
    const du = (p[0] - origin[0]) * u[0] + (p[1] - origin[1]) * u[1] + (p[2] - origin[2]) * u[2]
    const dv = (p[0] - origin[0]) * v[0] + (p[1] - origin[1]) * v[1] + (p[2] - origin[2]) * v[2]
    umin = Math.min(umin, du)
    umax = Math.max(umax, du)
    vmin = Math.min(vmin, dv)
    vmax = Math.max(vmax, dv)
  }
  let minSpacing = Number.POSITIVE_INFINITY
  for (const axis of [0, 1, 2] as const) for (let i = 0; i < grid.dims[axis]; i++) minSpacing = Math.min(minSpacing, grid.spacing(axis, i))
  const width = resolutionFor((umax - umin) / minSpacing, maxRes)
  const height = resolutionFor((vmax - vmin) / minSpacing, maxRes)
  const rgba = new Uint8Array(width * height * 4)
  const out = [0]
  const eps = 1e-6 * minSpacing
  for (let row = 0; row < height; row++) {
    const dv = vmin + ((vmax - vmin) * (row + 0.5)) / height
    for (let col = 0; col < width; col++) {
      const du = umin + ((umax - umin) * (col + 0.5)) / width
      const x = clampInto(origin[0] + u[0] * du + v[0] * dv, grid.min[0], grid.max[0], eps)
      const y = clampInto(origin[1] + u[1] * du + v[1] * dv, grid.min[1], grid.max[1], eps)
      const z = clampInto(origin[2] + u[2] * du + v[2] * dv, grid.min[2], grid.max[2], eps)
      const ok = Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z) && (interpolate ? grid.sampleTrilinear(scalar, 1, x, y, z, out) : grid.sampleNearest(scalar, 1, x, y, z, out))
      writePixel(rgba, (row * width + col) * 4, ok ? out[0] : Number.NaN, mapper)
    }
  }
  const corner = (du: number, dv: number): V3 => [origin[0] + u[0] * du + v[0] * dv, origin[1] + u[1] * du + v[1] * dv, origin[2] + u[2] * du + v[2] * dv]
  const corners = [corner(umin, vmin), corner(umax, vmin), corner(umax, vmax), corner(umin, vmax)]
  return { width, height, rgba, corners: Float32Array.from(corners.flat()), outline: Float32Array.from(polygon.flat()) }
}

/** Snap values within eps of the box faces onto them (pixels on the clipped boundary), NaN further outside. */
function clampInto(x: number, lo: number, hi: number, eps: number): number {
  if (x < lo) return lo - x <= eps ? lo : Number.NaN
  if (x > hi) return x - hi <= eps ? hi : Number.NaN
  return x
}

export interface GlyphSamples {
  positions: Float32Array
  vectors: Float32Array
  count: number
  stride: number
}

/** Cell-centre positions and vectors every `stride` cells (per axis), increasing the stride until `cap` is met. */
export function sampleGlyphs(grid: StructuredGrid, U: Float32Array, stride: number, cap: number, slice: { axis: Axis; position: number } | null): GlyphSamples {
  const [nx, ny, nz] = grid.dims
  let s = Math.max(1, Math.floor(stride))
  const countFor = (st: number) => {
    const n = (d: number) => Math.ceil(d / st)
    if (!slice) return n(nx) * n(ny) * n(nz)
    const ua = (slice.axis + 1) % 3
    const va = (slice.axis + 2) % 3
    return n(grid.dims[ua]) * n(grid.dims[va])
  }
  while (countFor(s) > cap) s++
  const total = countFor(s)
  const positions = new Float32Array(total * 3)
  const vectors = new Float32Array(total * 3)
  let n = 0
  const c = [0, 0, 0]
  const fixed = slice ? grid.clampAxis(slice.axis, slice.position) : -1
  const range = (axis: Axis) => (slice && slice.axis === axis ? [fixed, fixed + 1, 1] : [0, grid.dims[axis], s])
  const [i0, i1, is] = range(0)
  const [j0, j1, js] = range(1)
  const [k0, k1, ks] = range(2)
  for (let k = k0; k < k1; k += ks) {
    for (let j = j0; j < j1; j += js) {
      for (let i = i0; i < i1; i += is) {
        const cell = grid.cellIndex(i, j, k)
        grid.centerOf(cell, c)
        if (slice) c[slice.axis] = slice.position
        positions[n * 3] = c[0]
        positions[n * 3 + 1] = c[1]
        positions[n * 3 + 2] = c[2]
        vectors[n * 3] = U[cell * 3]
        vectors[n * 3 + 1] = U[cell * 3 + 1]
        vectors[n * 3 + 2] = U[cell * 3 + 2]
        n++
      }
    }
  }
  return { positions: positions.subarray(0, n * 3), vectors: vectors.subarray(0, n * 3), count: n, stride: s }
}
