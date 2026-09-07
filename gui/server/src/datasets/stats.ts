// Field statistics for the `field_stats` tool and the manifest ranges.
// Accumulation in Float64 even though the blobs are Float32.
import type { FieldStats } from './types.js'

export type StatComponent = 'magnitude' | 'x' | 'y' | 'z' | 'scalar'

const COMPONENT_INDEX: Record<StatComponent, number> = { magnitude: -1, scalar: 0, x: 0, y: 1, z: 2 }

/** Scalar value of tuple i under the chosen component. */
export function scalarAt(data: ArrayLike<number>, components: number, i: number, component: StatComponent): number {
  if (components === 1) return data[i]
  if (component === 'magnitude' || component === 'scalar') {
    const x = data[3 * i]
    const y = data[3 * i + 1]
    const z = data[3 * i + 2]
    return Math.sqrt(x * x + y * y + z * z)
  }
  return data[3 * i + COMPONENT_INDEX[component]]
}

/** Min/max of the scalar (magnitude for vectors), ignoring NaN; null when nothing finite. */
export function scalarRange(data: ArrayLike<number>, components: number): { min: number; max: number } | null {
  const n = Math.floor(data.length / components)
  let min = Infinity
  let max = -Infinity
  for (let i = 0; i < n; i++) {
    const v = scalarAt(data, components, i, 'magnitude')
    if (v !== v) continue
    if (v < min) min = v
    if (v > max) max = v
  }
  return min <= max ? { min, max } : null
}

export interface FieldStatsInput {
  field: string
  time: string
  data: ArrayLike<number>
  components: number
  component: StatComponent
  /** xyz per cell; required when `region` is given. */
  cellCenters?: ArrayLike<number> | null
  region?: { min: [number, number, number]; max: [number, number, number] } | null
}

const BINS = 16

export function computeFieldStats(input: FieldStatsInput): FieldStats {
  const { data, components, region } = input
  const component: StatComponent = components === 1 ? 'scalar' : input.component === 'scalar' ? 'magnitude' : input.component
  const n = Math.floor(data.length / components)
  const centers = input.cellCenters ?? null
  if (region && !centers) throw new Error('field stats: a region needs cell centres')
  const inside = (i: number): boolean => {
    if (!region || !centers) return true
    for (let a = 0; a < 3; a++) {
      const c = centers[3 * i + a]
      if (c < region.min[a] || c > region.max[a]) return false
    }
    return true
  }
  let count = 0
  let min = Infinity
  let max = -Infinity
  let argmin = -1
  let argmax = -1
  let sum = 0
  let sumSq = 0
  for (let i = 0; i < n; i++) {
    if (!inside(i)) continue
    const v = scalarAt(data, components, i, component)
    if (v !== v) continue
    count++
    sum += v
    sumSq += v * v
    if (v < min) {
      min = v
      argmin = i
    }
    if (v > max) {
      max = v
      argmax = i
    }
  }
  const counts = new Array<number>(BINS).fill(0)
  const edges: number[] = []
  if (count > 0) {
    const width = max > min ? (max - min) / BINS : 1
    for (let b = 0; b <= BINS; b++) edges.push(min + b * width)
    for (let i = 0; i < n; i++) {
      if (!inside(i)) continue
      const v = scalarAt(data, components, i, component)
      if (v !== v) continue
      let b = Math.floor((v - min) / width)
      if (b >= BINS) b = BINS - 1
      if (b < 0) b = 0
      counts[b]++
    }
  } else {
    min = NaN
    max = NaN
  }
  return {
    field: input.field,
    component,
    time: input.time,
    count,
    min,
    max,
    mean: count ? sum / count : NaN,
    rms: count ? Math.sqrt(sumSq / count) : NaN,
    histogram: { edges, counts },
    argmin,
    argmax,
  }
}
