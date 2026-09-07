// 256-entry colour tables for the six colormaps in the shared command schema.
// Pure data: the same table feeds the GPU LUT texture, the worker's slice
// sampler and the SVG/canvas legend.
import type { ColormapName } from '@cfd/shared'

export const LUT_SIZE = 256

export type ColorTable = Uint8Array // LUT_SIZE * 4 RGBA, sRGB

type Stop = [number, number, number, number] // position, r, g, b in 0..1

const VIRIDIS: Stop[] = [
  [0.0, 0.267004, 0.004874, 0.329415],
  [0.1, 0.282327, 0.140926, 0.457517],
  [0.2, 0.253935, 0.265254, 0.529983],
  [0.3, 0.206756, 0.371758, 0.553117],
  [0.4, 0.163625, 0.471133, 0.558148],
  [0.5, 0.127568, 0.566949, 0.550556],
  [0.6, 0.134692, 0.658636, 0.517649],
  [0.7, 0.266941, 0.748751, 0.440573],
  [0.8, 0.477504, 0.821444, 0.318195],
  [0.9, 0.741388, 0.873449, 0.149561],
  [1.0, 0.993248, 0.906157, 0.143936],
]

const INFERNO: Stop[] = [
  [0.0, 0.001462, 0.000466, 0.013866],
  [0.1, 0.087411, 0.044556, 0.224813],
  [0.2, 0.258234, 0.038571, 0.406485],
  [0.3, 0.416331, 0.090203, 0.432943],
  [0.4, 0.578304, 0.148039, 0.40443],
  [0.5, 0.735683, 0.215906, 0.330245],
  [0.6, 0.865006, 0.316822, 0.226055],
  [0.7, 0.954506, 0.468744, 0.099874],
  [0.8, 0.987622, 0.64532, 0.039886],
  [0.9, 0.964394, 0.843848, 0.273391],
  [1.0, 0.988362, 0.998364, 0.644924],
]

const COOLWARM: Stop[] = [
  [0.0, 0.23, 0.299, 0.754],
  [0.25, 0.552, 0.69, 0.996],
  [0.5, 0.865, 0.865, 0.865],
  [0.75, 0.958, 0.603, 0.48],
  [1.0, 0.706, 0.016, 0.15],
]

const JET: Stop[] = [
  [0.0, 0.0, 0.0, 0.5],
  [0.11, 0.0, 0.0, 1.0],
  [0.34, 0.0, 1.0, 1.0],
  [0.65, 1.0, 1.0, 0.0],
  [0.89, 1.0, 0.0, 0.0],
  [1.0, 0.5, 0.0, 0.0],
]

const GREYSCALE: Stop[] = [
  [0.0, 0.0, 0.0, 0.0],
  [1.0, 1.0, 1.0, 1.0],
]

function interpolateStops(stops: Stop[], t: number): [number, number, number] {
  if (t <= stops[0][0]) return [stops[0][1], stops[0][2], stops[0][3]]
  for (let i = 1; i < stops.length; i++) {
    const s1 = stops[i]
    if (t <= s1[0]) {
      const s0 = stops[i - 1]
      const f = (t - s0[0]) / (s1[0] - s0[0])
      return [s0[1] + (s1[1] - s0[1]) * f, s0[2] + (s1[2] - s0[2]) * f, s0[3] + (s1[3] - s0[3]) * f]
    }
  }
  const last = stops[stops.length - 1]
  return [last[1], last[2], last[3]]
}

/** Google's Turbo colormap (Mikhailov 2019), degree-5 polynomial fit. */
function turbo(t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, t))
  const x2 = x * x
  const x3 = x2 * x
  const x4 = x3 * x
  const x5 = x4 * x
  const r = 0.13572138 + 4.6153926 * x - 42.66032258 * x2 + 132.13108234 * x3 - 152.94239396 * x4 + 59.28637943 * x5
  const g = 0.09140261 + 2.19418839 * x + 4.84296658 * x2 - 14.18503333 * x3 + 4.27729857 * x4 + 2.82956604 * x5
  const b = 0.1066733 + 12.64194608 * x - 60.58204836 * x2 + 110.36276771 * x3 - 89.90310912 * x4 + 27.34824973 * x5
  return [clamp01(r), clamp01(g), clamp01(b)]
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v
}

export function colormapRgb(name: ColormapName, t: number): [number, number, number] {
  switch (name) {
    case 'viridis':
      return interpolateStops(VIRIDIS, t)
    case 'inferno':
      return interpolateStops(INFERNO, t)
    case 'coolwarm':
      return interpolateStops(COOLWARM, t)
    case 'jet':
      return interpolateStops(JET, t)
    case 'greyscale':
      return interpolateStops(GREYSCALE, t)
    case 'turbo':
      return turbo(t)
  }
}

const tableCache = new Map<ColormapName, ColorTable>()

/** RGBA8 table with LUT_SIZE entries (alpha 255). Cached per name. */
export function colormapTable(name: ColormapName): ColorTable {
  const cached = tableCache.get(name)
  if (cached) return cached
  const table = new Uint8Array(LUT_SIZE * 4)
  for (let i = 0; i < LUT_SIZE; i++) {
    const [r, g, b] = colormapRgb(name, i / (LUT_SIZE - 1))
    table[i * 4] = Math.round(r * 255)
    table[i * 4 + 1] = Math.round(g * 255)
    table[i * 4 + 2] = Math.round(b * 255)
    table[i * 4 + 3] = 255
  }
  tableCache.set(name, table)
  return table
}

export function isDiverging(name: ColormapName): boolean {
  return name === 'coolwarm'
}

/**
 * The range actually mapped onto the table. Diverging maps are centred on
 * zero whenever the data range straddles it, so the neutral colour means 0.
 */
export function effectiveRange(name: ColormapName, range: [number, number]): [number, number] {
  const [min, max] = range
  if (isDiverging(name) && min < 0 && max > 0) {
    const m = Math.max(-min, max)
    return [-m, m]
  }
  return [min, max]
}

/** CSS colour string of the table entry for t in 0..1. */
export function colormapCss(name: ColormapName, t: number): string {
  const table = colormapTable(name)
  const i = Math.round(clamp01(t) * (LUT_SIZE - 1)) * 4
  return `rgb(${table[i]},${table[i + 1]},${table[i + 2]})`
}
