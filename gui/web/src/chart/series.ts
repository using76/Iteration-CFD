// Turns residual records into uPlot aligned data: a fixed series superset in
// display order plus any dynamic field, log-safe nulls and stride
// downsampling. Pure, so the chart component stays thin.
import { RESIDUAL_SERIES_COLORS, RESIDUAL_SERIES_ORDER, type MetricRecord, type ResidualRecord } from '@cfd/shared'
import type { ChartKind } from '../state/uiStore'

export type XAxis = 'iter' | 'time'

export interface ShapeOptions {
  log: boolean
  xAxis: XAxis
  /** Keep at most this many points (stride decimation, last point always kept). */
  maxPoints?: number
}

export interface ShapedSeries {
  keys: string[]
  x: number[]
  ys: Array<Array<number | null>>
  /** Number of records represented (before decimation). */
  count: number
}

const FIXED = new Set<string>(RESIDUAL_SERIES_ORDER)
const FALLBACK_COLORS = ['#2f7ce8', '#f2a33a', '#22a35c', '#8a4fd6', '#e0483f', '#d6336c', '#0ea5b7', '#b5651d', '#8a94a3']

/** Fixed order first (only those present), then new keys in first-seen order. */
export function seriesKeys(records: ResidualRecord[]): string[] {
  const present = new Set<string>()
  const dynamic: string[] = []
  for (const r of records) {
    for (const k of Object.keys(r.fields)) {
      if (present.has(k)) continue
      present.add(k)
      if (!FIXED.has(k)) dynamic.push(k)
    }
  }
  return [...RESIDUAL_SERIES_ORDER.filter((k) => present.has(k)), ...dynamic]
}

export function seriesColor(key: string, index: number): string {
  return RESIDUAL_SERIES_COLORS[key] ?? FALLBACK_COLORS[index % FALLBACK_COLORS.length]
}

export function hasTimeAxis(records: ResidualRecord[]): boolean {
  return records.some((r) => r.time !== null)
}

/** The y scale a chart starts on: metrics and surface integrals go negative so
 *  linear, residuals are log. The chart component reads this on mount; the tab
 *  host keys the chart by kind so a chart switch remounts and re-runs it. */
export function initialYScale(chart: ChartKind): 'log' | 'linear' {
  return chart === 'residuals' ? 'log' : 'linear'
}

/** Metric names match case- and punctuation-blind: "Tmax" finds "T[max]". */
export function normaliseMetricName(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]/g, '')
}

/** Every metric key the run reported, unique, in first-seen order. */
export function metricNames(recs: MetricRecord[]): string[] {
  const seen = new Set<string>()
  const out: string[] = []
  for (const r of recs) {
    for (const k of Object.keys(r.metrics)) {
      if (seen.has(k)) continue
      seen.add(k)
      out.push(k)
    }
  }
  return out
}

/** The exact (case-sensitive) name when present, else the first name whose
 *  normalised form equals the query's; null when the query is empty or nothing matches. */
export function findMetric(names: string[], query: string): string | null {
  const q = normaliseMetricName(query)
  if (!q) return null
  if (names.includes(query)) return query
  return names.find((n) => normaliseMetricName(n) === q) ?? null
}

/** A per-patch surface-integral series is a metric named `<patch>.<quantity>`
 *  (the fan line produces fan_a.Q / fan_a.dp): split at the LAST dot, the patch
 *  part non-empty and the quantity a plain identifier. */
export function isSurfaceKey(key: string): boolean {
  const i = key.lastIndexOf('.')
  if (i <= 0) return false
  return /^[A-Za-z_][A-Za-z0-9_]*$/.test(key.slice(i + 1))
}

/** The `<patch>.<quantity>` series a run reports, in metricNames order. */
export function surfaceSeriesKeys(recs: MetricRecord[]): string[] {
  return metricNames(recs).filter(isSurfaceKey)
}

/** The sweeps view of the residuals chart: every field line's linear-solver
 *  iteration counts become its plotted series; lines without them plot nothing. */
export function sweepsAsResidualRecords(recs: ResidualRecord[]): ResidualRecord[] {
  return recs.map((r) => ({ ...r, fields: r.solverIters ?? {} }))
}

/** Applies the show_metric slot picks: both slots empty keeps every key;
 *  otherwise slot 1's name then slot 2's, and `right` names the series that
 *  moves to the second (right) axis. */
export function slotKeys(keys: string[], slots: [string | null, string | null]): { keys: string[]; right: string | null } {
  const [l, r] = slots
  if (l == null && r == null) return { keys, right: null }
  const out: string[] = []
  if (l != null && keys.includes(l)) out.push(l)
  if (r != null && r !== l && keys.includes(r)) out.push(r)
  return { keys: out, right: r != null && r !== l && out.includes(r) ? r : null }
}

/** Indices to keep so that at most `maxPoints` survive; the last index is always kept. */
export function decimateIndices(length: number, maxPoints: number): number[] {
  if (maxPoints <= 0 || length <= maxPoints) return Array.from({ length }, (_, i) => i)
  const stride = Math.ceil(length / maxPoints)
  const out: number[] = []
  for (let i = 0; i < length; i += stride) out.push(i)
  if (out[out.length - 1] !== length - 1) out.push(length - 1)
  return out
}

export function logSafe(v: number | undefined, log: boolean): number | null {
  if (v === undefined || !Number.isFinite(v)) return null
  if (log && v <= 0) return null
  return v
}

export function shapeResiduals(records: ResidualRecord[], keys: string[], opts: ShapeOptions): ShapedSeries {
  const idx = decimateIndices(records.length, opts.maxPoints ?? Number.POSITIVE_INFINITY)
  const x: number[] = new Array(idx.length)
  const ys = keys.map(() => new Array<number | null>(idx.length))
  idx.forEach((ri, i) => {
    const r = records[ri]
    x[i] = opts.xAxis === 'time' && r.time !== null ? r.time : r.iter
    keys.forEach((k, ki) => {
      ys[ki][i] = logSafe(r.fields[k], opts.log)
    })
  })
  return { keys, x, ys, count: records.length }
}

/** The metrics chart reads the same uPlot path: a metric line becomes a
 *  residual record (fields = metrics, wall/solverIters empty); a metric line
 *  with no iteration is placed at its sequence number. */
export function metricsAsResidualRecords(recs: MetricRecord[]): ResidualRecord[] {
  return recs.map((rec) => ({ seq: rec.seq, iter: rec.iter ?? rec.seq, time: rec.time, wall: null, fields: rec.metrics, solverIters: null, raw: rec.raw }))
}

/** The residual field to headline on cards: the largest of U, p, continuity, k... in the latest record. */
export function leadingField(fields: Record<string, number> | null): { field: string; value: number } | null {
  if (!fields) return null
  for (const k of ['U', 'p', 'p_rgh', 'continuity', 'k', 'epsilon', 'omega', 'nuTilda', 'T']) {
    const v = fields[k]
    if (typeof v === 'number' && Number.isFinite(v)) return { field: k, value: v }
  }
  const first = Object.entries(fields).find(([, v]) => Number.isFinite(v))
  return first ? { field: first[0], value: first[1] } : null
}

export function formatResidual(v: number): string {
  if (!Number.isFinite(v)) return String(v)
  if (v === 0) return '0'
  const a = Math.abs(v)
  if (a >= 1e-3 && a < 1e4) return v.toPrecision(3).replace(/\.?0+$/, '')
  return v.toExponential(1)
}

/** The chart key a compare run's field is drawn under: `<runId>:<field>` (run ids are `r_<n>`, hub-issued). */
export function compareKey(runId: string, key: string): string {
  return `${runId}:${key}`
}

export function isCompareKey(key: string, compareRunId: string | null): boolean {
  return compareRunId !== null && key.startsWith(compareRunId + ':')
}

/** Primary keys in seriesKeys order, then the compare run's keys prefixed; [] compare when compareRunId is null. */
export function pairKeys(primary: ResidualRecord[], compare: ResidualRecord[], compareRunId: string | null): string[] {
  const keys = seriesKeys(primary)
  if (compareRunId === null) return keys
  for (const k of seriesKeys(compare)) keys.push(compareKey(compareRunId, k))
  return keys
}

/** Two runs on one x axis: the sorted union of both x sets, null where a run has no record at that x.
 *  With compareRunId null this is exactly shapeResiduals(primary, keys, opts). */
export function shapeResidualsPair(
  primary: ResidualRecord[],
  compare: ResidualRecord[],
  keys: string[],
  compareRunId: string | null,
  opts: ShapeOptions,
): ShapedSeries {
  if (compareRunId === null) return shapeResiduals(primary, keys, opts)
  const a = shapeResiduals(primary, keys, opts)
  // The compare run's records store the bare field names; shape it under the
  // unprefixed base keys so each prefixed row reads its own run's field.
  const b = shapeResiduals(
    compare,
    keys.map((k) => (isCompareKey(k, compareRunId) ? k.slice(compareRunId.length + 1) : k)),
    opts,
  )
  const x = [...new Set([...a.x, ...b.x])].sort((p, q) => p - q)
  const ys = keys.map((k, i) => {
    const own = isCompareKey(k, compareRunId) ? b : a
    const m = new Map<number, number | null>()
    for (let j = 0; j < own.x.length; j++) m.set(own.x[j], own.ys[i][j])
    return x.map((t) => m.get(t) ?? null)
  })
  return { keys, x, ys, count: primary.length + compare.length }
}
