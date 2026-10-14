// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The centreline chart (STU-09): one field sampled along the line that
// gui_control's set_centerline resolved on the server (the longest axis
// through the domain centre, or an explicit p0..p1), drawn as value against
// distance from p0. The samples come from GET /api/results/sample, the route
// line_sample serves, with exactly the keys the server sampled, so the chart
// shows the numbers the assistant was given. CenterlineChartView is a plain
// function of its props (no hooks), so a test can render it without a DOM.
import { useEffect, useState } from 'react'
import { useT } from '../app/hooks'
import { api, type LineSampleReply } from '../api/rest'
import type { Translate } from '../app/hooks'
import type { CenterlineLine } from '../state/uiStore'

const W = 640
const H = 320
const PAD = { left: 64, right: 16, top: 12, bottom: 36 }

/** The y range drawn: the samples' finite min..max, widened to a unit-free span when they are all equal. */
export function valueRange(values: readonly number[]): [number, number] | null {
  let lo = Infinity
  let hi = -Infinity
  for (const v of values) {
    if (!Number.isFinite(v)) continue
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  if (lo > hi) return null
  if (lo === hi) {
    const d = Math.abs(lo) > 0 ? Math.abs(lo) * 0.05 : 1
    return [lo - d, hi + d]
  }
  return [lo, hi]
}

/** SVG polyline point lists for x against values in the plot box, one per run of finite values (a NaN breaks the line). */
export function polylineRuns(x: readonly number[], values: readonly number[], range: [number, number]): string[] {
  const x0 = x[0] ?? 0
  const x1 = x[x.length - 1] ?? 1
  const sx = (W - PAD.left - PAD.right) / (x1 - x0 || 1)
  const sy = (H - PAD.top - PAD.bottom) / (range[1] - range[0])
  const runs: string[] = []
  let cur: string[] = []
  for (let i = 0; i < values.length; i++) {
    const v = values[i]!
    if (!Number.isFinite(v)) {
      if (cur.length) runs.push(cur.join(' '))
      cur = []
      continue
    }
    const px = PAD.left + (x[i]! - x0) * sx
    const py = H - PAD.bottom - (v - range[0]) * sy
    cur.push(`${px.toFixed(2)},${py.toFixed(2)}`)
  }
  if (cur.length) runs.push(cur.join(' '))
  return runs
}

const fmt = (v: number): string => (v === 0 ? '0' : Math.abs(v) >= 1e4 || Math.abs(v) < 1e-3 ? v.toExponential(3) : Number(v.toPrecision(4)).toString())
const fmtPoint = (p: readonly number[]): string => `(${p.map(fmt).join(', ')})`

export interface CenterlineChartViewProps {
  line: CenterlineLine
  data: LineSampleReply | null
  error: string | null
  t: Translate
}

export function CenterlineChartView({ line, data, error, t }: CenterlineChartViewProps) {
  const range = data ? valueRange(data.values) : null
  const x0 = data?.x[0] ?? 0
  const x1 = data?.x[data.x.length - 1] ?? 0
  return (
    <div className="cl-chart" data-testid="centerline-chart">
      <div className="cl-head">
        <strong>{`${t('charts.centerline')}: ${line.field}`}</strong>
        <span className="cl-meta">{t('centerline.meta', { field: line.field, component: line.component, time: data?.time ?? line.time ?? '-', cells: data?.cells ?? '-' })}</span>
        <span className="cl-meta">{t('centerline.line', { p0: fmtPoint(line.p0), p1: fmtPoint(line.p1) })}</span>
      </div>
      {error !== null ? (
        <div className="cl-error" role="alert">{error}</div>
      ) : data === null ? (
        <div className="cl-note">{t('centerline.loading')}</div>
      ) : range === null ? (
        <div className="cl-note">{t('centerline.empty')}</div>
      ) : (
        <svg className="cl-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${t('charts.centerline')}: ${line.field}`}>
          <rect className="cl-box" x={PAD.left} y={PAD.top} width={W - PAD.left - PAD.right} height={H - PAD.top - PAD.bottom} />
          {polylineRuns(data.x, data.values, range).map((pts, i) => (
            <polyline key={i} className="cl-line" points={pts} />
          ))}
          <text className="cl-tick" x={PAD.left - 6} y={PAD.top + 10} textAnchor="end">{fmt(range[1])}</text>
          <text className="cl-tick" x={PAD.left - 6} y={H - PAD.bottom} textAnchor="end">{fmt(range[0])}</text>
          <text className="cl-tick" x={PAD.left} y={H - PAD.bottom + 16} textAnchor="start">{fmt(x0)}</text>
          <text className="cl-tick" x={W - PAD.right} y={H - PAD.bottom + 16} textAnchor="end">{fmt(x1)}</text>
          <text className="cl-tick" x={(PAD.left + W - PAD.right) / 2} y={H - 6} textAnchor="middle">{t('centerline.axis')}</text>
        </svg>
      )}
    </div>
  )
}

export function CenterlineChart({ line }: { line: CenterlineLine; active: boolean }) {
  const t = useT()
  const [state, setState] = useState<{ data: LineSampleReply | null; error: string | null }>({ data: null, error: null })
  const key = JSON.stringify(line)
  useEffect(() => {
    let live = true
    setState({ data: null, error: null })
    api
      .resultsSample({ dir: line.path, field: line.field, component: line.component, time: line.time, p0: line.p0, p1: line.p1 })
      .then((data) => {
        if (live) setState({ data, error: null })
      })
      .catch((err: unknown) => {
        if (live) setState({ data: null, error: err instanceof Error ? err.message : String(err) })
      })
    return () => {
      live = false
    }
    // The line's JSON is the dependency: a re-aimed tab fetches again, a re-render does not.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])
  return <CenterlineChartView line={line} data={state.data} error={state.error} t={t} />
}
