// uPlot residual chart: log/linear Y, iteration/time X, fixed series
// superset plus dynamic fields, 100 ms batched setData, auto-follow unless
// the user zoomed, legend like the mockup, CSV export.
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { formatDuration, useDismiss, useNow, useT } from '../app/hooks'
import { api } from '../api/rest'
import { Icon } from '../components/common/Icon'
import { pickActiveRun, selectRunData, sortRuns, useSessionStore } from '../state/sessionStore'
import { useUiStore } from '../state/uiStore'
import { hasTimeAxis, seriesColor, seriesKeys, shapeResiduals, type XAxis } from './series'

const SUP = '⁰¹²³⁴⁵⁶⁷⁸⁹'
const BATCH_MS = 100
const MAX_POINTS = 4000

function sup(n: number): string {
  const s = String(Math.abs(n))
  return (n < 0 ? '⁻' : '') + [...s].map((c) => SUP[Number(c)]).join('')
}

function fmtLogTick(v: number): string {
  if (v <= 0 || !Number.isFinite(v)) return ''
  const e = Math.log10(v)
  return Math.abs(e - Math.round(e)) < 1e-6 ? `10${sup(Math.round(e))}` : ''
}

function fmtX(v: number): string {
  if (Math.abs(v) >= 1000 && Number.isInteger(v)) return `${(v / 1000).toFixed(v % 1000 ? 1 : 0)}K`
  return Number.isInteger(v) ? String(v) : v.toPrecision(3)
}

function cssVar(name: string, fallback: string): string {
  if (typeof getComputedStyle !== 'function') return fallback
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback
}

export interface ResidualsChartProps {
  runId: string | null
  active?: boolean
  compact?: boolean
}

export function ResidualsChart({ runId: runIdProp, active = true, compact = false }: ResidualsChartProps) {
  const t = useT()
  const runs = useSessionStore((s) => s.runs)
  const uiActiveRunId = useUiStore((s) => s.activeRunId)
  const setActiveRun = useUiStore((s) => s.setActiveRun)
  const theme = useUiStore((s) => s.theme)
  const runList = useMemo(() => sortRuns(runs), [runs])
  const run = useMemo(() => (runIdProp ? (runs[runIdProp] ?? null) : pickActiveRun(runList, uiActiveRunId)), [runIdProp, runs, runList, uiActiveRunId])
  const runId = run?.id ?? null
  const residuals = useSessionStore((s) => selectRunData(runId)(s).residuals)

  const [yScale, setYScale] = useState<'log' | 'linear'>('log')
  const [xAxis, setXAxis] = useState<XAxis>('iter')
  const [hidden, setHidden] = useState<Set<string>>(new Set())
  const [follow, setFollow] = useState(true)
  const [fieldsOpen, setFieldsOpen] = useState(false)
  const closeFields = useCallback(() => setFieldsOpen(false), [])
  const fieldsRef = useDismiss(fieldsOpen, closeFields)

  const keys = useMemo(() => seriesKeys(residuals), [residuals])
  const keysSig = keys.join('|')
  const timeAvailable = useMemo(() => hasTimeAxis(residuals), [residuals])
  const effectiveX: XAxis = xAxis === 'time' && timeAvailable ? 'time' : 'iter'

  const hostRef = useRef<HTMLDivElement | null>(null)
  const plotRef = useRef<uPlot | null>(null)
  const dirtyRef = useRef(true)
  const followRef = useRef(follow)
  followRef.current = follow
  const latest = useRef({ residuals, keys, yScale, effectiveX })
  latest.current = { residuals, keys, yScale, effectiveX }

  // (Re)create the plot when the series set, scale or theme changes.
  useEffect(() => {
    const host = hostRef.current
    if (!host || !keys.length) return
    const fg = cssVar('--fg-muted', '#5f6b7a')
    const grid = cssVar('--border', '#e1e6ed')
    const opts: uPlot.Options = {
      width: Math.max(100, host.clientWidth),
      height: Math.max(80, host.clientHeight),
      cursor: { drag: { x: true, y: false } },
      legend: { show: false },
      scales: { x: { time: false }, y: yScale === 'log' ? { distr: 3, log: 10 } : { distr: 1 } },
      axes: [
        { stroke: fg, grid: { stroke: grid, width: 1 }, ticks: { stroke: grid, width: 1 }, values: (_u, vals) => vals.map(fmtX), label: compact ? undefined : effectiveX === 'time' ? t('residuals.xTime') : t('residuals.iterations'), labelFont: '12px ' + cssVar('--font-ui', 'sans-serif'), font: '11px ' + cssVar('--font-ui', 'sans-serif'), size: compact ? 28 : 44 },
        { stroke: fg, grid: { stroke: grid, width: 1 }, ticks: { stroke: grid, width: 1 }, values: (_u, vals) => vals.map((v) => (yScale === 'log' ? fmtLogTick(v) : v.toPrecision(2))), font: '11px ' + cssVar('--font-ui', 'sans-serif'), size: compact ? 40 : 52 },
      ],
      series: [{ label: effectiveX }, ...keys.map((k, i) => ({ label: k, stroke: seriesColor(k, i), width: 1.5, spanGaps: true, show: !hidden.has(k) }))],
      hooks: {
        setSelect: [
          (u) => {
            if (u.select.width > 0) setFollow(false)
          },
        ],
      },
    }
    const shaped = shapeResiduals(residuals, keys, { log: yScale === 'log', xAxis: effectiveX, maxPoints: MAX_POINTS })
    const plot = new uPlot(opts, [shaped.x, ...shaped.ys] as uPlot.AlignedData, host)
    plotRef.current = plot
    dirtyRef.current = false
    const ro = new ResizeObserver(() => {
      if (host.clientWidth > 0 && host.clientHeight > 0) plot.setSize({ width: host.clientWidth, height: host.clientHeight })
    })
    ro.observe(host)
    return () => {
      ro.disconnect()
      plot.destroy()
      plotRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [keysSig, yScale, effectiveX, theme, compact, runId])

  useEffect(() => {
    dirtyRef.current = true
  }, [residuals])

  // Batched redraw.
  useEffect(() => {
    const id = setInterval(() => {
      const plot = plotRef.current
      if (!plot || !dirtyRef.current) return
      dirtyRef.current = false
      const { residuals: recs, keys: ks, yScale: ys, effectiveX: ex } = latest.current
      const shaped = shapeResiduals(recs, ks, { log: ys === 'log', xAxis: ex, maxPoints: MAX_POINTS })
      plot.setData([shaped.x, ...shaped.ys] as uPlot.AlignedData, followRef.current)
    }, BATCH_MS)
    return () => clearInterval(id)
  }, [])

  useEffect(() => {
    const plot = plotRef.current
    if (!plot) return
    keys.forEach((k, i) => plot.setSeries(i + 1, { show: !hidden.has(k) }))
  }, [hidden, keys])

  useEffect(() => {
    if (follow && plotRef.current) {
      dirtyRef.current = true
      plotRef.current.setScale('x', { min: plotRef.current.data[0][0] ?? 0, max: plotRef.current.data[0].at(-1) ?? 1 })
    }
  }, [follow])

  useEffect(() => {
    if (active && plotRef.current && hostRef.current) {
      const host = hostRef.current
      if (host.clientWidth > 0) plotRef.current.setSize({ width: host.clientWidth, height: host.clientHeight })
    }
  }, [active])

  const toggleKey = (k: string) =>
    setHidden((h) => {
      const n = new Set(h)
      if (n.has(k)) n.delete(k)
      else n.add(k)
      return n
    })

  const running = run?.status === 'running' || run?.status === 'queued'
  const now = useNow(running ? 1000 : null)

  return (
    <div className={`residuals${compact ? ' compact' : ''}`} data-testid="residuals-chart" data-run-id={runId ?? ''}>
      <div className="residuals-head">
        <span className="title">
          <Icon name="chart" size={14} className="muted" /> {t('residuals.title')}
        </span>
        {run ? (
          <span className="faint truncate mono" style={{ fontSize: 'var(--fs-xs)' }}>
            {run.label ?? run.binary} · {run.targetIter ? `${run.iter.toLocaleString()} / ${run.targetIter.toLocaleString()}` : run.iter.toLocaleString()}
            {running ? ` · ${formatDuration(now - Date.parse(run.startedAt))}` : ''}
          </span>
        ) : null}
        <span className="grow" />
        {!follow ? (
          <button className="btn btn-sm" onClick={() => setFollow(true)} title={t('residuals.follow')}>
            <Icon name="refresh" size={11} /> {t('residuals.follow')}
          </button>
        ) : null}
      </div>
      <div className="residuals-body">
        {!runId ? (
          <div className="residuals-empty">{runList.length ? t('residuals.noRun') : t('residuals.empty')}</div>
        ) : !keys.length ? (
          <div className="residuals-empty">{t('residuals.empty')}</div>
        ) : null}
        <div ref={hostRef} className="residuals-plot" />
        {keys.length ? (
          <div className="residuals-legend" data-testid="residuals-legend">
            {keys.map((k, i) => (
              <button key={k} className={hidden.has(k) ? 'off' : ''} onClick={() => toggleKey(k)}>
                <i style={{ background: seriesColor(k, i) }} />
                <span>{k === 'epsilon' ? 'ε' : k === 'omega' ? 'ω' : k === 'continuity' ? 'Continuity' : k}</span>
              </button>
            ))}
          </div>
        ) : null}
      </div>
      <div className="residuals-foot">
        <label>
          {t('residuals.yscale')}
          <select className="select" value={yScale} onChange={(e) => setYScale(e.target.value as 'log' | 'linear')} data-testid="residuals-yscale">
            <option value="log">{t('residuals.log')}</option>
            <option value="linear">{t('residuals.linear')}</option>
          </select>
        </label>
        {timeAvailable ? (
          <label>
            X
            <select className="select" value={xAxis} onChange={(e) => setXAxis(e.target.value as XAxis)}>
              <option value="iter">{t('residuals.xIter')}</option>
              <option value="time">{t('residuals.xTime')}</option>
            </select>
          </label>
        ) : null}
        <div ref={fieldsRef} style={{ position: 'relative' }}>
          <label>
            {t('residuals.fields')}
            <button className="select" style={{ minWidth: 70, textAlign: 'left' }} onClick={() => setFieldsOpen(!fieldsOpen)} disabled={!keys.length}>
              {hidden.size === 0 ? t('residuals.all') : `${keys.length - hidden.size}/${keys.length}`}
            </button>
          </label>
          {fieldsOpen ? (
            <div className="popover fields-pop">
              {keys.map((k, i) => (
                <label key={k}>
                  <input type="checkbox" checked={!hidden.has(k)} onChange={() => toggleKey(k)} />
                  <i className="dot" style={{ background: seriesColor(k, i) }} />
                  {k}
                </label>
              ))}
            </div>
          ) : null}
        </div>
        {!compact ? (
          <label>
            {t('residuals.run')}
            <select className="select" value={runId ?? ''} onChange={(e) => setActiveRun(e.target.value || null)} style={{ maxWidth: 160 }}>
              {!runId ? <option value="">—</option> : null}
              {runList.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.label ?? r.binary} ({r.status})
                </option>
              ))}
            </select>
          </label>
        ) : null}
        <span className="grow" />
        {runId ? (
          <a className="btn btn-sm" href={api.residualsCsvUrl(runId)} download={`${runId}-residuals.csv`} data-testid="residuals-export" title={t('residuals.export')}>
            <Icon name="download" size={12} /> {compact ? 'CSV' : t('residuals.export')}
          </a>
        ) : null}
      </div>
    </div>
  )
}
