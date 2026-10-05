// The MFR and the mean a-priori BLC_8 by attempt round: a uPlot line pair with
// the same numbers as a plain table, so the values survive without a canvas.
import { useEffect, useRef } from 'react'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { useT } from '../app/hooks'
import { useUiStore } from '../state/uiStore'
import { fmtPy } from '../assistant/autonomy/explain'
import type { RoundPoint } from '../assistant/autonomy/campaign'

function cssVar(name: string, fallback: string): string {
  if (typeof getComputedStyle !== 'function') return fallback
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback
}

export function RoundChart({ points }: { points: RoundPoint[] }) {
  const t = useT()
  const hostRef = useRef<HTMLDivElement | null>(null)
  // The axis and line colours come from the theme tokens, so a theme switch rebuilds the plot.
  const theme = useUiStore((s) => s.theme)
  useEffect(() => {
    const host = hostRef.current
    if (host === null || points.length === 0) return
    const fg = cssVar('--fg-muted', '#5f6b7a')
    const grid = cssVar('--border', '#e1e6ed')
    const opts: uPlot.Options = {
      width: Math.max(240, host.clientWidth),
      height: 180,
      legend: { show: true },
      scales: { x: { time: false }, y: { range: [0, 1] } },
      // A round is a whole attempt number: no tick label between two of them.
      axes: [
        { stroke: fg, grid: { stroke: grid, width: 1 }, ticks: { stroke: grid, width: 1 }, values: (_u, vals) => vals.map((v) => (Number.isInteger(v) ? String(v) : '')) },
        { stroke: fg, grid: { stroke: grid, width: 1 }, ticks: { stroke: grid, width: 1 } },
      ],
      series: [
        { label: t('autonomy.round') },
        { label: 'MFR', stroke: cssVar('--danger', '#dc3a3a'), width: 2 },
        { label: 'BLC₈', stroke: cssVar('--success', '#22a35c'), width: 2 },
      ],
    }
    const data = [
      points.map((p) => p.round),
      points.map((p) => p.mfr),
      points.map((p) => p.blc8),
    ] as uPlot.AlignedData
    const plot = new uPlot(opts, data, host)
    const ro = new ResizeObserver(() => {
      if (host.clientWidth > 0) plot.setSize({ width: host.clientWidth, height: 180 })
    })
    ro.observe(host)
    return () => {
      ro.disconnect()
      plot.destroy()
    }
  }, [points, t, theme])
  if (points.length === 0) return null
  return (
    <div className="round-chart" data-testid="round-chart" data-rounds={points.length}>
      <h3>{t('autonomy.rounds')}</h3>
      <div ref={hostRef} className="round-chart-plot" />
      <table data-testid="round-table">
        <thead>
          <tr>
            <th>{t('autonomy.round')}</th>
            <th>{t('autonomy.mfr')}</th>
            <th>{t('autonomy.blc8')}</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {points.map((p) => (
            <tr key={p.round} data-round={p.round}>
              <td>{p.round}</td>
              <td>{fmtPy(p.mfr)}</td>
              <td>{fmtPy(p.blc8)}</td>
              <td>{t('autonomy.fails', { fails: p.fails, n: p.n })}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
