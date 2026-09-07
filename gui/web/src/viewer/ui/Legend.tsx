// Vertical colour bar with nice ticks, field name and unit ("|U| (m/s)").
import { useId } from 'react'
import type { LegendSpec } from '../controller/SceneView'
import { formatTick, logTicks, niceTicks, tickFraction } from './ticks'

export function Legend({ legend, height = 220 }: { legend: LegendSpec; height?: number }) {
  const id = useId()
  const barW = 16
  const barX = 6
  const barY = 24
  const barH = height - barY - 8
  const stops = []
  for (let i = 0; i < 256; i += 8) {
    const o = i * 4
    stops.push(<stop key={i} offset={`${(1 - i / 255) * 100}%`} stopColor={`rgb(${legend.table[o]},${legend.table[o + 1]},${legend.table[o + 2]})`} />)
  }
  const last = 255 * 4
  stops.push(<stop key="last" offset="0%" stopColor={`rgb(${legend.table[last]},${legend.table[last + 1]},${legend.table[last + 2]})`} />)
  const ticks = (legend.log ? logTicks(legend.min, legend.max) : niceTicks(legend.min, legend.max, 5)).filter((v) => {
    const f = tickFraction(v, legend.min, legend.max, legend.log)
    return f >= -1e-6 && f <= 1 + 1e-6
  })
  return (
    <svg className="v3d-legend" width={86} height={height} data-testid="viewer-legend">
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
          {stops.reverse()}
        </linearGradient>
      </defs>
      <text className="title" x={barX} y={14} fontSize={12}>
        {legend.title}
      </text>
      <rect x={barX} y={barY} width={barW} height={barH} fill={`url(#${id})`} stroke="rgba(31,39,51,0.5)" rx={2} />
      {ticks.map((v) => {
        const y = barY + (1 - tickFraction(v, legend.min, legend.max, legend.log)) * barH
        return (
          <g key={v}>
            <line x1={barX + barW} x2={barX + barW + 4} y1={y} y2={y} stroke="currentColor" />
            <text x={barX + barW + 7} y={y + 3.5} fontSize={11}>
              {formatTick(v)}
            </text>
          </g>
        )
      })}
    </svg>
  )
}
