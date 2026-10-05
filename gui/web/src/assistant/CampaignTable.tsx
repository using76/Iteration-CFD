// The campaign table: one row per geometry with its end state, expandable into
// that geometry's attempt cards, plus the per family and stratum summary.
import { Fragment, useState } from 'react'
import { useT } from '../app/hooks'
import { fmtPy } from './autonomy/explain'
import type { CampaignData } from './autonomy/campaign'
import type { Json } from './autonomy/explain'
import { AttemptCard } from './AttemptCard'

const pillOf = (terminal: string): string =>
  terminal === 'PASS' ? 'pill pill-ok' : terminal === 'CAPABILITY-LIMITED' ? 'pill pill-warn' : terminal === '' ? 'pill pill-muted' : 'pill pill-danger'

export function CampaignTable({ data, initialExpanded = [] }: { data: CampaignData; initialExpanded?: string[] }) {
  const t = useT()
  const [open, setOpen] = useState(new Set(initialExpanded))
  const counts = new Map<string, number>()
  for (const g of data.geometries) {
    const term = typeof g.terminal === 'string' ? g.terminal : ''
    counts.set(term, (counts.get(term) ?? 0) + 1)
  }
  const strip = [...counts.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
  const toggle = (gid: string) =>
    setOpen((prev) => {
      const next = new Set(prev)
      if (next.has(gid)) next.delete(gid)
      else next.add(gid)
      return next
    })
  const detail = (gid: string, reason: string | null, key: string) => {
    const e = data.explained[gid]
    const end = data.geometries.find((x) => x.geometry_id === gid)
    const refused = end !== undefined && Array.isArray(end.refused) ? end.refused : []
    return (
      <tr data-testid="campaign-detail" data-geometry={gid} key={key}>
        <td colSpan={7}>
          {refused.map((rid, i) => (
            <span key={`ref${i}`} className="mono pill pill-muted" data-testid="campaign-refused">
              {String(rid)}
            </span>
          ))}
          {e !== undefined && e.ok && (
            <div className="campaign-attempts">
              {e.value.attempts.map((a) => (
                <AttemptCard key={a.attempt} attempt={a} />
              ))}
            </div>
          )}
          {e !== undefined && !e.ok && <div className="notice-card error">{t('autonomy.explainError', { error: e.error })}</div>}
          {e === undefined && <div>{t('autonomy.noRows', { reason: reason ?? '—' })}</div>}
        </td>
      </tr>
    )
  }
  return (
    <div className="campaign-table-wrap">
      <div className="campaign-strip">
        {strip.map(([term, n]) => (
          <span key={term} data-testid="campaign-terminal" data-terminal={term}>
            {term} {n}
          </span>
        ))}
      </div>
      <table data-testid="campaign-table">
        <thead>
          <tr>
            <th>{t('autonomy.col.geometry')}</th>
            <th>{t('autonomy.col.group')}</th>
            <th>{t('autonomy.col.terminal')}</th>
            <th>{t('autonomy.col.attempts')}</th>
            <th>{t('autonomy.col.blc8')}</th>
            <th>{t('autonomy.col.cells')}</th>
            <th>{t('autonomy.col.reason')}</th>
          </tr>
        </thead>
        <tbody>
          {data.geometries.map((g) => {
            const gid = typeof g.geometry_id === 'string' ? g.geometry_id : ''
            const terminal = typeof g.terminal === 'string' ? g.terminal : ''
            const reason = typeof g.reason === 'string' ? g.reason : null
            const family = typeof g.family === 'string' ? g.family : ''
            const stratum = typeof g.stratum === 'string' ? g.stratum : ''
            const expanded = open.has(gid)
            return (
              <Fragment key={gid}>
                <tr data-testid="campaign-row" data-geometry={gid} data-terminal={terminal} data-expanded={String(expanded)} onClick={() => toggle(gid)}>
                  <td className="mono">{gid}</td>
                  <td>
                    {family}/{stratum}
                  </td>
                  <td>
                    <span className={pillOf(terminal)}>{terminal}</span>
                  </td>
                  <td>{String(g.attempts)}</td>
                  <td>{fmtPy(g.blc8_a_priori)}</td>
                  <td>{typeof g.n_cells === 'number' ? g.n_cells : '—'}</td>
                  <td>{reason ?? ''}</td>
                </tr>
                {expanded && detail(gid, reason, `d:${gid}`)}
              </Fragment>
            )
          })}
          {data.pending.map((gid) => {
            const expanded = open.has(gid)
            const nRows = data.attempts.filter((r) => r.geometry_id === gid).length
            return (
              <Fragment key={`p:${gid}`}>
                <tr data-testid="campaign-row" data-geometry={gid} data-terminal="" data-expanded={String(expanded)} onClick={() => toggle(gid)}>
                  <td className="mono">{gid}</td>
                  <td>—</td>
                  <td>
                    <span className={pillOf('')}>{t('autonomy.pending')}</span>
                  </td>
                  <td>{nRows}</td>
                  <td>—</td>
                  <td>—</td>
                  <td>—</td>
                </tr>
                {expanded && detail(gid, null, `pd:${gid}`)}
              </Fragment>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

export function CampaignSummaryTable({ summary }: { summary: Json | null }) {
  const t = useT()
  const groups = summary !== null && Array.isArray(summary.groups) ? (summary.groups as Json[]) : null
  if (groups === null) return <div data-testid="campaign-no-summary">{t('autonomy.noSummary')}</div>
  const num = (v: unknown): string => (typeof v === 'number' ? fmtPy(v) : '—')
  const rate = (value: unknown, ci: unknown): string => {
    const arr = Array.isArray(ci) ? ci : []
    return `${fmtPy(value)} [${fmtPy(arr[0])}, ${fmtPy(arr[1])}]`
  }
  return (
    <div className="campaign-summary-wrap">
      <h3>{t('autonomy.summary')}</h3>
      <table data-testid="campaign-summary">
        <thead>
          <tr>
            <th>{t('autonomy.col.group')}</th>
            <th>{t('autonomy.col.n')}</th>
            <th>{t('autonomy.col.mfr')}</th>
            <th>{t('autonomy.col.strict')}</th>
            <th>{t('autonomy.col.blc8')}</th>
            <th>{t('autonomy.col.cells')}</th>
            <th>{t('autonomy.col.capLimited')}</th>
          </tr>
        </thead>
        <tbody>
          {groups.map((g, i) => {
            const family = typeof g.family === 'string' ? g.family : ''
            const stratum = typeof g.stratum === 'string' ? g.stratum : ''
            return (
              <tr key={i} data-testid="summary-group" data-family={family} data-stratum={stratum}>
                <td>
                  {family}/{stratum}
                </td>
                <td>{num(g.n)}</td>
                <td>{rate(g.mfr, g.mfr_ci)}</td>
                <td>{rate(g.strict_rate, g.strict_ci)}</td>
                <td>{num(g.blc8_mean)}</td>
                <td>{num(g.cells_median)}</td>
                <td>{num(g.capability_limited)}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
