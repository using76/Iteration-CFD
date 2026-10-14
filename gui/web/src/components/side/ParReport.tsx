// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// An ofgpu-par run's decomposition report in the run panel: one row per
// (rank count, partitioner, label rotation) with its owned and ghost cells
// and partition-map hash, and each Krylov solve's iteration count per rank
// beside the whole mesh's (SPEC-LIT 138.8). The report is parsed from the
// run's log by parseParReport; this component only draws it.
import { parRotationOk, t, type Locale, type ParReport as Report, type ParRotation } from '@cfd/shared'

export interface ParReportProps {
  locale: Locale
  report: Report
}

/** The Krylov labels in the order the driver printed them, over every rotation. */
function krylovLabels(rotations: ParRotation[]): string[] {
  const out: string[] = []
  for (const r of rotations) for (const k of r.krylov) if (!out.includes(k.label)) out.push(k.label)
  return out
}

export function ParReport({ locale, report }: ParReportProps) {
  const labels = krylovLabels(report.rotations)
  const verdict = report.verdict
  return (
    <div className="par-report" data-testid="par-report">
      <div className="par-report-head">
        <span>{t(locale, 'par.title')}</span>
        <span className={`pill ${verdict ? (verdict.ok ? 'pill-ok' : 'pill-danger') : 'pill-accent'}`} data-testid="par-verdict">
          {verdict ? t(locale, verdict.ok ? 'par.pass' : 'par.fail') : t(locale, 'par.running')}
        </span>
        <span className="par-report-meta">
          {report.transport ? t(locale, 'par.transport', { comm: report.transport }) : null}
          {report.ranks.length ? ` · ${t(locale, 'par.ranks', { ranks: report.ranks.join(', ') })}` : null}
          {report.mesh ? ` · ${t(locale, 'par.mesh', { cells: report.mesh.cells.toLocaleString('en-US') })}` : null}
        </span>
      </div>
      {report.references.map((ref) => (
        <div key={ref.label} className="par-report-meta" data-testid="par-reference">
          {t(locale, 'par.whole', { label: ref.label, iters: ref.iterations, res: ref.finalResidual.toExponential(3) })}
        </div>
      ))}
      {report.rotations.length ? (
        <div className="par-report-wrap">
          <table>
            <thead>
              <tr>
                <th>{t(locale, 'par.col.world')}</th>
                <th>{t(locale, 'par.col.method')}</th>
                <th>{t(locale, 'par.col.rot')}</th>
                <th>{t(locale, 'par.col.owned')}</th>
                <th>{t(locale, 'par.col.ghosts')}</th>
                <th>{t(locale, 'par.col.map')}</th>
                {labels.map((l) => (
                  <th key={l}>{t(locale, 'par.col.rank', { label: l })}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {report.rotations.map((r) => {
                const ok = parRotationOk(r)
                return (
                  <tr key={`${r.np}/${r.method}/${r.rot}`} data-testid="par-row" data-ok={ok ? '1' : '0'}>
                    <td>{r.np}</td>
                    <td>{r.method}</td>
                    <td>{r.rot}</td>
                    <td className={r.haloOk === false ? 'par-bad' : undefined}>{r.owned ?? ''}</td>
                    <td>{r.ghosts ?? ''}</td>
                    <td className={r.mapOk ? undefined : 'par-bad'}>{r.hash}</td>
                    {labels.map((l) => {
                      const k = r.krylov.find((x) => x.label === l)
                      if (!k) return <td key={l} />
                      return (
                        <td key={l} className={k.ok ? undefined : 'par-bad'} title={`worst |d| ${k.worst}, ${k.differ} cells differ`}>
                          {`[${k.iterations.join(', ')}] (${k.whole})`}
                        </td>
                      )
                    })}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  )
}
