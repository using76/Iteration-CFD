import { formatDuration, useNow, useT } from '../app/hooks'
import { formatResidual, leadingField } from '../chart/series'
import { Icon } from '../components/common/Icon'
import { runElapsedMs, runStatusPill, statusKey, viewerPathFor } from '../components/side/RunsPane'
import { useSessionStore } from '../state/sessionStore'
import { useUiStore } from '../state/uiStore'
import { actions } from '../ws/actions'

export function RunProgressCard({ runId }: { runId: string }) {
  const t = useT()
  const run = useSessionStore((s) => s.runs[runId])
  const openResidualsTab = useUiStore((s) => s.openResidualsTab)
  const live = run?.status === 'running' || run?.status === 'queued'
  const now = useNow(live ? 1000 : null)
  if (!run) return null

  const iterPct = run.targetIter ? Math.min(100, (run.iter / run.targetIter) * 100) : null
  const timePct = run.endTime && run.time !== null ? Math.min(100, (run.time / run.endTime) * 100) : null
  const pct = iterPct ?? timePct
  const residual = leadingField(run.lastResidual)
  const isMesh = run.binary === 'ofgpu-generate-mesh'
  const title = live ? (isMesh ? `${t('task.mesh')}...` : t('run.running')) : (run.label ?? run.binary)
  const counter = run.targetIter ? `${run.iter.toLocaleString()} / ${run.targetIter.toLocaleString()}` : run.endTime && run.time !== null ? `t = ${run.time} / ${run.endTime} s` : run.iter ? `${run.iter.toLocaleString()} ${t('run.iterations')}` : ''
  const viewerPath = viewerPathFor(run)

  return (
    <div className="run-card" data-testid="run-card" data-run-id={run.id} data-status={run.status}>
      <div className="run-card-head">
        {live ? <span className="spinner" /> : <span className={`pill ${runStatusPill(run.status)}`}>{t(statusKey(run.status))}</span>}
        <span className="title">
          {title}
          {run.converged ? <span className="pill pill-ok" style={{ marginLeft: 6 }}>converged</span> : null}
        </span>
        <span className="counter">{counter}</span>
      </div>
      <div className={`progress${live && pct === null ? ' indeterminate' : ''}`}>
        <i style={{ width: `${pct ?? (live ? 30 : 100)}%`, background: run.status === 'failed' || run.status === 'diverged' ? 'var(--danger)' : run.status === 'killed' ? 'var(--warning)' : run.status === 'done' ? 'var(--success)' : undefined }} />
      </div>
      <div className="run-card-meta">
        {residual ? (
          <span>
            {t('run.residual')} ({residual.field}): <b>{formatResidual(residual.value)}</b>
          </span>
        ) : null}
        <span>
          {t('run.elapsed')}: <b>{formatDuration(runElapsedMs(run, now))}</b>
        </span>
        {!live && run.exitCode !== null ? (
          <span>
            {t('run.exitCode')}: <b>{run.exitCode}</b>
          </span>
        ) : null}
        {run.error ? <span style={{ color: 'var(--danger)' }}>{run.error}</span> : null}
      </div>
      <div className="run-card-actions">
        {live ? (
          <button className="btn btn-sm btn-danger" onClick={() => actions.stopRun(run.id)} data-testid="run-stop">
            <Icon name="stop" size={11} /> {t('run.stop')}
          </button>
        ) : null}
        <button className="link-btn" disabled={!viewerPath} onClick={() => viewerPath && actions.openInViewer(viewerPath, run.id)} data-testid="run-open-viewer">
          <Icon name="cube" size={12} /> {t('run.openViewer')}
        </button>
        <button className="link-btn" onClick={() => openResidualsTab(run.id)} data-testid="run-open-residuals">
          <Icon name="chart" size={12} /> {t('run.openResiduals')}
        </button>
        <span className="faint mono" style={{ marginLeft: 'auto', fontSize: 'var(--fs-xs)' }}>
          {run.binary}
        </span>
      </div>
    </div>
  )
}
