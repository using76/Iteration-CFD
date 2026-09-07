import { useMemo } from 'react'
import type { RunInfo } from '@cfd/shared'
import { formatDuration, useNow, useT } from '../../app/hooks'
import { pickActiveRun, sortRuns, useSessionStore } from '../../state/sessionStore'
import { useUiStore } from '../../state/uiStore'
import { actions } from '../../ws/actions'
import { Icon } from '../common/Icon'

export function runStatusPill(status: RunInfo['status']): string {
  switch (status) {
    case 'running':
      return 'pill-accent'
    case 'queued':
      return 'pill-muted'
    case 'done':
      return 'pill-ok'
    case 'killed':
      return 'pill-warn'
    default:
      return 'pill-danger'
  }
}

export function statusKey(status: RunInfo['status']): 'run.running_short' | 'run.queued' | 'run.done' | 'run.failed' | 'run.killed' | 'run.diverged' {
  switch (status) {
    case 'running':
      return 'run.running_short'
    case 'queued':
      return 'run.queued'
    case 'done':
      return 'run.done'
    case 'failed':
      return 'run.failed'
    case 'killed':
      return 'run.killed'
    case 'diverged':
      return 'run.diverged'
  }
}

export function runElapsedMs(run: RunInfo, now: number): number {
  const start = Date.parse(run.startedAt)
  const end = run.endedAt ? Date.parse(run.endedAt) : now
  return Math.max(0, end - start)
}

export function viewerPathFor(run: RunInfo): string | null {
  return run.written[0] ?? run.outputRoot ?? run.casePath
}

export function RunsPane() {
  const t = useT()
  const runs = useSessionStore((s) => s.runs)
  const activeRunId = useUiStore((s) => s.activeRunId)
  const setActiveRun = useUiStore((s) => s.setActiveRun)
  const setTerminalRun = useUiStore((s) => s.setTerminalRun)
  const setBottomTab = useUiStore((s) => s.setBottomTab)
  const openResidualsTab = useUiStore((s) => s.openResidualsTab)
  const list = useMemo(() => sortRuns(runs), [runs])
  const anyRunning = list.some((r) => r.status === 'running' || r.status === 'queued')
  const now = useNow(anyRunning ? 1000 : null)
  const active = pickActiveRun(list, activeRunId)
  if (!list.length) return <div className="empty-note">{t('runs.empty')}</div>
  return (
    <div data-testid="runs-pane">
      {list.map((r) => (
        <div key={r.id} className={`run-row${active?.id === r.id ? ' active' : ''}`} onClick={() => setActiveRun(r.id)} data-testid="run-row">
          <div className="run-row-head">
            <span className={`pill ${runStatusPill(r.status)}`}>{t(statusKey(r.status))}</span>
            <span className="truncate" title={r.argv.join(' ')}>
              {r.label ?? r.binary}
            </span>
          </div>
          <div className="run-row-meta">
            <span>{r.targetIter ? `${r.iter.toLocaleString()} / ${r.targetIter.toLocaleString()}` : r.time !== null ? `t = ${r.time}${r.endTime !== null ? ` / ${r.endTime}` : ''}` : `${t('runs.iter')} ${r.iter.toLocaleString()}`}</span>
            <span>{formatDuration(runElapsedMs(r, now))}</span>
            {r.casePath ? <span className="truncate">{r.casePath}</span> : null}
          </div>
          <div className="run-row-actions" onClick={(e) => e.stopPropagation()}>
            {r.status === 'running' || r.status === 'queued' ? (
              <button className="btn btn-sm btn-danger" onClick={() => actions.stopRun(r.id)}>
                <Icon name="stop" size={11} /> {t('run.stop')}
              </button>
            ) : null}
            <button
              className="btn btn-sm"
              onClick={() => {
                setTerminalRun(r.id)
                setBottomTab('terminal')
              }}
            >
              <Icon name="terminal" size={11} /> {t('runs.openLog')}
            </button>
            <button className="btn btn-sm" onClick={() => openResidualsTab(r.id)}>
              <Icon name="chart" size={11} /> {t('tab.residuals')}
            </button>
            {viewerPathFor(r) ? (
              <button className="btn btn-sm" onClick={() => actions.openInViewer(viewerPathFor(r)!, r.id)}>
                <Icon name="cube" size={11} /> {t('tab.viewer')}
              </button>
            ) : null}
          </div>
        </div>
      ))}
    </div>
  )
}
