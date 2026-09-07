import { useMemo } from 'react'
import type { Problem } from '@cfd/shared'
import { useT } from '../app/hooks'
import { Icon } from '../components/common/Icon'
import { useEditorStore } from '../state/editorStore'
import { flattenProblems, useSessionStore } from '../state/sessionStore'
import { useUiStore } from '../state/uiStore'
import { useLogsReveal } from './logsReveal'

const SOURCE_KEY = { schema: 'problems.source.schema', semantic: 'problems.source.semantic', solver: 'problems.source.solver', server: 'problems.source.server' } as const

export function ProblemsPane() {
  const t = useT()
  const problems = useSessionStore((s) => s.problems)
  const openFile = useUiStore((s) => s.openFile)
  const setBottomTab = useUiStore((s) => s.setBottomTab)
  const setTerminalRun = useUiStore((s) => s.setTerminalRun)
  const prefill = useUiStore((s) => s.prefillComposer)
  const requestReveal = useEditorStore((s) => s.requestReveal)
  const revealLog = useLogsReveal((s) => s.reveal)
  const groups = useMemo(() => {
    const m = new Map<Problem['source'], Problem[]>()
    for (const p of flattenProblems(problems)) m.set(p.source, [...(m.get(p.source) ?? []), p])
    return [...m.entries()]
  }, [problems])

  const open = (p: Problem) => {
    if (p.path && p.line !== null) {
      openFile(p.path)
      requestReveal(p.path, p.line, p.col ?? 1)
    } else if (p.runId && p.logSeq !== null) {
      setTerminalRun(p.runId)
      setBottomTab('logs')
      revealLog(p.runId, p.logSeq)
    } else if (p.path) {
      openFile(p.path)
    }
  }

  if (!groups.length) return <div className="problems-pane"><div className="empty-note">{t('problems.none')}</div></div>
  return (
    <div className="problems-pane" data-testid="problems-pane">
      {groups.map(([source, items]) => (
        <div key={source} className="problem-group">
          <div className="problem-group-head">
            <span>{t(SOURCE_KEY[source])}</span>
            <span className="pill pill-muted">{items.length}</span>
          </div>
          {items.map((p) => (
            <div key={p.id} className="problem-row" onClick={() => open(p)} data-testid="problem-row">
              <Icon name={p.severity === 'error' ? 'error' : p.severity === 'warning' ? 'warning' : 'info'} size={14} style={{ color: p.severity === 'error' ? 'var(--danger)' : p.severity === 'warning' ? 'var(--warning)' : 'var(--info)', flex: 'none', marginTop: 2 }} />
              <div className="msg">
                <div>{p.message}</div>
                <div className="loc">
                  {p.path ? `${p.path}${p.line !== null ? `:${p.line}${p.col !== null ? `:${p.col}` : ''}` : ''}` : ''}
                  {p.runId ? ` · ${p.runId}${p.logSeq !== null ? ` #${p.logSeq}` : ''}` : ''}
                </div>
                {p.hint ? (
                  <button
                    className="chip problem-hint"
                    onClick={(e) => {
                      e.stopPropagation()
                      prefill(`${p.hint}\n\n${p.message}`)
                    }}
                  >
                    <Icon name="sparkles" size={11} /> {p.hint}
                  </button>
                ) : null}
              </div>
            </div>
          ))}
        </div>
      ))}
    </div>
  )
}
