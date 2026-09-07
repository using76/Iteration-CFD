import { useEffect } from 'react'
import { basename, useT } from '../../app/hooks'
import { fsEvents } from '../../state/fsEvents'
import { useMetaStore } from '../../state/metaStore'
import { useUiStore } from '../../state/uiStore'
import { Icon } from '../common/Icon'

export function ScmPane() {
  const t = useT()
  const git = useMetaStore((s) => s.git)
  const refresh = useMetaStore((s) => s.refreshGit)
  const openFile = useUiStore((s) => s.openFile)
  useEffect(() => {
    void refresh()
    return fsEvents.subscribe(() => void refresh())
  }, [refresh])
  if (!git) return <div className="empty-note">{t('common.loading')}</div>
  if (!git.available) return <div className="empty-note">{t('scm.unavailable')}{git.error ? ` (${git.error})` : ''}</div>
  return (
    <div data-testid="scm-pane">
      <div className="empty-note" style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 12px' }}>
        <Icon name="branch" size={14} />
        <b>{git.branch ?? t('status.noBranch')}</b>
        <span className="pill pill-muted">{t('scm.readOnly')}</span>
        <span className="grow" />
        <button className="icon-btn" onClick={() => void refresh()} title={t('explorer.refresh')}>
          <Icon name="refresh" size={14} />
        </button>
      </div>
      {git.changes.length === 0 ? (
        <div className="empty-note">{t('scm.clean')}</div>
      ) : (
        <>
          <div className="section-title">{t('scm.changes', { n: git.changes.length })}</div>
          {git.changes.map((c) => (
            <div key={c.path} className="scm-row" onClick={() => openFile(c.path)} title={c.path}>
              <span className="scm-status">{c.status.trim() || '?'}</span>
              <span className="truncate">{basename(c.path)}</span>
              <span className="faint truncate" style={{ fontSize: 'var(--fs-xs)' }}>
                {c.path}
              </span>
            </div>
          ))}
        </>
      )}
    </div>
  )
}
