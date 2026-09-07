import { useCallback, useEffect, useState } from 'react'
import { useDismiss, useT } from '../app/hooks'
import { Icon } from '../components/common/Icon'
import { useSessionStore } from '../state/sessionStore'
import { actions } from '../ws/actions'

export function HistoryMenu({ open, onClose }: { open: boolean; onClose: () => void }) {
  const t = useT()
  const sessions = useSessionStore((s) => s.sessions)
  const current = useSessionStore((s) => s.currentSessionId)
  const [renaming, setRenaming] = useState<{ id: string; title: string } | null>(null)
  const close = useCallback(() => {
    setRenaming(null)
    onClose()
  }, [onClose])
  const ref = useDismiss(open, close)
  useEffect(() => {
    if (open) actions.listSessions()
  }, [open])
  if (!open) return null
  const sorted = sessions.slice().sort((a, b) => (a.updatedAt < b.updatedAt ? 1 : -1))
  return (
    <div ref={ref} className="popover history-pop" data-testid="history-menu">
      <div className="menu-label">{t('assistant.history')}</div>
      {!sorted.length ? <div className="empty-note">{t('assistant.noSessions')}</div> : null}
      {sorted.map((s) => (
        <div key={s.id} className={`history-item${s.id === current ? ' current' : ''}`}>
          {renaming?.id === s.id ? (
            <input
              className="input"
              style={{ flex: 1, height: 24 }}
              autoFocus
              value={renaming.title}
              onChange={(e) => setRenaming({ id: s.id, title: e.target.value })}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  if (renaming.title.trim()) actions.renameSession(s.id, renaming.title.trim())
                  setRenaming(null)
                } else if (e.key === 'Escape') setRenaming(null)
              }}
            />
          ) : (
            <button
              className="grow"
              onClick={() => {
                actions.openSession(s.id)
                close()
              }}
              title={`${s.messageCount} · ${new Date(s.updatedAt).toLocaleString()}`}
            >
              {s.title || s.id}
            </button>
          )}
          <button className="icon-btn" onClick={() => setRenaming({ id: s.id, title: s.title })} title={t('assistant.rename')}>
            <Icon name="edit" size={12} />
          </button>
          <button
            className="icon-btn"
            onClick={() => {
              if (window.confirm(t('assistant.deleteConfirm'))) actions.deleteSession(s.id)
            }}
            title={t('assistant.delete')}
          >
            <Icon name="trash" size={12} />
          </button>
        </div>
      ))}
    </div>
  )
}
