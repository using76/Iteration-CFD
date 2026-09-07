import { useCallback, useState } from 'react'
import { useT } from '../app/hooks'
import { Icon } from '../components/common/Icon'
import { useContextMenu } from '../components/common/Menu'
import { useSessionStore } from '../state/sessionStore'
import { useUiStore } from '../state/uiStore'
import { actions } from '../ws/actions'
import { Composer } from './Composer'
import { HistoryMenu } from './HistoryMenu'
import { MessageList } from './MessageList'
import { QuickActions } from './QuickActions'

export function AssistantPanel() {
  const t = useT()
  const connection = useSessionStore((s) => s.connection)
  const mode = useSessionStore((s) => s.hello?.mode ?? null)
  const llm = useSessionStore((s) => s.hello?.llm ?? null)
  const title = useSessionStore((s) => s.session?.title ?? null)
  const approvals = useSessionStore((s) => s.session?.pendingApprovals.length ?? 0)
  const setSettingsOpen = useUiStore((s) => s.setSettingsOpen)
  const toggleAssistant = useUiStore((s) => s.toggleAssistant)
  const [historyOpen, setHistoryOpen] = useState(false)
  const closeHistory = useCallback(() => setHistoryOpen(false), [])
  const [menu, openMenu] = useContextMenu()

  const status = connection !== 'online' ? 'offline' : llm === 'mock' || mode === 'demo' ? 'demo' : 'online'
  const statusText = status === 'offline' ? (connection === 'connecting' ? t('conn.connecting') : t('assistant.offline')) : status === 'demo' ? t('assistant.demo') : t('assistant.online')

  return (
    <aside className="panel assistant" data-testid="assistant-panel" style={{ position: 'relative' }}>
      <div className="assistant-head">
        <div className="assistant-head-row">
          <span className="title">
            <Icon name="sparkles" size={20} style={{ color: 'var(--accent)' }} />
            <span className="truncate">{t('assistant.title')}</span>
          </span>
          <button className="icon-btn" onClick={() => actions.newSession()} title={t('assistant.new')} data-testid="new-chat-btn">
            <Icon name="plus" size={18} />
          </button>
          <button className={`icon-btn${historyOpen ? ' active' : ''}`} onClick={() => setHistoryOpen(!historyOpen)} title={t('assistant.history')} data-testid="history-btn">
            <Icon name="history" size={17} />
          </button>
          <button
            className="icon-btn"
            title={t('common.more')}
            onClick={(e) =>
              openMenu(e, [
                { id: 'settings', label: t('assistant.settings'), icon: 'gear', onSelect: () => setSettingsOpen(true) },
                { id: 'new', label: t('assistant.new'), icon: 'plus', onSelect: () => actions.newSession() },
                { id: 'sep', label: '' },
                { id: 'hide', label: t('common.collapse'), icon: 'close', onSelect: toggleAssistant },
              ])
            }
          >
            <Icon name="more" size={17} />
          </button>
        </div>
        <div className="assistant-sub">
          <span className="truncate" title={title ?? ''}>
            {title ? title : t('assistant.subtitle')}
          </span>
          <span className={`assistant-status ${status}`} data-testid="assistant-status">
            <span className={`dot ${status === 'online' ? 'dot-ok' : status === 'demo' ? 'dot-warn' : 'dot-danger'}`} /> {statusText}
            {approvals ? <span className="pill pill-warn" style={{ marginLeft: 6 }}>{t('assistant.approvals', { n: approvals })}</span> : null}
          </span>
        </div>
        <HistoryMenu open={historyOpen} onClose={closeHistory} />
      </div>
      <MessageList />
      <div className="assistant-foot">
        <Composer />
        <QuickActions />
        <div className="disclaimer">{t('assistant.disclaimer')}</div>
      </div>
      {menu}
    </aside>
  )
}
