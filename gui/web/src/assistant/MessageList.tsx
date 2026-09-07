// Auto-scrolls to the newest content unless the user has scrolled up.
import { useEffect, useRef } from 'react'
import { useT } from '../app/hooks'
import { Icon } from '../components/common/Icon'
import { useSessionStore } from '../state/sessionStore'
import { ApprovalCard } from './ApprovalCard'
import { AssistantMessage } from './AssistantMessage'
import { ErrorCard, NoticeCard, RefusalCard } from './Cards'
import { UserBubble } from './UserBubble'

export function MessageList() {
  const t = useT()
  const session = useSessionStore((s) => s.session)
  const turn = useSessionStore((s) => s.turn)
  const ref = useRef<HTMLDivElement | null>(null)
  const stick = useRef(true)
  const messages = session?.messages ?? []
  const approvals = session?.pendingApprovals ?? []
  const lastAssistant = [...messages].reverse().find((m) => m.role === 'assistant')

  useEffect(() => {
    const el = ref.current
    if (el && stick.current) el.scrollTop = el.scrollHeight
  })

  return (
    <div
      ref={ref}
      className="assistant-body"
      data-testid="message-list"
      onScroll={(e) => {
        const el = e.currentTarget
        stick.current = el.scrollTop + el.clientHeight >= el.scrollHeight - 40
      }}
    >
      {!messages.length && !turn.active ? (
        <div className="empty-state" style={{ flex: 'none', padding: '40px 12px' }}>
          <Icon name="sparkles" size={32} style={{ color: 'var(--accent)' }} />
          <h3>{t('assistant.emptyTitle')}</h3>
          <p>{t('assistant.emptyBody')}</p>
        </div>
      ) : null}
      {messages.map((m) =>
        m.role === 'user' ? <UserBubble key={m.id} message={m} /> : <AssistantMessage key={m.id} message={m} streaming={turn.active && turn.messageId === m.id} showSuggestions={!turn.active && m.id === lastAssistant?.id} />,
      )}
      {approvals.map((a) => (
        <ApprovalCard key={`${a.turnId}:${a.requestedAt}`} approval={a} />
      ))}
      {turn.warnings.map((w, i) => (
        <NoticeCard key={i} level="warning" text={w} />
      ))}
      {turn.refusal ? <RefusalCard category={turn.refusal.category} explanation={turn.refusal.explanation} /> : null}
      {turn.error ? <ErrorCard message={turn.error.message} retryable={turn.error.retryable} /> : null}
    </div>
  )
}
