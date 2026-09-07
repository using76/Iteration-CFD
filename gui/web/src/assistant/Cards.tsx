// Small cards: thinking, notice, image, refusal, error, suggestion chips.
import { useState } from 'react'
import { useT } from '../app/hooks'
import { Icon } from '../components/common/Icon'
import { useSessionStore } from '../state/sessionStore'
import { useUiStore } from '../state/uiStore'
import { actions } from '../ws/actions'

export function ThinkingBlock({ text, streaming }: { text: string; streaming: boolean }) {
  const t = useT()
  const [open, setOpen] = useState(false)
  return (
    <div className="thinking" data-testid="thinking-block">
      <div className="thinking-head" onClick={() => setOpen(!open)}>
        <Icon name={open ? 'chevronDown' : 'chevronRight'} size={12} />
        <Icon name="brain" size={13} />
        <span>{t('assistant.thinking')}</span>
        {streaming ? <span className="spinner" style={{ width: 10, height: 10 }} /> : null}
      </div>
      {open ? <div className="thinking-body">{text || '…'}</div> : null}
    </div>
  )
}

export function NoticeCard({ level, text }: { level: 'info' | 'warning' | 'error'; text: string }) {
  return (
    <div className={`notice-card ${level}`} data-testid="notice-card">
      <Icon name={level === 'error' ? 'error' : level === 'warning' ? 'warning' : 'info'} size={14} style={{ flex: 'none', marginTop: 2 }} />
      <span className="grow">{text}</span>
    </div>
  )
}

export function ImageBlock({ mime, base64, alt }: { mime: string; base64: string; alt: string }) {
  return (
    <div className="image-block">
      <img src={`data:${mime};base64,${base64}`} alt={alt} />
    </div>
  )
}

export function RefusalCard({ category, explanation }: { category: string | null; explanation: string | null }) {
  const t = useT()
  return (
    <div className="notice-card warning" data-testid="refusal-card">
      <Icon name="shield" size={14} style={{ flex: 'none', marginTop: 2 }} />
      <div className="grow">
        <b>{t('refusal.title')}</b>
        <div>{explanation ?? t('refusal.body')}</div>
        {category ? (
          <div className="faint" style={{ fontSize: 'var(--fs-xs)' }}>
            {t('refusal.category')}: {category}
          </div>
        ) : null}
      </div>
    </div>
  )
}

export function ErrorCard({ message, retryable }: { message: string; retryable: boolean }) {
  const t = useT()
  const online = useSessionStore((s) => s.connection === 'online')
  return (
    <div className="notice-card error" data-testid="error-card">
      <Icon name="error" size={14} style={{ flex: 'none', marginTop: 2 }} />
      <div className="grow">
        <b>{t('assistant.errorTitle')}</b>
        <div style={{ wordBreak: 'break-word' }}>{message}</div>
      </div>
      {retryable ? (
        <button className="btn btn-sm" disabled={!online} onClick={() => actions.retryLast()} data-testid="retry-btn">
          <Icon name="refresh" size={11} /> {t('common.retry')}
        </button>
      ) : null}
    </div>
  )
}

export function SuggestionChips({ items }: { items: string[] }) {
  const prefill = useUiStore((s) => s.prefillComposer)
  if (!items.length) return null
  return (
    <div className="chips" data-testid="suggestion-chips">
      {items.map((s, i) => (
        <button key={i} className="chip-suggest" onClick={() => prefill(s)}>
          {s}
        </button>
      ))}
    </div>
  )
}
