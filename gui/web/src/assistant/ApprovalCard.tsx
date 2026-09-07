import type { PendingApproval } from '@cfd/shared'
import { toolLabel } from '@cfd/shared'
import { useLocale, useNow, useT } from '../app/hooks'
import { Icon } from '../components/common/Icon'
import { useSessionStore } from '../state/sessionStore'
import { actions } from '../ws/actions'

export function ApprovalCard({ approval }: { approval: PendingApproval }) {
  const t = useT()
  const locale = useLocale()
  const online = useSessionStore((s) => s.connection === 'online')
  const now = useNow(1000)
  const left = Math.max(0, Math.round((approval.expiresAt - now) / 1000))
  const ids = approval.toolUseIds
  return (
    <div className="approval-card" data-testid="approval-card">
      <div className="approval-head">
        <Icon name="shield" size={16} style={{ color: 'var(--warning)' }} />
        <span>{t('approval.title')}</span>
        <span className="grow" />
        <span className="faint" style={{ fontWeight: 400 }}>
          {left > 0 ? t('assistant.expiresIn', { s: left }) : t('assistant.expired')}
        </span>
      </div>
      {approval.calls.map((c) => (
        <div key={c.toolUseId} className="approval-call">
          <div>
            <b>{toolLabel(c.name, locale)}</b> <span className="faint mono">{c.name}</span>
          </div>
          <div>{c.summary}</div>
          {c.preview ? <pre>{c.preview}</pre> : null}
        </div>
      ))}
      <div className="approval-actions">
        <button className="btn btn-sm btn-primary" disabled={!online} onClick={() => actions.approve(ids, 'none')} data-testid="approve-btn">
          <Icon name="check" size={12} /> {t('approval.allow')}
        </button>
        <button className="btn btn-sm" disabled={!online} onClick={() => actions.approve(ids, 'session')} data-testid="approve-session-btn">
          {t('approval.allowSession')}
        </button>
        <button className="btn btn-sm btn-danger" disabled={!online} onClick={() => actions.deny(ids)} data-testid="deny-btn">
          <Icon name="close" size={12} /> {t('approval.deny')}
        </button>
      </div>
    </div>
  )
}
