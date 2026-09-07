import type { QuickAction } from '@cfd/shared'
import { useT } from '../app/hooks'
import { Icon, type IconName } from '../components/common/Icon'
import { useSessionStore } from '../state/sessionStore'
import { actions } from '../ws/actions'

const ITEMS: Array<{ action: QuickAction; icon: IconName; color: string; key: 'quick.mesh' | 'quick.run' | 'quick.explain' | 'quick.tool' }> = [
  { action: 'mesh', icon: 'mesh', color: 'var(--success)', key: 'quick.mesh' },
  { action: 'run', icon: 'play', color: 'var(--accent)', key: 'quick.run' },
  { action: 'explain', icon: 'error', color: 'var(--danger)', key: 'quick.explain' },
  { action: 'create_tool', icon: 'wrench', color: 'var(--success)', key: 'quick.tool' },
]

export function QuickActions() {
  const t = useT()
  const enabled = useSessionStore((s) => s.connection === 'online' && s.currentSessionId !== null && !s.turn.active)
  return (
    <div className="quick-actions" data-testid="quick-actions">
      {ITEMS.map((it) => (
        <button key={it.action} className="quick-btn" disabled={!enabled} onClick={() => actions.quick(it.action)} data-testid={`quick-${it.action}`} title={t(it.key)}>
          <span className="ico" style={{ color: it.color }}>
            <Icon name={it.icon} size={13} />
          </span>
          <span className="truncate">{t(it.key)}</span>
        </button>
      ))}
    </div>
  )
}
