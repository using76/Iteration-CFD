import { useT } from '../../app/hooks'
import { useUiStore, type ActivityView } from '../../state/uiStore'
import { Icon, type IconName } from '../common/Icon'

const ITEMS: Array<{ id: ActivityView; icon: IconName; key: 'activity.explorer' | 'activity.search' | 'activity.scm' | 'activity.run' | 'activity.extensions' | 'activity.cfd' }> = [
  { id: 'explorer', icon: 'folder', key: 'activity.explorer' },
  { id: 'search', icon: 'search', key: 'activity.search' },
  { id: 'scm', icon: 'git', key: 'activity.scm' },
  { id: 'run', icon: 'bug', key: 'activity.run' },
  { id: 'extensions', icon: 'blocks', key: 'activity.extensions' },
  { id: 'cfd', icon: 'wrench', key: 'activity.cfd' },
]

export function ActivityBar() {
  const t = useT()
  const activity = useUiStore((s) => s.activity)
  const sideVisible = useUiStore((s) => s.sideVisible)
  const setActivity = useUiStore((s) => s.setActivity)
  const setSettingsOpen = useUiStore((s) => s.setSettingsOpen)
  return (
    <nav className="activity" aria-label="activity">
      {ITEMS.map((it) => (
        <button key={it.id} className={`activity-item${activity === it.id && sideVisible ? ' active' : ''}`} onClick={() => setActivity(it.id)} title={t(it.key)} data-testid={`activity-${it.id}`}>
          <Icon name={it.icon} size={22} strokeWidth={1.6} />
          <span>{t(it.key)}</span>
        </button>
      ))}
      <div className="activity-spacer" />
      <div className="activity-avatar" title={t('common.profile')}>
        <Icon name="user" size={16} />
      </div>
      <button className="activity-item" style={{ height: 44 }} onClick={() => setSettingsOpen(true)} title={t('common.settings')} data-testid="activity-settings">
        <Icon name="gear" size={20} strokeWidth={1.6} />
      </button>
    </nav>
  )
}
