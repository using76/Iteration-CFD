import { isMac, useT } from '../../app/hooks'
import { Icon } from '../common/Icon'

export function EmptyState() {
  const t = useT()
  const mod = isMac() ? '⌘' : 'Ctrl'
  return (
    <div className="empty-state" data-testid="empty-state">
      <Icon name="logo" size={48} />
      <h3>{t('tab.emptyTitle')}</h3>
      <p>{t('tab.emptyBody')}</p>
      <div className="keys">
        <span>
          <span className="kbd">{mod}+K</span> {t('palette.placeholder')}
        </span>
        <span>
          <span className="kbd">{mod}+J</span> {t('palette.toggleBottom')}
        </span>
      </div>
    </div>
  )
}
