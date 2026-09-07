import { useT } from '../../app/hooks'
import { Icon } from '../common/Icon'

export function ExtensionsPane() {
  const t = useT()
  return (
    <div className="empty-state" data-testid="extensions-pane">
      <Icon name="blocks" size={32} className="faint" />
      <h3>{t('ext.title')}</h3>
      <p>{t('ext.stub')}</p>
    </div>
  )
}
