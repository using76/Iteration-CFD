import { useT } from '../app/hooks'
import { Icon } from '../components/common/Icon'
import { useUiStore } from '../state/uiStore'
import { ResidualsChart } from './ResidualsChart'

/** The bottom-right split: the same chart in compact form with an "open as tab" button. */
export function ResidualsMini() {
  const t = useT()
  const openResidualsTab = useUiStore((s) => s.openResidualsTab)
  return (
    <div className="panel" style={{ position: 'relative' }} data-testid="residuals-mini">
      <button className="icon-btn" style={{ position: 'absolute', right: 8, top: 4, zIndex: 4 }} onClick={() => openResidualsTab(null)} title={t('tab.residuals')}>
        <Icon name="layout" size={14} />
      </button>
      <ResidualsChart runId={null} compact />
    </div>
  )
}
