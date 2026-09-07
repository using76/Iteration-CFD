import { useMemo } from 'react'
import { Group, Panel, Separator } from 'react-resizable-panels'
import { useT } from '../app/hooks'
import { ResidualsMini } from '../chart/ResidualsMini'
import { Icon, type IconName } from '../components/common/Icon'
import { flattenProblems, useSessionStore } from '../state/sessionStore'
import { useUiStore, type BottomTab } from '../state/uiStore'
import { LogsPane } from './LogsPane'
import { OutputPane } from './OutputPane'
import { ProblemsPane } from './ProblemsPane'
import { TerminalPane } from './TerminalPane'

const TABS: Array<{ id: BottomTab; icon: IconName; key: 'bottom.terminal' | 'bottom.logs' | 'bottom.problems' | 'bottom.output' }> = [
  { id: 'terminal', icon: 'terminal', key: 'bottom.terminal' },
  { id: 'logs', icon: 'list', key: 'bottom.logs' },
  { id: 'problems', icon: 'alert', key: 'bottom.problems' },
  { id: 'output', icon: 'output', key: 'bottom.output' },
]

export function BottomPanel() {
  const t = useT()
  const bottomTab = useUiStore((s) => s.bottomTab)
  const setBottomTab = useUiStore((s) => s.setBottomTab)
  const setBottomVisible = useUiStore((s) => s.setBottomVisible)
  const layout = useUiStore((s) => s.bottomLayout)
  const setLayout = useUiStore((s) => s.setLayout)
  const problems = useSessionStore((s) => s.problems)
  const counts = useMemo(() => {
    const list = flattenProblems(problems)
    return { total: list.length, errors: list.filter((p) => p.severity === 'error').length }
  }, [problems])

  return (
    <div className="bottom" data-testid="bottom-panel">
      <Group className="rrp-group" orientation="horizontal" defaultLayout={layout ?? undefined} onLayoutChanged={(l, meta) => meta.isUserInteraction && setLayout('bottomLayout', l)}>
        <Panel id="bottom-main" defaultSize="60%" minSize="30%">
          <div className="panel">
            <div className="bottom-tabs">
              {TABS.map((tab) => (
                <button key={tab.id} className={`bottom-tab${bottomTab === tab.id ? ' active' : ''}`} onClick={() => setBottomTab(tab.id)} data-testid={`bottom-tab-${tab.id}`}>
                  <Icon name={tab.icon} size={14} />
                  <span>{t(tab.key)}</span>
                  {tab.id === 'problems' && counts.total ? <span className={`count${counts.errors ? ' err' : ''}`}>{counts.total}</span> : null}
                </button>
              ))}
              <span className="grow" />
              <button className="icon-btn" onClick={() => setBottomVisible(false)} title={t('common.close')}>
                <Icon name="close" size={14} />
              </button>
            </div>
            <div className="bottom-body">
              <div style={{ position: 'absolute', inset: 0 }} hidden={bottomTab !== 'terminal'}>
                <TerminalPane visible={bottomTab === 'terminal'} />
              </div>
              {bottomTab === 'logs' ? <LogsPane /> : null}
              {bottomTab === 'problems' ? <ProblemsPane /> : null}
              {bottomTab === 'output' ? <OutputPane /> : null}
            </div>
          </div>
        </Panel>
        <Separator className="rrp-sep" />
        <Panel id="bottom-residuals" defaultSize="40%" minSize="20%">
          <ResidualsMini />
        </Panel>
      </Group>
    </div>
  )
}
