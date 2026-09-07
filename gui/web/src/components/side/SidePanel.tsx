import { useState } from 'react'
import { useT } from '../../app/hooks'
import { useExplorerStore } from '../../state/explorerStore'
import { useUiStore } from '../../state/uiStore'
import { Icon } from '../common/Icon'
import { CfdToolsPane } from './CfdToolsPane'
import { ExplorerTree } from './ExplorerTree'
import { ExtensionsPane } from './ExtensionsPane'
import { OutlineView } from './OutlineView'
import { RunsPane } from './RunsPane'
import { ScmPane } from './ScmPane'
import { SearchPane } from './SearchPane'
import { SimulationTasks } from './SimulationTasks'

function Collapsible({ title, children, testId, defaultOpen = true }: { title: string; children: React.ReactNode; testId: string; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div data-testid={testId}>
      <div className="section-title" onClick={() => setOpen(!open)}>
        <Icon name={open ? 'chevronDown' : 'chevronRight'} size={13} />
        <span>{title}</span>
      </div>
      {open ? <div className="side-section-body">{children}</div> : null}
    </div>
  )
}

export function SidePanel() {
  const t = useT()
  const activity = useUiStore((s) => s.activity)
  const collapseAll = useUiStore((s) => s.collapseAll)
  const toggleSide = useUiStore((s) => s.toggleSide)
  const loadDir = useExplorerStore((s) => s.loadDir)

  const title = activity === 'explorer' ? t('panel.project') : activity === 'search' ? t('activity.search') : activity === 'scm' ? t('scm.title') : activity === 'run' ? t('runs.title') : activity === 'extensions' ? t('ext.title') : t('activity.cfd')

  return (
    <aside className="panel side" data-testid="side-panel">
      <div className="side-header">
        <span className="truncate">{title}</span>
        <span className="grow" />
        {activity === 'explorer' ? (
          <>
            <button className="icon-btn" title={t('explorer.refresh')} onClick={() => void loadDir('', true)}>
              <Icon name="refresh" size={14} />
            </button>
            <button className="icon-btn" title={t('explorer.collapseAll')} onClick={collapseAll}>
              <Icon name="collapse" size={14} />
            </button>
          </>
        ) : null}
        <button className="icon-btn" title={t('common.collapse')} onClick={toggleSide}>
          <Icon name="close" size={14} />
        </button>
      </div>
      {activity === 'explorer' ? (
        <>
          <div className="side-body">
            <ExplorerTree />
          </div>
          <div className="side-footer">
            <Collapsible title={t('panel.outline')} testId="outline-section">
              <OutlineView />
            </Collapsible>
            <Collapsible title={t('panel.tasks')} testId="tasks-section">
              <SimulationTasks />
            </Collapsible>
          </div>
        </>
      ) : activity === 'search' ? (
        <SearchPane />
      ) : (
        <div className="side-body">{activity === 'scm' ? <ScmPane /> : activity === 'run' ? <RunsPane /> : activity === 'extensions' ? <ExtensionsPane /> : <CfdToolsPane />}</div>
      )}
    </aside>
  )
}
