// Keeps every open tab mounted (editors, the viewer and the chart are
// expensive to re-create) and hides the inactive ones.
import { lazy, Suspense } from 'react'
import { ResidualsChart } from '../../chart/ResidualsChart'
import { useUiStore } from '../../state/uiStore'
import { EmptyState } from './EmptyState'
import { ViewerTab } from './ViewerTab'

const CodeEditor = lazy(() => import('../../editor/CodeEditor').then((m) => ({ default: m.CodeEditor })))
const DiffEditorTab = lazy(() => import('../../editor/DiffEditorTab').then((m) => ({ default: m.DiffEditorTab })))

export function TabHost() {
  const tabs = useUiStore((s) => s.tabs)
  const activeTabId = useUiStore((s) => s.activeTabId)
  if (!tabs.length) return <EmptyState />
  return (
    <div className="tab-host">
      {tabs.map((tab) => {
        const active = tab.id === activeTabId
        return (
          <div key={tab.id} className="tab-page" hidden={!active} data-testid={`tab-page-${tab.kind}`}>
            <Suspense fallback={<div className="editor-loading"><span className="spinner" /></div>}>
              {tab.kind === 'file' ? <CodeEditor path={tab.path} active={active} /> : tab.kind === 'viewer' ? <ViewerTab active={active} /> : tab.kind === 'residuals' ? <ResidualsChart runId={tab.runId} active={active} /> : <DiffEditorTab id={tab.id} />}
            </Suspense>
          </div>
        )
      })}
    </div>
  )
}
