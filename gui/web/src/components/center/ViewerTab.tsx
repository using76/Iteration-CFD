// Hosts the viewer module's Viewer3D and feeds it the overlay facts (case,
// solver, iteration, residual) from the active run.
import { useEffect, useMemo } from 'react'
import { getBinary } from '@cfd/shared'
import { basename, useLocale } from '../../app/hooks'
import { leadingField } from '../../chart/series'
import { pickActiveRun, sortRuns, useSessionStore } from '../../state/sessionStore'
import { useUiStore } from '../../state/uiStore'
import { Viewer3D, type ViewerOverlayInfo } from '../../viewer'
import { getWsClient } from '../../ws/client'

export function ViewerTab({ active }: { active: boolean }) {
  const locale = useLocale()
  const runs = useSessionStore((s) => s.runs)
  const activeRunId = useUiStore((s) => s.activeRunId)
  const openResidualsTab = useUiStore((s) => s.openResidualsTab)
  const viewSplit = useUiStore((s) => s.viewSplit)
  const focusedView = useUiStore((s) => s.focusedView)
  const setFocusedView = useUiStore((s) => s.setFocusedView)
  const run = useMemo(() => pickActiveRun(sortRuns(runs), activeRunId), [runs, activeRunId])

  useEffect(() => {
    if (!active) return
    getWsClient().viewer.attach()
    const id = setTimeout(() => getWsClient().viewer.attach(), 500)
    return () => clearTimeout(id)
  }, [active])

  const overlay = useMemo<ViewerOverlayInfo | null>(() => {
    if (!run) return null
    const bin = getBinary(run.binary)
    return {
      caseName: run.casePath ? basename(run.casePath).replace(/\.jsonc?$/, '') : null,
      solver: bin ? bin.name.replace(/^ofgpu-/, '') : run.binary,
      model: bin?.builds[0] ?? null,
      iter: run.iter,
      targetIter: run.targetIter,
      residual: leadingField(run.lastResidual),
    }
  }, [run])

  return (
    <div className={`viewer-tab${viewSplit ? ' split' : ''}`} data-testid="viewer-tab">
      <div
        className={`viewer-pane${viewSplit && focusedView === 'A' ? ' focused' : ''}`}
        data-view="A"
        onMouseDown={() => viewSplit && setFocusedView('A')}
      >
        <Viewer3D view="A" overlay={overlay} locale={locale} onOpenResiduals={() => openResidualsTab(run?.id ?? null)} />
      </div>
      {viewSplit ? (
        <div className={`viewer-pane${focusedView === 'B' ? ' focused' : ''}`} data-view="B" onMouseDown={() => setFocusedView('B')}>
          <Viewer3D view="B" overlay={null} locale={locale} />
        </div>
      ) : null}
    </div>
  )
}
