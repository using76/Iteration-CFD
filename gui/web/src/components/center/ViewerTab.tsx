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
    <div className="viewer-tab" data-testid="viewer-tab">
      <Viewer3D overlay={overlay} locale={locale} onOpenResiduals={() => openResidualsTab(run?.id ?? null)} />
    </div>
  )
}
