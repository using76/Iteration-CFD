// Top-left facts: case / solver / mesh / iteration / residual.
import type { Locale } from '@cfd/shared'
import { t, type ViewerDataset } from '@cfd/shared'
import type { ViewerOverlayInfo } from '../api/viewerApi'

export function overlayLines(locale: Locale, overlay: ViewerOverlayInfo | null, manifest: ViewerDataset | null): [string, string][] {
  const meta = manifest?.meta ?? {}
  const str = (v: unknown) => (typeof v === 'string' && v ? v : typeof v === 'number' ? String(v) : null)
  const lines: [string, string][] = []
  const caseName = overlay?.caseName ?? str(meta.caseName) ?? manifest?.name ?? null
  if (caseName) lines.push([t(locale, 'viewer.case'), caseName])
  const solver = overlay?.solver ?? str(meta.model)
  const model = overlay?.model ?? str(meta.turbulence)
  if (solver) lines.push([t(locale, 'viewer.solver'), model && model !== solver ? `${solver} (${model})` : solver])
  if (manifest) lines.push([t(locale, 'viewer.mesh'), `${manifest.cellCount.toLocaleString()} cells`])
  if (overlay?.iter !== null && overlay?.iter !== undefined) lines.push([t(locale, 'viewer.iter'), overlay.targetIter ? `${overlay.iter.toLocaleString()} / ${overlay.targetIter.toLocaleString()}` : overlay.iter.toLocaleString()])
  if (overlay?.residual) lines.push([`${t(locale, 'run.residual')} (${overlay.residual.field})`, overlay.residual.value.toExponential(1)])
  return lines
}

export function OverlayInfo({ lines }: { lines: [string, string][] }) {
  if (!lines.length) return null
  return (
    <div className="v3d-card v3d-overlay" data-testid="viewer-overlay">
      {lines.map(([k, v]) => (
        <div key={k}>
          <span className="k">{k}: </span>
          <span>{v}</span>
        </div>
      ))}
    </div>
  )
}
