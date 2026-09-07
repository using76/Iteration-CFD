// Bottom-right facts: domain size, cell count, projection (click to toggle).
import { t, type Locale, type ViewerDataset } from '@cfd/shared'

function dim(v: number): string {
  return v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : Number(v.toPrecision(3)).toString()
}

export function DomainCard({ locale, manifest, projection, onToggleProjection }: { locale: Locale; manifest: ViewerDataset; projection: 'perspective' | 'orthographic'; onToggleProjection: () => void }) {
  const b = manifest.bounds
  const size = [b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]]
  return (
    <div className="v3d-card v3d-domain" data-testid="viewer-domain">
      <div>
        <b>{t(locale, 'viewer.domain')}:</b> {size.map(dim).join(' × ')} {manifest.units.length}
      </div>
      <div>
        <b>{t(locale, 'viewer.cells')}:</b> {manifest.cellCount.toLocaleString()} ({manifest.grid ? 'Structured' : 'Unstructured'}
        {manifest.geometryFidelity === 'proxy' ? ', proxy' : ''})
      </div>
      <div>
        <b>{t(locale, 'viewer.view')}:</b>{' '}
        <span className="toggle" onClick={onToggleProjection} role="button" title="Toggle projection">
          {projection === 'perspective' ? 'Perspective' : 'Orthographic'}
        </span>
      </div>
    </div>
  )
}
