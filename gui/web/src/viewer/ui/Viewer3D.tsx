// The whole "3D Viewer" tab: toolbar, canvas, legend, overlays. STUB.
import type { ViewerOverlayInfo } from '../api/viewerApi'

export interface Viewer3DProps {
  overlay: ViewerOverlayInfo | null
  locale: 'ko' | 'en'
  onOpenResiduals?: () => void
}

export function Viewer3D(_props: Viewer3DProps) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--fg-muted)' }}>
      3D viewer placeholder
    </div>
  )
}
