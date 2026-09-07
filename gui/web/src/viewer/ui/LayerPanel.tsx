// Layer list (toggle / remove) and the section (clip box) slider.
import type { LayerEntry } from '../controller/model'
import { IconEye, IconEyeOff, IconTrash } from './icons'

export interface ClipUi {
  axis: 'x' | 'y' | 'z'
  fraction: number
}

export function LayerPanel({ layers, onToggle, onRemove, clip, onClipChange }: { layers: LayerEntry[]; onToggle: (id: string, visible: boolean) => void; onRemove: (id: string) => void; clip: ClipUi | null; onClipChange: (c: ClipUi) => void }) {
  if (!layers.length && !clip) return null
  return (
    <div className="v3d-card v3d-layers" data-testid="viewer-layers">
      {layers.length > 0 && (
        <>
          <h4>Layers</h4>
          <ul>
            {layers.map((l) => (
              <li key={l.spec.id} className={l.visible ? '' : 'hidden'}>
                <button onClick={() => onToggle(l.spec.id, !l.visible)} title={l.visible ? 'Hide' : 'Show'}>
                  {l.visible ? <IconEye size={14} /> : <IconEyeOff size={14} />}
                </button>
                <span title={l.summary}>{l.summary}</span>
                <button onClick={() => onRemove(l.spec.id)} title="Remove">
                  <IconTrash size={14} />
                </button>
              </li>
            ))}
          </ul>
        </>
      )}
      {clip && (
        <div className="clip">
          <select className="v3d-select" style={{ height: 24, padding: '0 22px 0 6px' }} value={clip.axis} onChange={(e) => onClipChange({ ...clip, axis: e.target.value as ClipUi['axis'] })}>
            <option value="x">x</option>
            <option value="y">y</option>
            <option value="z">z</option>
          </select>
          <input type="range" min={0.02} max={1} step={0.01} value={clip.fraction} onChange={(e) => onClipChange({ ...clip, fraction: Number(e.target.value) })} title="Section position" />
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--fs-xs)' }}>{Math.round(clip.fraction * 100)}%</span>
        </div>
      )}
    </div>
  )
}
