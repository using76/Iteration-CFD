// Gear popover: colormap, range lock, log, shading, quality, opacity,
// slice interpolation, projection, camera presets, screenshot.
import { ColormapNameSchema, type CameraPreset, type ColormapName } from '@cfd/shared'
import type { DisplaySettings, FieldSelection, QualityLevel, Shading } from '../controller/model'

export interface SettingsProps {
  display: DisplaySettings
  field: FieldSelection | null
  range: [number, number] | null
  projection: 'perspective' | 'orthographic'
  onColormap: (name: ColormapName) => void
  onRange: (mode: 'auto' | 'global' | [number, number]) => void
  onLog: (on: boolean) => void
  onShading: (s: Shading) => void
  onQuality: (q: QualityLevel) => void
  onOpacity: (o: number) => void
  onInterpolate: (on: boolean) => void
  onProjection: (p: 'perspective' | 'orthographic') => void
  onPreset: (p: CameraPreset) => void
  onScreenshot: () => void
}

const PRESETS: CameraPreset[] = ['iso', 'fit', '+x', '-x', '+y', '-y', '+z', '-z']

function Seg<T extends string>({ value, options, onChange }: { value: T; options: { v: T; label: string }[]; onChange: (v: T) => void }) {
  return (
    <div className="seg">
      {options.map((o) => (
        <button key={o.v} className={o.v === value ? 'active' : ''} onClick={() => onChange(o.v)}>
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function SettingsPopover(p: SettingsProps) {
  const rangeMode = p.field?.rangeMode ?? 'auto'
  const lock = p.field?.lockedRange ?? p.range ?? [0, 1]
  return (
    <div className="v3d-settings" data-testid="viewer-settings" onMouseDown={(e) => e.stopPropagation()}>
      <div className="row">
        <label>Colormap</label>
        <select value={p.display.colormap} onChange={(e) => p.onColormap(e.target.value as ColormapName)}>
          {ColormapNameSchema.options.map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </select>
      </div>
      <div className="row">
        <label>Range</label>
        <Seg value={rangeMode} options={[{ v: 'auto', label: 'Auto' }, { v: 'global', label: 'Global' }, { v: 'locked', label: 'Lock' }]} onChange={(m) => p.onRange(m === 'locked' ? [lock[0], lock[1]] : m)} />
      </div>
      {rangeMode === 'locked' && (
        <div className="row">
          <input type="number" step="any" defaultValue={lock[0]} onBlur={(e) => p.onRange([Number(e.target.value), lock[1]])} />
          <span>…</span>
          <input type="number" step="any" defaultValue={lock[1]} onBlur={(e) => p.onRange([lock[0], Number(e.target.value)])} />
        </div>
      )}
      <div className="row">
        <label>Log scale</label>
        <input type="checkbox" checked={p.field?.log ?? false} onChange={(e) => p.onLog(e.target.checked)} disabled={!p.field} />
      </div>
      <div className="row">
        <label>Shading</label>
        <Seg value={p.display.shading} options={[{ v: 'pbr', label: 'PBR' }, { v: 'flat', label: 'Flat' }]} onChange={p.onShading} />
      </div>
      <div className="row">
        <label>Quality</label>
        <Seg value={p.display.quality} options={[{ v: 'low', label: 'Low' }, { v: 'medium', label: 'Med' }, { v: 'high', label: 'High' }]} onChange={p.onQuality} />
      </div>
      <div className="row">
        <label>Opacity</label>
        <input type="range" min={0.05} max={1} step={0.05} value={p.display.opacity} onChange={(e) => p.onOpacity(Number(e.target.value))} />
      </div>
      <div className="row">
        <label>Smooth slices</label>
        <input type="checkbox" checked={p.display.interpolate} onChange={(e) => p.onInterpolate(e.target.checked)} />
      </div>
      <hr />
      <div className="row">
        <label>Projection</label>
        <Seg value={p.projection} options={[{ v: 'perspective', label: 'Perspective' }, { v: 'orthographic', label: 'Ortho' }]} onChange={p.onProjection} />
      </div>
      <div className="presets">
        {PRESETS.map((c) => (
          <button key={c} onClick={() => p.onPreset(c)}>
            {c}
          </button>
        ))}
      </div>
      <hr />
      <button className="wide" onClick={p.onScreenshot}>
        Screenshot (PNG)
      </button>
    </div>
  )
}
