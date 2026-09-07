// Field dropdown (with component sub-choice), tool buttons, representation
import type { ReactElement } from 'react'
// presets and the settings gear.
import { t, type FieldComponent, type FieldInfo, type Locale } from '@cfd/shared'
import type { ViewerTool } from '../controller/model'
import { IconPan, IconRotate, IconSection, IconSelect, IconSettings, IconZoom } from './icons'

export type RepresentationPreset = 'surface' | 'surfaceEdges' | 'surfaceStreamlines' | 'wireframe' | 'slice' | 'iso' | 'outline' | 'points'

export const REPRESENTATION_PRESETS: { key: RepresentationPreset; label: string }[] = [
  { key: 'surface', label: 'Surface' },
  { key: 'surfaceEdges', label: 'Surface + Edges' },
  { key: 'surfaceStreamlines', label: 'Surface + Streamlines' },
  { key: 'wireframe', label: 'Wireframe' },
  { key: 'slice', label: 'Slice' },
  { key: 'iso', label: 'Iso-surface' },
  { key: 'outline', label: 'Outline' },
  { key: 'points', label: 'Points' },
]

const FIELD_LABELS: Record<string, string> = {
  U: 'Velocity',
  p: 'Pressure',
  p_rgh: 'Pressure (p_rgh)',
  T: 'Temperature',
  k: 'Turbulent kinetic energy',
  epsilon: 'Dissipation rate',
  omega: 'Specific dissipation',
  nut: 'Turbulent viscosity',
  nuTilda: 'nuTilda',
  rho: 'Density',
  alpha: 'Phase fraction',
}

const COMPONENTS: { c: FieldComponent; label: string }[] = [
  { c: 'magnitude', label: 'Magnitude' },
  { c: 'x', label: 'X' },
  { c: 'y', label: 'Y' },
  { c: 'z', label: 'Z' },
]

export function fieldOptionValue(name: string, component: FieldComponent | null): string {
  return `${name}|${component ?? ''}`
}

export function fieldOptions(fields: FieldInfo[]): { value: string; label: string }[] {
  const out: { value: string; label: string }[] = []
  for (const f of fields) {
    const base = FIELD_LABELS[f.name] ?? f.name
    if (f.components === 3) for (const { c, label } of COMPONENTS) out.push({ value: fieldOptionValue(f.name, c), label: `${base} ${label}` })
    else out.push({ value: fieldOptionValue(f.name, null), label: base === f.name ? base : `${base} (${f.name})` })
  }
  return out
}

const TOOLS: { tool: ViewerTool; title: string; Icon: (p: { size?: number }) => ReactElement }[] = [
  { tool: 'select', title: 'Select / probe', Icon: IconSelect },
  { tool: 'pan', title: 'Pan', Icon: IconPan },
  { tool: 'rotate', title: 'Rotate', Icon: IconRotate },
  { tool: 'zoom', title: 'Zoom', Icon: IconZoom },
  { tool: 'section', title: 'Section (clip box)', Icon: IconSection },
]

export interface ToolbarProps {
  locale: Locale
  fields: FieldInfo[]
  fieldValue: string
  onField: (name: string, component: FieldComponent | null) => void
  tool: ViewerTool
  onTool: (t: ViewerTool) => void
  representation: RepresentationPreset
  onRepresentation: (r: RepresentationPreset) => void
  settingsOpen: boolean
  onToggleSettings: () => void
  disabled: boolean
}

export function Toolbar(p: ToolbarProps) {
  const options = fieldOptions(p.fields)
  return (
    <div className="v3d-toolbar" data-testid="viewer-toolbar">
      <select
        className="v3d-select field"
        title={t(p.locale, 'viewer.field')}
        value={p.fieldValue}
        disabled={p.disabled || !options.length}
        onChange={(e) => {
          const [name, comp] = e.target.value.split('|')
          p.onField(name, (comp || null) as FieldComponent | null)
        }}
      >
        {!options.length && <option value="">{t(p.locale, 'viewer.field')}</option>}
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
      <div className="spacer" />
      <div className="v3d-tools" role="toolbar">
        {TOOLS.map(({ tool, title, Icon }) => (
          <button key={tool} className={p.tool === tool ? 'active' : ''} title={title} onClick={() => p.onTool(tool)} data-tool={tool}>
            <Icon size={16} />
          </button>
        ))}
      </div>
      <div className="spacer" />
      <select className="v3d-select rep" title={t(p.locale, 'viewer.representation')} value={p.representation} disabled={p.disabled} onChange={(e) => p.onRepresentation(e.target.value as RepresentationPreset)}>
        {REPRESENTATION_PRESETS.map((r) => (
          <option key={r.key} value={r.key}>
            {r.label}
          </option>
        ))}
      </select>
      <button className={`v3d-iconbtn${p.settingsOpen ? ' active' : ''}`} title={t(p.locale, 'common.settings')} onClick={p.onToggleSettings} data-testid="viewer-settings-button">
        <IconSettings size={16} />
      </button>
    </div>
  )
}
