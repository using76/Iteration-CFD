// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The Post right panel (STU-08a): the field, the representation and the layer
// list of the focused viewport half, the same controls as the viewer's toolbar
// and its LayerPanel, driving the same controller the same way (the field is a
// setField, a representation goes through representation.ts, a layer toggles or
// is removed). The section slider stays on the canvas: it belongs to its tool.
// The selects and the layer card use viewer.css, which ViewerTab (a static
// import of TabHost) always loads.
import { useEffect, useMemo, useState, type ReactElement } from 'react'
import type { FieldComponent, FieldInfo, Locale, ViewerState } from '@cfd/shared'
import { tx, type UiKey } from '../../i18n/extra'
import { useUiStore } from '../../state/uiStore'
import { getViewerController } from '../../viewer/api/viewerApi'
import type { ViewerController } from '../../viewer/controller/ViewerController'
import type { LayerEntry } from '../../viewer/controller/model'
import { LayerPanel } from '../../viewer/ui/LayerPanel'
import { applyRepresentationPreset, representationOf } from '../../viewer/ui/representation'
import { REPRESENTATION_PRESETS, fieldOptionValue, fieldOptions, type RepresentationPreset } from '../../viewer/ui/Toolbar'

export interface PostPanelViewProps {
  locale: Locale
  fields: FieldInfo[]
  /** fieldOptionValue of the coloured field, '' when none. */
  fieldValue: string
  onField: (name: string, component: FieldComponent | null) => void
  representation: RepresentationPreset
  onRepresentation: (r: RepresentationPreset) => void
  layers: LayerEntry[]
  onToggle: (id: string, visible: boolean) => void
  onRemove: (id: string) => void
  /** No result is loaded: the selects are disabled. */
  disabled: boolean
}

/** The option value a field select change carries, back to (name, component). */
export function parseFieldOption(value: string): { name: string; component: FieldComponent | null } {
  const [name, comp] = value.split('|')
  return { name, component: (comp || null) as FieldComponent | null }
}

export function PostPanelView(p: PostPanelViewProps): ReactElement {
  const t = (key: UiKey) => tx(p.locale, key)
  const options = fieldOptions(p.fields)
  return (
    <div className="right-body post-panel" data-testid="post-panel">
      {p.disabled ? <div className="right-empty">{t('postPanel.noResult')}</div> : null}
      <label className="right-field">
        <span>{t('postPanel.field')}</span>
        <select
          className="v3d-select field"
          data-testid="post-field"
          value={p.fieldValue}
          disabled={p.disabled || !options.length}
          onChange={(e) => {
            const f = parseFieldOption(e.target.value)
            p.onField(f.name, f.component)
          }}
        >
          {!options.length && <option value="">{t('postPanel.field')}</option>}
          {options.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      </label>
      <label className="right-field">
        <span>{t('postPanel.representation')}</span>
        <select className="v3d-select rep" data-testid="post-representation" value={p.representation} disabled={p.disabled} onChange={(e) => p.onRepresentation(e.target.value as RepresentationPreset)}>
          {REPRESENTATION_PRESETS.map((r) => (
            <option key={r.key} value={r.key}>
              {r.label}
            </option>
          ))}
        </select>
      </label>
      {p.layers.length ? <LayerPanel layers={p.layers} onToggle={p.onToggle} onRemove={p.onRemove} clip={null} onClipChange={() => {}} /> : <div className="right-empty">{t('postPanel.noLayers')}</div>}
    </div>
  )
}

function useViewerState(controller: ViewerController): ViewerState {
  const [state, setState] = useState(() => controller.getState())
  useEffect(() => {
    setState(controller.getState())
    return controller.subscribe(setState)
  }, [controller])
  return state
}

export function PostPanel(): ReactElement {
  const locale = useUiStore((s) => s.locale)
  const focusedView = useUiStore((s) => s.focusedView)
  const controller = useMemo(() => getViewerController(focusedView), [focusedView])
  const state = useViewerState(controller)
  const layers = useMemo(() => [...controller.layers.values()], [controller, state.layers])
  const representation = useMemo(() => representationOf(controller, state), [controller, state])
  const manifest = controller.dataset?.manifest ?? null
  return (
    <PostPanelView
      locale={locale}
      fields={manifest?.fields ?? []}
      fieldValue={controller.field ? fieldOptionValue(controller.field.name, controller.field.component) : ''}
      onField={(name, component) => void controller.execute({ type: 'setField', field: name, component, range: null, colormap: null, log: null })}
      representation={representation}
      onRepresentation={(r) => void applyRepresentationPreset(controller, r, { field: state.field, range: state.range })}
      layers={layers}
      onToggle={(id, visible) => controller.setLayerVisible(id, visible)}
      onRemove={(id) => void controller.execute({ type: 'remove', id })}
      disabled={!manifest}
    />
  )
}
