// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The Inspector right panel (STU-08a): the last probe or screen pick, which is
// the store's selection. A gui_control probe and a select-tool click on the
// canvas both write it, so the panel and gui_state's selection agree.
import type { ReactElement } from 'react'
import type { Locale, UiSelection } from '@cfd/shared'
import { tx, type UiKey } from '../../i18n/extra'
import { useUiStore } from '../../state/uiStore'

export interface InspectorViewProps {
  locale: Locale
  selection: UiSelection | null
}

/** A probed number as the viewer's tip shows it: four significant figures are too few to read a value back, so six. */
export function formatValue(v: number | null | undefined): string {
  return v == null ? '-' : String(Number(v.toPrecision(6)))
}

export function InspectorView({ locale, selection }: InspectorViewProps): ReactElement {
  const t = (key: UiKey) => tx(locale, key)
  if (selection === null) return <div className="right-body right-empty" data-testid="inspector-empty">{t('inspectorPanel.empty')}</div>
  if (selection.kind === 'none') return <div className="right-body right-empty" data-testid="inspector-miss">{t('inspectorPanel.miss')}</div>
  const c = selection.center
  return (
    <div className="right-body">
      <table className="kv-table" data-testid="inspector-table">
        <tbody>
          <tr>
            <th>{t('inspectorPanel.cell')}</th>
            <td data-testid="inspector-cell">{selection.id}</td>
          </tr>
          <tr>
            <th>{t('inspectorPanel.field')}</th>
            <td data-testid="inspector-field">{selection.field ?? '-'}</td>
          </tr>
          <tr>
            <th>{t('inspectorPanel.value')}</th>
            <td data-testid="inspector-value">{formatValue(selection.value)}</td>
          </tr>
          <tr>
            <th>{t('inspectorPanel.center')}</th>
            <td data-testid="inspector-center">{c ? c.map((x) => formatValue(x)).join(', ') : '-'}</td>
          </tr>
          {selection.patch ? (
            <tr>
              <th>{t('inspectorPanel.patch')}</th>
              <td data-testid="inspector-patch">{selection.patch}</td>
            </tr>
          ) : null}
        </tbody>
      </table>
    </div>
  )
}

export function InspectorPanel(): ReactElement {
  const locale = useUiStore((s) => s.locale)
  const selection = useUiStore((s) => s.selection)
  return <InspectorView locale={locale} selection={selection} />
}
