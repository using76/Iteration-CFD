// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The Properties right panel (STU-08a): the keys of the selected tree node, the
// store's propertiesNode. STU-08b's project tree selects the node (select_step)
// and opens this panel; until then nothing sets it and the panel says so.
import type { ReactElement } from 'react'
import type { Locale } from '@cfd/shared'
import { tx, type UiKey } from '../../i18n/extra'
import { useUiStore, type PropertiesNode } from '../../state/uiStore'

export interface PropertiesViewProps {
  locale: Locale
  node: PropertiesNode | null
}

/** The longest value text a row shows before it is cut with an ellipsis. */
export const PROPERTY_TEXT_MAX = 200

/** One key's value as the panel prints it: scalars as themselves, null as 'null', anything else as JSON. */
export function formatProperty(v: unknown): string {
  if (typeof v === 'string') return v
  if (typeof v === 'number' || typeof v === 'boolean') return String(v)
  if (v === null || v === undefined) return 'null'
  let text: string
  try {
    text = JSON.stringify(v) ?? String(v)
  } catch {
    text = String(v)
  }
  return text.length > PROPERTY_TEXT_MAX ? `${text.slice(0, PROPERTY_TEXT_MAX)}…` : text
}

export function PropertiesView({ locale, node }: PropertiesViewProps): ReactElement {
  const t = (key: UiKey) => tx(locale, key)
  if (node === null) return <div className="right-body right-empty" data-testid="properties-empty">{t('properties.empty')}</div>
  const entries = Object.entries(node.keys)
  return (
    <div className="right-body">
      <div className="right-title" data-testid="properties-node" title={node.id}>
        {node.label}
      </div>
      {entries.length === 0 ? (
        <div className="right-empty">{t('properties.noKeys')}</div>
      ) : (
        <table className="kv-table" data-testid="properties-table">
          <thead>
            <tr>
              <th>{t('properties.key')}</th>
              <th>{t('properties.value')}</th>
            </tr>
          </thead>
          <tbody>
            {entries.map(([k, v]) => (
              <tr key={k} data-key={k}>
                <th>{k}</th>
                <td>{formatProperty(v)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

export function PropertiesPanel(): ReactElement {
  const locale = useUiStore((s) => s.locale)
  const node = useUiStore((s) => s.propertiesNode)
  return <PropertiesView locale={locale} node={node} />
}
