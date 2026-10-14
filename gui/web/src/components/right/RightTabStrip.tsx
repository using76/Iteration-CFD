// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The right column's tab strip (STU-08a): AI Assistant, Post, Inspector and
// Properties, in RIGHT_PANELS order. A tab press is open_panel from the screen.
import type { ReactElement } from 'react'
import type { Locale } from '@cfd/shared'
import { tx, type UiKey } from '../../i18n/extra'
import { RIGHT_PANELS, type RightPanelId } from '../../state/uiStore'

/** Each panel's i18n label; the panel ids themselves stay the English names open_panel takes. */
export const RIGHT_PANEL_LABEL: Record<RightPanelId, UiKey> = {
  'AI Assistant': 'rightCol.ai',
  Post: 'rightCol.post',
  Inspector: 'rightCol.inspector',
  Properties: 'rightCol.properties',
}

export interface RightTabStripProps {
  locale: Locale
  active: RightPanelId
  onPick: (panel: RightPanelId) => void
}

export function RightTabStrip({ locale, active, onPick }: RightTabStripProps): ReactElement {
  return (
    <div className="right-tabs" role="tablist" aria-label={tx(locale, 'rightCol.panels')} data-testid="right-tabs">
      {RIGHT_PANELS.map((p) => (
        <button key={p} role="tab" aria-selected={p === active} className={`right-tab${p === active ? ' active' : ''}`} data-panel={p} onClick={() => onPick(p)}>
          {tx(locale, RIGHT_PANEL_LABEL[p])}
        </button>
      ))}
    </div>
  )
}
