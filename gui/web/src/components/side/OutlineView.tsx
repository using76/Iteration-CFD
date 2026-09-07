import { useMemo } from 'react'
import { useT } from '../../app/hooks'
import { useEditorStore } from '../../state/editorStore'
import { selectActiveFile, useUiStore } from '../../state/uiStore'
import { Icon, type IconName } from '../common/Icon'
import { scanTopLevelKeys, type OutlineValueKind } from './outline'

const KIND_ICON: Record<OutlineValueKind, IconName> = { object: 'braces', array: 'list', string: 'file', number: 'chart', boolean: 'check', null: 'more', unknown: 'more' }

export function OutlineView() {
  const t = useT()
  const path = useUiStore(selectActiveFile)
  const content = useEditorStore((s) => (path ? s.buffers[path]?.content : undefined))
  const requestReveal = useEditorStore((s) => s.requestReveal)
  const isJson = !!path && /\.jsonc?$/i.test(path)
  const keys = useMemo(() => (isJson && content ? scanTopLevelKeys(content) : []), [isJson, content])
  if (!isJson || !keys.length) return <div className="empty-note">{t('outline.empty')}</div>
  return (
    <div data-testid="outline">
      {keys.map((k) => (
        <div key={`${k.key}:${k.line}`} className="outline-row" onClick={() => path && requestReveal(path, k.line, k.col)} title={`${k.key} (line ${k.line})`}>
          <Icon name={KIND_ICON[k.valueKind]} size={13} className="muted" />
          <span className="truncate">{k.key}</span>
          <span className="kind">{k.valueKind}</span>
        </div>
      ))}
    </div>
  )
}
