import { useMemo } from 'react'
import { DiffEditor } from '@monaco-editor/react'
import { useT } from '../app/hooks'
import { languageFor, useEditorStore } from '../state/editorStore'
import { useUiStore } from '../state/uiStore'
import { Icon } from '../components/common/Icon'
import { diffStats } from './diffStats'
import { setupMonaco } from './monacoSetup'

setupMonaco()

export function DiffEditorTab({ id }: { id: string }) {
  const t = useT()
  const diff = useEditorStore((s) => s.diffs[id])
  const applyDiff = useEditorStore((s) => s.applyDiff)
  const revertDiff = useEditorStore((s) => s.revertDiff)
  const theme = useUiStore((s) => s.theme)
  const stats = useMemo(() => (diff ? diffStats(diff.before, diff.after) : { added: 0, removed: 0 }), [diff])
  if (!diff) return <div className="empty-note">{t('common.none')}</div>
  return (
    <div className="editor-wrap" data-testid="diff-editor">
      <div className="diff-toolbar">
        <Icon name="diff" size={14} className="muted" />
        <span className="mono truncate">{diff.path}</span>
        <span className="diff-add">+{stats.added}</span>
        <span className="diff-del">−{stats.removed}</span>
        <span className="grow" />
        {diff.error ? <span className="pill pill-danger">{t('diff.applyFailed', { msg: diff.error })}</span> : null}
        {diff.applied ? <span className="pill pill-ok">{t('diff.applied')}</span> : null}
        <button className="btn btn-sm btn-primary" disabled={diff.applied} onClick={() => void applyDiff(id)}>
          {t('diff.apply')}
        </button>
        <button className="btn btn-sm" disabled={!diff.applied} onClick={() => void revertDiff(id)}>
          {t('diff.revert')}
        </button>
      </div>
      <div className="editor-body">
        <DiffEditor original={diff.before} modified={diff.after} language={languageFor(diff.path)} theme={theme === 'dark' ? 'cfd-dark' : 'cfd-light'} options={{ readOnly: true, renderSideBySide: true, automaticLayout: true, minimap: { enabled: false }, fontSize: 13, originalEditable: false }} loading={<span className="spinner" />} />
      </div>
    </div>
  )
}
