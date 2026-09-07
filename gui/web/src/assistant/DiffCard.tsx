import { useMemo } from 'react'
import type { UiBlock } from '@cfd/shared'
import { useT } from '../app/hooks'
import { Icon } from '../components/common/Icon'
import { diffStats } from '../editor/diffStats'
import { useEditorStore } from '../state/editorStore'
import { useUiStore } from '../state/uiStore'

type DiffBlock = Extract<UiBlock, { kind: 'diff' }>

export function DiffCard({ block, id }: { block: DiffBlock; id: string }) {
  const t = useT()
  const diff = useEditorStore((s) => s.diffs[id])
  const setDiff = useEditorStore((s) => s.setDiff)
  const applyDiff = useEditorStore((s) => s.applyDiff)
  const revertDiff = useEditorStore((s) => s.revertDiff)
  const openDiffTab = useUiStore((s) => s.openDiffTab)
  const stats = useMemo(() => diffStats(block.before, block.after), [block.before, block.after])
  const applied = diff ? diff.applied : block.applied
  const ensure = () => {
    if (!diff) setDiff(id, { path: block.path, before: block.before, after: block.after, applied: block.applied })
  }
  return (
    <div className="diff-card" data-testid="diff-card">
      <div className="diff-card-head">
        <Icon name="diff" size={14} className="muted" />
        <span className="path" title={block.path}>
          {block.path}
        </span>
        <span className="diff-add">+{stats.added}</span>
        <span className="diff-del">−{stats.removed}</span>
        {applied ? <span className="pill pill-ok">{t('diff.applied')}</span> : null}
      </div>
      {diff?.error ? <div className="error-note" style={{ padding: 0 }}>{t('diff.applyFailed', { msg: diff.error })}</div> : null}
      <div className="run-card-actions">
        <button
          className="btn btn-sm"
          onClick={() => {
            ensure()
            openDiffTab(id, block.path, block.toolUseId)
          }}
        >
          <Icon name="eye" size={11} /> {t('diff.open')}
        </button>
        <button
          className="btn btn-sm btn-primary"
          disabled={applied}
          onClick={() => {
            ensure()
            void applyDiff(id)
          }}
        >
          {t('diff.apply')}
        </button>
        <button
          className="btn btn-sm"
          disabled={!applied}
          onClick={() => {
            ensure()
            void revertDiff(id)
          }}
        >
          {t('diff.revert')}
        </button>
      </div>
    </div>
  )
}
