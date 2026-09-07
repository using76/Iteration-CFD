import { useCallback, useEffect, useRef } from 'react'
import Editor, { type OnMount } from '@monaco-editor/react'
import { useT } from '../app/hooks'
import { useEditorStore } from '../state/editorStore'
import { useUiStore } from '../state/uiStore'
import { Icon } from '../components/common/Icon'
import { ensureCaseSchema, fileUri, monaco, setupMonaco } from './monacoSetup'

setupMonaco()

export function CodeEditor({ path, active }: { path: string; active: boolean }) {
  const t = useT()
  const buffer = useEditorStore((s) => s.buffers[path])
  const loadBuffer = useEditorStore((s) => s.loadBuffer)
  const setContent = useEditorStore((s) => s.setContent)
  const save = useEditorStore((s) => s.save)
  const reload = useEditorStore((s) => s.reloadBuffer)
  const unlock = useEditorStore((s) => s.unlock)
  const dismissConflict = useEditorStore((s) => s.dismissConflict)
  const setDiff = useEditorStore((s) => s.setDiff)
  const reveal = useEditorStore((s) => s.reveal)
  const theme = useUiStore((s) => s.theme)
  const setCursor = useUiStore((s) => s.setCursor)
  const openDiffTab = useUiStore((s) => s.openDiffTab)
  const editorRef = useRef<monaco.editor.IStandaloneCodeEditor | null>(null)
  const appliedHash = useRef<string | null>(null)

  useEffect(() => {
    void loadBuffer(path)
  }, [loadBuffer, path])
  useEffect(() => {
    if (buffer?.language === 'json') void ensureCaseSchema()
  }, [buffer?.language])

  // Push disk content into the model after reloads (never while the user has unsaved edits).
  useEffect(() => {
    const ed = editorRef.current
    if (!ed || !buffer || buffer.loading) return
    if (appliedHash.current === buffer.hash) return
    appliedHash.current = buffer.hash
    const model = ed.getModel()
    if (model && !buffer.dirty && model.getValue() !== buffer.content) model.setValue(buffer.content)
  }, [buffer])

  useEffect(() => {
    const ed = editorRef.current
    if (!ed || !reveal || reveal.path !== path) return
    ed.revealLineInCenter(reveal.line)
    ed.setPosition({ lineNumber: reveal.line, column: reveal.col })
    ed.focus()
  }, [reveal, path])

  useEffect(() => {
    if (active && editorRef.current) {
      const p = editorRef.current.getPosition()
      if (p) setCursor({ line: p.lineNumber, col: p.column })
      editorRef.current.layout()
    }
  }, [active, setCursor])

  const onMount: OnMount = useCallback(
    (ed) => {
      editorRef.current = ed
      // The model may have outlived an earlier tab for this path, in which case
      // it still holds that tab's text and defaultValue was ignored. Trusting
      // appliedHash here would call the stale text applied and let the next
      // save write it back under a baseHash the server accepts - no conflict,
      // no warning, the newer file on disk simply gone.
      const buf = useEditorStore.getState().buffers[path]
      appliedHash.current = buf?.hash ?? null
      const mounted = ed.getModel()
      if (mounted && buf && !buf.loading && !buf.dirty && mounted.getValue() !== buf.content) mounted.setValue(buf.content)
      ed.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => void save(path))
      ed.onDidChangeCursorPosition((e) => {
        if (useUiStore.getState().activeTabId === `file:${path}`) setCursor({ line: e.position.lineNumber, col: e.position.column })
      })
      const p = ed.getPosition()
      if (p) setCursor({ line: p.lineNumber, col: p.column })
      const pending = useEditorStore.getState().reveal
      if (pending && pending.path === path) {
        ed.revealLineInCenter(pending.line)
        ed.setPosition({ lineNumber: pending.line, column: pending.col })
      }
    },
    [path, save, setCursor],
  )

  if (!buffer) return null

  const showDiff = () => {
    const id = `diff:${path}`
    setDiff(id, { path, before: buffer.savedContent, after: buffer.content })
    openDiffTab(id, path, null)
  }

  return (
    <div className="editor-wrap" data-testid="code-editor" data-path={path}>
      {buffer.error ? (
        <div className="editor-banner danger">
          <Icon name="error" size={14} />
          <span className="grow">{t('editor.loadError', { msg: buffer.error })}</span>
          <button className="btn btn-sm" onClick={() => void reload(path)}>
            {t('common.retry')}
          </button>
        </div>
      ) : null}
      {buffer.conflict ? (
        <div className="editor-banner" data-testid="conflict-banner">
          <Icon name="warning" size={14} />
          <span className="grow">
            <b>{t('editor.conflict')}</b> {t('editor.conflictBody')}
          </span>
          <button className="btn btn-sm" onClick={() => void reload(path)}>
            {t('editor.reload')}
          </button>
          <button className="btn btn-sm btn-primary" onClick={() => void save(path, { overwrite: true })}>
            {t('editor.overwrite')}
          </button>
          <button className="btn btn-sm" onClick={showDiff}>
            {t('editor.showDiff')}
          </button>
          <button className="icon-btn" onClick={() => dismissConflict(path)} title={t('common.close')}>
            <Icon name="close" size={13} />
          </button>
        </div>
      ) : null}
      {buffer.readOnly ? (
        <div className="editor-banner info">
          <Icon name="eye" size={14} />
          <span className="grow">{t('editor.readOnly')}</span>
          <button className="btn btn-sm" onClick={() => unlock(path)}>
            <Icon name="edit" size={12} /> {t('editor.unlock')}
          </button>
        </div>
      ) : null}
      {buffer.saving || buffer.savedAt ? (
        <div className="editor-banner info" style={{ padding: '2px 12px', fontSize: 'var(--fs-xs)' }}>
          {buffer.saving ? t('editor.saving') : t('editor.saved')}
        </div>
      ) : null}
      <div className="editor-body">
        {buffer.loading ? (
          <div className="editor-loading">
            <span className="spinner" /> {t('common.loading')}
          </div>
        ) : (
          <Editor
            path={fileUri(path)}
            defaultValue={buffer.content}
            language={buffer.language}
            theme={theme === 'dark' ? 'cfd-dark' : 'cfd-light'}
            onMount={onMount}
            onChange={(v) => setContent(path, v ?? '')}
            keepCurrentModel
            saveViewState
            loading={<span className="spinner" />}
            options={{
              readOnly: buffer.readOnly,
              automaticLayout: true,
              minimap: { enabled: false },
              fontSize: 13,
              fontFamily: 'var(--font-mono)',
              tabSize: 2,
              insertSpaces: true,
              scrollBeyondLastLine: false,
              renderLineHighlight: 'line',
              wordWrap: 'off',
              smoothScrolling: true,
              padding: { top: 8 },
            }}
          />
        )}
      </div>
    </div>
  )
}
