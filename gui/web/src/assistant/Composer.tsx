// The chat input: Enter sends, Shift+Enter newlines, @path autocomplete from
// the explorer, drag-drop attachments, context pills, Stop while a turn runs.
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { basename, useT } from '../app/hooks'
import { DRAG_MIME } from '../components/side/ExplorerTree'
import { Icon } from '../components/common/Icon'
import { useExplorerStore } from '../state/explorerStore'
import { pickActiveRun, sortRuns, useSessionStore } from '../state/sessionStore'
import { selectActiveFile, useUiStore } from '../state/uiStore'
import { actions } from '../ws/actions'
import { activeMention, completeMention, matchPaths, parseMentions } from './mentions'

export function Composer() {
  const t = useT()
  const online = useSessionStore((s) => s.connection === 'online')
  const hasSession = useSessionStore((s) => s.currentSessionId !== null)
  const turnActive = useSessionStore((s) => s.turn.active)
  const runs = useSessionStore((s) => s.runs)
  const draft = useUiStore((s) => s.composerDraft)
  const setDraft = useUiStore((s) => s.setComposerDraft)
  const prefill = useUiStore((s) => s.composerPrefill)
  const activeFile = useUiStore(selectActiveFile)
  const activeRunId = useUiStore((s) => s.activeRunId)
  const nodes = useExplorerStore((s) => s.nodes)
  const loadDir = useExplorerStore((s) => s.loadDir)
  const [attachments, setAttachments] = useState<string[]>([])
  const [mention, setMention] = useState<{ start: number; query: string } | null>(null)
  const [mentionIndex, setMentionIndex] = useState(0)
  const [dragOver, setDragOver] = useState(false)
  const areaRef = useRef<HTMLTextAreaElement | null>(null)
  const activeRun = useMemo(() => pickActiveRun(sortRuns(runs), activeRunId), [runs, activeRunId])
  const files = useMemo(
    () =>
      Object.values(nodes)
        .filter((n) => n.kind === 'file')
        .map((n) => n.path),
    [nodes],
  )
  const candidates = useMemo(() => (mention ? matchPaths(files, mention.query) : []), [mention, files])

  useEffect(() => {
    if (!prefill) return
    setDraft(prefill.text)
    setTimeout(() => {
      const el = areaRef.current
      if (el) {
        el.focus()
        el.setSelectionRange(el.value.length, el.value.length)
      }
    }, 0)
  }, [prefill, setDraft])

  useEffect(() => {
    const el = areaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(160, el.scrollHeight)}px`
  }, [draft])

  // Autocomplete only knows files the explorer has seen: widen it while a mention is typed.
  const mentionQuery = mention?.query ?? null
  useEffect(() => {
    if (mentionQuery === null) return
    const st = useExplorerStore.getState()
    void st.loadDir('')
    const slash = mentionQuery.lastIndexOf('/')
    if (slash >= 0) void st.loadDir(mentionQuery.slice(0, slash))
    else for (const n of st.children[''] ?? []) if (n.kind === 'dir') void st.loadDir(n.path)
  }, [mentionQuery, loadDir])

  const updateMention = useCallback((text: string, caret: number) => {
    setMention(activeMention(text, caret))
    setMentionIndex(0)
  }, [])

  const canSend = online && hasSession && draft.trim().length > 0 && !turnActive

  const send = useCallback(() => {
    if (!canSend) return
    const text = draft
    const mentions = parseMentions(text)
    const all = [...new Set([...attachments, ...mentions])]
    if (actions.sendUserMessage(text, all)) {
      setDraft('')
      setAttachments([])
      setMention(null)
    }
  }, [canSend, draft, attachments, setDraft])

  const pick = (path: string) => {
    if (!mention) return
    const el = areaRef.current
    const caret = el?.selectionStart ?? draft.length
    const r = completeMention(draft, caret, mention.start, path)
    setDraft(r.text)
    setMention(null)
    setTimeout(() => {
      if (el) {
        el.focus()
        el.setSelectionRange(r.caret, r.caret)
      }
    }, 0)
  }

  const onKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (mention && candidates.length) {
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        setMentionIndex((i) => (i + 1) % candidates.length)
        return
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault()
        setMentionIndex((i) => (i - 1 + candidates.length) % candidates.length)
        return
      }
      if (e.key === 'Tab' || e.key === 'Enter') {
        e.preventDefault()
        pick(candidates[mentionIndex])
        return
      }
      if (e.key === 'Escape') {
        setMention(null)
        return
      }
    }
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault()
      send()
    }
  }

  const addAttachment = (p: string) => setAttachments((a) => (a.includes(p) ? a : [...a, p]))

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {!online ? (
        <div className="offline-banner" data-testid="offline-banner">
          <Icon name="warning" size={14} /> {t('conn.offlineHint')}
        </div>
      ) : null}
      <div
        className={`composer${dragOver ? ' dragover' : ''}`}
        onDragOver={(e) => {
          if (e.dataTransfer.types.includes(DRAG_MIME)) {
            e.preventDefault()
            setDragOver(true)
          }
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          const p = e.dataTransfer.getData(DRAG_MIME)
          setDragOver(false)
          if (p) {
            e.preventDefault()
            addAttachment(p)
          }
        }}
      >
        {mention && candidates.length ? (
          <div className="mention-pop" data-testid="mention-popup">
            {candidates.map((p, i) => (
              <button key={p} className={`mention-item${i === mentionIndex ? ' active' : ''}`} onMouseDown={(e) => e.preventDefault()} onClick={() => pick(p)}>
                <Icon name={/\.jsonc?$/.test(p) ? 'braces' : 'file'} size={12} className="muted" />
                <span className="truncate">{p}</span>
              </button>
            ))}
          </div>
        ) : null}
        <textarea
          ref={areaRef}
          data-testid="composer-input"
          value={draft}
          placeholder={online ? t('assistant.placeholder') : t('common.offlineDisabled')}
          disabled={!online}
          rows={1}
          onChange={(e) => {
            setDraft(e.target.value)
            updateMention(e.target.value, e.target.selectionStart)
          }}
          onKeyUp={(e) => {
            if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) updateMention(e.currentTarget.value, e.currentTarget.selectionStart)
          }}
          onClick={(e) => updateMention(e.currentTarget.value, e.currentTarget.selectionStart)}
          onKeyDown={onKey}
          onBlur={() => setTimeout(() => setMention(null), 150)}
        />
        {turnActive ? (
          <button className="composer-send stop" onClick={() => actions.cancelTurn()} title={t('assistant.stop')} data-testid="stop-btn">
            <Icon name="stop" size={14} />
          </button>
        ) : (
          <button className="composer-send" onClick={send} disabled={!canSend} title={t('assistant.send')} data-testid="send-btn">
            <Icon name="send" size={14} />
          </button>
        )}
        <div className="composer-ctx">
          {activeFile ? (
            <span className="ctx-pill" title={activeFile}>
              <Icon name="file" size={10} /> {basename(activeFile)}
            </span>
          ) : null}
          {activeRun ? (
            <span className="ctx-pill" title={activeRun.argv.join(' ')}>
              <Icon name="play" size={10} /> {activeRun.label ?? activeRun.binary}
            </span>
          ) : null}
          {attachments.map((a) => (
            <span key={a} className="ctx-pill" title={a}>
              <Icon name="pin" size={10} /> {basename(a)}
              <button onClick={() => setAttachments((list) => list.filter((x) => x !== a))} title={t('assistant.clearContext')}>
                <Icon name="close" size={10} />
              </button>
            </span>
          ))}
          {turnActive ? (
            <span className="ctx-pill">
              <span className="spinner" style={{ width: 9, height: 9 }} /> {t('assistant.turnActive')}
            </span>
          ) : null}
          {!activeFile && !activeRun && !attachments.length && !turnActive ? <span className="faint" style={{ fontSize: 'var(--fs-xs)' }}>{t('assistant.mentionHint')}</span> : null}
        </div>
      </div>
    </div>
  )
}
