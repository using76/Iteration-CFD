import { useEffect, useMemo, useRef, useState } from 'react'
import type { FsSearchHit } from '@cfd/shared'
import { basename, useT } from '../../app/hooks'
import { api, type SearchResponse } from '../../api/rest'
import { useEditorStore } from '../../state/editorStore'
import { fsEvents } from '../../state/fsEvents'
import { useUiStore } from '../../state/uiStore'
import { Icon } from '../common/Icon'

export function SearchPane() {
  const t = useT()
  const [q, setQ] = useState('')
  const [glob, setGlob] = useState('')
  const [regex, setRegex] = useState(false)
  const [caseSensitive, setCase] = useState(false)
  const [result, setResult] = useState<SearchResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set())
  const seq = useRef(0)
  const openFile = useUiStore((s) => s.openFile)
  const requestReveal = useEditorStore((s) => s.requestReveal)

  const run = async () => {
    const id = ++seq.current
    if (!q.trim()) {
      setResult(null)
      setError(null)
      return
    }
    setBusy(true)
    try {
      const r = await api.search(q, { glob: glob || null, regex, caseSensitive })
      if (id === seq.current) {
        setResult(r)
        setError(null)
      }
    } catch (err) {
      if (id === seq.current) setError(err instanceof Error ? err.message : String(err))
    } finally {
      if (id === seq.current) setBusy(false)
    }
  }

  useEffect(() => {
    const h = setTimeout(() => void run(), 300)
    return () => clearTimeout(h)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [q, glob, regex, caseSensitive])
  useEffect(() => fsEvents.subscribe(() => void run()), [q, glob, regex, caseSensitive]) // eslint-disable-line react-hooks/exhaustive-deps

  const groups = useMemo(() => {
    const m = new Map<string, FsSearchHit[]>()
    for (const h of result?.hits ?? []) {
      const list = m.get(h.path) ?? []
      list.push(h)
      m.set(h.path, list)
    }
    return [...m.entries()]
  }, [result])

  const open = (h: FsSearchHit) => {
    openFile(h.path)
    requestReveal(h.path, h.line, h.col)
  }

  return (
    <div className="panel" data-testid="search-pane">
      <div className="search-form">
        <input className="input" value={q} placeholder={t('common.search')} onChange={(e) => setQ(e.target.value)} autoFocus data-testid="search-input" />
        <input className="input" value={glob} placeholder={t('search.glob')} onChange={(e) => setGlob(e.target.value)} />
        <div className="search-opts">
          <label>
            <input type="checkbox" checked={regex} onChange={(e) => setRegex(e.target.checked)} /> {t('search.regex')}
          </label>
          <label>
            <input type="checkbox" checked={caseSensitive} onChange={(e) => setCase(e.target.checked)} /> {t('search.case')}
          </label>
          {busy ? <span className="spinner" style={{ width: 10, height: 10 }} /> : null}
        </div>
      </div>
      <div className="side-body">
        {error ? <div className="error-note">{error}</div> : null}
        {result && !groups.length ? <div className="empty-note">{t('search.noResults')}</div> : null}
        {result && groups.length ? (
          <div className="empty-note" style={{ padding: '6px 10px' }}>
            {t('search.results', { n: result.hits.length, files: groups.length })}
            {result.truncated ? ` · ${t('search.truncated')}` : ''}
          </div>
        ) : null}
        {groups.map(([path, hits]) => {
          const isCollapsed = collapsed.has(path)
          return (
            <div key={path}>
              <div
                className="search-file"
                onClick={() =>
                  setCollapsed((s) => {
                    const n = new Set(s)
                    if (n.has(path)) n.delete(path)
                    else n.add(path)
                    return n
                  })
                }
              >
                <Icon name={isCollapsed ? 'chevronRight' : 'chevronDown'} size={12} />
                <span className="truncate">{basename(path)}</span>
                <span className="faint truncate" style={{ fontWeight: 400, fontSize: 'var(--fs-xs)' }}>
                  {path}
                </span>
                <span className="pill pill-muted" style={{ marginLeft: 'auto' }}>
                  {hits.length}
                </span>
              </div>
              {!isCollapsed
                ? hits.map((h, i) => (
                    <div key={i} className="search-hit" onClick={() => open(h)} title={`${h.path}:${h.line}:${h.col}`}>
                      <span className="ln">{h.line}</span>
                      <span className="truncate">{highlight(h.text.trim(), q, regex, caseSensitive)}</span>
                    </div>
                  ))
                : null}
            </div>
          )
        })}
      </div>
    </div>
  )
}

function highlight(text: string, q: string, regex: boolean, caseSensitive: boolean): React.ReactNode {
  if (!q) return text
  try {
    const re = new RegExp(regex ? q : q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), caseSensitive ? '' : 'i')
    const m = re.exec(text)
    if (!m || !m[0]) return text
    return (
      <>
        {text.slice(0, m.index)}
        <mark>{m[0]}</mark>
        {text.slice(m.index + m[0].length)}
      </>
    )
  } catch {
    return text
  }
}
