// Plain virtualised log list with a text filter and stream chips.
import { useEffect, useMemo, useRef, useState } from 'react'
import type { LogLine } from '@cfd/shared'
import { useT } from '../app/hooks'
import { pickActiveRun, selectRunData, sortRuns, useSessionStore } from '../state/sessionStore'
import { useUiStore } from '../state/uiStore'
import { useLogsReveal } from './logsReveal'

const ROW = 20
const OVERSCAN = 12

export function LogsPane() {
  const t = useT()
  const runs = useSessionStore((s) => s.runs)
  const terminalRunId = useUiStore((s) => s.terminalRunId)
  const setTerminalRun = useUiStore((s) => s.setTerminalRun)
  const list = useMemo(() => sortRuns(runs), [runs])
  const current = useMemo(() => pickActiveRun(list, terminalRunId), [list, terminalRunId])
  const logs = useSessionStore((s) => selectRunData(current?.id ?? null)(s).logs)
  const [filter, setFilter] = useState('')
  const [streams, setStreams] = useState<Set<LogLine['stream']>>(new Set(['stdout', 'stderr', 'system']))
  const [scrollTop, setScrollTop] = useState(0)
  const [height, setHeight] = useState(300)
  const listRef = useRef<HTMLDivElement | null>(null)
  const stickRef = useRef(true)
  const reveal = useLogsReveal((s) => s.request)

  const rows = useMemo(() => {
    const f = filter.trim().toLowerCase()
    return logs.filter((l) => streams.has(l.stream) && (!f || l.text.toLowerCase().includes(f)))
  }, [logs, filter, streams])

  useEffect(() => {
    const el = listRef.current
    if (!el) return
    const ro = new ResizeObserver(() => setHeight(el.clientHeight))
    ro.observe(el)
    setHeight(el.clientHeight)
    return () => ro.disconnect()
  }, [])

  useEffect(() => {
    const el = listRef.current
    if (el && stickRef.current) el.scrollTop = el.scrollHeight
  }, [rows.length])

  useEffect(() => {
    if (!reveal) return
    if (current?.id !== reveal.runId) setTerminalRun(reveal.runId)
    const idx = rows.findIndex((l) => l.seq >= reveal.seq)
    const el = listRef.current
    if (idx >= 0 && el) {
      stickRef.current = false
      el.scrollTop = Math.max(0, idx * ROW - height / 2)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reveal, rows.length])

  const first = Math.max(0, Math.floor(scrollTop / ROW) - OVERSCAN)
  const last = Math.min(rows.length, Math.ceil((scrollTop + height) / ROW) + OVERSCAN)
  const highlightSeq = reveal?.runId === current?.id ? reveal?.seq : null

  const toggleStream = (s: LogLine['stream']) =>
    setStreams((set) => {
      const n = new Set(set)
      if (n.has(s)) n.delete(s)
      else n.add(s)
      return n
    })

  return (
    <div className="logs-pane" data-testid="logs-pane">
      <div className="logs-toolbar">
        <select className="select" value={current?.id ?? ''} onChange={(e) => setTerminalRun(e.target.value || null)} style={{ maxWidth: 200 }}>
          {!current ? <option value="">—</option> : null}
          {list.map((r) => (
            <option key={r.id} value={r.id}>
              {r.label ?? r.binary}
            </option>
          ))}
        </select>
        <input className="input" placeholder={t('logs.filter')} value={filter} onChange={(e) => setFilter(e.target.value)} data-testid="logs-filter" />
        {(['stdout', 'stderr', 'system'] as const).map((s) => (
          <button key={s} className={`chip${streams.has(s) ? ' on' : ''}`} onClick={() => toggleStream(s)}>
            {s}
          </button>
        ))}
        <span className="faint" style={{ fontSize: 'var(--fs-xs)' }}>
          {rows.length.toLocaleString()}
        </span>
      </div>
      <div
        ref={listRef}
        className="logs-list"
        onScroll={(e) => {
          const el = e.currentTarget
          setScrollTop(el.scrollTop)
          stickRef.current = el.scrollTop + el.clientHeight >= el.scrollHeight - ROW * 2
        }}
      >
        {!rows.length ? <div className="empty-note">{t('logs.empty')}</div> : null}
        <div style={{ height: rows.length * ROW, position: 'relative' }}>
          {rows.slice(first, last).map((l, i) => (
            <div key={l.seq} className={`log-line ${l.stream}`} style={{ position: 'absolute', top: (first + i) * ROW, left: 0, right: 0, background: highlightSeq === l.seq ? 'var(--warning-soft)' : undefined }}>
              <span className="seq">{l.seq}</span>
              <span>{l.text}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
