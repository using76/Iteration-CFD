// Plain virtualised log list with a text filter, stream chips and a follow
// chip; the filter lives in the ui store so set_log_filter drives the same
// state the screen shows.
import { useEffect, useMemo, useRef, useState } from 'react'
import type { LogLine } from '@cfd/shared'
import { useT } from '../app/hooks'
import { pickActiveRun, selectRunData, sortRuns, useSessionStore } from '../state/sessionStore'
import { useUiStore, type LogFilter, type LogStream } from '../state/uiStore'
import { useLogsReveal } from './logsReveal'

const ROW = 20
const OVERSCAN = 12

/** Keeps a line when its stream is shown and (the trimmed, lower-cased text
 *  filter is empty or the line's lower-cased text contains it). Order kept. */
export function filterLogLines(logs: LogLine[], f: LogFilter): LogLine[] {
  const q = f.text.trim().toLowerCase()
  return logs.filter((l) => f.streams.includes(l.stream) && (!q || l.text.toLowerCase().includes(q)))
}

export function LogsPane() {
  const t = useT()
  const runs = useSessionStore((s) => s.runs)
  const terminalRunId = useUiStore((s) => s.terminalRunId)
  const setTerminalRun = useUiStore((s) => s.setTerminalRun)
  const logFilter = useUiStore((s) => s.logFilter)
  const setLogFilter = useUiStore((s) => s.setLogFilter)
  const list = useMemo(() => sortRuns(runs), [runs])
  const current = useMemo(() => pickActiveRun(list, terminalRunId), [list, terminalRunId])
  const logs = useSessionStore((s) => selectRunData(current?.id ?? null)(s).logs)
  const [scrollTop, setScrollTop] = useState(0)
  const [height, setHeight] = useState(300)
  const listRef = useRef<HTMLDivElement | null>(null)
  const reveal = useLogsReveal((s) => s.request)

  const rows = useMemo(() => filterLogLines(logs, logFilter), [logs, logFilter])

  useEffect(() => {
    const el = listRef.current
    if (!el) return
    const ro = new ResizeObserver(() => setHeight(el.clientHeight))
    ro.observe(el)
    setHeight(el.clientHeight)
    return () => ro.disconnect()
  }, [])

  // Follow the tail: scroll to the bottom when rows are added and when follow turns on.
  useEffect(() => {
    const el = listRef.current
    if (el && logFilter.follow) el.scrollTop = el.scrollHeight
  }, [rows.length, logFilter.follow])

  // A Problems reveal stops the tail and centres the revealed line.
  useEffect(() => {
    if (!reveal) return
    if (current?.id !== reveal.runId) setTerminalRun(reveal.runId)
    const idx = rows.findIndex((l) => l.seq >= reveal.seq)
    const el = listRef.current
    if (idx >= 0 && el) {
      if (logFilter.follow) setLogFilter({ follow: false })
      el.scrollTop = Math.max(0, idx * ROW - height / 2)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reveal, rows.length])

  const first = Math.max(0, Math.floor(scrollTop / ROW) - OVERSCAN)
  const last = Math.min(rows.length, Math.ceil((scrollTop + height) / ROW) + OVERSCAN)
  const highlightSeq = reveal?.runId === current?.id ? reveal?.seq : null

  const toggleStream = (s: LogStream) => {
    const streams = logFilter.streams.includes(s) ? logFilter.streams.filter((x) => x !== s) : [...logFilter.streams, s]
    setLogFilter({ streams })
  }

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
        <input className="input" placeholder={t('logs.filter')} value={logFilter.text} onChange={(e) => setLogFilter({ text: e.target.value })} data-testid="logs-filter" />
        {(['stdout', 'stderr', 'system'] as const).map((s) => (
          <button key={s} className={'chip' + (logFilter.streams.includes(s) ? ' on' : '')} data-testid={'logs-stream-' + s} onClick={() => toggleStream(s)}>
            {s}
          </button>
        ))}
        <button className={'chip' + (logFilter.follow ? ' on' : '')} data-testid="logs-follow" onClick={() => setLogFilter({ follow: !logFilter.follow })}>
          {t('logs.follow')}
        </button>
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
          const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - ROW * 2
          if (atBottom !== logFilter.follow) setLogFilter({ follow: atBottom })
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
