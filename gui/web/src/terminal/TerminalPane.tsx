// One xterm per run (kept alive across tab switches), fit + search addons,
// the synthesised "$ argv" echo and light colouring of solver output.
import { useEffect, useMemo, useRef, useState } from 'react'
import { Terminal } from '@xterm/xterm'
import { FitAddon } from '@xterm/addon-fit'
import { SearchAddon } from '@xterm/addon-search'
import '@xterm/xterm/css/xterm.css'
import type { LogLine } from '@cfd/shared'
import { useT } from '../app/hooks'
import { Icon } from '../components/common/Icon'
import { runStatusPill, statusKey } from '../components/side/RunsPane'
import { pickActiveRun, sortRuns, useSessionStore } from '../state/sessionStore'
import { useUiStore } from '../state/uiStore'
import { colourLogLine, commandEcho } from './ansi'

interface Term {
  term: Terminal
  fit: FitAddon
  search: SearchAddon
  el: HTMLDivElement
  writtenSeq: number
}

function themeFor(dark: boolean): NonNullable<ConstructorParameters<typeof Terminal>[0]>['theme'] {
  return dark
    ? { background: '#0f1216', foreground: '#e6eaf0', cursor: '#e6eaf0', selectionBackground: '#2a3442', black: '#1b1f26', brightBlack: '#6f7886', blue: '#4f93f0', green: '#3fbf72', red: '#ef5350', yellow: '#e39b12' }
    : { background: '#ffffff', foreground: '#1f2733', cursor: '#1f2733', selectionBackground: '#dfe9f7', black: '#1f2733', brightBlack: '#8a94a3', blue: '#2f7ce8', green: '#22a35c', red: '#dc3a3a', yellow: '#b8790a' }
}

export function TerminalPane({ visible }: { visible: boolean }) {
  const t = useT()
  const runs = useSessionStore((s) => s.runs)
  const runData = useSessionStore((s) => s.runData)
  const theme = useUiStore((s) => s.theme)
  const terminalRunId = useUiStore((s) => s.terminalRunId)
  const setTerminalRun = useUiStore((s) => s.setTerminalRun)
  const list = useMemo(() => sortRuns(runs), [runs])
  const current = useMemo(() => pickActiveRun(list, terminalRunId), [list, terminalRunId])
  const hostRef = useRef<HTMLDivElement | null>(null)
  const terms = useRef(new Map<string, Term>())
  const [query, setQuery] = useState('')

  // Create / attach the terminal for the current run.
  useEffect(() => {
    const host = hostRef.current
    if (!host || !current) return
    let entry = terms.current.get(current.id)
    if (!entry) {
      const el = document.createElement('div')
      el.style.height = '100%'
      const term = new Terminal({ convertEol: true, disableStdin: true, cursorBlink: false, cursorStyle: 'underline', fontSize: 12, fontFamily: getComputedStyle(document.documentElement).getPropertyValue('--font-mono') || 'monospace', lineHeight: 1.25, scrollback: 20000, theme: themeFor(theme === 'dark'), allowProposedApi: true })
      const fit = new FitAddon()
      const search = new SearchAddon()
      term.loadAddon(fit)
      term.loadAddon(search)
      entry = { term, fit, search, el, writtenSeq: 0 }
      terms.current.set(current.id, entry)
      host.appendChild(el)
      term.open(el)
      term.writeln(commandEcho(current.argv))
    }
    for (const [id, e] of terms.current) e.el.style.display = id === current.id ? 'block' : 'none'
    const fitNow = () => {
      try {
        if (visible && host.clientHeight > 0) entry!.fit.fit()
      } catch {
        // host not laid out yet
      }
    }
    fitNow()
    const ro = new ResizeObserver(fitNow)
    ro.observe(host)
    return () => ro.disconnect()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [current?.id, visible])

  // Feed new lines.
  useEffect(() => {
    if (!current) return
    const entry = terms.current.get(current.id)
    const data = runData[current.id]
    if (!entry || !data) return
    writeLines(entry, data.logs)
  }, [current, runData])

  useEffect(() => {
    for (const e of terms.current.values()) e.term.options.theme = themeFor(theme === 'dark')
  }, [theme])

  useEffect(() => {
    const map = terms.current
    return () => {
      for (const e of map.values()) e.term.dispose()
      map.clear()
    }
  }, [])

  const find = (dir: 'next' | 'prev') => {
    const entry = current ? terms.current.get(current.id) : null
    if (!entry || !query) return
    if (dir === 'next') entry.search.findNext(query, { incremental: false, caseSensitive: false })
    else entry.search.findPrevious(query, { caseSensitive: false })
  }

  return (
    <div className="terminal-pane" data-testid="terminal-pane">
      <div className="terminal-tabs">
        {list.map((r) => (
          <button key={r.id} className={`terminal-tab${current?.id === r.id ? ' active' : ''}`} onClick={() => setTerminalRun(r.id)} title={r.argv.join(' ')} data-testid="terminal-tab">
            <span className={`dot ${runStatusPill(r.status).replace('pill-', 'dot-').replace('dot-muted', '')}`} />
            <span className="truncate" style={{ maxWidth: 160 }}>
              {r.label ?? r.binary}
            </span>
            <span className="faint">{t(statusKey(r.status))}</span>
          </button>
        ))}
        <span style={{ flex: 1 }} />
        <input
          className="input"
          style={{ height: 22, width: 160 }}
          placeholder={t('terminal.search')}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') find(e.shiftKey ? 'prev' : 'next')
          }}
        />
        <button className="icon-btn" onClick={() => find('next')} title={t('terminal.search')}>
          <Icon name="search" size={13} />
        </button>
        <button className="icon-btn" onClick={() => current && terms.current.get(current.id)?.term.clear()} title={t('terminal.clear')}>
          <Icon name="trash" size={13} />
        </button>
      </div>
      <div className="terminal-host" ref={hostRef}>
        {!current ? <div className="empty-note">{t('terminal.empty')}</div> : null}
      </div>
    </div>
  )
}

function writeLines(entry: Term, logs: LogLine[]) {
  let start = logs.length
  for (let i = logs.length - 1; i >= 0; i--) {
    if (logs[i].seq <= entry.writtenSeq) break
    start = i
  }
  if (start >= logs.length) return
  const atBottom = entry.term.buffer.active.viewportY >= entry.term.buffer.active.baseY
  const out: string[] = []
  for (let i = start; i < logs.length; i++) {
    const l = logs[i]
    if (l.stream === 'system' && l.text.startsWith('$ ') && l.seq <= 1) continue
    out.push(colourLogLine(l.text, l.stream))
    entry.writtenSeq = l.seq
  }
  if (out.length) entry.term.write(out.join('\r\n') + '\r\n')
  if (atBottom) entry.term.scrollToBottom()
}
