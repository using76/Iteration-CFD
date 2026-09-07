// Ctrl/Cmd+K palette: open files, switch tabs, quick actions and toggles.
import { useEffect, useMemo, useRef, useState } from 'react'
import type { QuickAction } from '@cfd/shared'
import { basename, useT } from '../../app/hooks'
import { matchPaths } from '../../assistant/mentions'
import { useExplorerStore } from '../../state/explorerStore'
import { useSessionStore } from '../../state/sessionStore'
import { useUiStore, type Tab } from '../../state/uiStore'
import { actions } from '../../ws/actions'
import { Icon, type IconName } from '../common/Icon'

interface Item {
  id: string
  group: 'files' | 'tabs' | 'commands'
  label: string
  hint?: string
  icon: IconName
  run: () => void
}

function tabLabel(tab: Tab, t: ReturnType<typeof useT>): string {
  switch (tab.kind) {
    case 'file':
      return basename(tab.path)
    case 'viewer':
      return t('tab.viewer')
    case 'residuals':
      return t('tab.residuals')
    case 'diff':
      return `${t('tab.diff')}: ${basename(tab.path)}`
  }
}

export function CommandPalette() {
  const t = useT()
  const open = useUiStore((s) => s.paletteOpen)
  const setOpen = useUiStore((s) => s.setPaletteOpen)
  const tabs = useUiStore((s) => s.tabs)
  const ui = useUiStore
  const nodes = useExplorerStore((s) => s.nodes)
  const online = useSessionStore((s) => s.connection === 'online')
  const [query, setQuery] = useState('')
  const [active, setActive] = useState(0)
  const inputRef = useRef<HTMLInputElement | null>(null)

  useEffect(() => {
    if (open) {
      setQuery('')
      setActive(0)
      setTimeout(() => inputRef.current?.focus(), 0)
    }
  }, [open])

  const files = useMemo(
    () =>
      Object.values(nodes)
        .filter((n) => n.kind === 'file')
        .map((n) => n.path),
    [nodes],
  )

  const items = useMemo<Item[]>(() => {
    if (!open) return []
    const q = query.trim()
    const s = ui.getState()
    const commands: Item[] = [
      { id: 'c:theme', group: 'commands', label: t('palette.toggleTheme'), icon: s.theme === 'light' ? 'moon' : 'sun', run: () => s.toggleTheme() },
      { id: 'c:locale', group: 'commands', label: t('palette.toggleLocale'), icon: 'book', run: () => s.toggleLocale() },
      { id: 'c:side', group: 'commands', label: t('palette.toggleSidebar'), icon: 'collapse', hint: 'Ctrl+B', run: () => s.toggleSide() },
      { id: 'c:bottom', group: 'commands', label: t('palette.toggleBottom'), icon: 'terminal', hint: 'Ctrl+J', run: () => s.toggleBottom() },
      { id: 'c:viewer', group: 'commands', label: t('palette.openViewer'), icon: 'cube', run: () => s.openViewerTab() },
      { id: 'c:residuals', group: 'commands', label: t('palette.openResiduals'), icon: 'chart', run: () => s.openResidualsTab(null) },
      { id: 'c:newchat', group: 'commands', label: t('palette.newChat'), icon: 'plus', hint: 'Ctrl+Shift+N', run: () => actions.newSession() },
      ...(['mesh', 'run', 'postprocess', 'validate', 'export'] as QuickAction[]).map<Item>((a) => ({
        id: `q:${a}`,
        group: 'commands',
        label: t(a === 'mesh' ? 'task.mesh' : a === 'run' ? 'task.run' : a === 'postprocess' ? 'task.post' : a === 'validate' ? 'task.validate' : 'task.export'),
        icon: 'zap',
        hint: online ? undefined : t('conn.offline'),
        run: () => actions.quick(a),
      })),
    ]
    const tabItems: Item[] = tabs.map((tab) => ({ id: `t:${tab.id}`, group: 'tabs', label: tabLabel(tab, t), hint: tab.kind === 'file' ? tab.path : undefined, icon: tab.kind === 'viewer' ? 'cube' : tab.kind === 'residuals' ? 'chart' : tab.kind === 'diff' ? 'diff' : 'file', run: () => s.activateTab(tab.id) }))
    const fileItems: Item[] = matchPaths(files, q, 12).map((p) => ({ id: `f:${p}`, group: 'files', label: basename(p), hint: p, icon: /\.jsonc?$/.test(p) ? 'braces' : /\.rs$/.test(p) ? 'gear' : 'file', run: () => s.openFile(p) }))
    const lq = q.toLowerCase()
    const filt = (i: Item) => !lq || i.label.toLowerCase().includes(lq) || (i.hint ?? '').toLowerCase().includes(lq)
    return [...tabItems.filter(filt), ...fileItems, ...commands.filter(filt)]
  }, [open, query, tabs, files, t, ui, online])

  useEffect(() => setActive(0), [query])
  if (!open) return null

  const run = (i: Item) => {
    setOpen(false)
    i.run()
  }
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setActive((a) => Math.min(items.length - 1, a + 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActive((a) => Math.max(0, a - 1))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      if (items[active]) run(items[active])
    } else if (e.key === 'Escape') {
      setOpen(false)
    }
  }

  let lastGroup: Item['group'] | null = null
  return (
    <div className="palette-backdrop" onMouseDown={() => setOpen(false)}>
      <div className="palette" onMouseDown={(e) => e.stopPropagation()} role="dialog" data-testid="command-palette">
        <div className="palette-input">
          <Icon name="search" size={16} className="muted" />
          <input ref={inputRef} value={query} placeholder={t('palette.placeholder')} onChange={(e) => setQuery(e.target.value)} onKeyDown={onKey} />
          <span className="kbd">Esc</span>
        </div>
        <div className="palette-list">
          {items.length === 0 ? <div className="empty-note">{t('palette.noResults')}</div> : null}
          {items.map((i, idx) => {
            const header = i.group !== lastGroup ? <div className="menu-label">{t(i.group === 'files' ? 'palette.files' : i.group === 'tabs' ? 'palette.tabs' : 'palette.commands')}</div> : null
            lastGroup = i.group
            return (
              <div key={i.id}>
                {header}
                <button className={`palette-item${idx === active ? ' active' : ''}`} onMouseEnter={() => setActive(idx)} onClick={() => run(i)}>
                  <Icon name={i.icon} size={15} className="muted" />
                  <span className="truncate">{i.label}</span>
                  {i.hint ? <span className="hint truncate">{i.hint}</span> : null}
                </button>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
