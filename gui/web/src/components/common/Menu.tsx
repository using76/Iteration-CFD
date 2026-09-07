// Context menu / dropdown primitives shared by the explorer, tabs and the
// assistant header.
import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { Icon, type IconName } from './Icon'

export interface MenuEntry {
  id: string
  label: string
  icon?: IconName
  danger?: boolean
  disabled?: boolean
  onSelect?: () => void
  children?: MenuEntry[]
}

export interface MenuPosition {
  x: number
  y: number
}

export function ContextMenu({ entries, position, onClose }: { entries: MenuEntry[]; position: MenuPosition; onClose: () => void }) {
  const ref = useRef<HTMLDivElement | null>(null)
  const [pos, setPos] = useState(position)
  const [openSub, setOpenSub] = useState<string | null>(null)

  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const r = el.getBoundingClientRect()
    const x = Math.min(position.x, window.innerWidth - r.width - 8)
    const y = Math.min(position.y, window.innerHeight - r.height - 8)
    setPos({ x: Math.max(4, x), y: Math.max(4, y) })
  }, [position])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      // Consumed here: the window handler would take the same Escape as a
      // request to cancel the running turn.
      e.stopPropagation()
      e.preventDefault()
      onClose()
    }
    document.addEventListener('keydown', onKey, true)
    return () => document.removeEventListener('keydown', onKey, true)
  }, [onClose])

  const run = useCallback(
    (e: MenuEntry) => {
      if (e.disabled) return
      e.onSelect?.()
      onClose()
    },
    [onClose],
  )

  return (
    <div className="overlay" onMouseDown={onClose} onContextMenu={(e) => e.preventDefault()}>
      <div ref={ref} className="popover" style={{ left: pos.x, top: pos.y }} onMouseDown={(e) => e.stopPropagation()} role="menu">
        {entries.map((e) =>
          e.children ? (
            <div key={e.id} style={{ position: 'relative' }} onMouseEnter={() => setOpenSub(e.id)}>
              <button className="menu-item" disabled={e.disabled} onClick={() => setOpenSub(openSub === e.id ? null : e.id)}>
                {e.icon ? <Icon name={e.icon} size={14} /> : null}
                <span style={{ flex: 1 }}>{e.label}</span>
                <Icon name="chevronRight" size={12} />
              </button>
              {openSub === e.id && e.children.length ? (
                <div className="popover" style={{ left: '100%', top: -6 }}>
                  {e.children.map((c) => (
                    <button key={c.id} className={`menu-item${c.danger ? ' danger' : ''}`} disabled={c.disabled} onClick={() => run(c)}>
                      {c.icon ? <Icon name={c.icon} size={14} /> : null}
                      <span>{c.label}</span>
                    </button>
                  ))}
                </div>
              ) : null}
            </div>
          ) : e.id.startsWith('sep') ? (
            <div key={e.id} className="menu-sep" />
          ) : (
            <button key={e.id} className={`menu-item${e.danger ? ' danger' : ''}`} disabled={e.disabled} onClick={() => run(e)}>
              {e.icon ? <Icon name={e.icon} size={14} /> : null}
              <span>{e.label}</span>
            </button>
          ),
        )}
      </div>
    </div>
  )
}

/** Hook: returns [menuElement, open(event, entries)] for right-click menus. */
export function useContextMenu(): [ReactNode, (e: React.MouseEvent, entries: MenuEntry[]) => void] {
  const [state, setState] = useState<{ entries: MenuEntry[]; position: MenuPosition } | null>(null)
  const open = useCallback((e: React.MouseEvent, entries: MenuEntry[]) => {
    e.preventDefault()
    e.stopPropagation()
    setState({ entries, position: { x: e.clientX, y: e.clientY } })
  }, [])
  const close = useCallback(() => setState(null), [])
  const node = state ? <ContextMenu entries={state.entries} position={state.position} onClose={close} /> : null
  return [node, open]
}
