import { useState } from 'react'
import { basename, useT } from '../../app/hooks'
import { useEditorStore } from '../../state/editorStore'
import { useUiStore, type Tab } from '../../state/uiStore'
import { Icon, type IconName } from '../common/Icon'
import { useContextMenu } from '../common/Menu'

function tabIcon(tab: Tab): IconName {
  switch (tab.kind) {
    case 'viewer':
      return 'cube'
    case 'residuals':
      return 'chart'
    case 'diff':
      return 'diff'
    case 'file':
      return /\.jsonc?$/i.test(tab.path) ? 'braces' : /\.rs$/i.test(tab.path) ? 'gear' : 'file'
  }
}

export function TabBar() {
  const t = useT()
  const tabs = useUiStore((s) => s.tabs)
  const activeTabId = useUiStore((s) => s.activeTabId)
  const activateTab = useUiStore((s) => s.activateTab)
  const closeTab = useUiStore((s) => s.closeTab)
  const closeOthers = useUiStore((s) => s.closeOtherTabs)
  const moveTab = useUiStore((s) => s.moveTab)
  const setPaletteOpen = useUiStore((s) => s.setPaletteOpen)
  const toggleAssistant = useUiStore((s) => s.toggleAssistant)
  const buffers = useEditorStore((s) => s.buffers)
  const closeBuffer = useEditorStore((s) => s.closeBuffer)
  const removeDiff = useEditorStore((s) => s.removeDiff)
  const [dragIndex, setDragIndex] = useState<number | null>(null)
  const [overIndex, setOverIndex] = useState<number | null>(null)
  const [menu, openMenu] = useContextMenu()

  const close = (tab: Tab) => {
    if (tab.kind === 'file' && buffers[tab.path]?.dirty && !window.confirm(t('editor.closeDirty', { name: basename(tab.path) }))) return
    closeTab(tab.id)
    if (tab.kind === 'file') closeBuffer(tab.path)
    if (tab.kind === 'diff') removeDiff(tab.id)
  }

  const label = (tab: Tab) => (tab.kind === 'file' ? basename(tab.path) : tab.kind === 'viewer' ? t('tab.viewer') : tab.kind === 'residuals' ? t('tab.residuals') : `${t('tab.diff')}: ${basename(tab.path)}`)

  return (
    <div className="tabbar" data-testid="tabbar">
      <div className="tabbar-scroll" role="tablist">
        {tabs.map((tab, i) => {
          const dirty = tab.kind === 'file' && buffers[tab.path]?.dirty
          const testId = tab.kind === 'file' ? `tab-file-${basename(tab.path)}` : tab.kind === 'diff' ? `tab-diff` : `tab-${tab.kind}`
          return (
            <div
              key={tab.id}
              role="tab"
              aria-selected={tab.id === activeTabId}
              className={`tab${tab.id === activeTabId ? ' active' : ''}${overIndex === i && dragIndex !== null && dragIndex !== i ? ' dragover' : ''}`}
              data-testid={testId}
              data-tab-id={tab.id}
              title={tab.kind === 'file' || tab.kind === 'diff' ? tab.path : label(tab)}
              draggable
              onDragStart={() => setDragIndex(i)}
              onDragOver={(e) => {
                e.preventDefault()
                setOverIndex(i)
              }}
              onDragLeave={() => setOverIndex(null)}
              onDrop={(e) => {
                e.preventDefault()
                if (dragIndex !== null) moveTab(dragIndex, i)
                setDragIndex(null)
                setOverIndex(null)
              }}
              onDragEnd={() => {
                setDragIndex(null)
                setOverIndex(null)
              }}
              onClick={() => activateTab(tab.id)}
              onMouseDown={(e) => {
                if (e.button === 1) {
                  e.preventDefault()
                  close(tab)
                }
              }}
              onContextMenu={(e) =>
                openMenu(e, [
                  { id: 'close', label: t('tab.close'), icon: 'close', onSelect: () => close(tab) },
                  { id: 'others', label: t('tab.closeOthers'), onSelect: () => closeOthers(tab.id) },
                  ...(tab.kind === 'file' ? [{ id: 'copy', label: t('explorer.copyPath'), icon: 'copy' as IconName, onSelect: () => void navigator.clipboard?.writeText(tab.path) }] : []),
                ])
              }
            >
              <span className="tab-icon">
                <Icon name={tabIcon(tab)} size={14} />
              </span>
              <span>{label(tab)}</span>
              {dirty ? (
                <span className="tab-dirty" title="unsaved" />
              ) : (
                <button
                  className="tab-close"
                  onClick={(e) => {
                    e.stopPropagation()
                    close(tab)
                  }}
                  title={t('tab.close')}
                  aria-label={t('tab.close')}
                >
                  <Icon name="close" size={12} />
                </button>
              )}
            </div>
          )
        })}
        <button className="icon-btn" style={{ margin: '4px 4px' }} onClick={() => setPaletteOpen(true)} title={t('palette.placeholder')} data-testid="tab-add">
          <Icon name="plus" size={15} />
        </button>
      </div>
      <div className="tabbar-actions">
        <button className="icon-btn" onClick={toggleAssistant} title={t('assistant.title')}>
          <Icon name="layout" size={15} />
        </button>
      </div>
      {menu}
    </div>
  )
}
