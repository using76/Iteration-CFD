// Lazy workspace tree with result/time/VTK badges, viewer shortcuts and a
// right-click menu. One GET per expanded directory.
import { useCallback, useEffect, useMemo } from 'react'
import { MESH_PRESETS, type FsTreeNode } from '@cfd/shared'
import { copyText, useT } from '../../app/hooks'
import { useExplorerStore } from '../../state/explorerStore'
import { fsEvents } from '../../state/fsEvents'
import { useMetaStore } from '../../state/metaStore'
import { useSessionStore } from '../../state/sessionStore'
import { selectActiveFile, useUiStore } from '../../state/uiStore'
import { actions } from '../../ws/actions'
import { Icon, type IconName } from '../common/Icon'
import { useContextMenu, type MenuEntry } from '../common/Menu'

export const DRAG_MIME = 'text/x-cfd-path'

function iconFor(node: FsTreeNode, expanded: boolean): { name: IconName; color?: string } {
  if (node.kind === 'dir') {
    if (node.tags.includes('time')) return { name: 'clock', color: 'var(--accent)' }
    if (node.tags.includes('vtk')) return { name: 'cube', color: 'var(--warning)' }
    return { name: expanded ? 'folderOpen' : 'folder', color: '#4f93f0' }
  }
  const ext = node.name.slice(node.name.lastIndexOf('.') + 1).toLowerCase()
  if (node.tags.includes('jsonc') || ext === 'json') return { name: 'braces', color: 'var(--accent)' }
  if (node.tags.includes('vtk')) return { name: 'cube', color: 'var(--warning)' }
  if (ext === 'rs' || node.name === 'Cargo.toml') return { name: 'gear', color: '#b5651d' }
  if (ext === 'md') return { name: 'book', color: 'var(--fg-muted)' }
  if (ext === 'py' || ext === 'ts' || ext === 'js' || ext === 'cu' || ext === 'sh') return { name: 'code', color: 'var(--fg-muted)' }
  if (node.name === '.gitignore' || node.name === 'justfile') return { name: 'file', color: 'var(--fg-faint)' }
  return { name: 'file', color: 'var(--fg-muted)' }
}

function isViewerTarget(node: FsTreeNode): boolean {
  if (node.kind === 'dir') return node.tags.includes('time') || node.tags.includes('results') || node.tags.includes('case')
  return node.tags.includes('vtk') || node.tags.includes('jsonc')
}

function caseFormat(node: FsTreeNode): 'jsonc' | 'foamDir' | null {
  if (node.kind === 'file' && node.tags.includes('jsonc')) return 'jsonc'
  if (node.kind === 'dir' && node.tags.includes('case')) return 'foamDir'
  return null
}

export function ExplorerTree() {
  const t = useT()
  const loadDir = useExplorerStore((s) => s.loadDir)
  const refreshFor = useExplorerStore((s) => s.refreshFor)
  const rootChildren = useExplorerStore((s) => s.children[''])
  const rootLoading = useExplorerStore((s) => s.loading[''])
  const rootError = useExplorerStore((s) => s.errors[''])
  const rootName = useExplorerStore((s) => s.rootName)
  const hello = useSessionStore((s) => s.hello)

  useEffect(() => {
    void loadDir('')
  }, [loadDir, hello?.workspaceRoot])
  useEffect(() => fsEvents.subscribe((paths) => refreshFor(paths)), [refreshFor])

  const [menu, openMenu] = useContextMenu()

  if (rootError) return <div className="error-note">{rootError}</div>
  if (!rootChildren) return <div className="empty-note">{rootLoading ? t('explorer.loading') : t('explorer.empty')}</div>
  return (
    <div role="tree" data-testid="explorer-tree">
      <RootRow name={rootName ?? 'workspace'} />
      {rootChildren.length === 0 ? <div className="empty-note">{t('explorer.empty')}</div> : null}
      {rootChildren.map((n) => (
        <TreeNode key={n.path} node={n} depth={1} openMenu={openMenu} />
      ))}
      {menu}
    </div>
  )
}

function RootRow({ name }: { name: string }) {
  return (
    <div className="tree-row" style={{ paddingLeft: 6, fontWeight: 600 }} data-testid="explorer-root">
      <span className="tree-chevron">
        <Icon name="chevronDown" size={13} />
      </span>
      <span className="tree-icon" style={{ color: '#4f93f0' }}>
        <Icon name="folderOpen" size={16} />
      </span>
      <span className="tree-name">{name}</span>
    </div>
  )
}

function TreeNode({ node, depth, openMenu }: { node: FsTreeNode; depth: number; openMenu: (e: React.MouseEvent, entries: MenuEntry[]) => void }) {
  const t = useT()
  const expanded = useUiStore((s) => s.explorerExpanded.includes(node.path))
  const setExpanded = useUiStore((s) => s.setExpanded)
  const openFile = useUiStore((s) => s.openFile)
  const activeFile = useUiStore(selectActiveFile)
  const children = useExplorerStore((s) => s.children[node.path])
  const loading = useExplorerStore((s) => s.loading[node.path])
  const loadDir = useExplorerStore((s) => s.loadDir)
  const registry = useMetaStore((s) => s.registry)
  const loadRegistry = useMetaStore((s) => s.loadRegistry)
  const isDir = node.kind === 'dir'

  useEffect(() => {
    if (isDir && expanded && !children) void loadDir(node.path)
  }, [isDir, expanded, children, loadDir, node.path])

  const toggle = useCallback(() => setExpanded(node.path, !expanded), [setExpanded, node.path, expanded])

  const onClick = () => {
    if (!isDir) {
      if (node.tags.includes('vtk')) actions.openInViewer(node.path)
      else openFile(node.path)
      return
    }
    if (node.tags.includes('time')) {
      actions.openInViewer(node.path)
      return
    }
    toggle()
  }

  const entries = useMemo<MenuEntry[]>(() => {
    const fmt = caseFormat(node)
    const solvers = fmt && registry ? registry.binaries.filter((b) => b.kind === 'solver' && b.accepts.includes(fmt)) : []
    const list: MenuEntry[] = []
    if (!isDir) list.push({ id: 'open', label: t('explorer.open'), icon: 'file', onSelect: () => openFile(node.path) })
    if (isViewerTarget(node)) list.push({ id: 'viewer', label: t('explorer.openViewer'), icon: 'cube', onSelect: () => actions.openInViewer(node.path) })
    if (fmt) {
      list.push({
        id: 'run',
        label: t('explorer.runWith'),
        icon: 'play',
        children: solvers.length
          ? solvers.map((b) => ({ id: `run:${b.name}`, label: b.name, icon: 'play' as IconName, onSelect: () => actions.sendUserMessage(`Run ${b.name} on @${node.path}`, [node.path]) }))
          : [{ id: 'none', label: t('explorer.noSolver'), disabled: true }],
      })
    }
    list.push({ id: 'copy', label: t('explorer.copyPath'), icon: 'copy', onSelect: () => void copyText(node.path) })
    if (isDir) {
      list.push({ id: 'sep1', label: '' })
      list.push({
        id: 'new',
        label: t('explorer.newCase'),
        icon: 'plus',
        children: MESH_PRESETS.map((p) => ({ id: `new:${p.kind}`, label: `${p.title} (${p.kind})`, icon: 'braces' as IconName, onSelect: () => actions.sendUserMessage(`Create a new case from ${p.kind} at ${node.path ? `${node.path}/` : ''}${p.kind}.jsonc`) })),
      })
      list.push({ id: 'refresh', label: t('explorer.refresh'), icon: 'refresh', onSelect: () => void loadDir(node.path, true) })
    }
    return list
  }, [node, isDir, registry, t, openFile, loadDir])

  const onContext = (e: React.MouseEvent) => {
    if (!registry) void loadRegistry()
    openMenu(e, entries)
  }

  const icon = iconFor(node, expanded)
  const selected = !isDir && activeFile === node.path
  const badge = node.tags.includes('time') ? (
    <span className="tree-badge badge-time">time</span>
  ) : node.tags.includes('results') && isDir ? (
    <span className="tree-badge badge-results">results</span>
  ) : node.tags.includes('vtk') ? (
    <span className="tree-badge badge-vtk">vtk</span>
  ) : node.tags.includes('case') && isDir ? (
    <span className="tree-badge badge-case">case</span>
  ) : null

  return (
    <>
      <div
        className={`tree-row${selected ? ' selected' : ''}`}
        style={{ paddingLeft: 6 + depth * 14 }}
        role="treeitem"
        aria-expanded={isDir ? expanded : undefined}
        data-path={node.path}
        data-testid={`tree-node`}
        draggable
        onDragStart={(e) => {
          e.dataTransfer.setData(DRAG_MIME, node.path)
          e.dataTransfer.setData('text/plain', `@${node.path}`)
          e.dataTransfer.effectAllowed = 'copy'
        }}
        onClick={onClick}
        onContextMenu={onContext}
        title={node.path}
      >
        <span
          className="tree-chevron"
          onClick={(e) => {
            if (!isDir) return
            e.stopPropagation()
            toggle()
          }}
        >
          {isDir ? <Icon name={expanded ? 'chevronDown' : 'chevronRight'} size={13} /> : null}
        </span>
        <span className="tree-icon" style={{ color: icon.color }}>
          <Icon name={icon.name} size={16} />
        </span>
        <span className="tree-name">{node.name}</span>
        {badge}
        {loading ? <span className="spinner" style={{ width: 10, height: 10 }} /> : null}
      </div>
      {isDir && expanded && children ? children.map((c) => <TreeNode key={c.path} node={c} depth={depth + 1} openMenu={openMenu} />) : null}
      {isDir && expanded && children && children.length === 0 ? (
        <div className="faint" style={{ paddingLeft: 26 + depth * 14, fontSize: 'var(--fs-xs)', height: 22, lineHeight: '22px' }}>
          {t('common.none')}
        </div>
      ) : null}
    </>
  )
}
