// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The project tree (STU-08b): Geometry -> Mesh -> Physics -> Boundaries ->
// Solver -> Run -> Results, with each step's children read from the open case
// (projectTreeModel.ts builds it). Pressing a node does what gui_control select_step
// does: the node becomes the activeStep, its keys fill Properties and that
// panel opens; the Boundaries step and its patches also open the Boundary editor.
import { useEffect, useMemo, useState, type ReactElement } from 'react'
import type { Locale } from '@cfd/shared'
import { api } from '../../api/rest'
import { tx } from '../../i18n/extra'
import { useEditorStore } from '../../state/editorStore'
import { fsEvents } from '../../state/fsEvents'
import { useSessionStore } from '../../state/sessionStore'
import { selectActiveFile, useUiStore } from '../../state/uiStore'
import { activeCasePath } from '../../ws/actions'
import { Icon, type IconName } from '../common/Icon'
import { buildProjectTree, parseCaseText, selectProjectNode, type ProjectNode, type StepId } from './projectTreeModel'

const STEP_ICON: Record<StepId, IconName> = { geometry: 'cube', mesh: 'mesh', physics: 'zap', boundaries: 'blocks', solver: 'gear', run: 'play', results: 'post' }

export interface ProjectTreeViewProps {
  locale: Locale
  /** The case the tree was read from; null when no case is open. */
  path: string | null
  nodes: ProjectNode[]
  activeStep: string | null
  /** Steps whose children are folded away. */
  collapsed: ReadonlySet<string>
  onToggle(id: string): void
  onSelect(node: ProjectNode): void
}

/** The tree as host elements only (no hooks), so a test can walk it and press a row. */
export function ProjectTreeView({ locale, path, nodes, activeStep, collapsed, onToggle, onSelect }: ProjectTreeViewProps): ReactElement {
  const row = (n: ProjectNode, depth: number, icon: IconName | null): ReactElement => (
    <div
      key={n.id}
      className={`outline-row${n.id === activeStep ? ' selected' : ''}`}
      style={{ paddingLeft: 6 + depth * 14 }}
      data-step={n.id}
      title={n.id}
      onClick={() => onSelect(n)}
    >
      {icon !== null ? (
        <span
          data-toggle={n.id}
          onClick={(e) => {
            e.stopPropagation()
            onToggle(n.id)
          }}
        >
          <Icon name={n.children.length === 0 ? 'more' : collapsed.has(n.id) ? 'chevronRight' : 'chevronDown'} size={12} className="muted" />
        </span>
      ) : null}
      {icon !== null ? <Icon name={icon} size={13} className="muted" /> : null}
      <span className="truncate">{n.label}</span>
      {n.children.length > 0 ? <span className="kind">{n.children.length}</span> : null}
    </div>
  )
  return (
    <div data-testid="project-tree" title={path ?? undefined}>
      {path === null ? <div className="empty-note">{tx(locale, 'tree.noCase')}</div> : null}
      {nodes.map((n) => (
        <div key={n.id}>
          {row(n, 0, STEP_ICON[n.id as StepId] ?? 'list')}
          {collapsed.has(n.id) ? null : n.children.map((c) => row(c, 1, null))}
        </div>
      ))}
    </div>
  )
}

export function ProjectTree(): ReactElement {
  const locale = useUiStore((s) => s.locale)
  const activeStep = useUiStore((s) => s.activeStep)
  // activeCasePath reads these stores; subscribing re-renders the tree when they change.
  useUiStore(selectActiveFile)
  useUiStore((s) => s.activeRunId)
  useSessionStore((s) => s.runs)
  const path = activeCasePath()
  const buffer = useEditorStore((s) => (path !== null ? s.buffers[path] : undefined))
  const bufferText = buffer && !buffer.loading ? buffer.content : null
  const [diskText, setDiskText] = useState<string | null>(null)
  const [tick, setTick] = useState(0)
  const [collapsed, setCollapsed] = useState<ReadonlySet<string>>(new Set())

  const hasBuffer = bufferText !== null
  // A different case starts with no disk text.
  useEffect(() => setDiskText(null), [path])
  // Without an open buffer the tree reads the case from disk, again after a write to it.
  useEffect(() => {
    if (path === null || hasBuffer) return
    let live = true
    api.readFile(path).then(
      (r) => {
        if (live) setDiskText(r.content)
      },
      () => {},
    )
    return () => {
      live = false
    }
  }, [path, hasBuffer, tick])
  useEffect(() => {
    if (path === null) return
    return fsEvents.subscribe((paths) => {
      if (paths.includes(path)) setTick((n) => n + 1)
    })
  }, [path])

  const text = bufferText ?? diskText
  const nodes = useMemo(() => buildProjectTree(text === null ? null : parseCaseText(text), locale === 'ko' ? 'ko' : 'en'), [text, locale])
  return (
    <ProjectTreeView
      locale={locale}
      path={path}
      nodes={nodes}
      activeStep={activeStep}
      collapsed={collapsed}
      onToggle={(id) =>
        setCollapsed((s) => {
          const next = new Set(s)
          if (next.has(id)) next.delete(id)
          else next.add(id)
          return next
        })
      }
      onSelect={(n) => selectProjectNode(useUiStore.getState(), n, path)}
    />
  )
}
