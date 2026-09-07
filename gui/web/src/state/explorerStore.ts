// Lazily loaded workspace tree: one GET /api/fs/tree?depth=1 per expanded
// directory, refreshed on fs.changed.
import { create } from 'zustand'
import type { FsTreeNode } from '@cfd/shared'
import { api } from '../api/rest'

export interface ExplorerStore {
  /** Directory path ('' = root) -> its direct children. */
  children: Record<string, FsTreeNode[]>
  nodes: Record<string, FsTreeNode>
  loading: Record<string, boolean>
  errors: Record<string, string>
  rootName: string | null
  loadDir(path: string, force?: boolean): Promise<void>
  refreshFor(paths: string[]): void
  /** Every file path seen so far (for @-mentions and the palette). */
  knownFiles(): string[]
}

function parentOf(path: string): string {
  const i = path.lastIndexOf('/')
  return i < 0 ? '' : path.slice(0, i)
}

export const useExplorerStore = create<ExplorerStore>()((set, get) => ({
  children: {},
  nodes: {},
  loading: {},
  errors: {},
  rootName: null,
  async loadDir(path, force = false) {
    const s = get()
    if (!force && (s.children[path] || s.loading[path])) return
    set((st) => ({ loading: { ...st.loading, [path]: true } }))
    try {
      const node = await api.tree(path, 1)
      set((st) => {
        const nodes = { ...st.nodes, [node.path]: node }
        for (const c of node.children ?? []) nodes[c.path] = c
        const errors = { ...st.errors }
        delete errors[path]
        return { children: { ...st.children, [path]: node.children ?? [] }, nodes, errors, loading: { ...st.loading, [path]: false }, rootName: path === '' ? node.name : st.rootName }
      })
    } catch (err) {
      set((st) => ({ loading: { ...st.loading, [path]: false }, errors: { ...st.errors, [path]: err instanceof Error ? err.message : String(err) } }))
    }
  },
  refreshFor(paths) {
    const dirs = new Set<string>()
    const { children } = get()
    for (const p of paths) {
      const parent = parentOf(p)
      if (children[parent] !== undefined) dirs.add(parent)
      if (children[p] !== undefined) dirs.add(p)
    }
    if (!dirs.size && children[''] !== undefined) dirs.add('')
    for (const d of dirs) void get().loadDir(d, true)
  },
  knownFiles() {
    return Object.values(get().nodes)
      .filter((n) => n.kind === 'file')
      .map((n) => n.path)
      .sort()
  },
}))
