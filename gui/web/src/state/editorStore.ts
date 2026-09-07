// Open editor buffers (one per file tab) and diff-tab payloads. Monaco owns
// the live text model; this store mirrors it for dirty tracking, saving,
// conflict handling and cross-component reveal requests.
import { create } from 'zustand'
import { ApiError, api } from '../api/rest'

export interface EditorBuffer {
  path: string
  content: string
  savedContent: string
  hash: string | null
  dirty: boolean
  loading: boolean
  saving: boolean
  error: string | null
  language: string
  readOnly: boolean
  conflict: { currentHash: string } | null
  /** Set briefly after a successful save. */
  savedAt: number | null
}

export interface DiffPayload {
  path: string
  before: string
  after: string
  applied: boolean
  error: string | null
}

export interface RevealRequest {
  path: string
  line: number
  col: number
  nonce: number
}

export interface EditorStore {
  buffers: Record<string, EditorBuffer>
  diffs: Record<string, DiffPayload>
  reveal: RevealRequest | null
  loadBuffer(path: string): Promise<void>
  reloadBuffer(path: string): Promise<void>
  setContent(path: string, content: string): void
  save(path: string, opts?: { overwrite?: boolean }): Promise<boolean>
  unlock(path: string): void
  closeBuffer(path: string): void
  dismissConflict(path: string): void
  onFsChanged(paths: string[]): void
  requestReveal(path: string, line: number, col?: number): void
  setDiff(id: string, payload: Omit<DiffPayload, 'applied' | 'error'> & Partial<Pick<DiffPayload, 'applied'>>): void
  applyDiff(id: string): Promise<boolean>
  revertDiff(id: string): Promise<boolean>
  removeDiff(id: string): void
}

const READ_ONLY_EXT = new Set(['rs', 'cu', 'cuh', 'h', 'hpp', 'c', 'cpp', 'lock'])

export function languageFor(path: string): string {
  const name = path.slice(path.lastIndexOf('/') + 1)
  const ext = name.includes('.') ? name.slice(name.lastIndexOf('.') + 1).toLowerCase() : ''
  switch (ext) {
    case 'jsonc':
    case 'json':
      return 'json'
    case 'rs':
      return 'rust'
    case 'md':
      return 'markdown'
    case 'py':
      return 'python'
    case 'toml':
      return 'ini'
    case 'yaml':
    case 'yml':
      return 'yaml'
    case 'ts':
    case 'tsx':
      return 'typescript'
    case 'js':
    case 'mjs':
      return 'javascript'
    case 'cu':
    case 'cuh':
    case 'h':
    case 'hpp':
    case 'c':
    case 'cpp':
      return 'cpp'
    case 'sh':
      return 'shell'
    case 'html':
      return 'html'
    case 'css':
      return 'css'
    case 'csv':
      return 'plaintext'
    default:
      return name === 'Cargo.toml' ? 'ini' : name === 'justfile' ? 'makefile' : 'plaintext'
  }
}

export function isReadOnlyPath(path: string): boolean {
  const ext = path.slice(path.lastIndexOf('.') + 1).toLowerCase()
  return READ_ONLY_EXT.has(ext)
}

function emptyBuffer(path: string): EditorBuffer {
  return { path, content: '', savedContent: '', hash: null, dirty: false, loading: true, saving: false, error: null, language: languageFor(path), readOnly: isReadOnlyPath(path), conflict: null, savedAt: null }
}

export const useEditorStore = create<EditorStore>()((set, get) => {
  function patch(path: string, p: Partial<EditorBuffer>) {
    set((s) => {
      const b = s.buffers[path]
      if (!b) return {}
      return { buffers: { ...s.buffers, [path]: { ...b, ...p } } }
    })
  }

  async function fetchInto(path: string, keepEdits: boolean) {
    try {
      const f = await api.readFile(path)
      set((s) => {
        const b = s.buffers[path]
        if (!b) return {}
        const content = keepEdits && b.dirty ? b.content : f.content
        return { buffers: { ...s.buffers, [path]: { ...b, content, savedContent: f.content, hash: f.hash, dirty: content !== f.content, loading: false, error: null, conflict: null } } }
      })
    } catch (err) {
      patch(path, { loading: false, error: err instanceof Error ? err.message : String(err) })
    }
  }

  return {
    buffers: {},
    diffs: {},
    reveal: null,
    async loadBuffer(path) {
      if (get().buffers[path]) return
      set((s) => ({ buffers: { ...s.buffers, [path]: emptyBuffer(path) } }))
      await fetchInto(path, false)
    },
    async reloadBuffer(path) {
      if (!get().buffers[path]) return
      patch(path, { loading: true })
      await fetchInto(path, false)
    },
    setContent(path, content) {
      const b = get().buffers[path]
      if (!b || b.content === content) return
      patch(path, { content, dirty: content !== b.savedContent, savedAt: null })
    },
    async save(path, opts = {}) {
      const b = get().buffers[path]
      if (!b || b.saving) return false
      patch(path, { saving: true })
      try {
        const f = await api.writeFile(path, b.content, opts.overwrite ? null : b.hash)
        set((s) => {
          const cur = s.buffers[path]
          if (!cur) return {}
          return { buffers: { ...s.buffers, [path]: { ...cur, savedContent: f.content, hash: f.hash, dirty: cur.content !== f.content, saving: false, conflict: null, error: null, savedAt: Date.now() } } }
        })
        return true
      } catch (err) {
        if (err instanceof ApiError && err.status === 409) {
          const currentHash = typeof err.body?.currentHash === 'string' ? err.body.currentHash : ''
          patch(path, { saving: false, conflict: { currentHash } })
        } else {
          patch(path, { saving: false, error: err instanceof Error ? err.message : String(err) })
        }
        return false
      }
    },
    unlock(path) {
      patch(path, { readOnly: false })
    },
    closeBuffer(path) {
      set((s) => {
        if (!s.buffers[path]) return {}
        const buffers = { ...s.buffers }
        delete buffers[path]
        return { buffers }
      })
    },
    dismissConflict(path) {
      patch(path, { conflict: null })
    },
    onFsChanged(paths) {
      // Clean buffers follow the disk; dirty ones keep their base hash so the next save surfaces a 409 conflict.
      const { buffers } = get()
      for (const p of paths) {
        const b = buffers[p]
        if (b && !b.loading && !b.dirty && !b.saving) void fetchInto(p, false)
      }
    },
    requestReveal(path, line, col = 1) {
      set((s) => ({ reveal: { path, line, col, nonce: (s.reveal?.nonce ?? 0) + 1 } }))
    },
    setDiff(id, payload) {
      set((s) => ({ diffs: { ...s.diffs, [id]: { path: payload.path, before: payload.before, after: payload.after, applied: payload.applied ?? false, error: null } } }))
    },
    async applyDiff(id) {
      return writeDiff(id, 'after')
    },
    async revertDiff(id) {
      return writeDiff(id, 'before')
    },
    removeDiff(id) {
      set((s) => {
        const diffs = { ...s.diffs }
        delete diffs[id]
        return { diffs }
      })
    },
  }

  async function writeDiff(id: string, which: 'after' | 'before'): Promise<boolean> {
    const d = get().diffs[id]
    if (!d) return false
    try {
      let baseHash: string | null = null
      try {
        baseHash = (await api.readFile(d.path)).hash
      } catch (err) {
        if (!(err instanceof ApiError && err.status === 404)) throw err
      }
      await api.writeFile(d.path, which === 'after' ? d.after : d.before, baseHash)
      set((s) => (s.diffs[id] ? { diffs: { ...s.diffs, [id]: { ...s.diffs[id], applied: which === 'after', error: null } } } : {}))
      const buf = get().buffers[d.path]
      if (buf && !buf.dirty) void fetchInto(d.path, false)
      return true
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      set((s) => (s.diffs[id] ? { diffs: { ...s.diffs, [id]: { ...s.diffs[id], error: message } } } : {}))
      return false
    }
  }
})
