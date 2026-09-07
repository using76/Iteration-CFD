// chokidar watcher over the workspace (hide-list applied, internal state
// directories excluded), debounced into `fs.changed` broadcasts.
import path from 'node:path'
import { watch, type FSWatcher } from 'chokidar'
import { isHiddenDir, toWorkspaceRel } from './paths.js'

export interface WatcherOptions {
  root: string
  /** Absolute directories never reported (runs, sessions, cache). */
  ignore?: string[]
  debounceMs?: number
  onChange(paths: string[]): void
  onError?(err: Error): void
}

export interface WorkspaceWatcher {
  /** Watch an extra directory (e.g. a run's output root outside the initial scan). */
  add(absDir: string): void
  close(): Promise<void>
}

export function createWorkspaceWatcher(opts: WatcherOptions): WorkspaceWatcher {
  const root = path.resolve(opts.root)
  const ignore = (opts.ignore ?? []).map((p) => path.resolve(p))
  const debounceMs = opts.debounceMs ?? 300
  const pending = new Set<string>()
  let timer: NodeJS.Timeout | null = null

  const ignored = (p: string): boolean => {
    const abs = path.resolve(p)
    if (ignore.some((i) => abs === i || abs.startsWith(i + path.sep))) return true
    const rel = toWorkspaceRel(root, abs)
    if (rel.startsWith('..')) return false
    const parts = rel.split('/').filter(Boolean)
    let sofar = ''
    for (const part of parts) {
      sofar = sofar ? `${sofar}/${part}` : part
      if (isHiddenDir(part, sofar)) return true
    }
    return false
  }

  const flush = () => {
    timer = null
    if (!pending.size) return
    const paths = [...pending].sort()
    pending.clear()
    opts.onChange(paths)
  }

  const watcher: FSWatcher = watch(root, { ignored, ignoreInitial: true, persistent: true, followSymlinks: false, ignorePermissionErrors: true })
  watcher.on('all', (_event, p) => {
    pending.add(toWorkspaceRel(root, path.resolve(p)))
    if (!timer) timer = setTimeout(flush, debounceMs)
  })
  watcher.on('error', (err) => opts.onError?.(err as Error))

  return {
    add: (absDir) => {
      if (!ignored(absDir)) watcher.add(absDir)
    },
    close: async () => {
      if (timer) clearTimeout(timer)
      await watcher.close()
    },
  }
}
