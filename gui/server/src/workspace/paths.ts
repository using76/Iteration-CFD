// The one place a path-shaped input becomes an absolute path. Every tool,
// REST route and dataset open goes through resolveInWorkspace(): the result is
// guaranteed to be inside the workspace root even in the presence of `..`,
// symlinks and absolute paths.
import fs from 'node:fs'
import path from 'node:path'

export type WorkspaceErrorCode = 'OUTSIDE_WORKSPACE' | 'NOT_FOUND' | 'INVALID'

export class WorkspaceError extends Error {
  constructor(
    public readonly code: WorkspaceErrorCode,
    message: string,
  ) {
    super(message)
    this.name = 'WorkspaceError'
  }
}

export interface ResolvedPath {
  /** Absolute, normalised path. */
  abs: string
  /** Workspace-relative path with forward slashes ('' for the root itself). */
  rel: string
  exists: boolean
}

function realpathOfNearestExisting(abs: string): string {
  // Walk up until a path exists, realpath it, then re-append the missing tail.
  let cur = abs
  const tail: string[] = []
  for (;;) {
    try {
      const real = fs.realpathSync.native(cur)
      return tail.length ? path.join(real, ...tail.reverse()) : real
    } catch {
      const parent = path.dirname(cur)
      if (parent === cur) return abs
      tail.push(path.basename(cur))
      cur = parent
    }
  }
}

export function toWorkspaceRel(root: string, abs: string): string {
  const rel = path.relative(root, abs).split(path.sep).join('/')
  return rel === '.' ? '' : rel
}

/**
 * Resolve `input` (relative to the workspace root, or absolute) and refuse
 * anything that escapes the root after symlink resolution.
 */
export function resolveInWorkspace(root: string, input: string, opts: { mustExist?: boolean } = {}): ResolvedPath {
  if (typeof input !== 'string') throw new WorkspaceError('INVALID', 'path must be a string')
  const trimmed = input.trim()
  if (trimmed.includes('\0')) throw new WorkspaceError('INVALID', 'path contains NUL')
  const rootReal = realpathOfNearestExisting(path.resolve(root))
  const candidate = path.isAbsolute(trimmed) ? path.resolve(trimmed) : path.resolve(rootReal, trimmed)
  const real = realpathOfNearestExisting(candidate)
  const relRaw = path.relative(rootReal, real)
  if (relRaw.startsWith('..') || path.isAbsolute(relRaw)) {
    throw new WorkspaceError('OUTSIDE_WORKSPACE', `path is outside the workspace: ${input}`)
  }
  const exists = fs.existsSync(real)
  if (opts.mustExist && !exists) throw new WorkspaceError('NOT_FOUND', `no such file or directory: ${input}`)
  return { abs: real, rel: toWorkspaceRel(rootReal, real), exists }
}

/** Directories the explorer and search never descend into. Explicit list, not .gitignore. */
export const HIDDEN_DIRS = new Set(['.git', 'node_modules', 'target', 'reference', '.cache', 'dist', '__pycache__', '.vite'])

export function isHiddenDir(name: string, rel: string): boolean {
  if (HIDDEN_DIRS.has(name)) return true
  if (rel === 'rust/target' || rel.startsWith('rust/target/')) return true
  return false
}
