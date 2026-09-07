// Explorer tree, file read (with content hash) and conflict-checked atomic
// write, all confined to the workspace by resolveInWorkspace().
import { createHash } from 'node:crypto'
import fsp from 'node:fs/promises'
import path from 'node:path'
import type { FsFileResponse, FsTreeNode, FsWriteRequest } from '@cfd/shared'
import { WorkspaceError, isHiddenDir, resolveInWorkspace } from './paths.js'

export const MAX_FILE_BYTES = 8 * 1024 * 1024
export const MAX_TREE_DEPTH = 8

/** A request the file API refuses for a non-path reason (size, binary content, conflicts). */
export class FsRequestError extends Error {
  constructor(
    readonly status: number,
    message: string,
    readonly extra: Record<string, unknown> = {},
  ) {
    super(message)
    this.name = 'FsRequestError'
  }
}

export function sha1(text: string | Buffer): string {
  return createHash('sha1').update(text).digest('hex')
}

export function isBinary(buf: Buffer): boolean {
  const n = Math.min(buf.length, 8192)
  for (let i = 0; i < n; i++) if (buf[i] === 0) return true
  return false
}

type Tag = FsTreeNode['tags'][number]
const TIME_DIR = /^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$/
const CASE_FILES = new Set(['controlDict', 'fvSchemes', 'fvSolution'])

export function isTimeDirName(name: string): boolean {
  return TIME_DIR.test(name)
}

function fileTags(name: string, rel: string): Tag[] {
  const tags: Tag[] = []
  const ext = path.extname(name).toLowerCase()
  if (ext === '.jsonc') tags.push('jsonc', 'case')
  else if (ext === '.json' && (rel.startsWith('cases/') || /case/i.test(name))) tags.push('jsonc', 'case')
  if (ext === '.vtu' || ext === '.pvd' || ext === '.vtp') tags.push('vtk')
  if (ext === '.rs' || name === 'Cargo.toml') tags.push('rust')
  if (ext === '.md') tags.push('docs')
  return tags
}

function dirTags(name: string, rel: string, entries: Array<{ name: string; dir: boolean }>): Tag[] {
  const tags: Tag[] = []
  if (isTimeDirName(name)) tags.push('time')
  if (name === 'VTK') tags.push('vtk')
  const names = new Set(entries.map((e) => e.name))
  const hasTime = entries.some((e) => e.dir && isTimeDirName(e.name) && e.name !== '0')
  const hasVtk = names.has('VTK') || entries.some((e) => !e.dir && /\.(vtu|pvd)$/i.test(e.name))
  if (hasTime || hasVtk || name.endsWith('_jsonc')) tags.push('results')
  const isCase = (names.has('constant') && names.has('system')) || (names.has('0') && names.has('constant')) || (names.has('constant') && names.has('0.orig'))
  if (isCase || (name === 'system' && entries.some((e) => CASE_FILES.has(e.name)))) tags.push('case')
  if (name === 'rust' || rel === 'rust' || rel.startsWith('rust/src')) tags.push('rust')
  if (name === 'docs') tags.push('docs')
  return tags
}

function sortEntries(a: FsTreeNode, b: FsTreeNode): number {
  if (a.kind !== b.kind) return a.kind === 'dir' ? -1 : 1
  return a.name.localeCompare(b.name, 'en', { numeric: true, sensitivity: 'base' })
}

async function buildNode(root: string, abs: string, rel: string, name: string, depth: number): Promise<FsTreeNode | null> {
  let st: Awaited<ReturnType<typeof fsp.stat>>
  try {
    st = await fsp.stat(abs)
  } catch {
    return null
  }
  if (st.isDirectory()) {
    let entries: Array<{ name: string; dir: boolean }> = []
    try {
      const list = await fsp.readdir(abs, { withFileTypes: true })
      entries = list
        .filter((d) => !(d.isDirectory() && isHiddenDir(d.name, rel ? `${rel}/${d.name}` : d.name)))
        .map((d) => ({ name: d.name, dir: d.isDirectory() || d.isSymbolicLink() }))
    } catch {
      entries = []
    }
    let children: FsTreeNode[] | null = null
    if (depth > 0) {
      children = []
      for (const e of entries) {
        const childRel = rel ? `${rel}/${e.name}` : e.name
        const node = await buildNode(root, path.join(abs, e.name), childRel, e.name, depth - 1)
        if (node) children.push(node)
      }
      children.sort(sortEntries)
    }
    return { name, path: rel, kind: 'dir', size: null, mtime: st.mtimeMs, children, tags: dirTags(name, rel, entries) }
  }
  return { name, path: rel, kind: 'file', size: st.size, mtime: st.mtimeMs, children: null, tags: fileTags(name, rel) }
}

/** The tree under `relPath` expanded `depth` levels (0 = only the node itself, with tags). */
export async function fsTree(root: string, relPath: string, depth = 1): Promise<FsTreeNode> {
  const r = resolveInWorkspace(root, relPath || '.', { mustExist: true })
  const name = r.rel === '' ? path.basename(r.abs) : path.basename(r.rel)
  const node = await buildNode(root, r.abs, r.rel, name, Math.min(Math.max(depth, 0), MAX_TREE_DEPTH))
  if (!node) throw new WorkspaceError('NOT_FOUND', `no such file or directory: ${relPath}`)
  return node
}

export async function readWorkspaceFile(root: string, relPath: string): Promise<FsFileResponse> {
  const r = resolveInWorkspace(root, relPath, { mustExist: true })
  const st = await fsp.stat(r.abs)
  if (st.isDirectory()) throw new WorkspaceError('INVALID', `is a directory: ${relPath}`)
  if (st.size > MAX_FILE_BYTES) throw new FsRequestError(413, `file too large (${st.size} bytes > ${MAX_FILE_BYTES})`, { size: st.size })
  const buf = await fsp.readFile(r.abs)
  if (isBinary(buf)) throw new FsRequestError(415, `binary file: ${relPath}`, { size: st.size })
  const content = buf.toString('utf8')
  return { path: r.rel, content, hash: sha1(buf), mtime: st.mtimeMs, size: st.size }
}

/**
 * Atomic write. When `baseHash` is given and the file on disk hashes to
 * something else, refuse with 409 and the current hash so the editor can
 * reconcile.
 */
export async function writeWorkspaceFile(root: string, req: FsWriteRequest): Promise<FsFileResponse> {
  const r = resolveInWorkspace(root, req.path)
  if (r.rel === '') throw new WorkspaceError('INVALID', 'cannot write the workspace root')
  if (r.exists) {
    const st = await fsp.stat(r.abs)
    if (st.isDirectory()) throw new WorkspaceError('INVALID', `is a directory: ${req.path}`)
    if (req.baseHash !== null) {
      const current = sha1(await fsp.readFile(r.abs))
      if (current !== req.baseHash) throw new FsRequestError(409, 'file changed on disk since it was read', { currentHash: current })
    }
  }
  await fsp.mkdir(path.dirname(r.abs), { recursive: true })
  const tmp = `${r.abs}.${process.pid}.${Date.now().toString(36)}.tmp`
  await fsp.writeFile(tmp, req.content, 'utf8')
  await fsp.rename(tmp, r.abs)
  const st = await fsp.stat(r.abs)
  return { path: r.rel, content: req.content, hash: sha1(req.content), mtime: st.mtimeMs, size: st.size }
}
