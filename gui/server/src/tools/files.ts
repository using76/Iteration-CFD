// Workspace file tools. Every path goes through resolveInWorkspace; case
// files are edited with case_edit, never rewritten with file_write.
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import { z } from 'zod'
import { fail, okResult, type ToolDef } from './context.js'
import { globMatcher, isHiddenDir, resolveTool } from './paths.js'

export const TEXT_CAP = 32 * 1024
/** Same ceiling the REST route uses (workspace/fs.ts MAX_FILE_BYTES). */
export const READ_FILE_CAP = 8 * 1024 * 1024
const SEARCH_FILE_CAP = 2 * 1024 * 1024
const MAX_HITS = 500

/** The first 8 KB is all looksBinary() ever examined; reading it alone keeps a huge file out of memory. */
async function readHead(abs: string, bytes: number): Promise<Buffer> {
  const head = Buffer.alloc(bytes)
  const fh = await fsp.open(abs, 'r')
  try {
    const { bytesRead } = await fh.read(head, 0, bytes, 0)
    return head.subarray(0, bytesRead)
  } finally {
    await fh.close()
  }
}

function looksBinary(buf: Buffer): boolean {
  const n = Math.min(buf.length, 8192)
  for (let i = 0; i < n; i++) if (buf[i] === 0) return true
  return false
}

const ReadSchema = z.object({
  path: z.string().describe('Workspace-relative file path'),
  startLine: z.number().int().min(1).nullable().describe('First line to return (1-based); null = 1'),
  endLine: z.number().int().min(1).nullable().describe('Last line to return (inclusive); null = to the end'),
})

export const fileRead: ToolDef<typeof ReadSchema> = {
  name: 'file_read',
  description: 'Read a text file from the workspace with line numbers (at most 32 KB per call; use startLine/endLine to page). For case .jsonc files prefer case_read, which also summarises them.',
  schema: ReadSchema,
  async run(input, ctx) {
    const r = resolveTool(ctx.workspaceRoot, input.path, { mustExist: true })
    if (!r.ok) return r.result
    const st = await fsp.stat(r.path.abs)
    if (!st.isFile()) return fail('NOT_A_FILE', `${r.path.rel} is a directory; use file_list`)
    // A result file can be gigabytes. The old order read the whole thing into
    // memory and only then asked how big it was.
    if (looksBinary(await readHead(r.path.abs, Math.min(8192, st.size)))) return fail('BINARY', `${r.path.rel} is a binary file (${st.size} bytes)`)
    if (st.size > READ_FILE_CAP) return fail('TOO_LARGE', `${r.path.rel} is ${st.size} bytes; file_read handles text files up to ${READ_FILE_CAP} bytes. Use file_search to find the lines you want, or the results tools for solver output.`)
    const buf = await fsp.readFile(r.path.abs)
    const lines = buf.toString('utf8').split('\n')
    if (lines[lines.length - 1] === '') lines.pop()
    const start = Math.max(1, input.startLine ?? 1)
    const end = Math.min(lines.length, input.endLine ?? lines.length)
    const out: string[] = []
    let bytes = 0
    let last = start - 1
    for (let n = start; n <= end; n++) {
      const line = `${n}: ${lines[n - 1]}`
      if (bytes + line.length + 1 > TEXT_CAP) break
      out.push(line)
      bytes += line.length + 1
      last = n
    }
    return okResult({ path: r.path.rel, startLine: start, endLine: last, totalLines: lines.length, truncated: last < end, text: out.join('\n') })
  },
}

const ListSchema = z.object({
  dir: z.string().describe('Workspace-relative directory ("" or "." for the root)'),
  glob: z.string().nullable().describe('Optional glob such as "*.jsonc" or "**/controlDict"'),
})

export const fileList: ToolDef<typeof ListSchema> = {
  name: 'file_list',
  description: 'List a workspace directory (name, kind, size). With a glob containing "**" the listing recurses (capped at 2000 entries).',
  schema: ListSchema,
  async run(input, ctx) {
    const r = resolveTool(ctx.workspaceRoot, input.dir || '.', { mustExist: true })
    if (!r.ok) return r.result
    const st = await fsp.stat(r.path.abs)
    if (!st.isDirectory()) return fail('NOT_A_DIR', `${r.path.rel} is a file; use file_read`)
    const match = globMatcher(input.glob)
    const recurse = Boolean(input.glob && input.glob.includes('**'))
    const entries: Array<{ path: string; kind: 'dir' | 'file'; size: number | null }> = []
    let truncated = false
    const walk = async (abs: string, rel: string, depth: number): Promise<void> => {
      let names: fs.Dirent[]
      try {
        names = await fsp.readdir(abs, { withFileTypes: true })
      } catch {
        return
      }
      names.sort((a, b) => a.name.localeCompare(b.name))
      for (const d of names) {
        if (entries.length >= 2000) {
          truncated = true
          return
        }
        const childRel = rel ? `${rel}/${d.name}` : d.name
        const relToDir = childRel.slice(r.path.rel ? r.path.rel.length + 1 : 0)
        if (d.isDirectory()) {
          if (isHiddenDir(d.name, childRel)) continue
          if (match(relToDir) || recurse) entries.push({ path: childRel, kind: 'dir', size: null })
          if (recurse && depth < 8) await walk(path.join(abs, d.name), childRel, depth + 1)
        } else if (d.isFile() && match(relToDir)) {
          let size: number | null = null
          try {
            size = (await fsp.stat(path.join(abs, d.name))).size
          } catch {
            // unreadable entry
          }
          entries.push({ path: childRel, kind: 'file', size })
        }
      }
    }
    await walk(r.path.abs, r.path.rel, 0)
    return okResult({ dir: r.path.rel || '.', entries, truncated })
  },
}

const SearchSchema = z.object({
  pattern: z.string().min(1).describe('Text (or regex when regex=true) to search for, case-insensitive'),
  glob: z.string().nullable().describe('Restrict to files matching this glob, e.g. "rust/src/**/*.rs"'),
  maxHits: z.number().int().min(1).max(MAX_HITS).nullable(),
  regex: z.boolean().nullable(),
})

export interface SearchHit {
  path: string
  line: number
  text: string
}

export async function searchFiles(root: string, opts: { pattern: string; glob: string | null; maxHits: number; regex: boolean; signal?: AbortSignal }): Promise<{ hits: SearchHit[]; truncated: boolean; filesScanned: number }> {
  const source = opts.regex ? opts.pattern : opts.pattern.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const re = new RegExp(source, 'i')
  const match = globMatcher(opts.glob)
  const hits: SearchHit[] = []
  let filesScanned = 0
  let truncated = false
  const walk = async (abs: string, rel: string): Promise<void> => {
    if (truncated || opts.signal?.aborted) return
    let names: fs.Dirent[]
    try {
      names = await fsp.readdir(abs, { withFileTypes: true })
    } catch {
      return
    }
    names.sort((a, b) => a.name.localeCompare(b.name))
    for (const d of names) {
      if (truncated) return
      const childRel = rel ? `${rel}/${d.name}` : d.name
      const childAbs = path.join(abs, d.name)
      if (d.isDirectory()) {
        if (!isHiddenDir(d.name, childRel)) await walk(childAbs, childRel)
        continue
      }
      if (!d.isFile() || !match(childRel)) continue
      let buf: Buffer
      try {
        const st = await fsp.stat(childAbs)
        if (st.size === 0 || st.size > SEARCH_FILE_CAP) continue
        buf = await fsp.readFile(childAbs)
      } catch {
        continue
      }
      if (looksBinary(buf)) continue
      filesScanned++
      const lines = buf.toString('utf8').split('\n')
      for (let i = 0; i < lines.length; i++) {
        if (!re.test(lines[i])) continue
        hits.push({ path: childRel, line: i + 1, text: lines[i].trim().slice(0, 300) })
        if (hits.length >= opts.maxHits) {
          truncated = true
          break
        }
      }
    }
  }
  await walk(root, '')
  return { hits, truncated, filesScanned }
}

export const fileSearch: ToolDef<typeof SearchSchema> = {
  name: 'file_search',
  description: 'Search the workspace text files for a phrase or regex (case-insensitive) and return path/line hits. Skips .git, node_modules, rust/target and binaries.',
  schema: SearchSchema,
  async run(input, ctx) {
    if (input.regex) {
      try {
        new RegExp(input.pattern)
      } catch (err) {
        return fail('INVALID', `bad regex: ${(err as Error).message}`)
      }
    }
    const res = await searchFiles(ctx.workspaceRoot, { pattern: input.pattern, glob: input.glob, maxHits: input.maxHits ?? 100, regex: input.regex ?? false, signal: ctx.signal })
    return okResult({ pattern: input.pattern, ...res })
  },
}

const WriteSchema = z.object({
  path: z.string().describe('Workspace-relative file path'),
  content: z.string(),
  createOnly: z.boolean().describe('Fail instead of overwriting an existing file'),
})

export const fileWrite: ToolDef<typeof WriteSchema> = {
  name: 'file_write',
  description: 'Create or overwrite a text file in the workspace (needs approval). Refuses .jsonc case files: edit those with case_edit so comments survive and the result is validated.',
  schema: WriteSchema,
  async run(input, ctx) {
    if (/\.jsonc$/i.test(input.path)) return fail('USE_CASE_EDIT', 'case files are edited with case_edit (JSON pointer edits, comments preserved), not rewritten')
    const r = resolveTool(ctx.workspaceRoot, input.path)
    if (!r.ok) return r.result
    if (r.path.rel === '') return fail('INVALID', 'path names the workspace root')
    const existed = r.path.exists
    if (existed && input.createOnly) return fail('EXISTS', `${r.path.rel} already exists`)
    if (existed && (await fsp.stat(r.path.abs)).isDirectory()) return fail('NOT_A_FILE', `${r.path.rel} is a directory`)
    await fsp.mkdir(path.dirname(r.path.abs), { recursive: true })
    const tmp = `${r.path.abs}.${process.pid}.tmp`
    await fsp.writeFile(tmp, input.content, 'utf8')
    await fsp.rename(tmp, r.path.abs)
    ctx.hub.broadcast({ t: 'fs.changed', paths: [r.path.rel] })
    return okResult({ path: r.path.rel, bytes: Buffer.byteLength(input.content, 'utf8'), created: !existed })
  },
}
