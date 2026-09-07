// Streamed line search over the text files of the workspace, honouring the
// explorer hide-list, a glob filter and a hit cap.
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import readline from 'node:readline'
import type { FsSearchHit } from '@cfd/shared'
import { clampLine, compileUserRegex, UnsafeRegexError } from '../regex.js'
import { isBinary } from './fs.js'
import { isHiddenDir, resolveInWorkspace } from './paths.js'

export interface SearchOptions {
  q: string
  glob?: string | null
  max?: number
  regex?: boolean
  caseSensitive?: boolean
  /** Workspace-relative directory to search under (default: the root). */
  dir?: string | null
  maxFileBytes?: number
}

export interface SearchResult {
  hits: FsSearchHit[]
  truncated: boolean
  filesScanned: number
}

/** Minimal glob -> RegExp: `**`, `*`, `?`, `{a,b}`; matched against the relative path (and the basename when the glob has no slash). */
export function globToRegExp(glob: string): RegExp {
  let re = ''
  for (let i = 0; i < glob.length; i++) {
    const ch = glob[i]
    if (ch === '*') {
      if (glob[i + 1] === '*') {
        i++
        if (glob[i + 1] === '/') {
          i++
          re += '(?:.*/)?'
        } else re += '.*'
      } else re += '[^/]*'
    } else if (ch === '?') re += '[^/]'
    else if (ch === '{') {
      const end = glob.indexOf('}', i)
      if (end < 0) re += '\\{'
      else {
        re += `(?:${glob
          .slice(i + 1, end)
          .split(',')
          .map((s) => s.replace(/[.+^$()|[\]\\]/g, '\\$&').replace(/\*/g, '[^/]*'))
          .join('|')})`
        i = end
      }
    } else re += ch.replace(/[.+^$()|[\]\\]/g, '\\$&')
  }
  return new RegExp(`^${re}$`)
}

function matcher(opts: SearchOptions): RegExp {
  const flags = opts.caseSensitive ? 'g' : 'gi'
  if (opts.regex) {
    try {
      return compileUserRegex(opts.q, flags)
    } catch (err) {
      if (err instanceof UnsafeRegexError) throw err
      throw new Error(`invalid regular expression: ${(err as Error).message}`)
    }
  }
  return new RegExp(opts.q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), flags)
}

async function looksBinary(abs: string): Promise<boolean> {
  const fh = await fsp.open(abs, 'r')
  try {
    const buf = Buffer.alloc(8192)
    const { bytesRead } = await fh.read(buf, 0, 8192, 0)
    return isBinary(buf.subarray(0, bytesRead))
  } finally {
    await fh.close()
  }
}

async function* walk(root: string, rel: string): AsyncGenerator<{ abs: string; rel: string; size: number }> {
  const abs = rel ? path.join(root, rel) : root
  let entries: fs.Dirent[]
  try {
    entries = await fsp.readdir(abs, { withFileTypes: true })
  } catch {
    return
  }
  entries.sort((a, b) => a.name.localeCompare(b.name))
  for (const e of entries) {
    const childRel = rel ? `${rel}/${e.name}` : e.name
    if (e.isDirectory()) {
      if (isHiddenDir(e.name, childRel)) continue
      yield* walk(root, childRel)
    } else if (e.isFile()) {
      let size = 0
      try {
        size = (await fsp.stat(path.join(abs, e.name))).size
      } catch {
        continue
      }
      yield { abs: path.join(abs, e.name), rel: childRel, size }
    }
  }
}

function scanFile(abs: string, rel: string, re: RegExp, hits: FsSearchHit[], max: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const stream = fs.createReadStream(abs, { encoding: 'utf8' })
    const rl = readline.createInterface({ input: stream, crlfDelay: Infinity })
    let lineNo = 0
    let done = false
    const finish = () => {
      if (done) return
      done = true
      rl.close()
      stream.destroy()
      resolve()
    }
    rl.on('line', (text) => {
      lineNo++
      if (done) return
      re.lastIndex = 0
      const m = re.exec(clampLine(text))
      if (!m) return
      hits.push({ path: rel, line: lineNo, col: m.index + 1, text: text.length > 400 ? `${text.slice(0, 400)}…` : text })
      if (hits.length >= max) finish()
    })
    rl.on('close', finish)
    stream.on('error', (err) => {
      done = true
      reject(err)
    })
  })
}

export async function searchWorkspace(root: string, opts: SearchOptions): Promise<SearchResult> {
  const q = opts.q ?? ''
  if (!q.trim()) return { hits: [], truncated: false, filesScanned: 0 }
  const max = Math.max(1, Math.min(opts.max ?? 200, 5000))
  const maxBytes = opts.maxFileBytes ?? 2 * 1024 * 1024
  const re = matcher(opts)
  const glob = opts.glob ? globToRegExp(opts.glob) : null
  const globHasSlash = Boolean(opts.glob && opts.glob.includes('/'))
  const start = opts.dir ? resolveInWorkspace(root, opts.dir, { mustExist: true }).rel : ''
  const hits: FsSearchHit[] = []
  let filesScanned = 0
  for await (const f of walk(root, start)) {
    if (glob && !(glob.test(f.rel) || (!globHasSlash && glob.test(path.basename(f.rel))))) continue
    if (f.size > maxBytes || f.size === 0) continue
    try {
      if (await looksBinary(f.abs)) continue
      filesScanned++
      await scanFile(f.abs, f.rel, re, hits, max)
    } catch {
      // unreadable file: skip it
    }
    if (hits.length >= max) return { hits, truncated: true, filesScanned }
  }
  return { hits, truncated: false, filesScanned }
}
