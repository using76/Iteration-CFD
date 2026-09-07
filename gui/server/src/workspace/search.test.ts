import fs from 'node:fs'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { globToRegExp, searchWorkspace } from './search.js'
import { makeTempWorkspace, type TempWorkspace } from '../runs/test-helpers.js'

let ws: TempWorkspace
beforeAll(async () => {
  ws = await makeTempWorkspace()
  const mk = (p: string, content: string) => {
    fs.mkdirSync(path.dirname(path.join(ws.root, p)), { recursive: true })
    fs.writeFileSync(path.join(ws.root, p), content)
  }
  mk('rust/src/bin/k_epsilon.rs', 'fn usage() {\n  eprintln!("usage: ofgpu-k-epsilon");\n}\nlet kEpsilon = 1;\n')
  mk('rust/target/release/junk.rs', 'kEpsilon everywhere')
  mk('node_modules/dep/index.js', 'kEpsilon in deps')
  mk('docs/notes.md', 'The kEpsilon model\nand the kOmega model\n')
  mk('bin.dat', 'kEpsilon\0binary')
})
afterAll(() => ws.cleanup())

describe('globToRegExp', () => {
  it('handles **, *, ? and {a,b}', () => {
    expect(globToRegExp('**/*.rs').test('rust/src/bin/k_epsilon.rs')).toBe(true)
    expect(globToRegExp('**/*.rs').test('main.rs')).toBe(true)
    expect(globToRegExp('*.md').test('docs/notes.md')).toBe(false)
    expect(globToRegExp('*.md').test('notes.md')).toBe(true)
    expect(globToRegExp('cases/**').test('cases/a/b.jsonc')).toBe(true)
    expect(globToRegExp('*.{rs,md}').test('x.md')).toBe(true)
    expect(globToRegExp('k_?psilon.rs').test('k_epsilon.rs')).toBe(true)
  })
})

describe('searchWorkspace', () => {
  it('finds plain text hits with line/col, skipping hidden dirs and binary files', async () => {
    const r = await searchWorkspace(ws.root, { q: 'kEpsilon' })
    const paths = r.hits.map((h) => h.path)
    expect(paths).toContain('rust/src/bin/k_epsilon.rs')
    expect(paths).toContain('docs/notes.md')
    expect(paths).not.toContain('rust/target/release/junk.rs')
    expect(paths).not.toContain('node_modules/dep/index.js')
    expect(paths).not.toContain('bin.dat')
    expect(r.hits.find((h) => h.path === 'docs/notes.md')).toMatchObject({ line: 1, col: 5, text: 'The kEpsilon model' })
    expect(r.truncated).toBe(false)
  })
  it('honours glob, regex, case and max', async () => {
    const md = await searchWorkspace(ws.root, { q: 'k(Epsilon|Omega)', regex: true, glob: '*.md' })
    expect(md.hits.map((h) => h.line)).toEqual([1, 2])
    const cased = await searchWorkspace(ws.root, { q: 'kepsilon', caseSensitive: true })
    expect(cased.hits).toHaveLength(0)
    const capped = await searchWorkspace(ws.root, { q: 'model', max: 1 })
    expect(capped.hits).toHaveLength(1)
    expect(capped.truncated).toBe(true)
    const under = await searchWorkspace(ws.root, { q: 'kEpsilon', dir: 'docs' })
    expect(under.hits.every((h) => h.path.startsWith('docs/'))).toBe(true)
    await expect(searchWorkspace(ws.root, { q: '(', regex: true })).rejects.toThrow(/invalid regular expression/)
    expect(await searchWorkspace(ws.root, { q: '   ' })).toEqual({ hits: [], truncated: false, filesScanned: 0 })
  })
})
