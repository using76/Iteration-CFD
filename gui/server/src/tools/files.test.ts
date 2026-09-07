import fsp from 'node:fs/promises'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { fakeDatasets, fakeHub, fakeRuns, makeWorkspace, REPO_ROOT, type FakeHub, type TempWorkspace } from '../agent/test-fakes.js'
import type { ToolContext } from './context.js'
import { runTool } from './index.js'
import { globToRegExp } from './paths.js'
import { indexSpec, searchSpec, sectionText } from './spec.js'

let ws: TempWorkspace
let hub: FakeHub
beforeAll(async () => {
  ws = await makeWorkspace({ spec: true })
  hub = fakeHub()
  await fsp.mkdir(path.join(ws.root, 'src/deep'), { recursive: true })
  await fsp.writeFile(path.join(ws.root, 'src/a.rs'), 'fn main() {\n  println!("hello");\n}\n')
  await fsp.writeFile(path.join(ws.root, 'src/deep/b.rs'), '// Hello again\nfn other() {}\n')
  await fsp.writeFile(path.join(ws.root, 'notes.md'), 'hello notes\n')
  await fsp.mkdir(path.join(ws.root, 'node_modules/x'), { recursive: true })
  await fsp.writeFile(path.join(ws.root, 'node_modules/x/hidden.js'), 'hello hidden\n')
})
afterAll(() => ws.cleanup())

function ctx(root = ws.root): ToolContext {
  return { config: ws.config, hub, runs: fakeRuns(), datasets: fakeDatasets(), sessionId: 's1', signal: new AbortController().signal, workspaceRoot: root, settings: { autoApprove: 'reads', effort: 'high', notifyOnRunEnd: true, locale: 'en' }, toolUseId: 'toolu_1' }
}

describe('file tools', () => {
  it('file_read numbers lines and pages', async () => {
    const r = await runTool('file_read', { path: 'src/a.rs', startLine: 2, endLine: 2 }, ctx())
    expect(r.ok).toBe(true)
    expect((r.data as { text: string }).text).toBe('2:   println!("hello");')
    expect((r.data as { totalLines: number }).totalLines).toBe(3)
    const out = await runTool('file_read', { path: '../etc/passwd', startLine: null, endLine: null }, ctx())
    expect(out.error?.code).toBe('OUTSIDE_WORKSPACE')
  })

  it('file_read refuses a file it would have to hold in memory', async () => {
    // More than 8 KB of text at the head so the binary sniff passes, then
    // sparse out to 9 MB: the old code read the whole file into memory before
    // it ever asked how big it was.
    const big = path.join(ws.root, 'big.log')
    await fsp.writeFile(big, 'plain text header\n'.repeat(600))
    const fh = await fsp.open(big, 'r+')
    await fh.truncate(9 * 1024 * 1024)
    await fh.close()
    const r = await runTool('file_read', { path: 'big.log', startLine: null, endLine: null }, ctx())
    expect(r.error?.code).toBe('TOO_LARGE')
    expect(r.error?.message).toContain('9437184 bytes')

    await fsp.writeFile(path.join(ws.root, 'blob.bin'), Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x00, 0x01, 0x02]))
    expect((await runTool('file_read', { path: 'blob.bin', startLine: null, endLine: null }, ctx())).error?.code).toBe('BINARY')

    await fsp.rm(big)
    await fsp.rm(path.join(ws.root, 'blob.bin'))
  })

  it('file_list filters with globs and recurses on **', async () => {
    const flat = await runTool('file_list', { dir: 'src', glob: null }, ctx())
    expect((flat.data as { entries: Array<{ path: string; kind: string }> }).entries.map((e) => e.path)).toEqual(['src/a.rs', 'src/deep'])
    const rec = await runTool('file_list', { dir: '.', glob: '**/*.rs' }, ctx())
    const paths = (rec.data as { entries: Array<{ path: string; kind: string }> }).entries.filter((e) => e.kind === 'file').map((e) => e.path)
    expect(paths).toEqual(['src/a.rs', 'src/deep/b.rs'])
    expect(globToRegExp('*.rs').test('a.rs')).toBe(true)
    expect(globToRegExp('src/**/*.rs').test('src/deep/b.rs')).toBe(true)
    expect(globToRegExp('{a,b}.rs').test('b.rs')).toBe(true)
  })

  it('file_search finds hits, skips hidden dirs and honours the cap', async () => {
    const r = await runTool('file_search', { pattern: 'hello', glob: null, maxHits: null, regex: null }, ctx())
    expect(r.ok).toBe(true)
    const hits = (r.data as { hits: Array<{ path: string; line: number }> }).hits
    expect(hits.map((h) => h.path).sort()).toEqual(['notes.md', 'rust/SPEC-LIT.md', 'src/a.rs', 'src/deep/b.rs'])
    const capped = await runTool('file_search', { pattern: 'hello', glob: '*.rs', maxHits: 1, regex: null }, ctx())
    expect((capped.data as { hits: unknown[]; truncated: boolean }).hits).toHaveLength(1)
    expect((capped.data as { truncated: boolean }).truncated).toBe(true)
    const re = await runTool('file_search', { pattern: 'fn \\w+\\(', glob: null, maxHits: null, regex: true }, ctx())
    expect((re.data as { hits: unknown[] }).hits).toHaveLength(2)
    const bad = await runTool('file_search', { pattern: '(', glob: null, maxHits: null, regex: true }, ctx())
    expect(bad.error?.code).toBe('INVALID')
  })

  it('file_write creates files, refuses jsonc cases and honours createOnly', async () => {
    const r = await runTool('file_write', { path: 'out/new.txt', content: 'hi\n', createOnly: true }, ctx())
    expect(r.ok).toBe(true)
    expect(await fsp.readFile(path.join(ws.root, 'out/new.txt'), 'utf8')).toBe('hi\n')
    expect(hub.of('fs.changed').some((m) => m.paths.includes('out/new.txt'))).toBe(true)
    const again = await runTool('file_write', { path: 'out/new.txt', content: 'x', createOnly: true }, ctx())
    expect(again.error?.code).toBe('EXISTS')
    const jsonc = await runTool('file_write', { path: 'cases/plume.jsonc', content: '{}', createOnly: false }, ctx())
    expect(jsonc.error?.code).toBe('USE_CASE_EDIT')
  })
})

describe('spec_lookup', () => {
  it('indexes ## and ### headings', () => {
    const idx = indexSpec('## 1. A\nx\n### 1.1 B\ny\n## 2. C\nz\n')
    expect(idx.sections.map((s) => [s.number, s.title, s.level])).toEqual([
      ['1', 'A', 2],
      ['1.1', 'B', 3],
      ['2', 'C', 2],
    ])
    expect(sectionText(idx, '1.1', 1000)?.text).toBe('### 1.1 B\ny')
    expect(sectionText(idx, '1', 1000)?.text).toBe('## 1. A\nx\n### 1.1 B\ny')
    expect(searchSpec(idx, 'z')).toEqual([{ section: '2', title: 'C', line: 6, text: 'z' }])
  })

  it('finds section 31.1 in the fixture and searches it', async () => {
    const r = await runTool('spec_lookup', { section: '31.1', query: null, maxChars: null }, ctx())
    expect(r.ok).toBe(true)
    const data = r.data as { section: string; title: string; text: string }
    expect(data.section).toBe('31.1')
    expect(data.title).toBe('Cyclic patch pairs from a case file')
    expect(data.text).toContain('translation invariants')
    expect(data.text).not.toContain('31.2')
    const q = await runTool('spec_lookup', { section: null, query: 'translation invariants', maxChars: null }, ctx())
    expect((q.data as { hits: Array<{ section: string }> }).hits[0].section).toBe('31.1')
    const missing = await runTool('spec_lookup', { section: '99.9', query: null, maxChars: null }, ctx())
    expect(missing.error?.code).toBe('NOT_FOUND')
  })

  it('finds 31.1 in the real rust/SPEC-LIT.md', async () => {
    const r = await runTool('spec_lookup', { section: '31.1', query: null, maxChars: 500 }, ctx(REPO_ROOT))
    expect(r.ok).toBe(true)
    expect((r.data as { title: string }).title).toMatch(/Cyclic patch pairs/)
    expect((r.data as { truncated: boolean }).truncated).toBe(true)
  })
})
