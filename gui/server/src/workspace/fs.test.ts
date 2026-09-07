import fs from 'node:fs'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { fsTree, FsRequestError, isTimeDirName, readWorkspaceFile, sha1, writeWorkspaceFile } from './fs.js'
import { WorkspaceError } from './paths.js'
import { makeTempWorkspace, type TempWorkspace } from '../runs/test-helpers.js'

let ws: TempWorkspace
beforeAll(async () => {
  ws = await makeTempWorkspace()
  const r = ws.root
  const mk = (p: string, content = '') => {
    fs.mkdirSync(path.dirname(path.join(r, p)), { recursive: true })
    fs.writeFileSync(path.join(r, p), content)
  }
  mk('cases/plume_jsonc/0/U', 'FoamFile')
  mk('cases/plume_jsonc/1/k', 'FoamFile')
  mk('cases/plume_jsonc/VTK/kEpsilon.pvd', '<VTKFile/>')
  mk('cases/ch/constant/polyMesh/points', '0()')
  mk('cases/ch/system/controlDict', 'endTime 1;')
  mk('cases/ch/0/U', 'FoamFile')
  mk('rust/src/lib.rs', 'fn main() {}')
  mk('rust/target/release/ofgpu-k-epsilon', 'ELF')
  mk('node_modules/x/index.js', '')
  mk('.git/HEAD', 'ref: refs/heads/main')
  mk('docs/README.md', '# hi')
  mk('bin.dat', 'abc\0def')
  fs.writeFileSync(path.join(r, 'big.txt'), Buffer.alloc(9 * 1024 * 1024, 65))
})
afterAll(() => ws.cleanup())

describe('fsTree', () => {
  it('hides the hide-list and tags case/results/time/vtk/jsonc/rust/docs nodes', async () => {
    const tree = await fsTree(ws.root, '', 3)
    const names = tree.children!.map((c) => c.name)
    expect(names).not.toContain('node_modules')
    expect(names).not.toContain('.git')
    expect(names).toContain('cases')
    const rust = tree.children!.find((c) => c.name === 'rust')!
    expect(rust.tags).toContain('rust')
    expect(rust.children!.map((c) => c.name)).toEqual(['src'])
    const cases = tree.children!.find((c) => c.name === 'cases')!
    const results = cases.children!.find((c) => c.name === 'plume_jsonc')!
    expect(results.tags).toContain('results')
    expect(results.children!.find((c) => c.name === '1')!.tags).toContain('time')
    expect(results.children!.find((c) => c.name === 'VTK')!.tags).toContain('vtk')
    expect(cases.children!.find((c) => c.name === 'plume.jsonc')!.tags).toEqual(['jsonc', 'case'])
    const ch = cases.children!.find((c) => c.name === 'ch')!
    expect(ch.tags).toContain('case')
    expect(tree.children!.find((c) => c.name === 'docs')!.tags).toContain('docs')
    expect(tree.children!.find((c) => c.name === 'docs')!.children![0].tags).toContain('docs')
    expect(tree.children![0].kind).toBe('dir')
  })
  it('computes tags at the depth limit without expanding children', async () => {
    const node = await fsTree(ws.root, 'cases/plume_jsonc', 0)
    expect(node.children).toBeNull()
    expect(node.tags).toContain('results')
    expect(node.path).toBe('cases/plume_jsonc')
  })
  it('refuses paths outside the workspace and missing paths', async () => {
    await expect(fsTree(ws.root, '../', 1)).rejects.toThrow(WorkspaceError)
    await expect(fsTree(ws.root, 'nope', 1)).rejects.toMatchObject({ code: 'NOT_FOUND' })
  })
  it('recognises time directory names', () => {
    for (const n of ['0', '1', '0.5', '1e-05', '4000', '-1']) expect(isTimeDirName(n), n).toBe(true)
    for (const n of ['VTK', 'constant', '0.orig', 'results']) expect(isTimeDirName(n), n).toBe(false)
  })
})

describe('read/write', () => {
  it('reads a file with its sha1 and refuses binary and oversized files', async () => {
    const f = await readWorkspaceFile(ws.root, 'cases/plume.jsonc')
    expect(f.path).toBe('cases/plume.jsonc')
    expect(f.hash).toBe(sha1(f.content))
    expect(f.size).toBeGreaterThan(100)
    await expect(readWorkspaceFile(ws.root, 'bin.dat')).rejects.toMatchObject({ status: 415 })
    await expect(readWorkspaceFile(ws.root, 'big.txt')).rejects.toMatchObject({ status: 413 })
    await expect(readWorkspaceFile(ws.root, 'cases')).rejects.toMatchObject({ code: 'INVALID' })
    await expect(readWorkspaceFile(ws.root, '../../etc/hostname')).rejects.toMatchObject({ code: 'OUTSIDE_WORKSPACE' })
  })
  it('writes atomically, honours baseHash and reports conflicts with the current hash', async () => {
    const created = await writeWorkspaceFile(ws.root, { path: 'cases/new/case.jsonc', content: '{ "a": 1 }', baseHash: null })
    expect(created.hash).toBe(sha1('{ "a": 1 }'))
    expect(fs.readFileSync(path.join(ws.root, 'cases/new/case.jsonc'), 'utf8')).toBe('{ "a": 1 }')
    const ok = await writeWorkspaceFile(ws.root, { path: 'cases/new/case.jsonc', content: '{ "a": 2 }', baseHash: created.hash })
    expect(ok.content).toBe('{ "a": 2 }')
    let err: unknown
    try {
      await writeWorkspaceFile(ws.root, { path: 'cases/new/case.jsonc', content: '{ "a": 3 }', baseHash: created.hash })
    } catch (e) {
      err = e
    }
    expect(err).toBeInstanceOf(FsRequestError)
    expect((err as FsRequestError).status).toBe(409)
    expect((err as FsRequestError).extra.currentHash).toBe(ok.hash)
    expect(fs.readFileSync(path.join(ws.root, 'cases/new/case.jsonc'), 'utf8')).toBe('{ "a": 2 }')
    expect(fs.readdirSync(path.join(ws.root, 'cases/new')).filter((n) => n.endsWith('.tmp'))).toEqual([])
    await expect(writeWorkspaceFile(ws.root, { path: '../escape.txt', content: 'x', baseHash: null })).rejects.toMatchObject({ code: 'OUTSIDE_WORKSPACE' })
  })
})
