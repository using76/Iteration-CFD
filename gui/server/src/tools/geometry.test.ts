import fs from 'node:fs'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { summarizeToolCall } from '@cfd/shared'
import { fakeDatasets, fakeHub, fakeRuns, makeWorkspace, type FakeHub, type TempWorkspace } from '../agent/test-fakes.js'
import type { ToolContext } from './context.js'
import { geometryService } from './geometry.js'
import { runTool } from './index.js'

let ws: TempWorkspace
let hub: FakeHub
beforeAll(async () => {
  ws = await makeWorkspace()
  hub = fakeHub()
})
afterAll(() => ws.cleanup())

const ctx = (over: Partial<ToolContext> = {}): ToolContext =>
  ({ config: ws.config, hub, runs: fakeRuns(), datasets: fakeDatasets(), sessionId: 's1', signal: new AbortController().signal, workspaceRoot: ws.root, settings: { autoApprove: 'reads', effort: 'high', notifyOnRunEnd: true, locale: 'en' }, toolUseId: 'toolu_1', ...over })
const TET = 'solid tet\nfacet normal 0 0 0\nouter loop\nvertex 0 0 0\nvertex 0 1 0\nvertex 1 0 0\nendloop\nendfacet\nfacet normal 0 0 0\nouter loop\nvertex 0 0 0\nvertex 0 0 1\nvertex 0 1 0\nendloop\nendfacet\nfacet normal 0 0 0\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 0 1\nendloop\nendfacet\nfacet normal 0 0 0\nouter loop\nvertex 1 0 0\nvertex 0 1 0\nvertex 0 0 1\nendloop\nendfacet\nendsolid tet\n'
const tet2 = TET.replace(/vertex (\d) (\d) (\d)/g, (_, x, y, z) => `vertex ${x * 2} ${y * 2} ${z * 2}`)
const writeTet = (rel: string, text = TET): string => {
  const p = path.join(ws.root, rel)
  fs.mkdirSync(path.dirname(p), { recursive: true })
  fs.writeFileSync(p, text)
  return p
}
const TX1 = [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Loose = any

describe('geometry tools', () => {
  it('open_info_save_through_tools', async () => {
    writeTet('cases/tet.stl')
    const open = await runTool('geometry_open', { path: 'cases/tet.stl' }, ctx())
    expect(open.ok).toBe(true)
    const r = open.data as Loose
    expect(r.triangleCount).toBe(4)
    expect(r.closed).toBe(true)
    expect(r.blobs.positions.bytes).toBe(48)
    expect(r.blobs.indices.bytes).toBe(48)
    expect(summarizeToolCall('geometry_open', { path: 'cases/tet.stl' }, r, true, 'en')).toBe('Opened tet.stl: 4 triangles, closed')
    expect(summarizeToolCall('geometry_open', { path: 'cases/tet.stl' }, r, true, 'ko')).toBe('tet.stl 열기: 삼각형 4개, 닫힌 면')
    const id = r.id as string
    const byId = await runTool('geometry_info', { id, path: null }, ctx())
    expect(byId.ok).toBe(true)
    expect((byId.data as Loose).id).toBe(id)
    const byPath = await runTool('geometry_info', { id: null, path: 'cases/tet.stl' }, ctx())
    expect((byPath.data as Loose).id).toBe(id)
    const save = await runTool('geometry_save', { id, path: 'cases/tet_moved.stl', binary: false, transform: TX1, keepSolids: null, namesJson: null, overwrite: null }, ctx())
    expect(save.ok).toBe(true)
    const s = save.data as Loose
    expect(s.id).not.toBe(id)
    expect(s.bounds.min[0]).toBe(1)
    expect(s.closed).toBe(true)
    const again = await runTool('geometry_save', { id, path: 'cases/tet_moved.stl', binary: false, transform: null, keepSolids: null, namesJson: null, overwrite: null }, ctx())
    expect(again.error?.code).toBe('EXISTS')
    const empty = await runTool('geometry_save', { id, path: 'cases/tet_empty.stl', binary: false, transform: null, keepSolids: [], namesJson: null, overwrite: null }, ctx())
    expect(empty.error?.code).toBe('INVALID')
    const badNames = await runTool('geometry_save', { id, path: 'cases/tet_named.stl', binary: false, transform: null, keepSolids: null, namesJson: '[1]', overwrite: null }, ctx())
    expect(badNames.ok).toBe(false)
    expect(badNames.error?.code).toBe('INVALID')
  })

  it('paths_are_confined', async () => {
    const out = await runTool('geometry_open', { path: '../x.stl' }, ctx())
    expect(out.ok).toBe(false)
    expect(out.error?.code).toBe('OUTSIDE_WORKSPACE')
    const none = await runTool('geometry_open', { path: 'cases/none.stl' }, ctx())
    expect(none.error?.code).toBe('NOT_FOUND')
    const both = await runTool('geometry_info', { id: null, path: null }, ctx())
    expect(both.error?.code).toBe('INVALID')
    const twin = await runTool('geometry_info', { id: 'x', path: 'cases/tet.stl' }, ctx())
    expect(twin.error?.code).toBe('INVALID')
  })

  it('lru_holds_four', async () => {
    const svc = geometryService(ws.config)
    const names = ['a', 'b', 'c', 'd', 'e'].map((n) => writeTet(`cases/five_${n}.stl`))
    const ids: string[] = []
    for (let i = 0; i < 5; i++) ids.push((await svc.open(names[i], `cases/five_${'abcde'[i]}.stl`)).id)
    expect(svc.held().length).toBe(4)
    expect(svc.held()).not.toContain(ids[0])
    expect(svc.get(ids[0])).toBeNull()
    // Eviction drops the geometry from memory only: the BlobStore still serves the arrays from disk.
    const blob = await svc.blob(ids[0], 'positions')
    expect(blob).not.toBeNull()
    expect(blob?.length).toBe(48)
  })

  it('remove_deletes_blobs', async () => {
    const svc = geometryService(ws.config)
    const p = writeTet('cases/gone.stl')
    const id = (await svc.open(p, 'cases/gone.stl')).id
    expect(await svc.remove(id)).toBe(true)
    expect(svc.get(id)).toBeNull()
    expect(await svc.blob(id, 'positions')).toBeNull()
    expect(fs.existsSync(path.join(ws.config.cacheDir, 'geometry', id))).toBe(false)
    expect(await svc.remove('nope')).toBe(false)
  })

  it('open_parts_merges_two_files', async () => {
    const svc = geometryService(ws.config)
    const absA = writeTet('cases/a.stl')
    const absB = writeTet('cases/b.stl', tet2)
    const mkParts = () => [
      { path: absA, name: 'block', tag: 1, material: null, cadVolume: 2 },
      { path: absB, name: 'hole', tag: 2, material: 'steel', cadVolume: 0.1 },
    ]
    const src = { kind: 'step', path: 'cases/two.step', tool: 'geom_tool' } as const
    const first = await svc.openParts(mkParts(), 'cases/two.step', src)
    const info = first.info
    expect(info.triangleCount).toBe(8)
    expect(info.vertexCount).toBe(8)
    expect(info.solids.map((s) => s.name)).toEqual(['block', 'hole'])
    expect(info.solids[1].first).toBe(4)
    expect(info.solids[1].count).toBe(4)
    expect(info.solids[1].tag).toBe(2)
    expect(info.solids[1].material).toBe('steel')
    expect(info.solids[1].cadVolume).toBe(0.1)
    expect(Math.abs(info.solids[1].volume - 8 / 6)).toBeLessThan(1e-9)
    expect(info.closed).toBe(true)
    expect(info.path).toBe('cases/two.step')
    expect(info.source?.kind).toBe('step')
    const direct = await svc.open(absA, 'cases/a.stl')
    expect(first.id).not.toBe(direct.id)
    const second = await svc.openParts(mkParts(), 'cases/two.step', src)
    expect(second.id).toBe(first.id)
  })
})
