import fs from 'node:fs'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import type { ServerMsg } from '@cfd/shared'
import { loadCaseSchema, loadOptionalCaseSchema, type CaseSchema } from '../registry/schema.js'
import { makeTempWorkspace, REPO_ROOT, testConfig, type TempWorkspace } from '../runs/test-helpers.js'
import { registerApiRoutes } from './routes.js'
import { Router } from './router.js'
import { createHttpServer, type HttpServerHandle } from './server.js'
import { fakeAgent, fakeDatasets, fakeRun, fakeRunManager, type FakeRuns } from './test-fakes.js'

let ws: TempWorkspace
let srv: HttpServerHandle
let base: string
let runs: FakeRuns
let cht: CaseSchema | null = null
const broadcasts: ServerMsg[] = []
const agent = fakeAgent()
const datasets = fakeDatasets()

beforeAll(async () => {
  ws = await makeTempWorkspace()
  fs.mkdirSync(path.join(ws.root, 'rust', 'target'), { recursive: true })
  fs.writeFileSync(path.join(ws.root, 'rust', 'target', 'x.txt'), 'hidden')
  fs.writeFileSync(path.join(ws.root, 'README.md'), 'read me\nkEpsilon here\n')
  const config = testConfig(ws)
  runs = fakeRunManager()
  runs.runs.set('r_1', fakeRun('r_1'))
  runs.lines.set('r_1', [1, 2, 3].map((seq) => ({ seq, stream: 'stdout' as const, text: `line ${seq}`, ts: seq })))
  runs.residualRecs.set('r_1', [{ seq: 1, iter: 1, time: null, wall: null, fields: { k: 0.5 }, solverIters: null, raw: '1 k res 0.5' }])
  const schema = loadCaseSchema([path.join(REPO_ROOT, 'docs', 'schema', 'case-1.json')])
  const router = registerApiRoutes(new Router(), { config, hub: { broadcast: (m) => broadcasts.push(m) }, runs, agent, datasets, schema, chtSchema: () => cht, gitStatus: async () => ({ available: true, branch: 'main', upstream: null, ahead: 0, behind: 0, changes: [], error: null }) })
  srv = createHttpServer({ config, router, hub: null, staticDir: null })
  base = `http://127.0.0.1:${(await srv.listen()).port}`
})
afterAll(async () => {
  await srv.close()
  await ws.cleanup()
})

// Test-only loose typing: the assertions below are the type checks.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Loose = any
const get = (p: string) => fetch(`${base}${p}`)
const json = async (p: string): Promise<Loose> => (await get(p)).json()
const postJson = (u: string, body: unknown, method = 'POST') => fetch(u, { method, body: JSON.stringify(body), headers: { 'content-type': 'application/json' } })

describe('api routes', () => {
  it('health, hello, registry and the schema', async () => {
    expect(await json('/api/health')).toEqual({ ok: true, version: '0.0.0-test', mode: 'demo' })
    const hello = await json('/api/hello')
    expect(hello).toMatchObject({ mode: 'demo', llm: 'mock', workspaceRoot: ws.root, availableBinaries: ['ofgpu-k-epsilon'], platform: process.platform })
    expect(hello.gpu.state).toBe('demo')
    const reg = await json('/api/registry')
    expect(reg.binaries.map((b: { name: string }) => b.name)).toContain('ofgpu-k-epsilon')
    expect(reg.pickLists.patchKinds).toEqual(['wall', 'inlet', 'open', 'empty', 'symmetry'])
    expect(reg.meshPresets.length).toBe(7)
    const schema = await get('/api/schema/case-1.json')
    expect(schema.headers.get('content-type')).toContain('json')
    expect(((await schema.json()) as { title: string }).title).toBe('JsonCase')
  })

  it('serves the tree, files and search under the workspace only', async () => {
    const tree = await json('/api/fs/tree?depth=2')
    expect(tree.path).toBe('')
    expect(tree.children.map((c: { name: string }) => c.name)).toEqual(expect.arrayContaining(['cases', 'README.md']))
    const rust = tree.children.find((c: { name: string }) => c.name === 'rust')
    expect(rust.children.map((c: { name: string }) => c.name)).not.toContain('target')
    expect((await get('/api/fs/tree?path=..')).status).toBe(403)
    expect((await get('/api/fs/tree?path=missing')).status).toBe(404)

    const file = await json('/api/fs/file?path=cases/plume.jsonc')
    expect(file.path).toBe('cases/plume.jsonc')
    expect(file.hash).toMatch(/^[0-9a-f]{40}$/)
    expect((await get('/api/fs/file?path=../../../etc/passwd')).status).toBe(403)
    expect((await get('/api/fs/file?path=' + encodeURIComponent(path.join(ws.tmp, 'runs')))).status).toBe(403)
    expect((await get('/api/fs/file')).status).toBe(400)

    const search = await json('/api/fs/search?q=kEpsilon&glob=*.md')
    expect(search.hits).toEqual([{ path: 'README.md', line: 2, col: 1, text: 'kEpsilon here' }])
    expect((await get('/api/fs/search?q=(&regex=1')).status).toBe(400)
    expect(await json('/api/git/status')).toMatchObject({ available: true, branch: 'main' })
  })

  it('writes with conflict detection and broadcasts fs.changed', async () => {
    const put = (body: unknown) => postJson(`${base}/api/fs/file`, body, 'PUT')
    const created = await put({ path: 'cases/new.jsonc', content: '{}', baseHash: null })
    expect(created.status).toBe(200)
    const { hash } = (await created.json()) as { hash: string }
    expect(broadcasts.at(-1)).toEqual({ t: 'fs.changed', paths: ['cases/new.jsonc'] })
    const stale = await put({ path: 'cases/new.jsonc', content: '{"a":1}', baseHash: 'deadbeef' })
    expect(stale.status).toBe(409)
    expect(((await stale.json()) as { currentHash: string }).currentHash).toBe(hash)
    const fresh = await put({ path: 'cases/new.jsonc', content: '{"a":1}', baseHash: hash })
    expect(fresh.status).toBe(200)
    expect((await put({ path: '../x', content: '', baseHash: null })).status).toBe(403)
    expect((await put({ path: 'cases/x', content: 5, baseHash: null })).status).toBe(400)
  })

  it('exposes runs, logs, residuals and stop', async () => {
    expect((await json('/api/runs')).map((r: { id: string }) => r.id)).toEqual(['r_1'])
    expect((await json('/api/runs/r_1')).status).toBe('done')
    expect((await get('/api/runs/r_9')).status).toBe(404)
    const log = await json('/api/runs/r_1/log?fromSeq=2&max=10')
    expect(log.lines.map((l: { seq: number }) => l.seq)).toEqual([2, 3])
    expect(log.nextSeq).toBe(4)
    expect((await json('/api/runs/r_1/log?grep=line%203')).lines).toHaveLength(1)
    const csv = await get('/api/runs/r_1/residuals.csv')
    expect(csv.headers.get('content-type')).toContain('text/csv')
    expect(await csv.text()).toBe('iter,time,k\n1,,0.5\n')
    expect((await json('/api/runs/r_1/residuals')).residuals).toHaveLength(1)
    const started = await postJson(`${base}/api/runs`, { binary: 'ofgpu-k-epsilon', casePath: 'cases/plume.jsonc', args: [{ flag: '-iters', value: 10 }], positionals: [], label: 'x' })
    expect(started.status).toBe(200)
    expect(((await started.json()) as { label: string }).label).toBe('x')
    const bad = await postJson(`${base}/api/runs`, { binary: 'ofgpu-bad', casePath: null, args: [], positionals: [], label: null })
    expect(bad.status).toBe(400)
    expect((await postJson(`${base}/api/runs`, { casePath: 1 })).status).toBe(400)
    // A cross-site form can post text/plain but never application/json.
    expect((await fetch(`${base}/api/runs`, { method: 'POST', body: JSON.stringify({ casePath: 1 }) })).status).toBe(415)
    const stop = await fetch(`${base}/api/runs/r_1/stop`, { method: 'POST' })
    expect(((await stop.json()) as { status: string }).status).toBe('killed')
    expect(runs.stopped).toEqual(['r_1'])
  })

  it('delegates sessions and datasets', async () => {
    expect(await json('/api/sessions')).toEqual([])
    const s = agent.createSession()
    expect((await json('/api/sessions')).length).toBe(1)
    expect((await json(`/api/sessions/${s.id}`)).id).toBe(s.id)
    expect((await get('/api/sessions/nope')).status).toBe(404)
    expect(await (await fetch(`${base}/api/sessions/${s.id}`, { method: 'DELETE' })).json()).toEqual({ deleted: true })
    const open = await postJson(`${base}/api/datasets/open`, { path: 'cases/plume.jsonc', timeIndex: 'last' })
    expect(await open.json()).toMatchObject({ datasetId: 'd_1', status: 'loading' })
    expect(datasets.opened).toEqual(['cases/plume.jsonc'])
    expect((await postJson(`${base}/api/datasets/open`, { path: '../../etc' })).status).toBe(403)
    expect((await get('/api/datasets/d_9')).status).toBe(404)
    const blob = await get('/api/datasets/d_1/blob/f32')
    expect(blob.headers.get('content-type')).toBe('application/octet-stream')
    expect(new Float32Array(await blob.arrayBuffer())).toEqual(new Float32Array([1, 2, 3]))
    expect((await get('/api/datasets/d_1/blob/nope')).status).toBe(404)
    expect(await json('/api/results?root=cases')).toMatchObject({ root: 'cases' })
    expect((await get('/api/results?root=..')).status).toBe(403)
  })

  it('answers the case patch list from the mesh, then the case file, then with none', async () => {
    // No polyMesh: the case file's own rules, named by `match` and typed by `kind`.
    const declared = await json('/api/case/patches?path=cases/plume.jsonc')
    expect(declared.source).toBe('case')
    expect(declared.patches.map((p: { name: string }) => p.name)).toEqual(['inlet', 'outlet', '.*'])
    expect(declared.patches[0]).toEqual({ name: 'inlet', type: 'inlet', nFaces: 0, startFace: 0 })

    // A polyMesh beside the case wins: the boundary file's own names, types and face ranges.
    const meshed = path.join(ws.root, 'cases', 'meshed', 'constant', 'polyMesh')
    fs.mkdirSync(meshed, { recursive: true })
    fs.writeFileSync(
      path.join(meshed, 'boundary'),
      ['FoamFile { version 2.0; format ascii; class polyBoundaryMesh; object boundary; }', '2', '(', 'inlet { type patch; nFaces 40; startFace 100; }', 'walls { type wall; nFaces 60; startFace 140; }', ')'].join('\n'),
    )
    const fromMesh = await json('/api/case/patches?path=cases/meshed')
    expect(fromMesh.source).toBe('polyMesh')
    expect(fromMesh.patches).toEqual([
      { name: 'inlet', type: 'patch', nFaces: 40, startFace: 100 },
      { name: 'walls', type: 'wall', nFaces: 60, startFace: 140 },
    ])

    // Neither is an answer, not a 404.
    fs.mkdirSync(path.join(ws.root, 'cases', 'bare'), { recursive: true })
    expect(await json('/api/case/patches?path=cases/bare')).toEqual({ patches: [], source: 'none' })
    expect((await get('/api/case/patches')).status).toBe(400)
    expect((await get('/api/case/patches?path=..')).status).toBe(403)
  })

  it('geometry_routes', async () => {
    fs.mkdirSync(path.join(ws.root, 'cases'), { recursive: true })
    fs.writeFileSync(path.join(ws.root, 'cases', 'tet.stl'), 'solid tet\nfacet normal 0 0 0\nouter loop\nvertex 0 0 0\nvertex 0 1 0\nvertex 1 0 0\nendloop\nendfacet\nfacet normal 0 0 0\nouter loop\nvertex 0 0 0\nvertex 0 0 1\nvertex 0 1 0\nendloop\nendfacet\nfacet normal 0 0 0\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 0 1\nendloop\nendfacet\nfacet normal 0 0 0\nouter loop\nvertex 1 0 0\nvertex 0 1 0\nvertex 0 0 1\nendloop\nendfacet\nendsolid tet\n')
    const opened = (await (await postJson(`${base}/api/geometry/open`, { path: 'cases/tet.stl' })).json()) as Loose
    expect(opened.info.triangleCount).toBe(4)
    const id = opened.id as string
    expect(((await json(`/api/geometry/${id}`)) as Loose).id).toBe(id)
    const blob = await get(`/api/geometry/${id}/blob/positions`)
    expect(blob.status).toBe(200)
    expect(blob.headers.get('content-type')).toBe('application/octet-stream')
    expect(blob.headers.get('content-length')).toBe('48')
    expect((await get(`/api/geometry/${id}/blob/colours`)).status).toBe(400)
    expect((await get('/api/geometry/nope')).status).toBe(404)
    expect((await postJson(`${base}/api/geometry/open`, { path: '../x.stl' })).status).toBe(403)
    expect((await postJson(`${base}/api/geometry/open`, { path: 'README.md' })).status).toBe(400)
    expect((await postJson(`${base}/api/geometry/${id}/save`, { path: 'cases/tet2.stl' })).status).toBe(200)
    expect((await postJson(`${base}/api/geometry/${id}/save`, { path: 'cases/tet2.stl' })).status).toBe(409)
    const over = { path: 'cases/tet2.stl', overwrite: true, transform: [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] }
    expect((await postJson(`${base}/api/geometry/${id}/save`, over)).status).toBe(200)
    expect(broadcasts.at(-1)).toEqual({ t: 'fs.changed', paths: ['cases/tet2.stl'] })
    expect((((await (await fetch(`${base}/api/geometry/${id}`, { method: 'DELETE' })).json()) as Loose)).evicted).toBe(id)
    expect((await get(`/api/geometry/${id}/blob/positions`)).status).toBe(404)
    expect(((await json('/api/registry')) as Loose).pipelines.map((p: { name: string }) => p.name)).toContain('mesh-step')
  })
})

describe('auth token', () => {
  it('requires the token on /api when configured', async () => {
    const config = testConfig(ws, { authToken: 'secret' })
    const router = registerApiRoutes(new Router(), { config, hub: { broadcast: () => {} }, runs, agent, datasets, schema: loadCaseSchema([path.join(REPO_ROOT, 'docs', 'schema', 'case-1.json')]) })
    const s = createHttpServer({ config, router, hub: null, staticDir: null })
    const b = `http://127.0.0.1:${(await s.listen()).port}`
    try {
      expect((await fetch(`${b}/api/health`)).status).toBe(401)
      expect((await fetch(`${b}/api/health?token=secret`)).status).toBe(200)
      expect((await fetch(`${b}/api/health`, { headers: { authorization: 'Bearer secret' } })).status).toBe(200)
      expect((await fetch(`${b}/api/health`, { headers: { authorization: 'Bearer wrong' } })).status).toBe(401)
    } finally {
      await s.close()
    }
  })
})

describe('cht routes', () => {
  it('serves cht-1.json when the workspace has it, 404 otherwise', async () => {
    const missing = await get('/api/schema/cht-1.json')
    expect(missing.status).toBe(404)
    expect(((await missing.json()) as { error: string }).error).toMatch(/cht-1\.json/)
    const schemaDir = path.join(ws.root, 'docs', 'schema')
    fs.mkdirSync(schemaDir, { recursive: true })
    const file = path.join(schemaDir, 'cht-1.json')
    fs.writeFileSync(
      file,
      '{"$schema":"https://json-schema.org/draft/2020-12/schema","title":"ChtCase","type":"object","properties":{"name":{"type":"string"},"regions":{"type":"array"}},"required":["name","regions"]}',
    )
    cht = loadOptionalCaseSchema([file])
    const served = await get('/api/schema/cht-1.json')
    expect(served.status).toBe(200)
    expect(served.headers.get('content-type')).toContain('application/schema+json')
    expect((await served.text()).includes('"ChtCase"')).toBe(true)
    cht = null
  })

  it('opens a dataset by region and answers a region patch list from the case', async () => {
    const opened = await postJson(`${base}/api/datasets/open`, { path: 'cases/plume.jsonc', region: 'flap' })
    expect(opened.status).toBe(200)
    expect(datasets.opened.at(-1)).toBe('cases/plume.jsonc')
    expect(datasets.openedOpts.at(-1)?.region).toBe('flap')

    fs.mkdirSync(path.join(ws.root, 'cases'), { recursive: true })
    fs.copyFileSync(path.join(REPO_ROOT, 'cases', 'dieStack.cht.jsonc'), path.join(ws.root, 'cases', 'dieStack.cht.jsonc'))
    const die = await json('/api/case/patches?path=cases/dieStack.cht.jsonc&region=die')
    expect(die.source).toBe('case')
    expect(die.patches).toHaveLength(6)
    expect(die.patches.map((p: { name: string }) => p.name)).toEqual(['dieSideXMin', 'dieSideXMax', 'dieSideYMin', 'dieSideYMax', 'dieToSolder', 'dieTop'])
    const byName = new Map<string, string>(die.patches.map((p: { name: string; type: string }) => [p.name, p.type]))
    expect(byName.get('dieToSolder')).toBe('interface')
    expect(byName.get('dieTop')).toBe('wall')
    expect(byName.get('dieSideXMin')).toBe('wall')

    const nope = await get('/api/case/patches?path=cases/dieStack.cht.jsonc&region=nope')
    expect(nope.status).toBe(404)
    expect(((await nope.json()) as { error: string }).error).toMatch(/die, solder, spreader, grease/)

    // Without region the route is byte-for-byte what it was.
    const whole = await json('/api/case/patches?path=cases/dieStack.cht.jsonc')
    expect(whole).toEqual({ patches: [], source: 'none' })
  })
})
