import fs from 'node:fs'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import type { ServerMsg } from '@cfd/shared'
import { loadCaseSchema } from '../registry/schema.js'
import { makeTempWorkspace, REPO_ROOT, testConfig, type TempWorkspace } from '../runs/test-helpers.js'
import { registerApiRoutes } from './routes.js'
import { Router } from './router.js'
import { createHttpServer, type HttpServerHandle } from './server.js'
import { fakeAgent, fakeDatasets, fakeRun, fakeRunManager, type FakeRuns } from './test-fakes.js'

let ws: TempWorkspace
let srv: HttpServerHandle
let base: string
let runs: FakeRuns
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
  const router = registerApiRoutes(new Router(), { config, hub: { broadcast: (m) => broadcasts.push(m) }, runs, agent, datasets, schema, gitStatus: async () => ({ available: true, branch: 'main', upstream: null, ahead: 0, behind: 0, changes: [], error: null }) })
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
