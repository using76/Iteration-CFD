import fs from 'node:fs'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import type { ServerMsg } from '@cfd/shared'
import { ONTOLOGY } from '@cfd/shared'
import { fakeRuns } from '../agent/test-fakes.js'
import { closeOntologyHandles, ontologyHandle } from '../ontology/handle.js'
import { loadCaseSchema, type CaseSchema } from '../registry/schema.js'
import { makeTempWorkspace, REPO_ROOT, testConfig, type TempWorkspace } from '../runs/test-helpers.js'
import { registerApiRoutes } from './routes.js'
import { Router } from './router.js'
import { createHttpServer, type HttpServerHandle } from './server.js'
import { fakeAgent, fakeDatasets } from './test-fakes.js'

let ws: TempWorkspace
let srv: HttpServerHandle
let base: string
let runs: ReturnType<typeof fakeRuns>
const broadcasts: ServerMsg[] = []

const VALID = { binary: 'ofgpu-k-epsilon', casePath: 'cases/plume.jsonc', args: [{ flag: '-iters', value: '2000' }], positionals: [], label: 'plume refine' }

beforeAll(async () => {
  ws = await makeTempWorkspace()
  const config = testConfig(ws, { ontologyDir: path.join(ws.tmp, 'ontology') })
  runs = fakeRuns({ finishAfterMs: null })
  // Seed through the same handle the routes resolve, so there is exactly one SQLite handle.
  const h = await ontologyHandle({ config, runs })
  let seq = 0
  const runRow = (id: string, label: string, startedAt: number): Record<string, unknown> => {
    seq++
    return { runId: id, label, binary: 'ofgpu-k-epsilon', argv: [], casePath: null, outputRoot: null, status: 'done', startedAt, iter: 0, written: [], converged: false, logLines: 0, mode: 'demo' }
  }
  h.store.put({ type: 'Run', id: 'r_a', props: runRow('r_a', 'alpha', 1700000000001), sourcePath: 'test' })
  h.store.put({ type: 'Run', id: 'r_b', props: runRow('r_b', 'bravo', 1700000000002), sourcePath: 'test' })
  h.store.put({ type: 'Run', id: 'r_c', props: runRow('r_c', 'charlie', 1700000000003), sourcePath: 'test' })
  h.store.put({ type: 'Case', id: 'cases/plume.jsonc', props: { caseId: 'cases/plume.jsonc', name: 'plumeB', format: 'case-1', outputDir: 'cases/plume_jsonc', nRegions: 0, nInterfaces: 0, nPatchRules: 0 }, sourcePath: 'test' })
  const schema: CaseSchema = loadCaseSchema([path.join(REPO_ROOT, 'docs', 'schema', 'case-1.json')])
  const router = registerApiRoutes(new Router(), { config, hub: { broadcast: (m) => broadcasts.push(m) }, runs, agent: fakeAgent(), datasets: fakeDatasets(), schema })
  srv = createHttpServer({ config, router, hub: null, staticDir: null })
  base = `http://127.0.0.1:${(await srv.listen()).port}`
})
afterAll(async () => {
  await srv.close()
  await closeOntologyHandles()
  await ws.cleanup()
})

const get = (p: string): Promise<Response> => fetch(`${base}${p}`)
const postJson = (u: string, body: unknown, method = 'POST'): Promise<Response> =>
  fetch(`${base}${u}`, { method, body: JSON.stringify(body), headers: { 'content-type': 'application/json' } })
// Test-only loose typing: the assertions below are the type checks.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Loose = any
const json = async (p: string): Promise<Loose> => (await get(p)).json()
const posted = async (u: string, body: unknown): Promise<Loose> => (await postJson(u, body)).json()

describe('ontology routes', () => {
  it('the six ontology routes answer', async () => {
    const types = await json('/api/ontology/types')
    expect(types.objectTypes).toEqual(ONTOLOGY.objectTypeNames())
    expect(types.linkTypes).toEqual(ONTOLOGY.linkTypeNames())
    expect(types.actionTypes).toEqual(ONTOLOGY.actionTypeNames())
    expect((await get('/api/ontology/objects?type=Run')).status).toBe(200)
    expect((await get(`/api/ontology/object/${encodeURIComponent('cases/plume.jsonc')}?type=Case`)).status).toBe(200)
    expect((await get('/api/ontology/object/r_missing?type=Run')).status).toBe(404)
    expect((await get('/api/ontology/links?type=Run&id=r_a&link=case')).status).toBe(200)
    expect((await get('/api/ontology/links?type=Run&id=r_a')).status).toBe(400)
    expect((await postJson('/api/ontology/propose', { action: 'startRun', parameters: VALID })).status).toBe(200)
    expect((await postJson('/api/ontology/apply', { proposalId: 'p_nope' })).status).toBe(404)
    const nope = await get('/api/ontology/nope')
    expect(nope.status).toBe(404)
    expect(await nope.json()).toEqual({ error: 'no route for GET /api/ontology/nope' })
  })

  it('objects filters, orders and pages', async () => {
    const page1 = await json('/api/ontology/objects?type=Run&limit=1')
    expect(page1.objects.length).toBe(1)
    expect(page1.nextCursor).not.toBeNull()
    const page2 = await json(`/api/ontology/objects?type=Run&limit=1&cursor=${page1.nextCursor}`)
    expect(page2.objects[0].id).not.toBe(page1.objects[0].id)
    const desc = await json('/api/ontology/objects?type=Run&orderBy=startedAt&descending=true')
    expect(desc.objects.map((o: { id: string }) => o.id)).toEqual(['r_c', 'r_b', 'r_a'])
    const asc = await json('/api/ontology/objects?type=Run&orderBy=startedAt')
    expect(asc.objects.map((o: { id: string }) => o.id)).toEqual(['r_a', 'r_b', 'r_c'])
    const where = encodeURIComponent(JSON.stringify([{ property: 'label', op: 'eq', value: 'bravo' }]))
    const filtered = await json(`/api/ontology/objects?type=Run&where=${where}`)
    expect(filtered.objects.map((o: { id: string }) => o.id)).toEqual(['r_b'])
    expect((await get('/api/ontology/objects?type=Run&where=not-json')).status).toBe(400)
    expect((await get('/api/ontology/objects')).status).toBe(400)
  })

  it('resolves an id containing a slash', async () => {
    const ok = await get(`/api/ontology/object/${encodeURIComponent('cases/plume.jsonc')}?type=Case`)
    expect(ok.status).toBe(200)
    expect(((await ok.json()) as { id: string }).id).toBe('cases/plume.jsonc')
    expect((await get('/api/ontology/object/cases%2Fnope.jsonc?type=Case')).status).toBe(404)
  })


  it('links walks one hop from an object', async () => {
    const sha = '74ba8320e1d4c9a7b6f5e3d2c1b0a9f8e7d6c5b4'
    const h = await ontologyHandle({ config: testConfig(ws, { ontologyDir: path.join(ws.tmp, 'ontology') }), runs })
    h.store.put({ type: 'Commit', id: sha, props: { sha, shortSha: sha.slice(0, 7), subject: 's', author: 't', authoredAt: 1, committedAt: 1, nFiles: 0 }, sourcePath: 'test' })
    h.store.putLink({ type: 'atCommit', fromId: 'r_a', toId: sha, sourcePath: 'test' })
    const r = await json('/api/ontology/links?type=Run&id=r_a&link=atCommit')
    expect(r.links.length).toBe(1)
    expect(r.links[0].linkType).toBe('atCommit')
    expect(r.linked.length).toBe(1)
    expect(r.linked[0].type).toBe('Commit')
    expect((await get('/api/ontology/links?type=Run&id=r_a')).status).toBe(400)
  })

  it('propose returns the edit set and writes nothing', async () => {
    fs.writeFileSync(path.join(ws.root, 'cases', 'plume.jsonc'), fs.readFileSync(path.join(REPO_ROOT, 'cases', 'plume.jsonc')))
    const before = await json('/api/ontology/objects?type=Run')
    const r = await postJson('/api/ontology/propose', { action: 'startRun', parameters: VALID })
    expect(r.status).toBe(200)
    const p = await posted('/api/ontology/propose', { action: 'startRun', parameters: VALID })
    expect(p.state).toBe('rendered')
    expect(p.edits.objects.length).toBeGreaterThanOrEqual(1)
    expect(p.edits.links.length).toBeGreaterThanOrEqual(1)
    expect(typeof p.edits.summary).toBe('string')
    const after = await json('/api/ontology/objects?type=Run')
    expect(after.objects.length).toBe(before.objects.length)
    expect(runs.started.length).toBe(0)
    const bad = await postJson('/api/ontology/propose', { action: 'startRun', parameters: { ...VALID, args: [{ flag: '-nope', value: '1' }] } })
    expect(bad.status).toBe(200)
    const rejected = await posted('/api/ontology/propose', { action: 'startRun', parameters: { ...VALID, args: [{ flag: '-nope', value: '1' }] } })
    expect(rejected.state).toBe('rejected')
    expect(rejected.blocking.length).toBeGreaterThanOrEqual(1)
    expect((await postJson('/api/ontology/propose', { action: 'nope', parameters: {} })).status).toBe(400)
    const res = await fetch(`${base}/api/ontology/propose`, { method: 'POST', body: JSON.stringify({ action: 'startRun', parameters: VALID }) })
    expect(res.status).toBe(415)
  })

  it('apply applies a proposal the route proposed', async () => {
    const p = await posted('/api/ontology/propose', { action: 'startRun', parameters: VALID })
    const r = await postJson('/api/ontology/apply', { proposalId: p.proposalId })
    expect(r.status).toBe(200)
    const entry = (await r.json()) as { editId: string; appliedAt: number; approvedBy: { kind: string }; parameters: Record<string, unknown> }
    expect(entry.editId).toMatch(/^e_/)
    expect(typeof entry.appliedAt).toBe('number')
    expect(entry.approvedBy.kind).toBe('user')
    expect(entry.parameters).toEqual(VALID)
    const second = await postJson('/api/ontology/apply', { proposalId: p.proposalId })
    expect(second.status).toBe(409)
    expect(((await second.json()) as { code: string }).code).toBe('ALREADY_APPLIED')
    expect((await postJson('/api/ontology/apply', { proposalId: 'p_missing' })).status).toBe(404)
  })
})
