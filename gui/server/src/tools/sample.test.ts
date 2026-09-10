// line_sample: constant / analytic field sampling through the fixture case,
// the bucket-index nearest-centre lookup, and the REST route.
import fs from 'node:fs/promises'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { fakeDatasets, fakeHub, fakeRuns } from '../agent/test-fakes.js'
import { FIXTURE_JSONC, makeJsoncCase } from '../datasets/fixtures/makeCase.js'
import { registerApiRoutes } from '../http/routes.js'
import { Router } from '../http/router.js'
import { createHttpServer, type HttpServerHandle } from '../http/server.js'
import { fakeAgent, fakeDatasets as fakeHttpDatasets, fakeRunManager } from '../http/test-fakes.js'
import { loadCaseSchema } from '../registry/schema.js'
import { makeTempWorkspace, REPO_ROOT, type TempWorkspace } from '../runs/test-helpers.js'
import type { ToolContext } from './context.js'
import { runTool } from './index.js'
import { buildCentreIndex, nearestCentre, type LineSampleResult } from './sample.js'

let ws: TempWorkspace
beforeAll(async () => {
  ws = await makeTempWorkspace({ plume: false })
  await makeJsoncCase(path.join(ws.root, 'cases', 'fixture'))
})
afterAll(() => ws.cleanup())

const OUT = 'cases/fixture/fixture_jsonc'

function ctx(): ToolContext {
  return { config: ws.config, hub: fakeHub(), runs: fakeRuns(), datasets: fakeDatasets(), sessionId: 's1', signal: new AbortController().signal, workspaceRoot: ws.root, settings: { autoApprove: 'reads', effort: 'high', notifyOnRunEnd: true, locale: 'en' }, toolUseId: 'toolu_1' }
}

const sample = (over: Record<string, unknown>): ReturnType<typeof runTool> =>
  runTool('line_sample', { resultDir: OUT, time: null, field: 'k', component: null, p0: [0, 0, 0], p1: [4, 2, 1], n: 5, ...over }, ctx()) as ReturnType<typeof runTool>

describe('line_sample tool', () => {
  it('samples a constant field along a line, latest time by default', async () => {
    const t0 = performance.now()
    const r = await sample({})
    const firstMs = performance.now() - t0
    expect(r.ok).toBe(true)
    const d = r.data as LineSampleResult
    expect(d.time).toBe('100')
    expect(d.cells).toBe(64)
    expect(d.field).toBe('k')
    expect(d.s).toEqual([0, 0.25, 0.5, 0.75, 1])
    expect(d.values.every((v) => Math.abs(v - 1.1) < 1e-6)).toBe(true)
    for (let i = 1; i < d.x.length; i++) expect(d.x[i]).toBeGreaterThan(d.x[i - 1])
    const t1 = performance.now()
    const again = await sample({})
    const cachedMs = performance.now() - t1
    expect((again.data as LineSampleResult).values).toEqual(d.values)
    console.log(`[line_sample] fixture (64 cells): first call ${firstMs.toFixed(1)} ms, cached mesh ${cachedMs.toFixed(1)} ms`)
  })

  it('defaults to 121 points when n is null', async () => {
    const r = await sample({ n: null })
    expect(r.ok).toBe(true)
    expect((r.data as LineSampleResult).values).toHaveLength(121)
    expect((r.data as LineSampleResult).s[120]).toBe(1)
  })

  it('tracks the analytic p = -x field through an explicit time', async () => {
    // p0/p1 run x through every one of the 8 uniform x centres.
    const r = await sample({ time: '50', field: 'p', p0: [0.25, 0.5, 0.5], p1: [3.75, 1.5, 0.5], n: 8 })
    expect(r.ok).toBe(true)
    const d = r.data as LineSampleResult
    expect(d.time).toBe('50')
    for (let i = 0; i < d.values.length; i++) {
      const sx = 0.25 + 0.5 * i
      expect(Math.abs(d.values[i] + sx)).toBeLessThan(1e-6)
      expect(d.x[i]).toBeCloseTo((Math.hypot(3.5, 1, 0) * i) / 7, 12)
    }
  })

  it('reads a vector field as magnitude or as a component', async () => {
    // U = (x, y/2, 0), written only at time 0.
    const cx = await sample({ time: '0', field: 'U', component: 'x', p0: [0.25, 0.5, 0.5], p1: [3.75, 1.5, 0.5], n: 8 })
    expect(cx.ok).toBe(true)
    const xs = (cx.data as LineSampleResult).values
    for (let i = 0; i < xs.length; i++) expect(Math.abs(xs[i] - (0.25 + 0.5 * i))).toBeLessThan(1e-6)
    const mag = await sample({ time: '0', field: 'U', component: 'magnitude', p0: [0.25, 0.5, 0.5], p1: [3.75, 1.5, 0.5], n: 8 })
    const ms = (mag.data as LineSampleResult).values
    for (let i = 0; i < ms.length; i++) {
      const sx = 0.25 + 0.5 * i
      expect(ms[i]).toBeGreaterThanOrEqual(sx - 1e-6)
      expect(ms[i]).toBeLessThan(Math.hypot(sx, 1) + 1e-6)
    }
  })

  it('reports unknown fields, unknown times, scalar components and degenerate lines', async () => {
    expect((await sample({ field: 'zzz' })).error?.code).toBe('NOT_FOUND')
    expect((await sample({ time: '999' })).error?.code).toBe('NOT_FOUND')
    expect((await sample({ time: '50', field: 'p', component: 'x' })).error?.code).toBe('INVALID')
    expect((await sample({ p1: [0, 0, 0] })).error?.code).toBe('INVALID')
    expect((await sample({ n: 1 })).error?.code).toBe('INVALID_INPUT')
    expect((await sample({ resultDir: '../outside' })).error?.code).toBe('OUTSIDE_WORKSPACE')
  })

  it('re-reads the geometry after the case is re-meshed in place', async () => {
    // its own case, so the shared fixture keeps the mesh the other tests count on
    const dir = path.join(ws.root, 'cases', 'recut')
    await makeJsoncCase(dir)
    const resultDir = 'cases/recut/fixture_jsonc'
    // p = -x is written cell by cell; a uniform field would simply expand to the new
    // cell count and hide whether the mesh was re-read at all
    const line = { resultDir, time: null, field: 'p', component: null, p0: [0, 0, 0], p1: [4, 2, 1], n: 5 }
    const first = await runTool('line_sample', line, ctx())
    expect(first.ok).toBe(true)
    expect((first.data as LineSampleResult).cells).toBe(64)

    // the same case re-meshed twice as fine: p on disk still holds 64 values, so a stale
    // cache would answer happily and only a re-read can notice the mismatch
    const jsonc = path.join(dir, 'fixture.jsonc')
    await fs.writeFile(jsonc, FIXTURE_JSONC.replace('"cells": [8, 4, 2]', '"cells": [16, 4, 2]'))
    const later = new Date(Date.now() + 5000)
    await fs.utimes(jsonc, later, later)

    const second = await runTool('line_sample', line, ctx())
    expect(second.ok).toBe(false)
    expect(second.error?.code).toBe('INVALID')
    expect(second.error?.message).toMatch(/64 values for 128 cells/)
  })
})

describe('centre bucket index', () => {
  const cloud10 = (): Float64Array => {
    const centres = new Float64Array(3 * 1000)
    let w = 0
    for (let k = 0; k < 10; k++) {
      for (let j = 0; j < 10; j++) {
        for (let i = 0; i < 10; i++) {
          centres[w++] = i + 0.5
          centres[w++] = j + 0.5
          centres[w++] = k + 0.5
        }
      }
    }
    return centres
  }

  it('finds the analytically nearest centre in a 10x10x10 cloud', () => {
    const centres = cloud10()
    const idx = buildCentreIndex(centres, 1000)
    expect([...idx.dims]).toEqual([10, 10, 10])
    // (7.5, 2.5, 5.5) beats (6.5, ...) / (3.5, ...) / (5.5, ...) on each axis.
    expect(nearestCentre(idx, centres, [7.31, 2.76, 5.95])).toBe(7 + 10 * (2 + 10 * 5))
    expect(nearestCentre(idx, centres, [-10, -10, -10])).toBe(0)
    expect(nearestCentre(idx, centres, [100, 3.2, 5.1])).toBe(9 + 10 * (3 + 10 * 5))
  })

  it('expands across empty buckets to the true nearest', () => {
    const centres = Float64Array.from([0.5, 0.5, 0.5, 9.5, 9.5, 9.5])
    const idx = buildCentreIndex(centres, 2)
    expect(nearestCentre(idx, centres, [4.9, 4.9, 4.9])).toBe(0)
    expect(nearestCentre(idx, centres, [5.2, 5.2, 5.2])).toBe(1)
    expect(nearestCentre(idx, centres, [9.4, 9.4, 9.4])).toBe(1)
  })
})

describe('GET /api/results/sample', () => {
  let srv: HttpServerHandle
  let base: string
  let ws2: TempWorkspace
  const get = (q: string): Promise<Response> => fetch(`${base}/api/results/sample${q}`)

  beforeAll(async () => {
    ws2 = await makeTempWorkspace({ plume: false })
    await makeJsoncCase(path.join(ws2.root, 'cases', 'fixture'))
    const router = registerApiRoutes(new Router(), {
      config: ws2.config,
      hub: { broadcast: () => {} },
      runs: fakeRunManager(),
      agent: fakeAgent(),
      datasets: fakeHttpDatasets(),
      schema: loadCaseSchema([path.join(REPO_ROOT, 'docs', 'schema', 'case-1.json')]),
    })
    srv = createHttpServer({ config: ws2.config, router, hub: null, staticDir: null })
    base = `http://127.0.0.1:${(await srv.listen()).port}`
  })
  afterAll(async () => {
    await srv.close()
    await ws2.cleanup()
  })

  it('returns the same object the tool does', async () => {
    const res = await get(`?dir=${encodeURIComponent(OUT)}&field=p&p0=0.25,0.5,0.5&p1=3.75,1.5,0.5&n=4`)
    expect(res.status).toBe(200)
    const d = (await res.json()) as LineSampleResult
    expect(d.time).toBe('100')
    expect(d.cells).toBe(64)
    expect(d.values).toHaveLength(4)
    expect(d.values[0]).toBeCloseTo(-0.25, 6)
  })

  it('answers 400 on bad params and 404 on a missing result', async () => {
    expect((await get('?dir=x&field=p&p1=1,1,1')).status).toBe(400)
    expect((await get('?dir=x&p0=1,1,1&p1=1,1,1')).status).toBe(400)
    expect((await get(`?dir=${encodeURIComponent(OUT)}&field=p&p0=a,b,c&p1=1,1,1`)).status).toBe(400)
    expect((await get(`?dir=${encodeURIComponent(OUT)}&field=p&p0=1,1,1&p1=2,2,2&n=1`)).status).toBe(400)
    expect((await get(`?dir=${encodeURIComponent(OUT)}&field=p&p0=1,1,1&p1=2,2,2&component=q`)).status).toBe(400)
    expect((await get('?dir=cases/nope&field=p&p0=1,1,1&p1=2,2,2')).status).toBe(404)
    expect((await get(`?dir=${encodeURIComponent(OUT)}&field=zzz&p0=1,1,1&p1=2,2,2`)).status).toBe(404)
  })
})
