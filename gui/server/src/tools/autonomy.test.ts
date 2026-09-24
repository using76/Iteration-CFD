// Proves autonomy_attempts against a real campaign's files (det_a, copied verbatim
// into fixtures/autonomy/det_a) and against the input shapes a weak model sends.
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { summarizeToolCall, TOOL_META, toolPolicy } from '@cfd/shared'
import { fakeDatasets, fakeHub, fakeRuns, makeWorkspace, type TempWorkspace } from '../agent/test-fakes.js'
import type { ToolContext, ToolResult } from './context.js'
import { RESULT_CAP_BYTES, runTool, toolDefinitions, TOOLS } from './index.js'
import { briefRow, BRIEF_OMIT } from './autonomy.js'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const FIX = path.join(HERE, 'fixtures', 'autonomy', 'det_a')
const OUT = 'campaigns/det_a'
const lines = (f: string) => fs.readFileSync(path.join(FIX, f), 'utf8').split(/\r?\n/).filter((l) => l.trim()).map((l) => JSON.parse(l) as Record<string, unknown>)
const json = (f: string) => JSON.parse(fs.readFileSync(path.join(FIX, f), 'utf8')) as Record<string, unknown>
const ATTEMPTS = lines('attempts.jsonl'); const GEOMETRIES = lines('geometries.jsonl')

let ws: TempWorkspace
beforeAll(async () => {
  ws = await makeWorkspace()
  await fsp.cp(FIX, path.join(ws.root, OUT), { recursive: true })
})
afterAll(() => ws.cleanup())

function ctx(): ToolContext {
  return { config: ws.config, hub: fakeHub(), runs: fakeRuns(), datasets: fakeDatasets(), sessionId: 's1', signal: new AbortController().signal, workspaceRoot: ws.root, settings: { autoApprove: 'reads', effort: 'high', notifyOnRunEnd: true, locale: 'en' }, toolUseId: 'toolu_1' }
}
const bytes = (r: ToolResult) => Buffer.byteLength(JSON.stringify(r.data), 'utf8')
const call = (input: unknown) => runTool('autonomy_attempts', input, ctx())

describe('autonomy_attempts', () => {
  it('is a flat, auto-approved read tool directly after regions_check', () => {
    expect(TOOLS.map((t) => t.name).indexOf('autonomy_attempts')).toBe(TOOLS.map((t) => t.name).indexOf('regions_check') + 1)
    expect(toolPolicy('autonomy_attempts')).toBe('auto')
    expect(TOOL_META.autonomy_attempts.kind).toBe('read')
    const def = toolDefinitions().find((t) => t.name === 'autonomy_attempts')!
    expect(def.input_schema.type).toBe('object')
    expect(def.input_schema.required).toEqual(['out'])
    expect(Object.keys(def.input_schema.properties as Record<string, unknown>)).toEqual(['out', 'view', 'geometryId', 'decidedBy', 'offset', 'limit', 'detail'])
    const oneOfs: unknown[] = []
    const anyOfs: unknown[][] = []
    const walk = (v: unknown): void => {
      if (Array.isArray(v)) { v.forEach(walk); return }
      if (v !== null && typeof v === 'object') {
        const o = v as Record<string, unknown>
        if ('oneOf' in o) oneOfs.push(o.oneOf)
        if ('anyOf' in o) anyOfs.push(o.anyOf as unknown[])
        Object.values(o).forEach(walk)
      }
    }
    walk(def.input_schema)
    expect(oneOfs).toEqual([])
    for (const a of anyOfs) {
      expect(a).toHaveLength(2)
      expect(a).toEqual(expect.arrayContaining([{ type: 'null' }]))
    }
    expect(summarizeToolCall('autonomy_attempts', { out: OUT }, { view: 'attempts', campaignId: 'det_a', returned: 10, total: 29 }, true, 'en')).toBe('Read 10 of 29 attempts of campaign det_a')
    expect(summarizeToolCall('autonomy_attempts', { out: OUT }, { view: 'attempts', campaignId: 'det_a', returned: 10, total: 29 }, true, 'ko')).toBe('캠페인 det_a 시도 10/29건 조회')
    expect(summarizeToolCall('autonomy_attempts', { out: OUT }, { view: 'status', campaignId: 'det_a', finished: true }, true, 'en')).toBe('Campaign det_a: finished')
  })

  it('refuses the empty object and its kin by name, never by a throw', async () => {
    for (const input of [{}, { limit: '5' }, { out: 42 }]) {
      const r = await call(input)
      expect(r.ok).toBe(false)
      expect(r.error?.code).toBe('INVALID_INPUT')
      expect(r.error?.message).toContain('out')
    }
    const r = await call(null)
    expect(r.ok).toBe(false)
    expect(r.error?.code).toBe('INVALID_INPUT')
  })

  it('serves { out } alone: the default view, brief rows, the first page of ten', async () => {
    const r = await call({ out: OUT })
    const d = r.data as any
    expect(r.ok).toBe(true)
    expect(d.view).toBe('attempts')
    expect(d.detail).toBe('brief')
    expect(d.campaignId).toBe('det_a')
    expect(d.mode).toBe('rules')
    expect(d.total).toBe(29)
    expect(d.offset).toBe(0)
    expect(d.returned).toBe(10)
    expect(d.nextOffset).toBe(10)
    expect(d.badLines).toEqual([])
    expect(d.ignored).toEqual([])
    expect(d.rows[0].geometry_id).toBe('D-1-002')
    expect(d.rows[0].attempt).toBe(1)
    for (const row of d.rows) for (const k of BRIEF_OMIT) expect(row[k]).toBeUndefined()
    expect(d.rows).toEqual(ATTEMPTS.slice(0, 10).map(briefRow))
    expect(d.rows[0]).toHaveProperty('decided_by')
    expect(d.rows[0]).toHaveProperty('rule_id')
    expect(d.rows[0]).toHaveProperty('trigger')
    expect(d.rows[0]).toHaveProperty('outcome')
    expect(bytes(r)).toBeLessThan(RESULT_CAP_BYTES)
    expect(d.truncated).toBeUndefined()
  })

  it('takes stringified numbers, forgives the nullish strings, ignores an extra key', async () => {
    const r = await call({ out: OUT, offset: '10', limit: '5' })
    const d = r.data as any
    expect(r.ok).toBe(true)
    expect(d.offset).toBe(10)
    expect(d.returned).toBe(5)
    expect(d.nextOffset).toBe(15)
    expect(d.rows[0]).toEqual(briefRow(ATTEMPTS[10]))
    expect(d.rows[0].geometry_id).toBe('F-1-009')
    expect(d.rows[0].attempt).toBe(2)
    const r2 = await call({ out: OUT, view: 'null', geometryId: '', decidedBy: 'None', offset: 'null', limit: 'undefined', detail: '' })
    const d2 = r2.data as any
    expect(d2.total).toBe(29)
    expect(d2.returned).toBe(10)
    expect(d2.view).toBe('attempts')
    expect(d2.detail).toBe('brief')
    const r3 = await call({ out: OUT, foo: 1 })
    expect(r3.ok).toBe(true)
  })

  it('refuses bad values by field name', async () => {
    for (const [input, field] of [
      [{ out: OUT, limit: 'many' }, 'limit'],
      [{ out: OUT, limit: 500 }, 'limit'],
      [{ out: OUT, offset: -1 }, 'offset'],
      [{ out: OUT, view: 'everything' }, 'view'],
      [{ out: OUT, decidedBy: 'human' }, 'decidedBy'],
    ] as const) {
      const r = await call(input)
      expect(r.ok).toBe(false)
      expect(r.error?.code).toBe('INVALID_INPUT')
      expect(r.error?.message).toContain(field)
    }
  })

  it('refuses a directory that is not a campaign, by name', async () => {
    expect((await call({ out: 'nowhere' })).error?.code).toBe('NOT_FOUND')
    expect((await call({ out: '../outside' })).error?.code).toBe('OUTSIDE_WORKSPACE')
    expect((await call({ out: '' })).error?.code).toBe('NOT_A_CAMPAIGN')
    expect((await call({ out: 'cases/plume.jsonc' })).error?.code).toBe('NOT_A_CAMPAIGN')
    await fsp.mkdir(path.join(ws.root, 'campaigns/other'), { recursive: true })
    await fsp.writeFile(path.join(ws.root, 'campaigns/other/campaign.json'), '{"schema":"other/1"}', 'utf8')
    expect((await call({ out: 'campaigns/other' })).error?.code).toBe('NOT_A_CAMPAIGN')
    await fsp.mkdir(path.join(ws.root, 'campaigns/torn'), { recursive: true })
    await fsp.writeFile(path.join(ws.root, 'campaigns/torn/campaign.json'), '{not json', 'utf8')
    expect((await call({ out: 'campaigns/torn' })).error?.code).toBe('BAD_FILE')
  })

  it('pages every row once, under the cap, in file order', async () => {
    for (const [view, all] of [['attempts', ATTEMPTS], ['geometries', GEOMETRIES]] as const) {
      const got: Record<string, unknown>[] = []
      let offset: number | null = 0
      let pages = 0
      while (offset !== null) {
        expect(pages++).toBeLessThan(40)
        const r = await call({ out: OUT, view, offset, limit: 50, detail: 'full' })
        const d = r.data as any
        expect(bytes(r)).toBeLessThan(RESULT_CAP_BYTES)
        expect(d.truncated).toBeUndefined()
        expect(d.returned).toBeGreaterThanOrEqual(1)
        got.push(...d.rows)
        offset = d.nextOffset
      }
      expect(got).toEqual(all)
      expect(pages).toBeGreaterThanOrEqual(2)
    }
  })

  it('filters by geometryId and decidedBy', async () => {
    const b = await call({ out: OUT, geometryId: 'B-1-011' })
    const bd = b.data as any
    expect(bd.total).toBe(4)
    expect(bd.rows.map((x: any) => x.attempt)).toEqual([1, 2, 3, 4])
    const rem = await call({ out: OUT, decidedBy: 'remedy', limit: 50 })
    const remd = rem.data as any
    expect(remd.total).toBe(12)
    for (const row of remd.rows) expect(row.decided_by).toBe('remedy')
    const rule = await call({ out: OUT, decidedBy: 'rule', limit: 50 })
    expect((rule.data as any).total).toBe(17)
    const nope = await call({ out: OUT, geometryId: 'NOPE' })
    const nd = nope.data as any
    expect(nd.total).toBe(0)
    expect(nd.rows).toEqual([])
    expect(nd.nextOffset).toBeNull()
    const far = await call({ out: OUT, offset: 100 })
    const fd = far.data as any
    expect(fd.total).toBe(29)
    expect(fd.rows).toEqual([])
    expect(fd.nextOffset).toBeNull()
  })

  it('geometries: brief end records, the terminal counts, decidedBy listed as ignored', async () => {
    const got: Record<string, unknown>[] = []
    let offset: number | null = 0
    while (offset !== null) {
      const r = await call({ out: OUT, view: 'geometries', offset, limit: 50 })
      const d = r.data as any
      got.push(...d.rows)
      offset = d.nextOffset
    }
    expect(got).toHaveLength(20)
    for (const row of got) {
      expect(row).not.toHaveProperty('fingerprint')
      expect(row).not.toHaveProperty('records')
    }
    const counts: Record<string, number> = {}
    for (const row of got) counts[row.terminal as string] = (counts[row.terminal as string] ?? 0) + 1
    expect(counts).toEqual(json('campaign_end.json').terminals)
    const ig = await call({ out: OUT, view: 'geometries', decidedBy: 'rule', limit: 50 })
    const igd = ig.data as any
    expect(igd.ignored).toEqual(['decidedBy'])
    expect(igd.total).toBe(20)
    const g = await call({ out: OUT, view: 'geometries', geometryId: 'G-1-026' })
    const gd = g.data as any
    expect(gd.total).toBe(1)
    expect(gd.rows[0].terminal).toBe('SURFACE-OPEN')
  })

  it('summary and status read the remaining files', async () => {
    const s = await call({ out: OUT, view: 'summary' })
    const sd = s.data as any
    expect(sd.summary).toEqual(json('summary.json'))
    const st = await call({ out: OUT, view: 'status' })
    const std = st.data as any
    expect(std.header).toEqual(json('campaign.json'))
    expect(std.progress).toEqual(json('progress.json'))
    expect(std.end).toEqual(json('campaign_end.json'))
    expect(std.finished).toBe(true)
    expect(std.nRows).toBe(29)
    expect(std.badLines).toEqual([])
  })

  it('a campaign still being written: a torn last line is a badLine and there is no summary yet', async () => {
    const dir = path.join(ws.root, 'campaigns/live')
    await fsp.mkdir(dir, { recursive: true })
    await fsp.cp(path.join(FIX, 'campaign.json'), path.join(dir, 'campaign.json'))
    await fsp.cp(path.join(FIX, 'progress.json'), path.join(dir, 'progress.json'))
    const src = fs.readFileSync(path.join(FIX, 'attempts.jsonl'), 'utf8').split(/\r?\n/).filter((l) => l.trim())
    const body = src.slice(0, 3).map((l) => l + '\n').join('') + src[3].slice(0, 100)
    await fsp.writeFile(path.join(dir, 'attempts.jsonl'), body, 'utf8')
    const a = await call({ out: 'campaigns/live' })
    const ad = a.data as any
    expect(ad.total).toBe(3)
    expect(ad.badLines).toEqual([4])
    const st = await call({ out: 'campaigns/live', view: 'status' })
    const std = st.data as any
    expect(std.finished).toBe(false)
    expect(std.end).toBeNull()
    expect(std.progress).not.toBeNull()
    expect(std.nRows).toBe(3)
    expect(std.badLines).toEqual([4])
    const s = await call({ out: 'campaigns/live', view: 'summary' })
    expect(s.error?.code).toBe('NO_SUMMARY')
    const g = await call({ out: 'campaigns/live', view: 'geometries' })
    const gd = g.data as any
    expect(g.ok).toBe(true)
    expect(gd.total).toBe(0)
    expect(gd.rows).toEqual([])
  })
})
