import path from 'node:path'
import { DC_ONTOLOGY as ONTOLOGY, TOOL_META, TOOL_NAMES, summarizeToolCall, toolPolicy } from '@cfd/shared'
import { z } from 'zod'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { approvalPreview } from '../agent/loop.js'
import { fakeDatasets, fakeHub, fakeRuns } from '../agent/test-fakes.js'
import { closeOntologyHandles, ontologyHandle, ontologyPreviewFor } from '../ontology/handle.js'
import { ONTOLOGY_ACT_TOOL, proposalByToolUseId } from '../ontology/preview.js'
import { debugSql } from '../ontology/store.js'
import { makeTempWorkspace, testConfig, type TempWorkspace } from '../runs/test-helpers.js'
import { runTool, sanitizeSchema, toolDefinitions, TOOLS } from './index.js'
import { ACTION_HINT, ontologyAct, ontologyApply, ontologyQuery } from './ontology.js'
import type { ToolContext } from './context.js'

type Cfg = ReturnType<typeof testConfig>
type Runs = ReturnType<typeof fakeRuns>
type Handle = Awaited<ReturnType<typeof ontologyHandle>>

let ws: TempWorkspace
let cfg: Cfg
let runs: Runs
let h: Handle

// N4's pinned startRun fixture (C16): five parameters, all present, none omitted.
const VALID = { binary: 'ofgpu-k-epsilon', casePath: 'cases/plume.jsonc', args: [{ flag: '-iters', value: '2000' }], positionals: [], label: 'plume refine' }

beforeAll(async () => {
  ws = await makeTempWorkspace()
  // The only safe override: testConfig is a plain spread, so overriding guiDir would NOT move ontologyDir.
  cfg = testConfig(ws, { ontologyDir: path.join(ws.tmp, 'ontology') })
  runs = fakeRuns({ finishAfterMs: null })
  h = await ontologyHandle({ config: cfg, runs })
})
afterAll(async () => {
  await closeOntologyHandles()
  await ws.cleanup()
})

function ctx(over: Partial<ToolContext> = {}): ToolContext {
  return { config: cfg, hub: fakeHub(), runs, datasets: fakeDatasets(), sessionId: 's_1', signal: new AbortController().signal, workspaceRoot: ws.root, settings: { autoApprove: 'reads', effort: 'high', notifyOnRunEnd: true, locale: 'en' }, toolUseId: 'toolu_1', ...over }
}

function nulls(over: Record<string, unknown>): Record<string, unknown> {
  return { id: null, where: null, orderBy: null, descending: null, limit: null, cursor: null, traverse: null, properties: null, mode: null, text: null, ...over }
}

const rowCount = (): number => ['Run', 'Driver', 'Case', 'Commit'].reduce((n: number, t) => n + h.store.count(t), 0)
const sqlCount = (sql: string): number => (debugSql(h.store, sql)[0] as { n: number }).n

let seq = 0
function runRow(id: string, label: string, over: Record<string, unknown> = {}): Record<string, unknown> {
  seq++
  return { runId: id, label, binary: 'ofgpu-k-epsilon', argv: [], casePath: null, outputRoot: null, status: 'done', startedAt: 1700000000000 + seq, iter: 0, written: [], converged: false, logLines: 0, mode: 'demo', ...over }
}

describe('ontology tools', () => {
  it('registers the three tools in all five places', async () => {
    expect(TOOLS.map((t) => t.name)).toEqual([...TOOL_NAMES])
    expect(TOOLS.slice(-3).map((t) => t.name)).toEqual(['ontology_query', 'ontology_act', 'ontology_apply'])
    for (const n of ['ontology_query', 'ontology_act', 'ontology_apply']) expect(summarizeToolCall(n, {}, {}, true, 'en')).not.toBe(n)
    const hook = ontologyPreviewFor({ config: cfg, runs }, 's_1')
    const withHook = await approvalPreview(ONTOLOGY_ACT_TOOL, { action: 'startRun', parameters: VALID }, cfg.workspaceRoot, hook, 'tu_reg')
    expect(typeof withHook === 'string' && withHook.length > 0).toBe(true)
    const withoutHook = await approvalPreview(ONTOLOGY_ACT_TOOL, { action: 'startRun', parameters: VALID }, cfg.workspaceRoot)
    expect(withoutHook).toBeNull()
  })

  it('policies', () => {
    expect(toolPolicy('ontology_query')).toBe('auto')
    expect(toolPolicy('ontology_act')).toBe('ask')
    expect(toolPolicy('ontology_apply')).toBe('auto')
    expect(TOOL_META.ontology_query.kind).toBe('read')
    expect(TOOL_META.ontology_act.kind).toBe('mutate')
    expect(TOOL_META.ontology_apply.kind).toBe('mutate')
  })

  it('builds every enum from the registry, never by hand', () => {
    const defs = toolDefinitions()
    const enumOf = (spec: unknown): string[] | undefined => {
      const n = spec as { enum?: string[]; anyOf?: Array<{ enum?: string[] }> }
      return n.enum ?? n.anyOf?.find((b) => b.enum !== undefined)?.enum
    }
    const q = defs.find((d) => d.name === 'ontology_query')!.input_schema as { properties: Record<string, unknown> }
    const a = defs.find((d) => d.name === 'ontology_act')!.input_schema as { properties: Record<string, unknown> }
    expect(enumOf(q.properties.objectType)).toEqual(ONTOLOGY.objectTypeNames())
    expect(enumOf(a.properties.action)).toEqual(ONTOLOGY.actionTypeNames())
    const sides = ONTOLOGY.linkTypes.flatMap((l) => [l.from.apiName, l.to.apiName]).filter((v, i, arr) => arr.indexOf(v) === i).sort()
    expect(enumOf(q.properties.traverse)).toEqual(sides)
    // C6 Run 1: the registry is DC_ONTOLOGY, whose 73 side names need a higher ceiling than N5's 40.
    expect(enumOf(q.properties.traverse)!.length).toBeLessThanOrEqual(96)
    expect(ACTION_HINT.length).toBeLessThanOrEqual(1200)
    expect(ACTION_HINT).toContain('startRun (binary, casePath, args, positionals, label)')
  })

  it('costs under 6 KB of tool definition for all three', () => {
    const sizes = [ontologyQuery, ontologyAct, ontologyApply].map((t) => Buffer.byteLength(JSON.stringify(sanitizeSchema(z.toJSONSchema(t.schema, { io: 'input' }))), 'utf8'))
    console.log('ontology tool definition bytes:', sizes.join(' + '), '=', sizes.reduce((x, y) => x + y, 0))
    expect(sizes.reduce((x, y) => x + y, 0)).toBeLessThan(6144)
  })

  it('closes every object and uses nullable, never optional', () => {
    const defs = [ontologyQuery, ontologyAct, ontologyApply].map((t) => sanitizeSchema(z.toJSONSchema(t.schema, { io: 'input' })) as Record<string, unknown>)
    const isNullable = (spec: unknown): boolean => {
      const n = spec as { type?: unknown; anyOf?: Array<{ type?: string }> }
      if (Array.isArray(n.type)) return n.type.includes('null')
      if (Array.isArray(n.anyOf)) return n.anyOf.some((b) => b.type === 'null')
      return false
    }
    const walk = (node: unknown, visit: (obj: Record<string, unknown>) => void): void => {
      if (Array.isArray(node)) return node.forEach((n) => walk(n, visit))
      if (typeof node !== 'object' || node === null) return
      visit(node as Record<string, unknown>)
      for (const v of Object.values(node as Record<string, unknown>)) walk(v, visit)
    }
    for (const s of defs) {
      expect(s.type).toBe('object')
      expect(s).not.toHaveProperty('$schema')
      expect(s.oneOf).toBeUndefined()
      walk(s, (obj) => {
        if (obj.type === 'object' && obj.properties) expect(obj.additionalProperties).toBe(false)
      })
      const props = s.properties as Record<string, unknown>
      expect((s.required as string[]).slice().sort()).toEqual(Object.keys(props).slice().sort())
    }
    expect(Object.keys(defs[0].properties as object).length).toBe(11)
    expect(Object.keys(defs[1].properties as object).length).toBe(2)
    expect(Object.keys(defs[2].properties as object).length).toBe(1)
    const q = defs[0].properties as Record<string, unknown>
    for (const name of ['id', 'where', 'orderBy', 'descending', 'limit', 'cursor', 'traverse', 'properties']) expect(isNullable(q[name]), name).toBe(true)
    expect(isNullable(q.objectType)).toBe(false)
  })

  it('answers a query with objects, their titles and one link hop', async () => {
    const sha = '74ba8320e1d4c9a7b6f5e3d2c1b0a9f8e7d6c5b4'
    h.store.put({ type: 'Run', id: 'r_100', props: runRow('r_100', 'done run', { status: 'done' }), sourcePath: 'test' })
    h.store.put({ type: 'Run', id: 'r_101', props: runRow('r_101', 'failed run', { status: 'failed' }), sourcePath: 'test' })
    h.store.put({ type: 'Commit', id: sha, props: { sha, shortSha: sha.slice(0, 7), subject: 'seed commit', author: 'test', authoredAt: 1700000000000, committedAt: 1700000000000, nFiles: 1 }, sourcePath: 'test' })
    h.store.putLink({ type: 'atCommit', fromId: 'r_100', toId: sha, sourcePath: 'test' })
    const r = await runTool('ontology_query', nulls({ objectType: 'Run', where: [{ property: 'status', op: 'eq', value: 'done' }], traverse: 'atCommit' }), ctx())
    expect(r.ok).toBe(true)
    const d = r.data as Record<string, unknown>
    expect(d.kind).toBe('ontologyObjects')
    expect(d.objectType).toBe('Run')
    expect((d.objects as unknown[]).length).toBe(1)
    expect((d.links as Array<Record<string, unknown>>).length).toBe(1)
    expect((d.links as Array<Record<string, unknown>>)[0].linkType).toBe('atCommit')
    expect((d.linked as Array<Record<string, unknown>>).length).toBe(1)
    expect((d.linked as Array<Record<string, unknown>>)[0].type).toBe('Commit')
    expect(d.nextCursor).toBeNull()
    expect(d.trimmed).toBe(false)
    const c = await runTool('ontology_query', nulls({ objectType: 'Run', where: [{ property: 'label', op: 'contains', value: 'done' }] }), ctx())
    expect((c.data as Record<string, unknown[]>).objects.length).toBe(1)
  })

  it('refuses an unknown property by name', async () => {
    const r = await runTool('ontology_query', nulls({ objectType: 'Run', where: [{ property: 'gitShaX', op: 'eq', value: 'x' }] }), ctx())
    expect(r.ok).toBe(false)
    expect(r.error?.code).toBe('UNKNOWN_PROPERTY')
    for (const name of ['runId', 'label', 'binary']) expect(r.error?.message).toContain(name)
  })

  it('caps a query at 200 rows and trims a fat result to 16 KB', async () => {
    for (let i = 1; i <= 300; i++) {
      const id = `r_fat_${String(i).padStart(3, '0')}`
      h.store.put({ type: 'Run', id, props: runRow(id, `seed ${i}`, { outputRoot: `out/${'x'.repeat(180)}` }), sourcePath: 'test' })
    }
    const tooBig = await runTool('ontology_query', nulls({ objectType: 'Run', limit: 500 }), ctx())
    expect(tooBig.ok).toBe(false)
    expect(tooBig.error?.code).toBe('INVALID_INPUT')
    const r = await runTool('ontology_query', nulls({ objectType: 'Run', limit: 200 }), ctx())
    expect(r.ok).toBe(true)
    const d = r.data as Record<string, unknown>
    const objects = d.objects as Array<{ id: string; title: string | null; props: Record<string, unknown> }>
    expect(objects.length).toBe(200)
    const bytes = Buffer.byteLength(JSON.stringify(d), 'utf8')
    console.log('trimmed query bytes:', bytes)
    expect(bytes).toBeLessThanOrEqual(16384)
    expect(d.trimmed).toBe(true)
    for (const o of objects) {
      expect('id' in o).toBe(true)
      expect('title' in o).toBe(true)
      expect(o.props).toEqual({})
    }
    expect(d.nextCursor).not.toBeNull()
    const p2 = await runTool('ontology_query', nulls({ objectType: 'Run', limit: 200, cursor: d.nextCursor as string }), ctx())
    const o2 = (p2.data as Record<string, unknown>).objects as Array<{ id: string }>
    expect(o2[0].id).not.toBe(objects[0].id)
  })

  it('the approval preview is the edit set summary and writes nothing', async () => {
    const before = rowCount()
    const hook = ontologyPreviewFor({ config: cfg, runs }, 's_1')
    const text = await hook(ONTOLOGY_ACT_TOOL, { action: 'startRun', parameters: VALID }, 'tu_prev')
    expect(text).toContain('create Run')
    expect(text).toMatch(/(^|\n)\+ link/)
    expect(text).not.toContain('INSERT')
    expect(text).not.toContain('SELECT')
    expect(runs.started.length).toBe(0)
    expect(rowCount()).toBe(before)
  })

  it('reuses the previewed proposal, so what was approved is what applies', async () => {
    const hook = ontologyPreviewFor({ config: cfg, runs }, 's_1')
    const previewText = await hook(ONTOLOGY_ACT_TOOL, { action: 'startRun', parameters: VALID }, 'tu_1')
    const r = await runTool('ontology_act', { action: 'startRun', parameters: VALID }, ctx({ toolUseId: 'tu_1' }))
    expect(r.ok).toBe(true)
    const d = r.data as Record<string, unknown>
    expect(d.proposalId).toBe(proposalByToolUseId(h.engine, 'tu_1')!.proposalId)
    expect(d.summary).toBe(previewText)
    const r2 = await runTool('ontology_act', { action: 'startRun', parameters: VALID }, ctx({ toolUseId: 'tu_2' }))
    expect((r2.data as Record<string, unknown>).proposalId).not.toBe(d.proposalId)
  })

  it('refuses to apply a proposal no approved act produced', async () => {
    const before = rowCount()
    const refused = await runTool('ontology_apply', { proposalId: 'p_nope' }, ctx())
    expect(refused.ok).toBe(false)
    expect(refused.error?.code).toBe('NOT_APPROVED')
    // The denied-card case: the preview made a proposal, but execute never ran, so it never
    // entered the approved-gate map and can never be applied.
    const hook = ontologyPreviewFor({ config: cfg, runs }, 's_1')
    await hook(ONTOLOGY_ACT_TOOL, { action: 'startRun', parameters: VALID }, 'tu_deny')
    const deniedId = proposalByToolUseId(h.engine, 'tu_deny')!.proposalId
    const refused2 = await runTool('ontology_apply', { proposalId: deniedId }, ctx())
    expect(refused2.ok).toBe(false)
    expect(refused2.error?.code).toBe('NOT_APPROVED')
    expect(rowCount()).toBe(before)
  })

  it('act then apply writes the objects, the links and one edit-log row', async () => {
    const runsBefore = rowCount()
    const linksBefore = sqlCount('SELECT COUNT(*) AS n FROM links')
    const logBefore = sqlCount('SELECT COUNT(*) AS n FROM edit_log')
    const act = await runTool('ontology_act', { action: 'startRun', parameters: VALID }, ctx({ toolUseId: 'tu_exec' }))
    expect(act.ok).toBe(true)
    const proposalId = (act.data as Record<string, unknown>).proposalId as string
    const app = await runTool('ontology_apply', { proposalId }, ctx({ toolUseId: 'tu_ap' }))
    expect(app.ok).toBe(true)
    // fakeRuns mints r_1; the Run row exists with the action's source stamp, and the edit set's
    // two links (executed, runs) landed — atCommit is dropped because a temp workspace has no HEAD.
    const run = h.store.get('Run', 'r_1')
    expect(run?.sourcePath).toBe('action:startRun')
    expect(sqlCount('SELECT COUNT(*) AS n FROM links')).toBe(linksBefore + 2)
    expect(rowCount()).toBe(runsBefore + 1)
    expect(sqlCount('SELECT COUNT(*) AS n FROM edit_log')).toBe(logBefore + 1)
    const entry = JSON.parse((debugSql(h.store, 'SELECT entry FROM edit_log ORDER BY applied_at DESC, edit_id DESC')[0] as { entry: string }).entry) as Record<string, unknown>
    expect((entry.approvedBy as Record<string, unknown>).kind).toBe('user')
    expect(entry.parameters).toEqual(VALID)
  })

  it('refuses a second apply of the same proposal', async () => {
    const runsBefore = rowCount()
    const act = await runTool('ontology_act', { action: 'startRun', parameters: VALID }, ctx({ toolUseId: 'tu_again' }))
    const proposalId = (act.data as Record<string, unknown>).proposalId as string
    const first = await runTool('ontology_apply', { proposalId }, ctx({ toolUseId: 'tu_again2' }))
    expect(first.ok).toBe(true)
    const second = await runTool('ontology_apply', { proposalId }, ctx({ toolUseId: 'tu_again3' }))
    expect(second.ok).toBe(false)
    expect(second.error?.code).toBe('ALREADY_APPLIED')
    expect(rowCount()).toBe(runsBefore + 1)
    expect(runs.started.length).toBe(2)
  })

  it('a blocked proposal returns its blocking criteria and writes nothing', async () => {
    const before = rowCount()
    const bad = { ...VALID, args: [{ flag: '-nope', value: '1' }] }
    const r = await runTool('ontology_act', { action: 'startRun', parameters: bad }, ctx({ toolUseId: 'tu_bad' }))
    expect(r.ok).toBe(true)
    const d = r.data as Record<string, unknown>
    expect(d.state).toBe('rejected')
    expect((d.blocking as unknown[]).length).toBeGreaterThanOrEqual(1)
    expect((d.blocking as Array<{ id: string; message: string }>)[0].id).toBe('flagsTypeCheck')
    expect((d.blocking as Array<{ id: string; message: string }>)[0].message).toContain('-nope')
    expect(d.applied).toBe(false)
    expect(rowCount()).toBe(before)
  })
})
