// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// T1-T11 of GUI-1: the template list with its live vocabulary, the propose leg grounded in the
// user's own turn text, the approval card with the derived tick, and the locked requirement set.
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'
import { toolPolicy } from '@cfd/shared'
import { APPROVAL_TTL_MS } from '../agent/approvals.js'
import { runTurn } from '../agent/loop.js'
import { ALWAYS_ASK, classifyTool } from '../agent/policy.js'
import { appendUserTurn } from '../agent/session.js'
import { fakeDatasets, fakeHub, fakeRuns, makeWorkspace, REPO_ROOT, type TempWorkspace } from '../agent/test-fakes.js'
import { makeDeps, toolResultsOf, until } from '../agent/test-util.js'
import { cadRequirementsPropose, CONDITION_NULLABLE, ConditionSchema, FLOW_NULLABLE, FlowSchema, OPP_NULLABLE, OppSchema, ROW_NULLABLE, RowSchema, TICK_SOURCES } from './cadReqs.js'
import type { ToolContext } from './context.js'
import { runTool, toolDefinitions } from './index.js'
import { pythonCommand } from './pytool.js'
import { spawnCapture } from './shell.js'

vi.setConfig({ testTimeout: 120_000, hookTimeout: 120_000 })

const CASES = JSON.parse(fs.readFileSync(path.join(REPO_ROOT, 'tools', 'cad', 'fixtures', 'reqs', 'cases.json'), 'utf8')) as {
  briefs: Record<string, { schema: string; text: string; attachments: unknown[]; provider: string }>
  sets: Array<{ name: string; proposal: { study_id: string; template_id: string; operating_point: any; rows: any[] } }>
}
const set = (name: string) => CASES.sets.find((s) => s.name === name)!.proposal

let ws: TempWorkspace
let SHA = ''
let T2_PID = ''

const V1 = () => set('v1_nominal')
const V3 = () => set('v3_exit_velocity')
const V5 = () => set('v5_band_default_assumed')

/** An independent `reqs.py vocab` run against the workspace copy; its stdout is the vocab sha. */
async function vocabSha(): Promise<string> {
  const out = path.join(ws.tmp, `vocab-${Math.random().toString(36).slice(2)}.json`)
  const r = await spawnCapture(
    [pythonCommand(ws.config), path.join(ws.root, 'tools', 'cad', 'reqs.py'), 'vocab', path.join(ws.root, 'tools', 'cad', 'templates', 'nozzle_contraction'), out],
    { cwd: ws.root, timeoutMs: 60_000 },
  )
  expect(r.exitCode).toBe(0)
  fs.rmSync(out, { force: true })
  return r.stdout.trim()
}

beforeAll(async () => {
  ws = await makeWorkspace()
  const cad = path.join(ws.root, 'tools', 'cad')
  await fsp.mkdir(path.join(cad, 'schema'), { recursive: true })
  await fsp.mkdir(path.join(cad, 'templates', 'nozzle_contraction'), { recursive: true })
  const src = path.join(REPO_ROOT, 'tools', 'cad')
  for (const f of ['reqs.py', 'common.py', 'schema.py']) await fsp.copyFile(path.join(src, f), path.join(cad, f))
  for (const f of await fsp.readdir(path.join(src, 'schema'))) await fsp.copyFile(path.join(src, 'schema', f), path.join(cad, 'schema', f))
  await fsp.copyFile(path.join(src, 'templates.lock'), path.join(cad, 'templates.lock'))
  for (const f of ['template.json', 'template.py'])
    await fsp.copyFile(path.join(src, 'templates', 'nozzle_contraction', f), path.join(cad, 'templates', 'nozzle_contraction', f))
  SHA = await vocabSha()
})
afterAll(() => ws.cleanup())

let calls = 0
function ctx(over: Partial<ToolContext> = {}): ToolContext {
  calls += 1
  return {
    config: ws.config,
    hub: fakeHub(),
    runs: fakeRuns(),
    datasets: fakeDatasets(),
    sessionId: 's1',
    signal: new AbortController().signal,
    workspaceRoot: ws.root,
    settings: { autoApprove: 'reads', effort: 'high', notifyOnRunEnd: true, locale: 'en' },
    toolUseId: `toolu_t${calls}`,
    userText: CASES.briefs.B1.text,
    llm: { provider: 'mock', model: 'test' },
    ...over,
  }
}

const input = (p: { study_id: string; template_id: string; operating_point: unknown; rows?: unknown[] }, over: Record<string, unknown> = {}) => ({
  template_id: p.template_id,
  vocab_sha: SHA,
  study_id: p.study_id,
  operating_point: p.operating_point,
  rows: p.rows,
  ...over,
})

describe('cad requirements tools', () => {
  it('T1: list reads the live vocabulary and frozen at call time', async () => {
    const res = await runTool('cad_template_list', {}, ctx())
    expect(res.ok).toBe(true)
    const d = res.data as { kind: string; templates: any[] }
    expect(d.kind).toBe('cadTemplates')
    expect(d.templates).toHaveLength(1)
    const t = d.templates[0]
    expect(t.template_id).toBe('nozzle_contraction/1')
    expect(t.dir).toBe('tools/cad/templates/nozzle_contraction')
    expect(t.frozen).toBe(true)
    expect(t.vocab_sha).toBe(SHA)
    expect(SHA).toBe('bf74cfb2a0e9c25796e753c53545cb2721ddfcef5dc44a6c9eec6db9602ec790')
    expect(t.vocab.quantities.some((q: { quantity: string }) => q.quantity === 'inlet_diameter')).toBe(true)
    const declPath = path.join(ws.root, 'tools', 'cad', 'templates', 'nozzle_contraction', 'template.json')
    const original = await fsp.readFile(declPath)
    try {
      await fsp.writeFile(declPath, Buffer.concat([original, Buffer.from(' ')]))
      const res2 = await runTool('cad_template_list', {}, ctx())
      const t2 = (res2.data as { templates: any[] }).templates[0]
      expect(t2.frozen).toBe(false)
      expect(t2.vocab_sha).toBe(await vocabSha())
    } finally {
      await fsp.writeFile(declPath, original)
    }
  })

  it('T2: propose v1_nominal ok with server ids, EARS and derived confidence', async () => {
    const c = ctx()
    const res = await runTool('cad_requirements_propose', input(V1()), c)
    expect(res.ok).toBe(true)
    const d = res.data as any
    expect(d.status).toBe('ok')
    expect(d.proposalId).toMatch(/^crq-[0-9a-f]{16}$/)
    T2_PID = d.proposalId as string
    expect(d.rows.map((r: { id: string }) => r.id)).toEqual([
      'REQ-001', 'REQ-002', 'REQ-003', 'REQ-004', 'REQ-005', 'REQ-006',
      'SYS-SOLID', 'SYS-VALID', 'SYS-WATERTIGHT', 'SYS-AXIS', 'SYS-UNITS', 'SYS-MACH',
    ])
    expect(d.rows.every((r: { confidence: string }) => r.confidence === 'high')).toBe(true)
    expect(d.ticked).toEqual([])
    const brief = JSON.parse(fs.readFileSync(path.join(ws.root, 'cad', 'proposals', c.toolUseId, 'brief.json'), 'utf8')) as Record<string, unknown>
    expect(Object.keys(brief).sort()).toEqual(['attachments', 'provider', 'schema', 'text'])
    expect(brief.text).toBe(CASES.briefs.B1.text.normalize('NFC'))
    expect(brief.attachments).toEqual([])
  })

  it('T3: coerced inputs give the identical report; a quote string stays a string', async () => {
    const base = await runTool('cad_requirements_propose', input(V1()), ctx())
    const baseId = (base.data as any).proposalId as string
    const a = await runTool('cad_requirements_propose', input(V1(), { rows: JSON.stringify(V1().rows) }), ctx())
    const rowsArr = JSON.parse(JSON.stringify(V1().rows))
    const b = await runTool('cad_requirements_propose', input(V1(), { rows: [JSON.stringify(rowsArr[0]), ...rowsArr.slice(1)] }), ctx())
    const c = await runTool('cad_requirements_propose', input(V1(), { operating_point: JSON.stringify(V1().operating_point) }), ctx())
    const numeric = input(V1())
    ;(numeric as any).rows = JSON.parse(JSON.stringify(V1().rows))
    ;((numeric as any).rows[0] as any).value = '60'
    const d = await runTool('cad_requirements_propose', numeric, ctx())
    const flowStr = JSON.parse(JSON.stringify(V1().operating_point))
    flowStr.flow = JSON.stringify(flowStr.flow)
    const e = await runTool('cad_requirements_propose', input(V1(), { operating_point: flowStr }), ctx())
    for (const r of [a, b, c, d, e]) {
      expect(r.ok).toBe(true)
      expect((r.data as any).proposalId).toBe(baseId)
    }
    const reS = JSON.parse(JSON.stringify(V1().rows))
    ;(reS[0] as any).condition = { Re: '30000', level: null }
    const reN = JSON.parse(JSON.stringify(V1().rows))
    ;(reN[0] as any).condition = { Re: 30000, level: null }
    const f = await runTool('cad_requirements_propose', input(V1(), { rows: reS }), ctx())
    const g = await runTool('cad_requirements_propose', input(V1(), { rows: reN }), ctx())
    expect(f.ok).toBe(true)
    expect(g.ok).toBe(true)
    expect((f.data as any).proposalId).toBe((g.data as any).proposalId)
    const v3 = await runTool('cad_requirements_propose', input(V3()), ctx({ userText: CASES.briefs.B3.text }))
    expect(v3.ok).toBe(true)
    expect((v3.data as any).rows.filter((r: { id: string }) => r.id.startsWith('REQ-')).map((r: { id: string }) => r.id)).toEqual(['REQ-001', 'REQ-002'])
    const quoted = input(V1())
    ;(quoted as any).rows = JSON.parse(JSON.stringify(V1().rows))
    ;((quoted as any).rows[0] as any).quote = '60 mm'
    const q = ctx()
    const res = await runTool('cad_requirements_propose', quoted, q)
    expect(res.ok).toBe(true)
    const written = JSON.parse(fs.readFileSync(path.join(ws.root, 'cad', 'proposals', q.toolUseId, 'proposal.json'), 'utf8')) as any
    expect(written.rows[0].quote).toBe('60 mm')
  })

  it('T4: quotes and values ground in the user text only', async () => {
    const res = await runTool('cad_requirements_propose', input(V1()), ctx({ userText: 'Design a nozzle.' }))
    expect(res.ok).toBe(false)
    expect(res.error?.code).toBe('REQ-REFUSED')
    const d = res.data as any
    expect(d.status).toBe('refused')
    expect(d.refusals).toHaveLength(7)
    expect(d.refusals.every((r: { id: string; check: string }) => r.id === 'REQ-QUOTE' && r.check === 'not_in_brief')).toBe(true)
    const ap = await runTool('cad_requirements_apply', { proposalId: 'crq-00000000000000ff' }, ctx())
    expect(ap.error?.code).toBe('CAD-NOT-APPROVED')
    const nb = ctx({ userText: '   ' })
    const res2 = await runTool('cad_requirements_propose', input(V1()), nb)
    expect(res2.error?.code).toBe('CAD-NOBRIEF')
    expect(fs.existsSync(path.join(ws.root, 'cad', 'proposals', nb.toolUseId))).toBe(false)
    const res3 = await runTool('cad_requirements_propose', input(V1(), { study_id: 'Bad Id' }), ctx())
    expect(res3.error?.code).toBe('REQ-ENVELOPE')
    expect(res3.error?.message).toContain('study_id')
  })

  it('T5: a stale vocab_sha is REQ-VOCAB with the current sha echoed', async () => {
    const res = await runTool('cad_requirements_propose', input(V1(), { vocab_sha: '0'.repeat(64) }), ctx())
    expect(res.ok).toBe(false)
    const d = res.data as any
    expect(d.error.code).toBe('REQ-REFUSED')
    expect(d.refusals).toHaveLength(1)
    expect(d.refusals[0]).toMatchObject({ row: 'proposal', id: 'REQ-VOCAB', check: 'stale' })
    expect(d.vocab_sha).toBe(SHA)
  })

  it('T6: the approval card ticks the hard default/assumed row', async () => {
    const v5 = V5()
    delete (v5.rows[3] as any).ticked
    const base = { study_id: v5.study_id, template_id: v5.template_id, operating_point: v5.operating_point }
    const b5 = ctx({ userText: CASES.briefs.B5.text })
    const withTicked = JSON.parse(JSON.stringify(v5.rows))
    ;(withTicked[3] as any).ticked = true
    const r1 = await runTool('cad_requirements_propose', input(base, { rows: withTicked }), b5)
    expect(r1.ok).toBe(false)
    expect(r1.error?.code).toBe('CAD-FIELD')
    expect(r1.error?.message).toContain('rows[3].ticked')
    const withConf = JSON.parse(JSON.stringify(v5.rows))
    ;(withConf[0] as any).confidence = 'high'
    const r2 = await runTool('cad_requirements_propose', input(base, { rows: withConf }), b5)
    expect(r2.error?.code).toBe('CAD-FIELD')
    expect(r2.error?.message).toContain('rows[0].confidence')
    const clean = input(base, { rows: v5.rows })
    const c6 = ctx({ userText: CASES.briefs.B5.text })
    const card = await cadRequirementsPropose.preview!(clean as never, c6)
    expect(card).toContain('REQ-004 [low]')
    expect(card).toContain('TICKED by approving (default)')
    expect(card).toContain('Approving ticks 1 hard default/assumed row(s).')
    const res = await runTool('cad_requirements_propose', clean, c6)
    expect(res.ok).toBe(true)
    const d = res.data as any
    expect(d.status).toBe('ok')
    const rq4 = d.rows.find((r: { id: string }) => r.id === 'REQ-004')
    expect(rq4.confidence).toBe('low')
    expect(rq4.ticked).toBe(true)
    expect(rq4.source).toBe('default')
    expect((d.rows.find((r: { id: string }) => r.id === 'REQ-005') as any).ticked).toBe(false)
    expect(d.ticked).toEqual(['REQ-004'])
  })
  it('T7: apply refuses an unknown id and an expired approval', async () => {
    const res = await runTool('cad_requirements_apply', { proposalId: 'crq-0000000000000000' }, ctx())
    expect(res.ok).toBe(false)
    expect(res.error?.code).toBe('CAD-NOT-APPROVED')
    const prop = await runTool('cad_requirements_propose', input(V1()), ctx())
    const pid = (prop.data as any).proposalId as string
    const spy = vi.spyOn(Date, 'now').mockReturnValue(Date.now() + APPROVAL_TTL_MS + 1)
    const ap = await runTool('cad_requirements_apply', { proposalId: pid }, ctx())
    spy.mockRestore()
    expect(ap.error?.code).toBe('CAD-EXPIRED')
    expect(ap.error?.message).toContain('10 minutes')
    expect(fs.existsSync(path.join(ws.root, 'cad', 'v1_nominal', 'requirements'))).toBe(false)
  })

  it('T8: apply locks once, repeats byte-identical, refuses a changed file', async () => {
    const prop = await runTool('cad_requirements_propose', input(V1()), ctx())
    const pid = (prop.data as any).proposalId as string
    const res = await runTool('cad_requirements_apply', { proposalId: pid }, ctx())
    expect(res.ok).toBe(true)
    const d = res.data as any
    const dir = path.join(ws.root, 'cad', 'v1_nominal', 'requirements')
    const jsonPath = path.join(dir, 'requirements.json')
    const lockPath = path.join(dir, 'requirements.lock')
    expect(fs.existsSync(jsonPath)).toBe(true)
    expect(fs.existsSync(lockPath)).toBe(true)
    expect(fs.readFileSync(lockPath, 'utf8').trim()).toBe(d.lock_sha)
    expect((JSON.parse(fs.readFileSync(jsonPath, 'utf8')) as { lock_sha: string }).lock_sha).toBe(d.lock_sha)
    expect(d.approved_by).toBeTruthy()
    expect(d.approved_by).not.toBe('pending')
    const mtime = fs.statSync(jsonPath).mtimeMs
    const res2 = await runTool('cad_requirements_apply', { proposalId: pid }, ctx())
    expect(res2.ok).toBe(true)
    expect((res2.data as any).lock_sha).toBe(d.lock_sha)
    expect(fs.statSync(jsonPath).mtimeMs).toBe(mtime)
    fs.appendFileSync(jsonPath, ' ')
    const res3 = await runTool('cad_requirements_apply', { proposalId: pid }, ctx())
    expect(res3.ok).toBe(false)
    expect(res3.error?.code).toBe('REQ-IMMUTABLE')
  })

  it('T9: policy and schema shape', () => {
    expect(toolPolicy('cad_template_list')).toBe('auto')
    expect(toolPolicy('cad_requirements_propose')).toBe('ask')
    expect(toolPolicy('cad_requirements_apply')).toBe('auto')
    expect(ALWAYS_ASK.has('cad_requirements_propose')).toBe(true)
    const settings = { autoApprove: 'all' as const, effort: 'high' as const, notifyOnRunEnd: true, locale: 'en' as const }
    expect(classifyTool('cad_requirements_propose', {}, { settings, allowedTools: ['cad_requirements_propose'], overrides: {} })).toBe('ask')
    expect(classifyTool('cad_requirements_propose', {}, { settings, allowedTools: [], overrides: { cad_requirements_propose: 'auto' } })).toBe('ask')
    const defs = new Map(toolDefinitions().map((t) => [t.name, t]))
    for (const name of ['cad_template_list', 'cad_requirements_propose', 'cad_requirements_apply']) {
      const s = defs.get(name)!.input_schema as Record<string, unknown>
      expect(s.oneOf).toBeUndefined()
      expect(s.anyOf).toBeUndefined()
      expect(s.type).toBe('object')
    }
    const rows = (defs.get('cad_requirements_propose')!.input_schema as any).properties.rows as any
    expect(rows.type).toBe('array')
    const keys = Object.keys(rows.items.properties)
    expect(keys).not.toContain('confidence')
    expect(keys).not.toContain('ticked')
  })
  it('T10: the loop takes brief B1 to a locked requirements.json', async () => {
    const deps = makeDeps(ws)
    const rec = deps.store.create({ locale: 'en' })
    appendUserTurn(rec, { role: 'user', content: CASES.briefs.B1.text }, { synthetic: false, entitle: true })
    const turn = runTurn(rec, 't10', new AbortController().signal, deps)
    await until(() => deps.hub.of('tool.approval_request').length === 1, 20000)
    const req = (deps.hub.of('tool.approval_request')[0] as { approval: { toolUseIds: string[]; calls: Array<{ name: string; preview: string }> } }).approval
    expect(req.calls).toHaveLength(1)
    expect(req.calls[0].name).toBe('cad_requirements_propose')
    expect(req.calls[0].preview).toContain('REQ-001 [high]')
    deps.approvals.resolve(req.toolUseIds, 'approved')
    await turn
    const withResults = rec.messages.filter((m) => toolResultsOf(m).length > 0)
    const results = toolResultsOf(withResults[withResults.length - 1])
    expect(results).toHaveLength(1)
    expect(results[0].is_error).toBe(false)
    const body = JSON.parse(results[0].content) as { kind: string; study_id: string; lock_sha: string }
    expect(body.kind).toBe('cadRequirementsLocked')
    expect(body.study_id).toBe('mock_nozzle')
    const dir = path.join(ws.root, 'cad', 'mock_nozzle', 'requirements')
    expect(fs.readFileSync(path.join(dir, 'requirements.lock'), 'utf8').trim()).toBe(body.lock_sha)
    expect((JSON.parse(fs.readFileSync(path.join(dir, 'requirements.json'), 'utf8')) as { lock_sha: string }).lock_sha).toBe(body.lock_sha)
  })

  it('T11: TICK_SOURCES parity with reqs.py', async () => {
    const r = await spawnCapture(
      [pythonCommand(ws.config), '-c', 'import json, sys; import reqs; sys.stdout.reconfigure(encoding="utf-8"); print(json.dumps(list(reqs.TICK_SOURCES)))'],
      { cwd: path.join(ws.root, 'tools', 'cad'), timeoutMs: 60_000 },
    )
    expect(r.exitCode).toBe(0)
    expect(JSON.parse(r.stdout.trim())).toEqual([...TICK_SOURCES])
  })

  it('T12: a failed check call reports runPyTool itself, and a stale report.json is removed before the check', async () => {
    const c = ctx()
    const pdir = path.join(ws.root, 'cad', 'proposals', c.toolUseId)
    fs.mkdirSync(pdir, { recursive: true })
    fs.writeFileSync(
      path.join(pdir, 'report.json'),
      JSON.stringify({ version: 1, status: 'ok', refusals: [], questions: [], requirements: null, derived: null }),
    )
    const bad = await runTool('cad_requirements_propose', input(V1(), { study_id: 'Bad Id' }), c)
    expect(bad.ok).toBe(false)
    expect(bad.error?.code).toBe('REQ-ENVELOPE')
    expect(bad.error?.message).toContain('study_id')
    const good = await runTool('cad_requirements_propose', input(V1()), c)
    expect(good.ok).toBe(true)
    expect((good.data as any).status).toBe('ok')
    expect((good.data as any).proposalId).toBe(T2_PID)
  })

  it('T13: condition absent, null or "null" is no condition; the advertised schema stays a plain object', async () => {
    const variants: Array<(rows: any[]) => void> = [
      (rows) => { for (const r of rows) r.condition = null },
      (rows) => { for (const r of rows) delete r.condition },
      (rows) => { for (const r of rows) r.condition = 'null' },
    ]
    for (const apply of variants) {
      const rows = JSON.parse(JSON.stringify(V1().rows)) as any[]
      apply(rows)
      const res = await runTool('cad_requirements_propose', input(V1(), { rows }), ctx())
      expect(res.ok).toBe(true)
      expect((res.data as any).proposalId).toBe(T2_PID)
    }
    const defs = new Map(toolDefinitions().map((t) => [t.name, t]))
    const cond = ((defs.get('cad_requirements_propose')!.input_schema as any).properties.rows.items.properties.condition) as any
    expect(cond.type).toBe('object')
    expect(cond.anyOf).toBeUndefined()
    expect(cond.oneOf).toBeUndefined()
  })
  it('T14: an omitted or null T_K/p0_Pa becomes the reference state, recorded assumed', async () => {
    const noRefOpp = () => {
      const o = JSON.parse(JSON.stringify(V1().operating_point)) as Record<string, unknown>
      delete o.T_K
      delete o.p0_Pa
      return o
    }
    // (a) both deleted: ok, the proposal and the report hold the reference state as assumed
    const ca = ctx()
    const ra = await runTool('cad_requirements_propose', input(V1(), { operating_point: noRefOpp() }), ca)
    expect(ra.ok).toBe(true)
    expect((ra.data as any).status).toBe('ok')
    const proposalOf = (id: string) =>
      JSON.parse(fs.readFileSync(path.join(ws.root, 'cad', 'proposals', id, 'proposal.json'), 'utf8')) as any
    const pa = proposalOf(ca.toolUseId)
    expect(pa.operating_point.T_K).toBe(293.15)
    expect(pa.operating_point.p0_Pa).toBe(101325)
    expect(pa.operating_point.T_K_source).toBe('assumed')
    expect(pa.operating_point.p0_Pa_source).toBe('assumed')
    const pid = (ra.data as any).proposalId as string
    const report = JSON.parse(fs.readFileSync(path.join(ws.root, 'cad', 'proposals', ca.toolUseId, 'report.json'), 'utf8')) as any
    expect(report.requirements.operating_point.T_K_source).toBe('assumed')
    expect(report.requirements.operating_point.p0_Pa_source).toBe('assumed')
    // (b) null and the string "null": the same report, so the same proposalId
    const nullOpp = JSON.parse(JSON.stringify(V1().operating_point)) as Record<string, unknown>
    nullOpp.T_K = null
    nullOpp.p0_Pa = 'null'
    const rb = await runTool('cad_requirements_propose', input(V1(), { operating_point: nullOpp }), ctx())
    expect(rb.ok).toBe(true)
    expect((rb.data as any).proposalId).toBe(pid)
    // (c) a present T_K keeps the source it was sent with
    const briefOpp = JSON.parse(JSON.stringify(V1().operating_point)) as Record<string, unknown>
    briefOpp.T_K_source = 'brief'
    const cc = ctx()
    const rc = await runTool('cad_requirements_propose', input(V1(), { operating_point: briefOpp }), cc)
    expect(rc.ok).toBe(true)
    expect(proposalOf(cc.toolUseId).operating_point.T_K_source).toBe('brief')
    expect(proposalOf(cc.toolUseId).operating_point.T_K).toBe(293.15)
    // (d) the card names every value with its source, so the approver sees what was assumed;
    // the loop previews the schema-parsed input (loop.ts parses, then previews), so parse here too
    const card = await cadRequirementsPropose.preview!(
      cadRequirementsPropose.schema.parse(input(V1(), { operating_point: noRefOpp() })) as never,
      ctx(),
    )
    expect(card).toContain('operating point: air (assumed), T_K 293.15 (assumed), p0_Pa 101325 (assumed), Q_m3_s')
    // (e) the advertised schema keeps T_K a plain number, no union
    const defs = new Map(toolDefinitions().map((t) => [t.name, t]))
    const tp = (defs.get('cad_requirements_propose')!.input_schema as any).properties.operating_point as any
    expect(tp.properties.T_K.type).toBe('number')
    expect(tp.properties.T_K.anyOf).toBeUndefined()
    expect(tp.properties.T_K.oneOf).toBeUndefined()
  })
  it('T15: a nullable key the model leaves out reads as null, not INVALID_INPUT', async () => {
    // (a) each row drops exactly the keys v1_nominal already holds as null: ok, the T2 report
    const dropped = (rows: any[]) => {
      for (const r of rows)
        for (const k of ['feature', 'value', 'upper', 'tol_abs', 'tol_rel'] as const)
          if (r[k] == null) delete r[k]
      return rows
    }
    const ca = ctx()
    const ra = await runTool('cad_requirements_propose', input(V1(), { rows: dropped(JSON.parse(JSON.stringify(V1().rows))) }), ca)
    expect(ra.ok).toBe(true)
    expect((ra.data as any).status).toBe('ok')
    expect((ra.data as any).proposalId).toBe(T2_PID)
    const pa = JSON.parse(fs.readFileSync(path.join(ws.root, 'cad', 'proposals', ca.toolUseId, 'proposal.json'), 'utf8')) as any
    expect(pa.rows).toEqual(JSON.parse(JSON.stringify(V1().rows)))
    // (b) flow without its quote: reqs.py's REQ-QUOTE, not INVALID_INPUT
    const noQuote = JSON.parse(JSON.stringify(V1().operating_point)) as Record<string, unknown>
    delete (noQuote.flow as Record<string, unknown>).quote
    const rb = await runTool('cad_requirements_propose', input(V1(), { operating_point: noQuote }), ctx())
    expect(rb.ok).toBe(false)
    expect(rb.error?.code).toBe('REQ-REFUSED')
    expect((rb.data as any).refusals.some((r: { id: string }) => r.id === 'REQ-QUOTE')).toBe(true)
    // (c) operating point without flow: status questions with Q-FLOW, not INVALID_INPUT
    const noFlow = JSON.parse(JSON.stringify(V1().operating_point)) as Record<string, unknown>
    delete noFlow.flow
    const rc = await runTool('cad_requirements_propose', input(V1(), { operating_point: noFlow }), ctx())
    expect(rc.ok).toBe(true)
    expect((rc.data as any).status).toBe('questions')
    expect((rc.data as any).questions.some((q: { id: string }) => q.id === 'Q-FLOW')).toBe(true)
    // (d) every treated-as-nullable key is nullable in the zod schema, by safeParse(null); condition is
    // nullable too but its own preprocess already reads null as no condition, so it is not filled
    const shapeNullable = (schema: { shape: Record<string, { safeParse: (v: unknown) => { success: boolean } }> }, key: string) =>
      schema.shape[key].safeParse(null).success
    for (const k of ROW_NULLABLE) expect(shapeNullable(RowSchema, k), k).toBe(true)
    expect([...ROW_NULLABLE].sort()).toEqual(['ears', 'feature', 'quote', 'standard_ref', 'tol_abs', 'tol_rel', 'upper', 'value'])
    for (const k of CONDITION_NULLABLE) expect(shapeNullable(ConditionSchema, k), k).toBe(true)
    expect([...CONDITION_NULLABLE].sort()).toEqual(['Re', 'level'])
    for (const k of FLOW_NULLABLE) expect(shapeNullable(FlowSchema, k), k).toBe(true)
    expect([...FLOW_NULLABLE].sort()).toEqual(['quote'])
    for (const k of OPP_NULLABLE) expect(shapeNullable(OppSchema, k), k).toBe(true)
    expect([...OPP_NULLABLE].sort()).toEqual(['T_K_source', 'flow', 'fluid_source', 'p0_Pa_source'])
    // a row missing the required quantity stays INVALID_INPUT
    const noQty = JSON.parse(JSON.stringify(V1().rows)) as any[]
    delete noQty[0].quantity
    const rd = await runTool('cad_requirements_propose', input(V1(), { rows: noQty }), ctx())
    expect(rd.ok).toBe(false)
    expect(rd.error?.code).toBe('INVALID_INPUT')
  })
})
