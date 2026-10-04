// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// E1-E14 of GUI-3: cad_propose_edit is ALWAYS_ASK and cannot be relaxed, each of the eight rule ids
// refuses before any card, file or python (the study tree byte-identical), a refused call draws no
// card while a legal one draws exactly one, an approved edit is written once, re-checked by loop.py
// intake, logged in cad_edits.jsonl, and the numbering never overwrites.
import crypto from 'node:crypto'
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'
import type { BetaMessage, BetaMessageParam, BetaRawMessageStreamEvent } from '@anthropic-ai/sdk/resources/beta/messages/messages'
import { runTurn } from '../agent/loop.js'
import { makeMessage, mockEvents, type MockPlan } from '../agent/mockLlm.js'
import { ALWAYS_ASK, classifyTool } from '../agent/policy.js'
import { appendUserTurn } from '../agent/session.js'
import { fakeDatasets, fakeHub, fakeRuns, makeWorkspace, REPO_ROOT, type TempWorkspace } from '../agent/test-fakes.js'
import { makeDeps, toolResultsOf, until } from '../agent/test-util.js'
import { cadProposeEdit, writeEditFile } from './cadEdit.js'
import type { ToolContext } from './context.js'
import { runTool, toolDefinitions } from './index.js'
import { pythonCommand } from './pytool.js'
import { spawnCapture } from './shell.js'

vi.setConfig({ testTimeout: 300_000, hookTimeout: 300_000 })

const START = {
  params: { D_i: 0.06, CR: 9.0, L_over_Di: 0.5, law: 'poly7', x_m: null, Lx_over_De: 0.5, Lu_over_Di: 0.5, upstream_role: 'slip', t_wall: 0.003 },
  provenance: { D_i: 'user_text', CR: 'user_text', L_over_Di: 'llm_choice', law: 'llm_choice', Lx_over_De: 'default', Lu_over_Di: 'default', upstream_role: 'default', t_wall: 'user_text' },
} as const

const STABLE_KEY = '0359447df80e93efbc78ba1bc3c72defbb0f6c15e776e48947ce9937a3a57b82'

/** start.json's bytes written directly: JSON.stringify prints CR's 9.0 as the integer 9, the loop
 * hashes its numbers as it read them, and the eval key splits on that one spelling difference. */
const START_BYTES = JSON.stringify({ evaluator: 'stub', repeat_band: 5e-4, start: { params: START.params, provenance: START.provenance } }).replace('"CR":9,', '"CR":9.0,')

let ws: TempWorkspace
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
    settings: { autoApprove: 'all', effort: 'high', notifyOnRunEnd: true, locale: 'en' },
    toolUseId: `toolu_g3t${calls}`,
    llm: { provider: 'mock', model: 'test' },
    ...over,
  }
}

/** sha256-hex of every file under dir, relative - the study tree's before/after identity. */
function shaTree(dir: string): Record<string, string> {
  const out: Record<string, string> = {}
  const walk = (d: string): void => {
    for (const e of fs.readdirSync(d, { withFileTypes: true })) {
      const p = path.join(d, e.name)
      if (e.isDirectory()) walk(p)
      else out[path.relative(dir, p).split(path.sep).join('/')] = crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex')
    }
  }
  walk(dir)
  return out
}

const editsFiles = (id: string): string[] => {
  try {
    return fs.readdirSync(path.join(ws.root, 'cad', id, 'edits'))
  } catch {
    return []
  }
}
const logLines = (id: string): any[] => {
  try {
    return fs
      .readFileSync(path.join(ws.root, 'cad', id, 'cad_edits.jsonl'), 'utf8')
      .split(/\r?\n/)
      .filter((l) => l.trim().length > 0)
      .map((l) => JSON.parse(l))
  } catch {
    return []
  }
}

/** One refused call: the rule id, no edits file, no log line, and the v1_nominal study tree byte-identical. */
async function refusedBy(input: Record<string, unknown>, id: string, message?: (m: string) => void): Promise<any> {
  const before = shaTree(path.join(ws.root, 'cad', 'v1_nominal', 'study'))
  const r = await runTool('cad_propose_edit', input, ctx())
  expect(r.ok, JSON.stringify(r.data)).toBe(false)
  expect(r.error?.code, JSON.stringify(r.data)).toBe(id)
  expect(editsFiles('v1_nominal')).toEqual([])
  expect(logLines('v1_nominal')).toEqual([])
  expect(shaTree(path.join(ws.root, 'cad', 'v1_nominal', 'study'))).toEqual(before)
  if (message) message(r.error!.message)
  return r
}

const editInput = (edits: Array<{ pointer: string; value: unknown }>, intent?: string[]): Record<string, unknown> => ({
  study_id: 'v1_nominal',
  edits,
  reason: 'from the requirement deltas: shorten the contraction against REQ-003',
  ...(intent ? { intent_params: intent } : {}),
})
const lOverDi = (v: unknown): Array<{ pointer: string; value: unknown }> => [{ pointer: '/params/L_over_Di', value: v }]

// The planLlm helper of cadLoop.test.ts, copied: a scripted LLM the agent loop talks to.
function planLlm(kind: 'zai' | 'anthropic', model: string, plans: MockPlan[]): import('../agent/llm.js').LlmClient {
  let n = 0
  return {
    kind,
    model,
    stream(params) {
      const p = plans[Math.min(n++, plans.length - 1)]
      let final: BetaMessage | null = null
      const gen = mockEvents(p, model, { signal: params.signal, delayMs: 0, inputTokens: 10 }, (m) => (final = m))
      return {
        events: gen as AsyncIterable<BetaRawMessageStreamEvent>,
        finalMessage: async () => final ?? makeMessage(model, [], 'end_turn', null, 10),
      }
    },
  }
}
const say = (text: string): MockPlan => ({ blocks: [{ type: 'text', text }], stopReason: 'end_turn' })
const propose = (input: Record<string, unknown>): MockPlan => ({
  blocks: [
    { type: 'text', text: 'I propose the edit through cad_propose_edit.' },
    { type: 'tool_use', name: 'cad_propose_edit', input },
  ],
  stopReason: 'tool_use',
})

beforeAll(async () => {
  ws = await makeWorkspace()
  const cad = path.join(ws.root, 'tools', 'cad')
  await fsp.cp(path.join(REPO_ROOT, 'tools', 'cad'), cad, {
    recursive: true,
    filter: (src) => !src.split(path.sep).includes('__pycache__') && path.basename(src) !== 'studies.jsonl',
  })
  const geom = path.join(ws.root, 'tools', 'geom')
  await fsp.mkdir(geom, { recursive: true })
  await fsp.copyFile(path.join(REPO_ROOT, 'tools', 'geom', 'stl_repair.py'), path.join(geom, 'stl_repair.py'))
  // loop.py -> evaluate_cfd -> case_writer/wedge_mesh import these from <REPO>/tools/mesh.
  const mesh = path.join(ws.root, 'tools', 'mesh')
  await fsp.mkdir(mesh, { recursive: true })
  await fsp.copyFile(path.join(REPO_ROOT, 'tools', 'mesh', 'polymesh_write.py'), path.join(mesh, 'polymesh_write.py'))
  await fsp.copyFile(path.join(REPO_ROOT, 'tools', 'mesh', 'regions_check.py'), path.join(mesh, 'regions_check.py'))
  const r = await spawnCapture(
    [
      pythonCommand(ws.config),
      '-c',
      'import os, sys; import reqs, loop; d = os.path.abspath(sys.argv[1]); os.makedirs(d, exist_ok=True); reqs.write_locked(d, loop._fx_doc()); print("locked")',
      path.join(ws.root, 'cad', 'v1_nominal', 'requirements'),
    ],
    { cwd: cad, timeoutMs: 300_000 },
  )
  expect(r.exitCode).toBe(0)
  expect(fs.existsSync(path.join(ws.root, 'cad', 'v1_nominal', 'requirements', 'requirements.lock'))).toBe(true)
  const startAbs = path.join(ws.root, 'cad', 'v1_nominal', 'start.json')
  await fsp.mkdir(path.dirname(startAbs), { recursive: true })
  await fsp.writeFile(startAbs, START_BYTES)
  const ir = await spawnCapture(
    [
      pythonCommand(ws.config),
      path.join(cad, 'loop.py'),
      'init',
      path.join(ws.root, 'cad', 'v1_nominal', 'study'),
      path.join(ws.root, 'cad', 'v1_nominal', 'requirements'),
      startAbs,
      '--registry',
      path.join(ws.root, 'cad', 'studies.jsonl'),
    ],
    { cwd: cad, timeoutMs: 300_000, env: { PYTHONIOENCODING: 'utf-8' } },
  )
  expect(ir.exitCode).toBe(0)
  const ev = await runTool('cad_evaluate', { study_id: 'v1_nominal' }, ctx())
  expect(ev.ok).toBe(true)
  const d = ev.data as any
  expect(d.status.status).toBe('paused')
  expect(d.status.n_evals).toBe(6)
  expect(d.status.n_decisions).toBe(13)
  expect(d.status.last_decision).toEqual({ decision: 'llm_consult', rule_id: 'GATE-STALL' })
  expect(d.stable.eval_key).toBe(STABLE_KEY)
  expect(d.stable.params).toEqual({
    L_over_Di: 0.9077495532110333,
    x_m: null,
    Lx_over_De: 0.756935644429177,
    t_wall: 0.007458058019168675,
    D_i: 0.06,
    CR: 9.0,
    Lu_over_Di: 0.5,
    upstream_role: 'slip',
    law: 'poly3',
  })
})

afterAll(async () => {
  await ws.cleanup()
})

describe('cad_propose_edit (docs/16 §E.7, §E.8, §F, §I GUI-3)', () => {
  it('E1: ALWAYS_ASK that no autoApprove, allowlist or policy auto relaxes; a plain object schema', () => {
    expect(ALWAYS_ASK.has('cad_propose_edit')).toBe(true)
    const settings = { autoApprove: 'all' as const, effort: 'high' as const, notifyOnRunEnd: true, locale: 'en' as const }
    expect(classifyTool('cad_propose_edit', {}, { settings, allowedTools: ['cad_propose_edit'], overrides: {} })).toBe('ask')
    expect(classifyTool('cad_propose_edit', {}, { settings, allowedTools: [], overrides: { cad_propose_edit: 'auto' } })).toBe('ask')
    expect(classifyTool('cad_propose_edit', {}, { settings, allowedTools: [], overrides: { cad_propose_edit: 'never' } })).toBe('never')
    const s = toolDefinitions().find((t) => t.name === 'cad_propose_edit')!.input_schema as Record<string, unknown>
    expect(s.type).toBe('object')
    expect(s.oneOf).toBeUndefined()
    expect(s.anyOf).toBeUndefined()
  })

  it('E2 CAD-TARGET: bad id, missing study, a busy study and a closed study refuse before anything', async () => {
    await refusedBy({ study_id: 'Bad/../x', edits: lOverDi(1.2), reason: 'r' }, 'CAD-TARGET')
    await refusedBy({ study_id: 'nostudy', edits: lOverDi(1.2), reason: 'r' }, 'CAD-TARGET')
    const busy = path.join(ws.root, 'cad', 'v1_busy', 'study')
    await fsp.cp(path.join(ws.root, 'cad', 'v1_nominal', 'study'), busy, { recursive: true })
    const decBusy = path.join(busy, 'decisions.jsonl')
    const lines = fs.readFileSync(decBusy, 'utf8').split(/\r?\n/).filter((l) => l.trim().length > 0)
    expect(lines.length).toBe(13)
    fs.writeFileSync(decBusy, lines.slice(0, 11).join('\n') + '\n')
    const rb = await refusedBy({ study_id: 'v1_busy', edits: lOverDi(1.2), reason: 'r' }, 'CAD-TARGET')
    expect(rb.error!.message).toContain('not paused')
    const closed = path.join(ws.root, 'cad', 'v1_closed', 'study')
    await fsp.cp(path.join(ws.root, 'cad', 'v1_nominal', 'study'), closed, { recursive: true })
    const decClosed = path.join(closed, 'decisions.jsonl')
    const lines2 = fs.readFileSync(decClosed, 'utf8').split(/\r?\n/).filter((l) => l.trim().length > 0)
    const last = JSON.parse(lines2[lines2.length - 1])
    last.decision = 'stop'
    lines2[lines2.length - 1] = JSON.stringify(last)
    fs.writeFileSync(decClosed, lines2.join('\n') + '\n')
    const rc = await refusedBy({ study_id: 'v1_closed', edits: lOverDi(1.2), reason: 'r' }, 'CAD-TARGET')
    expect(rc.error!.message).toContain('closed')
  })

  it('E3 CAD-FROZEN: an emptied templates.lock refuses, and is restored byte-identical', async () => {
    const lockAbs = path.join(ws.root, 'tools', 'cad', 'templates.lock')
    const original = fs.readFileSync(lockAbs)
    try {
      fs.writeFileSync(lockAbs, JSON.stringify({ schema: 'cad-templates-lock/1', templates: [] }))
      await refusedBy(editInput(lOverDi(1.2)), 'CAD-FROZEN')
    } finally {
      fs.writeFileSync(lockAbs, original)
    }
    expect(fs.readFileSync(lockAbs).equals(original)).toBe(true)
  })

  it('E4 CAD-LOCKED: requirements, gates, indexed and bare pointers refuse; the bare name is corrected', async () => {
    await refusedBy(editInput([{ pointer: '/requirements/rows/0/value', value: 40 }]), 'CAD-LOCKED')
    await refusedBy(editInput([{ pointer: '/gates/max_evals', value: 100 }]), 'CAD-LOCKED')
    await refusedBy(editInput([{ pointer: '/params/law/0', value: 'poly5' }]), 'CAD-LOCKED')
    await refusedBy(editInput([{ pointer: 'L_over_Di', value: 1.2 }]), 'CAD-LOCKED', (m) => expect(m).toContain('/params/L_over_Di'))
  })

  it('E5 CAD-UNLISTED: an undeclared parameter refuses', async () => {
    await refusedBy(editInput([{ pointer: '/params/throat_d', value: 0.02 }]), 'CAD-UNLISTED')
  })

  it('E6 CAD-TYPE: empty list, a string for a real, a boolean, a number for a choice, a twice pointer, an empty reason', async () => {
    await refusedBy(editInput([]), 'CAD-TYPE')
    await refusedBy(editInput([{ pointer: '/params/L_over_Di', value: 'long' }]), 'CAD-TYPE')
    await refusedBy(editInput([{ pointer: '/params/L_over_Di', value: true }]), 'CAD-TYPE')
    await refusedBy(editInput([{ pointer: '/params/law', value: 7 }]), 'CAD-TYPE')
    await refusedBy(editInput([{ pointer: '/params/L_over_Di', value: 1.1 }, { pointer: '/params/L_over_Di', value: 1.2 }]), 'CAD-TYPE')
    await refusedBy({ study_id: 'v1_nominal', edits: lOverDi(1.2), reason: '' }, 'CAD-TYPE')
  })

  it('E7 CAD-NOOP: the stable value again refuses', async () => {
    await refusedBy(editInput(lOverDi(0.9077495532110333)), 'CAD-NOOP')
  })

  it('E8 CAD-INTENT: D_i and law need a card listing them; listed, the edit passes refuse (null)', async () => {
    const r1 = await refusedBy(editInput([{ pointer: '/params/D_i', value: 0.07 }]), 'CAD-INTENT')
    expect(r1.error!.message).toContain('D_i')
    expect(r1.error!.message).toContain('REQ-001')
    const r2 = await refusedBy(editInput([{ pointer: '/params/law', value: 'poly5' }]), 'CAD-INTENT')
    expect(r2.error!.message).toContain('law')
    expect(cadProposeEdit.refuse!(editInput([{ pointer: '/params/D_i', value: 0.07 }], ['D_i']), ctx())).toBeNull()
  })

  it('E9 CAD-RANGE: the stable vector with the changes applied stays inside the declared boxes', async () => {
    await refusedBy(editInput(lOverDi(2.0)), 'CAD-RANGE')
    await refusedBy(editInput([{ pointer: '/params/law', value: 'cubic_matched' }], ['law']), 'CAD-RANGE', (m) => expect(m).toContain('x_m'))
    await refusedBy(editInput([{ pointer: '/params/x_m', value: 0.5 }]), 'CAD-RANGE', (m) => expect(m).toContain('must be null'))
    await refusedBy(editInput([{ pointer: '/params/upstream_role', value: 'foo' }]), 'CAD-RANGE')
    const r5 = await refusedBy(editInput([{ pointer: '/params/Lu_over_Di', value: 3.0 }]), 'CAD-RANGE')
    expect(r5.error!.message).toContain('Lu_over_Di')
    expect(r5.error!.message).toContain('[0.5, 2]')
    expect(cadProposeEdit.refuse!(editInput(lOverDi(1.2)), ctx())).toBeNull()
    expect(cadProposeEdit.refuse!(editInput([{ pointer: '/params/Lu_over_Di', value: 1.5 }]), ctx())).toBeNull()
  })

  it('E10: through the loop a refused call draws 0 cards, a legal one exactly 1; denying writes nothing', async () => {
    const deps = makeDeps(ws, {
      llm: planLlm('zai', 'claude-opus-5', [
        propose({ study_id: 'v1_nominal', edits: [{ pointer: '/requirements/rows/0/value', value: 40 }], reason: 'loosen REQ-003' }),
        propose(editInput(lOverDi(1.2))),
        say('The edit was not approved, so nothing changed.'),
      ]),
    })
    const rec = deps.store.create({ locale: 'en', autoApprove: 'all' })
    appendUserTurn(rec, { role: 'user', content: 'Propose an edit to the paused study v1_nominal.' }, { synthetic: false, entitle: true })
    const turn = runTurn(rec, 'e10', new AbortController().signal, deps)
    await until(() => deps.hub.of('tool.approval_request').length === 1, 60_000)
    const req = (deps.hub.of('tool.approval_request')[0] as { approval: { toolUseIds: string[]; calls: Array<{ name: string; preview: string }> } }).approval
    expect(req.calls).toHaveLength(1)
    expect(req.calls[0].name).toBe('cad_propose_edit')
    expect(req.calls[0].preview).toContain(`L_over_Di: 0.9077495532110333 -> 1.2`)
    expect(req.calls[0].preview).toContain(`stable 0359447df80e`)
    deps.approvals.resolve(req.toolUseIds, 'denied')
    await turn
    const results = rec.messages.flatMap((m) => toolResultsOf(m))
    expect(results).toHaveLength(2)
    const refused = JSON.parse(results[0].content) as { error: { code: string } }
    expect(refused.error.code).toBe('CAD-LOCKED')
    expect(results[0].is_error).toBe(true)
    const denied = JSON.parse(results[1].content) as { error: { code: string } }
    expect(denied.error.code).toBe('DENIED')
    expect(editsFiles('v1_nominal')).toEqual([])
    expect(logLines('v1_nominal')).toEqual([])
  })

  it('E11: approving the same call writes cad1 once, the loop re-checks it, and the log records the outcome', async () => {
    const cad1Abs = path.join(ws.root, 'cad', 'v1_nominal', 'edits', 'v1_nominal.cad1.json')
    const deps = makeDeps(ws, {
      llm: planLlm('zai', 'claude-opus-5', [propose(editInput(lOverDi(1.2))), say('The edit was evaluated by the loop.')]),
    })
    const rec = deps.store.create({ locale: 'en', autoApprove: 'all' })
    appendUserTurn(rec, { role: 'user', content: 'Propose the L_over_Di edit to study v1_nominal.' }, { synthetic: false, entitle: true })
    const turn = runTurn(rec, 'e11', new AbortController().signal, deps)
    await until(() => deps.hub.of('tool.approval_request').length === 1, 60_000)
    const req = (deps.hub.of('tool.approval_request')[0] as { approval: { toolUseIds: string[] } }).approval
    deps.approvals.resolve(req.toolUseIds, 'approved')
    await turn
    const results = rec.messages.flatMap((m) => toolResultsOf(m))
    expect(results).toHaveLength(1)
    expect(results[0].is_error).toBe(false)
    const body = JSON.parse(results[0].content) as any
    expect(body.kind).toBe('cadEdit')
    expect(body.study_id).toBe('v1_nominal')
    expect(body.n).toBe(1)
    expect(body.file).toBe('cad/v1_nominal/edits/v1_nominal.cad1.json')
    expect(body.outcome).toBe('evaluated')
    expect(body.rule_id).toBe('CAD-INTAKE')
    expect(body.gate_decision).toBe('reject')
    expect(body.gate_rule_id).toBe('GATE-REGRESS')
    expect(body.regressed).toEqual(['REQ-003'])
    expect(body.status.status).toBe('paused')
    expect(body.status.n_evals).toBe(7)
    const doc = JSON.parse(fs.readFileSync(cad1Abs, 'utf8')) as Record<string, unknown>
    expect(Object.keys(doc).sort()).toEqual(['base_stable_eval_key', 'card', 'edits', 'reason', 'schema', 'study_id'])
    expect(body.sha256).toBe(crypto.createHash('sha256').update(fs.readFileSync(cad1Abs)).digest('hex'))
    expect(doc.schema).toBe('cad-edit/1')
    const card = doc.card as { approved_by: string; params: string[] }
    expect(card.approved_by.length).toBeGreaterThan(0)
    expect(card.params).toEqual([])
    const log = logLines('v1_nominal')
    expect(log).toHaveLength(1)
    expect(log[0].outcome).toBe('evaluated')
    expect(log[0].sha256).toBe(body.sha256)
  })

  it('E12: a second edit takes cad2, cad1 byte-identical, and the log grows to 2 lines', async () => {
    const cad1Abs = path.join(ws.root, 'cad', 'v1_nominal', 'edits', 'v1_nominal.cad1.json')
    const before = fs.readFileSync(cad1Abs).toString('hex')
    const r = await runTool('cad_propose_edit', editInput(lOverDi(1.3)), ctx())
    expect(r.ok, JSON.stringify(r.data)).toBe(true)
    const d = r.data as any
    expect(d.kind).toBe('cadEdit')
    expect(d.n).toBe(2)
    expect(d.outcome).toBe('evaluated')
    expect(d.rule_id).toBe('CAD-INTAKE')
    expect(d.gate_decision).toBe('reject')
    expect(d.gate_rule_id).toBe('GATE-REGRESS')
    expect(d.status.n_evals).toBe(8)
    expect(fs.readFileSync(cad1Abs).toString('hex')).toBe(before)
    expect(logLines('v1_nominal')).toHaveLength(2)
  })

  it('E13: the loop refuses the third edit with GATE-LLMOFF and the refusal is still logged', async () => {
    const r = await runTool('cad_propose_edit', editInput(lOverDi(1.4)), ctx())
    expect(r.ok).toBe(false)
    expect(r.error?.code).toBe('GATE-LLMOFF')
    expect((r.data as any).n).toBe(3)
    expect(fs.existsSync(path.join(ws.root, 'cad', 'v1_nominal', 'edits', 'v1_nominal.cad3.json'))).toBe(true)
    const log = logLines('v1_nominal')
    expect(log).toHaveLength(3)
    expect(log[2].outcome).toBe('refused')
    expect(log[2].rule_id).toBe('GATE-LLMOFF')
  })

  it('E14: writeEditFile never overwrites - the next N after the largest planted, then one more', async () => {
    const dir = await fsp.mkdtemp(path.join(os.tmpdir(), 'cad-edit-n-'))
    try {
      fs.writeFileSync(path.join(dir, 's.cad1.json'), '{"n":1}')
      fs.writeFileSync(path.join(dir, 's.cad2.json'), '{"n":2}')
      const w1 = await writeEditFile(dir, 's', '{"n":3}')
      expect(w1.n).toBe(3)
      expect(fs.readFileSync(path.join(dir, 's.cad1.json'), 'utf8')).toBe('{"n":1}')
      expect(fs.readFileSync(path.join(dir, 's.cad2.json'), 'utf8')).toBe('{"n":2}')
      expect(fs.readFileSync(w1.file, 'utf8')).toBe('{"n":3}')
      const w2 = await writeEditFile(dir, 's', '{"n":4}')
      expect(w2.n).toBe(4)
      expect(fs.readFileSync(path.join(dir, 's.cad3.json'), 'utf8')).toBe('{"n":3}')
    } finally {
      await fsp.rm(dir, { recursive: true, force: true })
    }
  })
})
