// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// T1-T11 of GUI-2: cad_build really builds one params vector of the locked v1_nominal fixture and
// returns the table, cad_evaluate initialises the study once and walks the stub loop to its rest
// points, cad_study_status reads without writing, the grounding lint covers the two campaign tools,
// the mock LLM takes brief B1 to a confirmed stub study, and the constants equal the Python ones.
// STUDIO-E2E adds T3b (an inactive non-null x_m is refused before the study is written) and T10b
// (the mock takes the Korean brief to a confirmed stub study under study id mock_nozzle_ko).
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'
import type { BetaMessage, BetaMessageParam, BetaRawMessageStreamEvent } from '@anthropic-ai/sdk/resources/beta/messages/messages'
import { CAMPAIGN_TOOLS, lintSession } from '../agent/grounding.js'
import { runTurn } from '../agent/loop.js'
import { makeMessage, mockEvents, type MockPlan } from '../agent/mockLlm.js'
import { appendUserTurn } from '../agent/session.js'
import { fakeDatasets, fakeHub, fakeRuns, makeWorkspace, REPO_ROOT, type TempWorkspace } from '../agent/test-fakes.js'
import { makeDeps, textOf, toolResultsOf, until } from '../agent/test-util.js'
import { CAD_BUILD_H_M, CAD_EVALUATOR, STUB_REPEAT_BAND, studyDir } from './cadLoop.js'
import type { ToolContext } from './context.js'
import { runTool } from './index.js'
import { pythonCommand } from './pytool.js'
import { spawnCapture } from './shell.js'

vi.setConfig({ testTimeout: 300_000, hookTimeout: 300_000 })

const CASES = JSON.parse(fs.readFileSync(path.join(REPO_ROOT, 'tools', 'cad', 'fixtures', 'reqs', 'cases.json'), 'utf8')) as {
  briefs: Record<string, { schema: string; text: string; attachments: unknown[]; provider: string }>
}

const START = {
  params: { D_i: 0.06, CR: 9.0, L_over_Di: 0.5, law: 'poly7', x_m: null, Lx_over_De: 0.5, Lu_over_Di: 0.5, upstream_role: 'slip', t_wall: 0.003 },
  provenance: { D_i: 'user_text', CR: 'user_text', L_over_Di: 'llm_choice', law: 'llm_choice', Lx_over_De: 'default', Lu_over_Di: 'default', upstream_role: 'default', t_wall: 'user_text' },
} as const

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
    settings: { autoApprove: 'reads', effort: 'high', notifyOnRunEnd: true, locale: 'en' },
    toolUseId: `toolu_g2t${calls}`,
    llm: { provider: 'mock', model: 'test' },
    ...over,
  }
}

const fileCode = (res: { ok: boolean; error?: { code: string } }): string => {
  expect(res.ok).toBe(false)
  return res.error!.code
}

/** sha256 of every file under dir, workspace-relative - T6 and T8's before/after identity. */
function shaTree(dir: string): Record<string, string> {
  const out: Record<string, string> = {}
  const walk = (d: string): void => {
    for (const e of fs.readdirSync(d, { withFileTypes: true })) {
      const p = path.join(d, e.name)
      if (e.isDirectory()) walk(p)
      else out[path.relative(dir, p).split(path.sep).join('/')] = fs.readFileSync(p).toString('hex')
    }
  }
  walk(dir)
  return out
}

// The planLlm helper of groundingRepair.test.ts, copied: a scripted LLM the agent loop talks to.
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

beforeAll(async () => {
  ws = await makeWorkspace()
  const cad = path.join(ws.root, 'tools', 'cad')
  await fsp.cp(path.join(REPO_ROOT, 'tools', 'cad'), cad, {
    recursive: true,
    filter: (src) => !src.split(path.sep).includes('__pycache__') && path.basename(src) !== 'studies.jsonl',
  })
  // The S3 build's watertightness oracle is common.REPO-relative: tools/geom/stl_repair.py.
  const geom = path.join(ws.root, 'tools', 'geom')
  await fsp.mkdir(geom, { recursive: true })
  await fsp.copyFile(path.join(REPO_ROOT, 'tools', 'geom', 'stl_repair.py'), path.join(geom, 'stl_repair.py'))
  // loop.py -> evaluate_cfd -> case_writer/wedge_mesh import these from <REPO>/tools/mesh.
  const mesh = path.join(ws.root, 'tools', 'mesh')
  await fsp.mkdir(mesh, { recursive: true })
  await fsp.copyFile(path.join(REPO_ROOT, 'tools', 'mesh', 'polymesh_write.py'), path.join(mesh, 'polymesh_write.py'))
  await fsp.copyFile(path.join(REPO_ROOT, 'tools', 'mesh', 'regions_check.py'), path.join(mesh, 'regions_check.py'))
  // The locked fixture set, relocked against today's template module by one python run in the copy.
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
})
afterAll(() => ws.cleanup())

describe('cad loop tools', () => {
  it('T1: cad_build builds the nominal for real and returns the table in checks order', async () => {
    const res = await runTool('cad_build', { study_id: 'v1_nominal', params: START.params }, ctx())
    expect(res.ok).toBe(true)
    const d = res.data as { kind: string; status: string; rule_id: string | null; objective: number; table: any[]; build_dir: string; params_sha: string; reason: string }
    expect(d.kind).toBe('cadBuild')
    expect(d.status).toBe('pass')
    expect(d.rule_id).toBeNull()
    expect(d.objective).toBe(0.040000000000000036)
    expect(d.params_sha).toMatch(/^[0-9a-f]{64}$/)
    expect(d.build_dir.startsWith('cad/v1_nominal/builds/')).toBe(true)
    expect(d.table.map((r: { req_id: string }) => r.req_id)).toEqual([
      'REQ-001', 'REQ-002', 'REQ-003', 'REQ-004', 'REQ-005', 'REQ-006',
      'SYS-SOLID', 'SYS-VALID', 'SYS-WATERTIGHT', 'SYS-AXIS', 'SYS-UNITS', 'SYS-MACH',
    ])
    const build = d.table.filter((r: { decided_by: string }) => r.decided_by === 'build')
    expect(build.map((r: { req_id: string }) => r.req_id)).toEqual(['REQ-001', 'REQ-002', 'REQ-003', 'REQ-004', 'SYS-SOLID', 'SYS-VALID', 'SYS-WATERTIGHT', 'SYS-AXIS', 'SYS-UNITS'])
    expect(build.every((r: { verdict: string }) => r.verdict === 'pass')).toBe(true)
    for (const id of ['REQ-005', 'SYS-MACH']) {
      const r = d.table.find((x: { req_id: string }) => x.req_id === id)
      expect(r.decided_by).toBe('evaluate')
      expect(r.m).toBeNull()
    }
    const obj = d.table.find((x: { req_id: string }) => x.req_id === 'REQ-006')
    expect(obj.decided_by).toBe('objective')
    expect(obj.m).toBe(0.040000000000000036)
    const reqs = JSON.parse(fs.readFileSync(path.join(ws.root, 'cad', 'v1_nominal', 'requirements', 'requirements.json'), 'utf8')) as { rows: any[] }
    const req1 = reqs.rows.find((r) => r.id === 'REQ-001')
    const r1 = d.table.find((x: { req_id: string }) => x.req_id === 'REQ-001')
    expect(r1.m).toBe(0.06)
    expect(r1.quantity).toBe('inlet_diameter')
    expect(r1.feature).toBe('contraction_start')
    expect(r1.ears).toBe(req1.ears)
  })

  it('T2: a failing hard row is data, not an error (t_wall "0.001" coerced, REQ-004 fail)', async () => {
    const res = await runTool('cad_build', { study_id: 'v1_nominal', params: { ...START.params, t_wall: '0.001' } }, ctx())
    expect(res.ok).toBe(true)
    const d = res.data as { status: string; rule_id: string | null; table: any[] }
    expect(d.status).toBe('refused')
    expect(d.rule_id).toBe('REQ-004')
    const r4 = d.table.find((r: { req_id: string }) => r.req_id === 'REQ-004')
    expect(r4.verdict).toBe('fail')
    expect(Math.abs((r4.m as number) - 0.0009999999996819886)).toBeLessThan(1e-12)
  })

  it('T3: bad study id, missing requirements and missing study are refused by name before python', async () => {
    expect(fileCode(await runTool('cad_build', { study_id: 'Bad/../x', params: START.params }, ctx()))).toBe('CAD-STUDY-ID')
    expect(fileCode(await runTool('cad_build', { study_id: 'nostudy', params: START.params }, ctx()))).toBe('CAD-NO-REQUIREMENTS')
    expect(fileCode(await runTool('cad_evaluate', { study_id: 'nostudy', start: START }, ctx()))).toBe('CAD-NO-REQUIREMENTS')
    expect(fileCode(await runTool('cad_study_status', { study_id: 'v1_nominal' }, ctx()))).toBe('CAD-NO-STUDY')
    const noStart = await runTool('cad_evaluate', { study_id: 'v1_nominal' }, ctx())
    expect(fileCode(noStart)).toBe('CAD-NO-STUDY')
    expect((noStart.error as { message: string }).message).toContain('start {params, provenance}')
  })

  it('T3b: an inactive non-null x_m on the start is refused before the study is written', async () => {
    const res = await runTool('cad_evaluate', { study_id: 'v1_nominal', start: { params: { ...START.params, x_m: 0.5 }, provenance: START.provenance } }, ctx())
    expect(fileCode(res)).toBe('CADOPT-DOC')
    expect(fs.existsSync(path.join(ws.root, 'cad', 'v1_nominal', 'study', 'study.json'))).toBe(false)
    expect(fs.existsSync(path.join(ws.root, 'cad', 'studies.jsonl'))).toBe(false)
  })

  it('T4: cad_evaluate initialises once and runs to the LLM-consult pause', async () => {
    const res = await runTool('cad_evaluate', { study_id: 'v1_nominal', start: START }, ctx())
    expect(res.ok).toBe(true)
    const d = res.data as any
    expect(d.kind).toBe('cadEvaluate')
    expect(d.evaluator).toBe('stub')
    expect(d.initialised).toBe(true)
    expect(d.status.status).toBe('paused')
    expect(d.status.n_evals).toBe(6)
    expect(d.status.n_decisions).toBe(13)
    expect(d.status.last_decision).toEqual({ decision: 'llm_consult', rule_id: 'GATE-STALL' })
    expect(d.stable.design_verdict).toBe('feasible')
    expect(d.stable.objective).toBe(0.06960368608124554)
    expect(d.stable.table).toHaveLength(12)
    expect(d.next).toContain('GATE-STALL')
    expect(fs.existsSync(path.join(ws.root, 'cad', 'studies.jsonl'))).toBe(true)
    expect(fs.existsSync(path.join(ws.root, 'tools', 'cad', 'studies.jsonl'))).toBe(false)
  })

  it('T5: unattended it runs on to a confirmed stable design', async () => {
    const res = await runTool('cad_evaluate', { study_id: 'v1_nominal', unattended: true }, ctx())
    expect(res.ok).toBe(true)
    const d = res.data as any
    expect(d.initialised).toBe(false)
    expect(d.status.status).toBe('confirmed')
    expect(d.status.n_evals).toBe(8)
    expect(d.status.n_decisions).toBe(20)
    expect(d.run.evaluator_calls).toBe(3)
    expect(d.stable.objective).toBe(0.050419780127704136)
    expect(d.stable.table.every((r: { verdict: string }) => r.verdict === 'pass')).toBe(true)
    expect(d.status.last_decision).toEqual({ decision: 'confirm_pass', rule_id: 'GATE-CONFIRM' })
    expect(d.next).toBe('the stable design passed its L2 confirmation')
    fs.writeFileSync(path.join(ws.tmp, 't5-status.json'), JSON.stringify(d.status))
  })

  it('T6: again it does zero work and changes no byte', async () => {
    const studyAbs = path.join(ws.root, studyDir('v1_nominal'))
    const before = shaTree(studyAbs)
    const res = await runTool('cad_evaluate', { study_id: 'v1_nominal', unattended: true }, ctx())
    expect(res.ok).toBe(true)
    const d = res.data as any
    expect(d.run.evaluator_calls).toBe(0)
    expect(d.status.status).toBe('confirmed')
    expect(shaTree(studyAbs)).toEqual(before)
  })

  it('T7: a changed start on the existing study is the loop LOOP-IMMUTABLE refusal', async () => {
    const changed = { params: { ...START.params, t_wall: 0.004 }, provenance: START.provenance }
    const res = await runTool('cad_evaluate', { study_id: 'v1_nominal', start: changed }, ctx())
    expect(res.ok).toBe(false)
    expect(res.error?.code).toBe('LOOP-IMMUTABLE')
  })

  it('T8: cad_study_status reads rows only and reports the stable table', async () => {
    const studyAbs = path.join(ws.root, studyDir('v1_nominal'))
    const before = shaTree(studyAbs)
    const res = await runTool('cad_study_status', { study_id: 'v1_nominal' }, ctx())
    expect(res.ok).toBe(true)
    const d = res.data as any
    expect(d.kind).toBe('cadStudyStatus')
    expect(d.status).toEqual(JSON.parse(fs.readFileSync(path.join(ws.tmp, 't5-status.json'), 'utf8')))
    expect(d.counts).toEqual({ iterations: 10, decisions: 20 })
    expect(d.evals).toHaveLength(9)
    expect(d.evals.map((e: { n: number }) => e.n)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8])
    expect(d.evals[8]).toEqual({ n: 8, origin: 'confirmation', level: 'L2', design_verdict: 'feasible', objective: 0.050419780127704136 })
    expect(d.decisions).toHaveLength(8)
    expect(shaTree(studyAbs)).toEqual(before)
  })

  it('T9: the two tools joined CAMPAIGN_TOOLS and a planted ungrounded number is repaired or marked', async () => {
    expect([...CAMPAIGN_TOOLS]).toEqual(['autonomy_attempts', 'cad_evaluate', 'cad_study_status'])
    const CALL: MockPlan = { blocks: [{ type: 'tool_use', name: 'cad_study_status', input: { study_id: 'v1_nominal' } }], stopReason: 'tool_use' }
    const DRAFT = 'Study v1_nominal is confirmed after 8 evaluations; the stable design is feasible with objective 0.0917 m.'
    const FIXED = 'Study v1_nominal is confirmed after 8 evaluations; the stable design is feasible with objective 0.0504 m.'
    // (a) the repair replaces 0.0917 with the objective the tool result holds
    const deps = makeDeps(ws, { llm: planLlm('zai', 'glm-5.3-flash', [CALL, say(DRAFT), say(FIXED)]) })
    const rec = deps.store.create({ locale: 'en', autoApprove: 'reads' })
    appendUserTurn(rec, { role: 'user', content: 'How did the nozzle study end?' }, { synthetic: false, entitle: true })
    const out = await runTurn(rec, 'g2-9a', new AbortController().signal, deps)
    expect(out.status).toBe('done')
    expect(rec.repairs).toHaveLength(1)
    expect(rec.repairs![0].fixed).toEqual(['0.0917'])
    expect(rec.repairs![0].replaced).toBe(true)
    expect(textOf(rec.messages[rec.messages.length - 1])).toBe(FIXED)
    expect(lintSession(rec.messages).ungrounded).toBe(0)
    // (b) a repair that changes nothing: the draft is shown marked
    const deps2 = makeDeps(ws, { llm: planLlm('zai', 'glm-5.3-flash', [CALL, say(DRAFT), say(DRAFT)]) })
    const rec2 = deps2.store.create({ locale: 'en', autoApprove: 'reads' })
    appendUserTurn(rec2, { role: 'user', content: 'How did the nozzle study end?' }, { synthetic: false, entitle: true })
    await runTurn(rec2, 'g2-9b', new AbortController().signal, deps2)
    const done = deps2.hub.of('msg.done').at(-1)!.message
    const text = done.blocks.find((b: { kind: string }) => b.kind === 'text') as { text: string }
    expect(text.text).toContain('0.0917 [?]')
    expect(rec2.repairs![0].remaining).toEqual(['0.0917'])
  })

  it('T10: the mock LLM takes brief B1 from requirements to a confirmed stub study', async () => {
    const deps = makeDeps(ws)
    const rec = deps.store.create({ locale: 'en' })
    appendUserTurn(rec, { role: 'user', content: CASES.briefs.B1.text + ' Then build it and run the study.' }, { synthetic: false, entitle: true })
    const turn = runTurn(rec, 'g2-10', new AbortController().signal, deps)
    const approval = (n: number) =>
      (deps.hub.of('tool.approval_request')[n] as { approval: { toolUseIds: string[]; calls: Array<{ name: string }> } }).approval
    await until(() => deps.hub.of('tool.approval_request').length === 1, 120000)
    expect(approval(0).calls[0].name).toBe('cad_requirements_propose')
    deps.approvals.resolve(approval(0).toolUseIds, 'approved')
    await until(() => deps.hub.of('tool.approval_request').length === 2, 120000)
    expect(approval(1).calls[0].name).toBe('cad_evaluate')
    deps.approvals.resolve(approval(1).toolUseIds, 'approved')
    await turn
    const withResults = rec.messages.filter((m) => toolResultsOf(m).length > 0)
    const results = toolResultsOf(withResults[withResults.length - 1])
    expect(results).toHaveLength(1)
    expect(results[0].is_error).toBe(false)
    const body = JSON.parse(results[0].content) as { kind: string; study_id: string; status: { status: string } }
    expect(body.kind).toBe('cadStudyStatus')
    expect(body.study_id).toBe('mock_nozzle')
    expect(body.status.status).toBe('confirmed')
    expect(textOf(rec.messages[rec.messages.length - 1]).startsWith('Study mock_nozzle is confirmed:')).toBe(true)
  })

  const BRIEF_KO =
    '20 °C 공기용 축대칭 수축 노즐을 설계해 줘. 입구 지름은 60 mm이고 수축비는 9:1이야. 전체 길이는 80 mm 이하, 벽 두께는 2 mm 이상, 벽 기울기는 35 deg 미만으로 해 줘. 유량은 7.07 L/s야. 가능한 한 짧게 만들어 줘. 요구사항을 잠근 다음 cad_build로 설계를 만들고, cad_evaluate로 평가하고, cad_study_status로 결과를 보여 줘.'

  it('T10b: the mock takes the Korean brief to a confirmed stub study', async () => {
    const deps = makeDeps(ws)
    const rec = deps.store.create({ locale: 'ko' })
    appendUserTurn(rec, { role: 'user', content: BRIEF_KO }, { synthetic: false, entitle: true })
    const turn = runTurn(rec, 'g2-10b', new AbortController().signal, deps)
    const approval = (n: number) =>
      (deps.hub.of('tool.approval_request')[n] as { approval: { toolUseIds: string[]; calls: Array<{ name: string }> } }).approval
    await until(() => deps.hub.of('tool.approval_request').length === 1, 120000)
    expect(approval(0).calls[0].name).toBe('cad_requirements_propose')
    deps.approvals.resolve(approval(0).toolUseIds, 'approved')
    await until(() => deps.hub.of('tool.approval_request').length === 2, 120000)
    expect(approval(1).calls[0].name).toBe('cad_evaluate')
    deps.approvals.resolve(approval(1).toolUseIds, 'approved')
    await turn
    const withResults = rec.messages.filter((m) => toolResultsOf(m).length > 0)
    const results = toolResultsOf(withResults[withResults.length - 1])
    expect(results).toHaveLength(1)
    expect(results[0].is_error).toBe(false)
    const body = JSON.parse(results[0].content) as { kind: string; study_id: string; status: { status: string } }
    expect(body.kind).toBe('cadStudyStatus')
    expect(body.study_id).toBe('mock_nozzle_ko')
    expect(body.status.status).toBe('confirmed')
    expect(textOf(rec.messages[rec.messages.length - 1]).startsWith('스터디 mock_nozzle_ko 상태는 confirmed입니다:')).toBe(true)
  })

  it('T11: the constants the tools pass equal the Python ones', async () => {
    const r = await spawnCapture(
      [
        pythonCommand(ws.config),
        '-c',
        'import json, sys; import loop, readiness; sys.stdout.reconfigure(encoding="utf-8"); print(json.dumps({"band": loop.STUB_REPEAT_BAND, "evaluators": list(loop.EVALUATORS), "h": readiness.H_SELFTEST_M}))',
      ],
      { cwd: path.join(ws.root, 'tools', 'cad'), timeoutMs: 120_000 },
    )
    expect(r.exitCode).toBe(0)
    const py = JSON.parse(r.stdout.trim()) as { band: number; evaluators: string[]; h: number }
    expect(py.band).toBe(STUB_REPEAT_BAND)
    expect(py.evaluators).toContain(CAD_EVALUATOR)
    expect(py.h).toBe(CAD_BUILD_H_M)
  })
})
