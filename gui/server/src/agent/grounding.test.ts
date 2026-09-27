// Proves the grounding lint reads numbers with their written precision, grounds them only on tool
// results an assistant saw earlier, and finds nothing ungrounded in twenty campaign explanations
// under the scripted mock and under a rounding zai provider alike.
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import type { BetaMessage, BetaMessageParam, BetaRawMessageStreamEvent } from '@anthropic-ai/sdk/resources/beta/messages/messages'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { closeOntologyHandles } from '../ontology/handle.js'
import { explainPrompt } from '../tools/neutrality.js'
import { extractNumbers, lintSession, lintText, sourceNumbers } from './grounding.js'
import type { LlmClient } from './llm.js'
import { runTurn } from './loop.js'
import { createMockLlm, makeMessage, mockEvents, type MockPlan } from './mockLlm.js'
import { createAgentService } from './service.js'
import { appendUserTurn, type SessionRecord } from './session.js'
import { fakeDatasets, fakeHub, fakeRuns, makeWorkspace, type TempWorkspace } from './test-fakes.js'
import { makeDeps, toolUsesOf, type TestDeps } from './test-util.js'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const DET = path.join(HERE, '..', 'tools', 'fixtures', 'autonomy', 'det_a')
const ROWS = fs.readFileSync(path.join(DET, 'attempts.jsonl'), 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l) as Record<string, any>)
const IDS = [...new Set(ROWS.map((r) => String(r.geometry_id)))].sort().slice(0, 10)
const OUTS = ['campaigns/studio', 'campaigns/headless']

let ws: TempWorkspace
beforeAll(async () => {
  ws = await makeWorkspace()
  for (const out of OUTS) await fsp.cp(DET, path.join(ws.root, out), { recursive: true })
})
afterAll(async () => {
  await closeOntologyHandles()
  await ws.cleanup()
})

function session(deps: TestDeps, text: string, settings: Partial<SessionRecord['settings']> = {}): SessionRecord {
  const rec = deps.store.create({ locale: 'en', ...settings })
  appendUserTurn(rec, { role: 'user', content: text }, { synthetic: false, entitle: true })
  return rec
}
function planLlm(kind: LlmClient['kind'], model: string, plans: MockPlan[]): LlmClient {
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

const p3 = (v: unknown): string => (typeof v === 'number' ? v.toPrecision(3) : Array.isArray(v) ? v.map(p3).join(' to ') : String(v ?? 'none'))
function roundedText(gid: string): string {
  const mine = ROWS.filter((r) => r.geometry_id === gid)
  const lines = [`**${gid}** had ${mine.length} attempts.`]
  for (const r of mine) {
    const t = r.trigger ?? {}
    const o = r.outcome ?? {}
    lines.push(`${r.attempt}. Attempt ${r.attempt} was decided by the ${r.decided_by} layer (${r.rule_id ?? 'no rule'}): ${t.observable ?? 'no trigger'} was ${p3(t.value)} against ${p3(t.threshold)}, so the verdict was ${o.verdict}${typeof o.n_cells === 'number' ? `, with ${o.n_cells.toLocaleString('en-US')} cells` : ''}${typeof o.blc8_a_priori === 'number' ? ` and BLC_8 ${(100 * o.blc8_a_priori).toFixed(1)} %` : ''}.`)
  }
  return lines.join('\n')
}

describe('grounding', () => {
  it('extractNumbers reads each number with its precision and skips identifiers and list markers', () => {
    expect(extractNumbers('Geometry F-1-009 in campaigns/det_a/configs/F-1-009_a2.json: 2 attempt rows.').map((s) => s.raw)).toEqual(['2'])
    expect(extractNumbers('1. BLC_8 was 100.0 % and y+ ~1.00, G4@castellate, F3a, §D.3 and SPEC-LIT §92.13, docs/15, 8th ed., 8-layer stack').map((s) => s.raw)).toEqual(['100.0 %', '1.00', '8'])
    const t3 = 'n_cells 87,166 or 87.2k, 15x the cells, 2.40e-14 and 8.72e+4, time 14:28:33 on 2026-09-24, 8개 층, -0.5 and v1.2.3, ratio 1,2,3'
    expect(extractNumbers(t3).map((s) => s.raw)).toEqual(['87,166', '87.2k', '15x', '2.40e-14', '8.72e+4', '2026', '8', '-0.5', '1', '2', '3'])
    expect(extractNumbers('value .5, (0.6 mm), 3x3 grid, 45%, **26.2**, `/snap/iterations` 60.').map((s) => s.raw)).toEqual(['.5', '0.6', '45%', '26.2', '60'])
    const xs = extractNumbers(t3)
    expect(xs[0]).toMatchObject({ value: 87166, half: 0.5, exact: false })
    expect(xs[1]).toMatchObject({ value: 87200, half: 50 })
    expect(xs[3].value).toBe(2.4e-14)
    expect(xs[3].half).toBeCloseTo(5e-17, 25)
    expect(xs[6].exact).toBe(true)
    expect(xs[7].value).toBe(-0.5)
    const p = extractNumbers('1. BLC_8 was 100.0 % and y+ ~1.00')[0]
    expect(p.percent).toBe(true)
    expect(p.value).toBe(100)
    expect(p.half).toBeCloseTo(0.05, 12)
    expect(extractNumbers('８７ cells and −0.5').map((s) => s.value)).toEqual([87, -0.5])
  })

  it('lintText grounds exact, rounded, percent, scientific and thousands forms, and flags the rest', () => {
    const src = sourceNumbers(JSON.stringify({ total: 2, rows: [{ attempt: 1, trigger: { value: 26.201602136181577, threshold: [16.000016, 41.999957999999985] }, outcome: { n_cells: 87166, blc8_a_priori: 0.914, t1: 0.0005992, tiny: 2.404564606299065e-14 }, note: 'pinned 3108 points' }] }))
    const good = lintText('Attempt 1 of 2: 26.2 against 16 to 42, 87,166 cells (87.2k), BLC_8 91.4 % or 91 %, t1 0.000599 or 5.99e-4, 2.40e-14, and 3108 pinned points.', src)
    expect(good.checked).toBe(13)
    expect(good.ungrounded).toEqual([])
    const bad = lintText('It needs 37.5 % more, a 0.6 mm layer, 3 attempts and 26.3 at most; 87,200 cells.', src)
    expect(bad.checked).toBe(5)
    expect(bad.ungrounded.map((s) => s.raw)).toEqual(['37.5 %', '0.6', '3', '26.3', '87,200'])
    expect(sourceNumbers('not json: 12 and 0.5')).toEqual([12, 0.5])
  })

  it('lintSession grounds a message only on tool results before it, never on the user words', () => {
    const msgs = [
      { role: 'user', content: 'Explain D-1-002; it had 7 attempts' },
      { role: 'assistant', content: [{ type: 'text', text: 'Reading 7 rows.' }, { type: 'tool_use', id: 't1', name: 'autonomy_attempts', input: {} }, { type: 'tool_use', id: 't2', name: 'file_read', input: {} }] },
      { role: 'user', content: [{ type: 'tool_result', tool_use_id: 't1', content: JSON.stringify({ total: 2, rows: [{ attempt: 1, outcome: { n_cells: 87166 } }] }) }, { type: 'tool_result', tool_use_id: 't2', content: [{ type: 'text', text: 'pinned 3108 points' }] }] },
      { role: 'assistant', content: [{ type: 'text', text: 'D-1-002 had 2 attempts with 87166 cells and 3108 pinned points, not 7.' }] },
      { role: 'user', content: 'And the run?' },
      { role: 'assistant', content: [{ type: 'text', text: 'No campaign tool here: 87166 cells again and 5 runs.' }] },
    ] as BetaMessageParam[]
    const g = lintSession(msgs)
    expect(g.schema).toBe('autonomy-grounding/1')
    expect(g.messages.map((m) => [m.index, m.turn, m.campaign, m.final, m.checked, m.ungrounded.map((u) => u.raw)])).toEqual([[1, 0, true, false, 1, ['7']], [3, 0, true, true, 4, ['7']], [5, 1, false, true, 2, ['5']]])
    expect(g.explanations).toBe(1)
    expect(g.checked).toBe(5)
    expect(g.ungrounded).toBe(2)
  })
  it('the mock explains twenty geometries grounded: ten of a studio campaign and ten of the headless one', async () => {
    let total = 0
    let checked = 0
    let n = 0
    for (const out of OUTS) {
      for (const id of IDS) {
        n++
        const deps = makeDeps(ws)
        const rec = session(deps, explainPrompt(out, id), { autoApprove: 'reads' })
        await runTurn(rec, `mx-${n}`, new AbortController().signal, deps)
        expect(toolUsesOf(rec.messages[1])).toEqual([{ id: expect.any(String), name: 'autonomy_attempts', input: { out, view: 'attempts', geometryId: id, detail: 'brief' } }])
        const g = lintSession(rec.messages)
        expect(g.explanations).toBe(1)
        if (g.ungrounded !== 0) console.log(JSON.stringify(g.messages))
        expect(g.ungrounded).toBe(0)
        expect(g.checked).toBeGreaterThanOrEqual(3)
        expect(g.messages[g.messages.length - 1].text).toContain(`Geometry ${id} in ${out}`)
        total += g.explanations
        checked += g.checked
      }
    }
    expect(total).toBe(20)
    expect(checked).toBeGreaterThan(100)
  }, 60_000)
  it('a zai glm-5.3-flash provider that rounds is grounded too, and one invented number is caught', async () => {
    let total = 0
    let n = 0
    for (const out of OUTS) {
      for (const id of IDS) {
        n++
        const deps = makeDeps(ws, { llm: planLlm('zai', 'glm-5.3-flash', [
          { blocks: [{ type: 'tool_use', name: 'autonomy_attempts', input: { out, geometryId: id } }], stopReason: 'tool_use' },
          { blocks: [{ type: 'text', text: roundedText(id) }], stopReason: 'end_turn' },
        ]) })
        const rec = session(deps, explainPrompt(out, id), { autoApprove: 'reads' })
        await runTurn(rec, `gx-${n}`, new AbortController().signal, deps)
        const g = lintSession(rec.messages)
        expect(g.explanations).toBe(1)
        expect(g.ungrounded).toBe(0)
        expect(g.checked).toBeGreaterThanOrEqual(3)
        total += g.explanations
      }
    }
    expect(total).toBe(20)
    const deps = makeDeps(ws, { llm: planLlm('zai', 'glm-5.3-flash', [
      { blocks: [{ type: 'tool_use', name: 'autonomy_attempts', input: { out: OUTS[0], geometryId: IDS[0] } }], stopReason: 'tool_use' },
      { blocks: [{ type: 'text', text: roundedText(IDS[0]) + '\nIt would need 12345.678 more cells.' }], stopReason: 'end_turn' },
    ]) })
    const rec = session(deps, explainPrompt(OUTS[0], IDS[0]), { autoApprove: 'reads' })
    await runTurn(rec, 'gx-invented', new AbortController().signal, deps)
    const g = lintSession(rec.messages)
    expect(g.ungrounded).toBe(1)
    expect(g.messages.flatMap((m) => m.ungrounded.map((u) => u.raw))).toEqual(['12345.678'])
  }, 60_000)
  it('the service lints a stored session, the source of GET /api/sessions/:id/grounding', async () => {
    const agent = createAgentService({ config: ws.config, hub: fakeHub(), runs: fakeRuns(), datasets: fakeDatasets(), llm: createMockLlm({ delayMs: 0 }), retryDelayMs: 5 })
    const r = await agent.chat({ sessionId: null, text: explainPrompt(OUTS[0], IDS[0]), attachments: [], attachmentIds: [], activeFile: null, autoApprove: 'reads', locale: 'en', timeoutMs: 20000 })
    expect(r.status).toBe('done')
    const g = agent.groundingOf!(r.sessionId)!
    expect(g.explanations).toBe(1)
    expect(g.ungrounded).toBe(0)
    expect(g.checked).toBeGreaterThanOrEqual(3)
    expect(agent.groundingOf!('no-such-session')).toBeNull()
    await agent.shutdown()
  })
})
