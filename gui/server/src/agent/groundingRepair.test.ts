// Proves the reply that ends a campaign turn is repaired once before it is shown, what stays
// ungrounded is marked on screen, and the session log records every repair with its tokens.
import fsp from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import type { BetaMessage, BetaMessageParam, BetaRawMessageStreamEvent } from '@anthropic-ai/sdk/resources/beta/messages/messages'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { closeOntologyHandles } from '../ontology/handle.js'
import { explainPrompt } from '../tools/neutrality.js'
import { extractNumbers, lintSession, nearestSources, sessionSources } from './grounding.js'
import { markUngrounded, REPAIR_MARK, repairPrompt } from './groundingRepair.js'
import type { LlmClient } from './llm.js'
import { runTurn } from './loop.js'
import { createMockLlm, makeMessage, mockEvents, type MockPlan } from './mockLlm.js'
import { createAgentService } from './service.js'
import { appendUserTurn, createSessionStore, type SessionRecord } from './session.js'
import { fakeDatasets, fakeHub, fakeRuns, makeWorkspace, type TempWorkspace } from './test-fakes.js'
import { makeDeps, textOf, type TestDeps } from './test-util.js'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const DET = path.join(HERE, '..', 'tools', 'fixtures', 'autonomy', 'det_a')

let ws: TempWorkspace
beforeAll(async () => {
  ws = await makeWorkspace()
  await fsp.cp(DET, path.join(ws.root, 'campaigns/studio'), { recursive: true })
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

const OUT = 'campaigns/studio'
const GID = 'A-1-002'
const CALL: MockPlan = { blocks: [{ type: 'tool_use', name: 'autonomy_attempts', input: { out: OUT, geometryId: GID } }], stopReason: 'tool_use' }
const say = (text: string): MockPlan => ({ blocks: [{ type: 'text', text }], stopReason: 'end_turn' })
// A-1-002 attempt 2 in det_a: trigger.value 0.10465316921661107 against 0.05, n_cells 298435. Two digits mis-copied:
const DRAFT = 'Geometry A-1-002: attempt 2 was decided by the remedy layer (RM-SNAP-FT): outcome.pinned_frac was 0.1064 against 0.05, so the verdict was fail, with 298,453 cells.'
const FIXED = 'Geometry A-1-002: attempt 2 was decided by the remedy layer (RM-SNAP-FT): outcome.pinned_frac was 0.1047 against 0.05, so the verdict was fail, with 298,435 cells.'
const HALF = 'Geometry A-1-002: attempt 2 was decided by the remedy layer (RM-SNAP-FT): outcome.pinned_frac was 0.1047 against 0.05, so the verdict was fail, with 298,453 cells.'
const MSGS = [
  { role: 'user', content: 'Explain D-1-002; it had 7 attempts' },
  { role: 'assistant', content: [{ type: 'text', text: 'Reading 7 rows.' }, { type: 'tool_use', id: 't1', name: 'autonomy_attempts', input: {} }, { type: 'tool_use', id: 't2', name: 'file_read', input: {} }] },
  { role: 'user', content: [{ type: 'tool_result', tool_use_id: 't1', content: JSON.stringify({ total: 2, rows: [{ attempt: 1, outcome: { n_cells: 87166 } }] }) }, { type: 'tool_result', tool_use_id: 't2', content: [{ type: 'text', text: 'pinned 3108 points' }] }] },
] as BetaMessageParam[]
const deltas = (deps: TestDeps): string => deps.hub.of('msg.delta').map((f) => f.delta).join('')

describe('groundingRepair', () => {
  it("sessionSources keeps lintSession's numbers with tool, call id and path, and nearestSources ranks them", () => {
    const S = sessionSources(MSGS)
    expect(S).toEqual([
      { value: 2, tool: 'autonomy_attempts', toolUseId: 't1', path: '/total' },
      { value: 1, tool: 'autonomy_attempts', toolUseId: 't1', path: '/rows/0/attempt' },
      { value: 87166, tool: 'autonomy_attempts', toolUseId: 't1', path: '/rows/0/outcome/n_cells' },
      { value: 3108, tool: 'file_read', toolUseId: 't2', path: '/' },
    ])
    expect(nearestSources({ value: 87200, percent: false }, S)[0]).toEqual(S[2])
    expect(nearestSources({ value: 5, percent: false }, S).map((e) => e.value)).toEqual([2, 1, 3108])
    expect(nearestSources({ value: 91.4, percent: true }, [{ value: 90, tool: 'x', toolUseId: 'a', path: '/p' }, { value: 0.914, tool: 'x', toolUseId: 'a', path: '/q' }])[0].path).toBe('/q')
    expect(nearestSources({ value: 7, percent: false }, [...S, ...S], 2)).toHaveLength(2)
  })

  it('repairPrompt names each number, its context and the closest source numbers; markUngrounded marks them', () => {
    const src = [
      { value: 0.10465316921661107, tool: 'autonomy_attempts', toolUseId: 'toolu_x', path: '/rows/1/trigger/value' },
      { value: 298435, tool: 'autonomy_attempts', toolUseId: 'toolu_x', path: '/rows/1/outcome/n_cells' },
    ]
    const bad = extractNumbers(DRAFT).filter((s) => s.raw === '0.1064' || s.raw === '298,453')
    expect(bad).toHaveLength(2)
    const p = repairPrompt(DRAFT, bad, src)
    expect(p.startsWith('[grounding check] Your reply has not been shown yet. It states 2 number(s)')).toBe(true)
    expect(p).toContain('- "0.1064" in "...')
    expect(p).toContain('0.10465316921661107 (autonomy_attempts toolu_x at /rows/1/trigger/value)')
    expect(p).toContain('- "298,453" in "...')
    expect(p).toContain('298435 (autonomy_attempts toolu_x at /rows/1/outcome/n_cells)')
    expect(p).toContain('Do not call a tool')
    expect(p.split('\n')).toHaveLength(4)
    expect(markUngrounded(DRAFT, bad)).toBe(DRAFT.replace('0.1064', '0.1064 [?]').replace('298,453', '298,453 [?]'))
    expect(markUngrounded(DRAFT, [])).toBe(DRAFT)
    expect(REPAIR_MARK).toBe(' [?]')
  })

  it('a mis-copied campaign reply is repaired by one more call before it is shown', async () => {
    const deps = makeDeps(ws, { llm: planLlm('zai', 'glm-5.3-flash', [CALL, say(DRAFT), say(FIXED)]) })
    const rec = session(deps, explainPrompt(OUT, GID), { autoApprove: 'reads' })
    const out = await runTurn(rec, 'rp-1', new AbortController().signal, deps)
    expect(out.status).toBe('done')
    expect(deps.calls).toHaveLength(3)
    expect(out.usage.inputTokens).toBe(30)
    const req = deps.calls[2]
    expect(req.at(-2)).toEqual({ role: 'assistant', content: [{ type: 'text', text: DRAFT }] })
    expect(req.at(-1)!.role).toBe('user')
    expect(textOf(req.at(-1)!).startsWith('[grounding check]')).toBe(true)
    for (const s of ['"0.1064"', '"298,453"', '0.10465316921661107', '298435', 'autonomy_attempts']) expect(textOf(req.at(-1)!)).toContain(s)
    expect(rec.messages).toHaveLength(4)
    expect(textOf(rec.messages[3])).toBe(FIXED)
    expect(lintSession(rec.messages).explanations).toBe(1)
    expect(lintSession(rec.messages).ungrounded).toBe(0)
    expect(rec.repairs).toHaveLength(1)
    expect(rec.repairs![0]).toMatchObject({ schema: 'autonomy-grounding-repair/1', turnId: 'rp-1', index: 3, model: 'glm-5.3-flash', fixed: ['0.1064', '298,453'], remaining: [], replaced: true, error: null, before: { text: DRAFT, checked: 4 }, after: { text: FIXED, checked: 4, ungrounded: [] } })
    expect(rec.repairs![0].before.ungrounded.map((u) => u.raw)).toEqual(['0.1064', '298,453'])
    expect(rec.repairs![0].usage.inputTokens).toBe(10)
    expect(rec.repairs![0].usage.outputTokens).toBeGreaterThan(0)
    expect(deltas(deps)).not.toContain('0.1064')
    expect(deltas(deps)).not.toContain('298,453')
    const done = deps.hub.of('msg.done').at(-1)!.message
    expect(done.blocks).toEqual([
      { kind: 'text', text: FIXED },
      { kind: 'notice', level: 'info', text: 'Checked against the tool results before it was shown: 2 number(s) corrected (0.1064, 298,453).' },
    ])
    expect(rec.ui.at(-1)!.blocks).toEqual(done.blocks)
  })

  it('a reply sent together with suggest_followups is the reply too, and is repaired before it is shown', async () => {
    // Seen live on glm-5.3-flash: the explanation and its follow-up chips arrive in one message.
    const withChips: MockPlan = { blocks: [{ type: 'text', text: DRAFT }, { type: 'tool_use', name: 'suggest_followups', input: { items: ['Explain A-1-005'] } }], stopReason: 'tool_use' }
    const deps = makeDeps(ws, { llm: planLlm('zai', 'glm-5.3-flash', [CALL, withChips, say(FIXED), say('Done.')]) })
    const rec = session(deps, explainPrompt(OUT, GID), { autoApprove: 'reads' })
    const out = await runTurn(rec, 'rp-6', new AbortController().signal, deps)
    expect(out.status).toBe('done')
    expect(deps.calls).toHaveLength(4)
    expect(textOf(deps.calls[2].at(-1)!).startsWith('[grounding check]')).toBe(true)
    expect(textOf(rec.messages[3])).toBe(FIXED)
    expect(rec.messages[3].content).toEqual([{ type: 'text', text: FIXED }, expect.objectContaining({ type: 'tool_use', name: 'suggest_followups' })])
    expect(rec.repairs).toHaveLength(1)
    expect(rec.repairs![0]).toMatchObject({ index: 3, fixed: ['0.1064', '298,453'], remaining: [], replaced: true })
    expect(lintSession(rec.messages).ungrounded).toBe(0)
    expect(deltas(deps)).not.toContain('0.1064')
  })

  it('what the repair leaves ungrounded is shown marked, and a failed repair call keeps the draft marked', async () => {
    const deps = makeDeps(ws, { llm: planLlm('zai', 'glm-5.3-flash', [CALL, say(DRAFT), say(HALF)]) })
    const rec = session(deps, explainPrompt(OUT, GID), { autoApprove: 'reads' })
    await runTurn(rec, 'rp-2', new AbortController().signal, deps)
    expect(textOf(rec.messages[3])).toBe(HALF)
    expect(lintSession(rec.messages).messages.flatMap((m) => m.ungrounded.map((u) => u.raw))).toEqual(['298,453'])
    expect(rec.repairs![0]).toMatchObject({ fixed: ['0.1064'], remaining: ['298,453'], replaced: true, error: null })
    expect(deps.hub.of('msg.done').at(-1)!.message.blocks).toEqual([
      { kind: 'text', text: HALF.replace('298,453', '298,453 [?]') },
      { kind: 'notice', level: 'warning', text: '1 number(s) in this reply are in no tool result the assistant saw, after one correction round: 298,453. Each is marked [?].' },
    ])
    const deps2 = makeDeps(ws, { llm: planLlm('zai', 'glm-5.3-flash', [CALL, say(DRAFT), { blocks: [], stopReason: 'end_turn', throwError: new Error('429 Limit Exhausted') }]) })
    const rec2 = session(deps2, explainPrompt(OUT, GID), { autoApprove: 'reads' })
    const out = await runTurn(rec2, 'rp-3', new AbortController().signal, deps2)
    expect(out.status).toBe('done')
    expect(deps2.calls).toHaveLength(3)
    expect(textOf(rec2.messages[3])).toBe(DRAFT)
    expect(rec2.repairs![0]).toMatchObject({ replaced: false, fixed: [], remaining: ['0.1064', '298,453'], model: null })
    expect(rec2.repairs![0].error).toContain('429 Limit Exhausted')
    const done = deps2.hub.of('msg.done').at(-1)!.message
    expect(done.blocks[0]).toEqual({ kind: 'text', text: DRAFT.replace('0.1064', '0.1064 [?]').replace('298,453', '298,453 [?]') })
    expect(done.blocks[1]).toMatchObject({ level: 'warning' })
  })

  it('no repair where none is due, and those frames reach the screen', async () => {
    const deps = makeDeps(ws)
    const rec = session(deps, explainPrompt(OUT, GID), { autoApprove: 'reads' })
    await runTurn(rec, 'rp-4', new AbortController().signal, deps)
    expect(deps.calls).toHaveLength(2)
    expect(rec.repairs).toBeUndefined()
    expect(lintSession(rec.messages).ungrounded).toBe(0)
    expect(deltas(deps)).toContain(`Geometry ${GID} in ${OUT}`)
    const deps2 = makeDeps(ws, { llm: planLlm('zai', 'glm-5.3-flash', [say('There were 12345.678 cells.')]) })
    const rec2 = session(deps2, explainPrompt(OUT, GID), { autoApprove: 'reads' })
    await runTurn(rec2, 'rp-5', new AbortController().signal, deps2)
    expect(deps2.calls).toHaveLength(1)
    expect(rec2.repairs).toBeUndefined()
    expect(deltas(deps2)).toContain('12345.678')
    expect(lintSession(rec2.messages).messages[0]).toMatchObject({ campaign: false, ungrounded: [{ raw: '12345.678' }] })
  })

  it('the service serves the repairs with the lint, and a reloaded store keeps them', { timeout: 30_000 }, async () => {
    const agent = createAgentService({ config: ws.config, hub: fakeHub(), runs: fakeRuns(), datasets: fakeDatasets(), llm: planLlm('zai', 'glm-5.3-flash', [CALL, say(DRAFT), say(FIXED)]), retryDelayMs: 5 })
    const r = await agent.chat({ sessionId: null, text: explainPrompt(OUT, GID), attachments: [], attachmentIds: [], activeFile: null, autoApprove: 'reads', locale: 'en', timeoutMs: 20000 })
    expect(r.status).toBe('done')
    const g = agent.groundingOf!(r.sessionId)!
    expect(g.explanations).toBe(1)
    expect(g.ungrounded).toBe(0)
    expect(g.repairs).toHaveLength(1)
    expect(g.repairs![0].fixed).toEqual(['0.1064', '298,453'])
    await agent.shutdown()
    expect(createSessionStore(ws.config.sessionsDir, 'm').get(r.sessionId)!.repairs!.map((x) => x.fixed)).toEqual([['0.1064', '298,453']])
    const agent2 = createAgentService({ config: ws.config, hub: fakeHub(), runs: fakeRuns(), datasets: fakeDatasets(), llm: createMockLlm({ delayMs: 0 }), retryDelayMs: 5 })
    expect(agent2.groundingOf!('no-such-session')).toBeNull()
    await agent2.shutdown()
  })
})
