import fsp from 'node:fs/promises'
import path from 'node:path'
import Anthropic from '@anthropic-ai/sdk'
import type { BetaMessage, BetaRawMessageStreamEvent } from '@anthropic-ai/sdk/resources/beta/messages/messages'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { setSchemaValidator, structuralValidate } from '../tools/case.js'
import type { LlmClient } from './llm.js'
import { MAX_TOOL_ROUNDS, runTurn } from './loop.js'
import { makeMessage, mockEvents, type MockPlan } from './mockLlm.js'
import { BUDGET_EXHAUSTED_TEXT, runNoticeText } from './prompt.js'
import { appendUserTurn, type SessionRecord } from './session.js'
import { fakeRuns, makeWorkspace, type TempWorkspace } from './test-fakes.js'
import { makeDeps, textOf, toolResultsOf, toolUsesOf, until, type TestDeps } from './test-util.js'

let ws: TempWorkspace
beforeAll(async () => {
  ws = await makeWorkspace()
  setSchemaValidator(structuralValidate)
})
afterAll(async () => {
  setSchemaValidator(null)
  await ws.cleanup()
})

function session(deps: TestDeps, text: string, settings: Partial<SessionRecord['settings']> = {}): SessionRecord {
  const rec = deps.store.create({ locale: 'en', ...settings })
  appendUserTurn(rec, { role: 'user', content: text }, { synthetic: false, entitle: true })
  return rec
}

/** An LlmClient that always answers with the same plan (for edge cases the script does not cover). */
function constantLlm(plan: MockPlan | ((n: number) => MockPlan)): LlmClient {
  let n = 0
  return {
    kind: 'mock',
    model: 'claude-test',
    stream(params) {
      const p = typeof plan === 'function' ? plan(n++) : plan
      let final: BetaMessage | null = null
      const gen = mockEvents(p, 'claude-test', { signal: params.signal, delayMs: 0, inputTokens: 10 }, (m) => (final = m))
      return {
        events: gen as AsyncIterable<BetaRawMessageStreamEvent>,
        finalMessage: async () => final ?? makeMessage('claude-test', [], 'end_turn', null, 10),
      }
    },
  }
}

describe('runTurn with the mock LLM', () => {
  it('drives the mesh scenario through the real mesh_generate tool', async () => {
    const deps = makeDeps(ws)
    const rec = session(deps, 'Generate a mesh for the channel case', { autoApprove: 'all' })
    const outcome = await runTurn(rec, 't1', new AbortController().signal, deps)
    expect(outcome.status).toBe('done')
    expect(outcome.model).toBe('claude-opus-5')
    const roles = rec.messages.map((m) => m.role)
    expect(roles).toEqual(['user', 'assistant', 'user', 'assistant', 'user', 'assistant'])
    const meshUse = toolUsesOf(rec.messages[1])
    expect(meshUse[0].name).toBe('mesh_generate')
    const results = toolResultsOf(rec.messages[2])
    expect(results).toHaveLength(1)
    expect(results[0].tool_use_id).toBe(meshUse[0].id)
    expect(results[0].is_error).toBe(false)
    expect(JSON.parse(results[0].content).cells).toBe(24000)
    expect(toolUsesOf(rec.messages[3])[0].name).toBe('suggest_followups')
    expect(textOf(rec.messages[5])).toMatch(/24,000 cells/)
    expect(deps.runs.started[0].binary).toBe('ofgpu-generate-mesh')
    expect(deps.runs.started[0].positionals).toEqual(['channel', 'cases/channel'])
    expect(rec.runs).toEqual(['r_1'])
    // UI projection: three assistant messages, tool cards ok, suggestions on the last one.
    const assistants = rec.ui.filter((m) => m.role === 'assistant')
    expect(assistants).toHaveLength(3)
    const card = assistants[0].blocks.find((b) => b.kind === 'tool')
    expect(card && card.kind === 'tool' ? card.call.status : null).toBe('ok')
    expect(card && card.kind === 'tool' ? card.call.summary : null).toBe('Generated structured mesh (24,000 cells)')
    expect(assistants[2].suggestions).toHaveLength(3)
    expect(rec.title).toBe('Generate a mesh for the channel case')
    const t = deps.hub.of('turn.done')
    expect(t).toHaveLength(1)
    expect(t[0].usage.inputTokens).toBeGreaterThan(0)
    expect(deps.hub.of('msg.delta').length).toBeGreaterThan(0)
    expect(deps.hub.of('tool.start')[0].name).toBe('mesh_generate')
    expect(deps.hub.of('tool.input_delta').length).toBeGreaterThan(0)
    expect(deps.hub.of('tool.update').map((u) => u.call.status)).toContain('running')
    // Persisted on disk.
    const file = JSON.parse(await fsp.readFile(path.join(ws.config.sessionsDir, `${rec.id}.json`), 'utf8')) as SessionRecord
    expect(file.messages).toHaveLength(6)
  })

  it('places the volatile system message last on every request and never persists it', async () => {
    const deps = makeDeps(ws, { customTools: () => ['echo_case'] })
    const rec = session(deps, '이 케이스에 대해 알려줘')
    await runTurn(rec, 't2', new AbortController().signal, deps)
    expect(deps.calls.length).toBeGreaterThanOrEqual(2)
    for (const messages of deps.calls) {
      const last = messages[messages.length - 1]
      expect(last.role).toBe('system')
      expect(String(last.content)).toContain('Workspace root:')
      expect(String(last.content)).toContain('Custom tools: echo_case')
      expect(messages.filter((m) => m.role === 'system')).toHaveLength(1)
    }
    expect(rec.messages.some((m) => m.role === 'system')).toBe(false)
    expect(textOf(rec.messages[rec.messages.length - 1])).toContain('meteor-cfd')
  })

  it('runs the solver scenario: run_start (approved by settings), run_wait until done, summary', async () => {
    const deps = makeDeps(ws)
    const rec = session(deps, 'Run the solver on cases/plume.jsonc', { autoApprove: 'all' })
    const outcome = await runTurn(rec, 't3', new AbortController().signal, deps)
    expect(outcome.status).toBe('done')
    const names = rec.messages.flatMap((m) => toolUsesOf(m).map((u) => u.name))
    expect(names).toEqual(['run_start', 'run_wait', 'suggest_followups'])
    expect(deps.runs.started[0].args).toEqual([
      { flag: '-iters', value: 4000 },
      { flag: '-check', value: 100 },
    ])
    expect(textOf(rec.messages[rec.messages.length - 1])).toMatch(/status \*\*done\*\*/)
    expect(textOf(rec.messages[rec.messages.length - 1])).toContain('cases/plume_jsonc/1')
  })

  it('runs the viewer scenario through hub.requestViewer', async () => {
    const deps = makeDeps(ws)
    const rec = session(deps, '결과를 3D 뷰어로 시각화해줘')
    await runTurn(rec, 't4', new AbortController().signal, deps)
    expect(deps.hub.viewerCalls.map((c) => c.type)).toEqual(['load', 'addSlice', 'addStreamlines', 'setRepresentation'])
    expect(deps.hub.viewerCalls[0]).toMatchObject({ type: 'load', path: 'cases/plume_jsonc', timeIndex: 'last', field: 'U' })
    expect(textOf(rec.messages[rec.messages.length - 1])).toMatch(/절단면/)
    const results = rec.messages.flatMap(toolResultsOf)
    expect(results.every((r) => !r.is_error)).toBe(true)
  })

  it('reports NO_VIEWER as an error result the script explains', async () => {
    const deps = makeDeps(ws)
    deps.hub.viewerResult = null
    const rec = session(deps, 'show the 3d viewer')
    await runTurn(rec, 't5', new AbortController().signal, deps)
    const results = rec.messages.flatMap(toolResultsOf)
    expect(results[0].is_error).toBe(true)
    expect(JSON.parse(results[0].content).error.code).toBe('NO_VIEWER')
    expect(textOf(rec.messages[rec.messages.length - 1])).toMatch(/No 3D viewer/)
  })

  it('explains a failed run from its log', async () => {
    const runs = fakeRuns({ endStatus: 'failed' })
    const deps = makeDeps(ws, { runs })
    await runs.start({ binary: 'ofgpu-k-epsilon', casePath: 'cases/plume.jsonc', args: [], positionals: [], label: null, sessionId: null })
    await until(() => runs.get('r_1')?.status === 'failed')
    const rec = session(deps, 'Explain the error')
    await runTurn(rec, 't6', new AbortController().signal, deps)
    expect(toolUsesOf(rec.messages[1])[0]).toMatchObject({ name: 'run_log', input: { runId: 'r_1', maxLines: 200 } })
    expect(textOf(rec.messages[rec.messages.length - 1])).toMatch(/kOmegaSST/)
  })

  it('denies policy-never tools without running them', async () => {
    const deps = makeDeps(ws)
    const rec = session(deps, 'run a shell command please', { autoApprove: 'all' })
    await runTurn(rec, 't7', new AbortController().signal, deps)
    const results = toolResultsOf(rec.messages[2])
    expect(results[0].is_error).toBe(true)
    expect(JSON.parse(results[0].content).error.code).toBe('DENIED')
    expect(deps.hub.of('tool.approval_request')).toHaveLength(0)
    const card = rec.ui[1].blocks.find((b) => b.kind === 'tool')
    expect(card && card.kind === 'tool' ? [card.call.policy, card.call.status] : null).toEqual(['never', 'denied'])
    expect(textOf(rec.messages[3])).toMatch(/disabled by policy/)
  })
})

describe('approvals', () => {
  it('denied ask tools return is_error DENIED and nothing is written', async () => {
    const deps = makeDeps(ws)
    const rec = session(deps, 'edit cases/plume.jsonc endTime to 3')
    const before = await fsp.readFile(path.join(ws.root, 'cases/plume.jsonc'), 'utf8')
    const turn = runTurn(rec, 't8', new AbortController().signal, deps)
    await until(() => deps.hub.of('tool.approval_request').length === 1)
    const req = deps.hub.of('tool.approval_request')[0].approval
    expect(req.calls[0].name).toBe('case_edit')
    expect(req.calls[0].preview).toContain('+    "endTime": 3,')
    expect(deps.approvals.pending()).toHaveLength(1)
    expect(deps.hub.of('tool.update').at(-1)?.call.status).toBe('awaiting_approval')
    deps.approvals.resolve(req.toolUseIds, 'denied', 'not now')
    await turn
    const results = toolResultsOf(rec.messages[2])
    expect(results[0].is_error).toBe(true)
    expect(JSON.parse(results[0].content).error).toMatchObject({ code: 'DENIED', message: 'denied by user: not now' })
    expect(await fsp.readFile(path.join(ws.root, 'cases/plume.jsonc'), 'utf8')).toBe(before)
    expect(textOf(rec.messages[3])).toMatch(/not approved/)
    expect(deps.approvals.pending()).toHaveLength(0)
  })

  it('approved ask tools execute and produce a diff block', async () => {
    const deps = makeDeps(ws)
    const rec = session(deps, 'change cases/plume.jsonc endTime to 4')
    const turn = runTurn(rec, 't9', new AbortController().signal, deps)
    await until(() => deps.hub.of('tool.approval_request').length === 1)
    deps.approvals.resolve(deps.hub.of('tool.approval_request')[0].approval.toolUseIds, 'approved')
    await turn
    const results = toolResultsOf(rec.messages[2])
    expect(results[0].is_error).toBe(false)
    expect(JSON.parse(results[0].content).applied).toBe(true)
    expect(await fsp.readFile(path.join(ws.root, 'cases/plume.jsonc'), 'utf8')).toContain('"endTime": 4')
    const diff = rec.ui[1].blocks.find((b) => b.kind === 'diff')
    expect(diff && diff.kind === 'diff' ? diff.applied : null).toBe(true)
    expect(deps.hub.of('fs.changed').length).toBeGreaterThan(0)
    expect(textOf(rec.messages[rec.messages.length - 1])).toMatch(/Edited/)
  })

  it('expires unanswered approvals', async () => {
    const deps = makeDeps(ws)
    const rec = session(deps, 'edit cases/plume.jsonc endTime to 5')
    const original = deps.approvals.request.bind(deps.approvals)
    deps.approvals.request = (turnId, calls) => original(turnId, calls, 20)
    await runTurn(rec, 't10', new AbortController().signal, deps)
    const results = toolResultsOf(rec.messages[2])
    expect(JSON.parse(results[0].content).error.message).toMatch(/timed out/)
  })
})

describe('stop reasons and errors', () => {
  it('does not persist an empty refusal and emits turn.refusal', async () => {
    const deps = makeDeps(ws)
    const rec = session(deps, 'refuse-test')
    const outcome = await runTurn(rec, 't11', new AbortController().signal, deps)
    expect(outcome.status).toBe('refusal')
    expect(rec.messages).toHaveLength(1)
    expect(rec.ui).toHaveLength(1)
    const refusal = deps.hub.of('turn.refusal')[0]
    expect(refusal.category).toBe('general_harms')
    expect(deps.hub.of('turn.done')).toHaveLength(1)
  })

  it('repairs a max_tokens response with a dangling tool_use', async () => {
    const deps = makeDeps(ws)
    const rec = session(deps, 'long-test')
    const outcome = await runTurn(rec, 't12', new AbortController().signal, deps)
    expect(outcome.status).toBe('done')
    expect(rec.messages.map((m) => m.role)).toEqual(['user', 'assistant', 'user'])
    const uses = toolUsesOf(rec.messages[1])
    expect(uses[0].name).toBe('gpu_info')
    const results = toolResultsOf(rec.messages[2])
    expect(results[0].tool_use_id).toBe(uses[0].id)
    expect(results[0].is_error).toBe(true)
    expect(JSON.parse(results[0].content).error.code).toBe('TRUNCATED')
    expect(rec.ui[1].stopReason).toBe('max_tokens')
    expect(deps.hub.of('turn.warning')).toHaveLength(1)
  })

  it('retries once after a retryable SDK error', async () => {
    const deps = makeDeps(ws)
    const rec = session(deps, 'error-test')
    const outcome = await runTurn(rec, 't13', new AbortController().signal, deps)
    expect(outcome.status).toBe('done')
    expect(deps.hub.of('turn.warning')[0].message).toMatch(/retrying/)
    expect(deps.hub.of('turn.error')).toHaveLength(0)
    expect(textOf(rec.messages[1])).toMatch(/Recovered/)
    expect(deps.calls).toHaveLength(2)
  })

  it('reports a non-retryable error as turn.error without touching the history', async () => {
    const deps = makeDeps(ws, {
      llm: {
        kind: 'mock',
        model: 'x',
        stream() {
          throw new Error('boom')
        },
      },
    })
    const rec = session(deps, 'hello')
    const outcome = await runTurn(rec, 't14', new AbortController().signal, deps)
    expect(outcome.status).toBe('error')
    expect(deps.hub.of('turn.error')[0]).toMatchObject({ message: 'boom', retryable: false })
    expect(rec.messages).toHaveLength(1)
  })

  it('stops after the tool budget with the budget user message', async () => {
    const deps = makeDeps(ws, { llm: constantLlm({ blocks: [{ type: 'tool_use', name: 'gpu_info', input: {} }], stopReason: 'tool_use' }) })
    const rec = session(deps, 'loop forever')
    const outcome = await runTurn(rec, 't15', new AbortController().signal, deps)
    expect(outcome.status).toBe('done')
    expect(outcome.rounds).toBe(MAX_TOOL_ROUNDS + 1)
    const budget = rec.messages.find((m) => m.role === 'user' && textOf(m) === BUDGET_EXHAUSTED_TEXT)
    expect(budget).toBeDefined()
    const last = toolResultsOf(rec.messages[rec.messages.length - 1])
    expect(last[0].is_error).toBe(true)
    expect(JSON.parse(last[0].content).error.code).toBe('BUDGET')
    for (const m of rec.messages) if (m.role === 'assistant') expect(toolResultsOf(rec.messages[rec.messages.indexOf(m) + 1]).length).toBe(1)
  })

  it('folds the volatile context into the user turn when the API rejects role system', async () => {
    let n = 0
    const inner = constantLlm({ blocks: [{ type: 'text', text: 'ok' }], stopReason: 'end_turn' })
    const deps = makeDeps(ws, {
      llm: {
        kind: 'mock',
        model: 'x',
        stream(params) {
          if (n++ === 0) {
            throw new Anthropic.BadRequestError(400, { type: 'error', error: { type: 'invalid_request_error', message: 'messages: role "system" is not supported' } }, 'messages: role "system" is not supported', new Headers())
          }
          return inner.stream(params)
        },
      },
    })
    const rec = session(deps, 'hi')
    const outcome = await runTurn(rec, 't16', new AbortController().signal, deps)
    expect(outcome.status).toBe('done')
    const second = deps.calls[1]
    expect(second.every((m) => m.role !== 'system')).toBe(true)
    expect(textOf(second[second.length - 1])).toContain('[context]')
    expect(rec.messages).toHaveLength(2)
    expect(textOf(rec.messages[0])).toBe('hi')
  })
})

describe('cancellation', () => {
  it('drops a text-only partial turn', async () => {
    const deps = makeDeps(ws, { llm: undefined })
    deps.llm = (await import('./mockLlm.js')).createMockLlm({ delayMs: 10 })
    const rec = session(deps, 'tell me about the registry')
    const controller = new AbortController()
    const turn = runTurn(rec, 't17', controller.signal, deps)
    await until(() => deps.hub.of('tool.input_delta').length >= 1)
    controller.abort()
    const outcome = await turn
    expect(outcome.status).toBe('cancelled')
    expect(rec.messages).toHaveLength(1)
    expect(deps.hub.of('turn.done')).toHaveLength(1)
  })

  it('gives complete tool_use blocks a cancelled result when aborted mid-stream', async () => {
    const deps = makeDeps(ws, { llm: (await import('./mockLlm.js')).createMockLlm({ delayMs: 10 }) })
    const rec = session(deps, 'generate a mesh', { autoApprove: 'all' })
    const controller = new AbortController()
    const original = deps.emit
    deps.emit = (msg) => {
      original(msg)
      if (msg.t === 'tool.input_delta' && msg.partialJson.endsWith('}')) controller.abort()
    }
    const outcome = await runTurn(rec, 't18', controller.signal, deps)
    expect(outcome.status).toBe('cancelled')
    expect(rec.messages.map((m) => m.role)).toEqual(['user', 'assistant', 'user'])
    const uses = toolUsesOf(rec.messages[1])
    expect(uses[0].name).toBe('mesh_generate')
    const results = toolResultsOf(rec.messages[2])
    expect(results[0]).toMatchObject({ tool_use_id: uses[0].id, is_error: true })
    expect(JSON.parse(results[0].content).error.message).toBe('cancelled by user')
    expect(deps.runs.started).toHaveLength(0)
    expect(rec.ui[1].stopReason).toBe('cancelled')
  })

  it('cancels tools in flight and records cancelled results', async () => {
    const runs = fakeRuns({ finishAfterMs: null })
    const deps = makeDeps(ws, { runs })
    const rec = session(deps, 'run the solver', { autoApprove: 'all' })
    const controller = new AbortController()
    const turn = runTurn(rec, 't19', controller.signal, deps)
    await until(() => deps.hub.of('tool.update').some((u) => u.call.name === 'run_wait' && u.call.status === 'running'))
    controller.abort()
    const outcome = await turn
    expect(outcome.status).toBe('cancelled')
    const last = rec.messages[rec.messages.length - 1]
    expect(last.role).toBe('user')
    const results = toolResultsOf(last)
    expect(results[0].is_error).toBe(true)
    expect(JSON.parse(results[0].content).error.code).toBe('CANCELLED')
    expect(deps.hub.of('tool.update').at(-1)?.call.status).toBe('cancelled')
    expect(runs.get('r_1')?.status).toBe('running')
  })
})

describe('run notices', () => {
  it('replies to a [run notice] user message', async () => {
    const runs = fakeRuns()
    const deps = makeDeps(ws, { runs })
    await runs.start({ binary: 'ofgpu-k-epsilon', casePath: 'cases/plume.jsonc', args: [{ flag: '-iters', value: 400 }], positionals: [], label: null, sessionId: null })
    await until(() => runs.get('r_1')?.status === 'done')
    const rec = deps.store.create({ locale: 'en' })
    const notice = runNoticeText(runs.get('r_1')!, 'en')
    expect(notice.startsWith('[run notice] run r_1 (ofgpu-k-epsilon, cases/plume.jsonc) ended with status done after 400 iterations')).toBe(true)
    const ui = appendUserTurn(rec, { role: 'user', content: notice }, { synthetic: true })
    expect(ui?.synthetic).toBe(true)
    expect(rec.messages[0].role).toBe('user')
    await runTurn(rec, 't20', new AbortController().signal, deps)
    expect(rec.messages).toHaveLength(2)
    expect(textOf(rec.messages[1])).toMatch(/r_1 .* ended with status done after 400 iterations/)
  })
})
