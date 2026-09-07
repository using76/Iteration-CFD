import fsp from 'node:fs/promises'
import path from 'node:path'
import type { BetaMessageParam, BetaRawMessageStreamEvent } from '@anthropic-ai/sdk/resources/beta/messages/messages'
import { DEFAULT_SESSION_SETTINGS, TOOL_NAMES, type RunInfo, type ServerMsg, type ToolCallRecord } from '@cfd/shared'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { STATIC_SYSTEM } from '../prompts/system.js'
import { buildStreamParams } from './anthropic.js'
import { createApprovalManager } from './approvals.js'
import { classifyTool, parsePolicyOverrides } from './policy.js'
import { buildVolatileContext, foldContextIntoUser, systemParam } from './prompt.js'
import { buildQuickMessage } from './quick.js'
import { appendUserTurn, createSessionStore, repairDanglingToolUses, stateOf, titleFromText } from './session.js'
import { fakeRuns, makeWorkspace, type TempWorkspace } from './test-fakes.js'
import { until } from './test-util.js'
import { createStreamProjector, projectAssistant, projectUser } from './ui-projection.js'
import { detectScenario, extractFacts } from './mockLlm.js'

let ws: TempWorkspace
beforeAll(async () => {
  ws = await makeWorkspace()
})
afterAll(() => ws.cleanup())

describe('session store', () => {
  it('round-trips a session through disk atomically', async () => {
    const dir = path.join(ws.tmp, 'sessions-rt')
    const store = createSessionStore(dir, 'claude-opus-5')
    const rec = store.create({ locale: 'en' })
    appendUserTurn(rec, { role: 'user', content: 'A rather long first message that should become the title of the session, truncated' }, { synthetic: false, entitle: true })
    rec.messages.push({ role: 'assistant', content: [{ type: 'thinking', thinking: 't', signature: 'sig' }, { type: 'text', text: 'hi' }] })
    rec.runs.push('r_1')
    rec.allowedTools.push('case_edit')
    await store.save(rec)
    expect(rec.title.length).toBeLessThanOrEqual(60)
    expect(rec.title.endsWith('…')).toBe(true)
    const files = await fsp.readdir(dir)
    expect(files).toEqual([`${rec.id}.json`])
    const again = createSessionStore(dir, 'claude-opus-5')
    const loaded = again.get(rec.id)!
    expect(loaded.messages).toEqual(rec.messages)
    expect(loaded.ui).toEqual(rec.ui)
    expect(loaded.runs).toEqual(['r_1'])
    expect(loaded.allowedTools).toEqual(['case_edit'])
    expect(loaded.settings).toEqual({ ...DEFAULT_SESSION_SETTINGS, locale: 'en' })
    expect(again.list().map((s) => s.id)).toEqual([rec.id])
    const state = stateOf(loaded, { pendingApprovals: [], turnActive: false, customTools: [] })
    expect(state.messages).toHaveLength(1)
    expect(await again.delete(rec.id)).toBe(true)
    expect(await fsp.readdir(dir)).toEqual([])
  })

  it('repairs a tool_use that never got its result', () => {
    const messages: BetaMessageParam[] = [
      { role: 'user', content: 'run it' },
      { role: 'assistant', content: [{ type: 'text', text: 'starting' }, { type: 'tool_use', id: 'toolu_a', name: 'run_start', input: {} }] },
    ]
    expect(repairDanglingToolUses(messages)).toBe(1)
    expect(messages).toHaveLength(3)
    const blocks = messages[2].content as Array<{ type: string; tool_use_id?: string; is_error?: boolean; content?: string }>
    expect(messages[2].role).toBe('user')
    expect(blocks[0]).toMatchObject({ type: 'tool_result', tool_use_id: 'toolu_a', is_error: true })
    expect(JSON.parse(blocks[0].content!).error.code).toBe('INTERRUPTED')
    // Idempotent: a repaired history is already answered.
    expect(repairDanglingToolUses(messages)).toBe(0)
  })

  it('repairs only the unanswered half of a partly answered round', () => {
    const messages: BetaMessageParam[] = [
      { role: 'user', content: 'two things' },
      {
        role: 'assistant',
        content: [
          { type: 'tool_use', id: 'toolu_a', name: 'file_read', input: {} },
          { type: 'tool_use', id: 'toolu_b', name: 'run_start', input: {} },
        ],
      },
      { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'toolu_b', content: 'ok', is_error: false }] },
    ]
    expect(repairDanglingToolUses(messages)).toBe(1)
    expect(messages).toHaveLength(3)
    const blocks = messages[2].content as Array<{ type: string; tool_use_id: string }>
    // The synthetic result leads: tool_result blocks come first in the message.
    expect(blocks.map((b) => b.tool_use_id)).toEqual(['toolu_a', 'toolu_b'])
  })

  it('repairs a session file that was written mid-tool-round', async () => {
    const dir = path.join(ws.tmp, 'sessions-dangling')
    await fsp.mkdir(dir, { recursive: true })
    const rec = {
      id: 's_dangling',
      title: 't',
      createdAt: '2026-09-07T00:00:00.000Z',
      updatedAt: '2026-09-07T00:00:00.000Z',
      model: 'claude-opus-5',
      settings: DEFAULT_SESSION_SETTINGS,
      messages: [
        { role: 'user', content: 'mesh it' },
        { role: 'assistant', content: [{ type: 'tool_use', id: 'toolu_x', name: 'mesh_generate', input: {} }] },
      ],
      ui: [],
      toolCalls: [],
      runs: [],
      allowedTools: [],
    }
    await fsp.writeFile(path.join(dir, 's_dangling.json'), JSON.stringify(rec))
    const loaded = createSessionStore(dir, 'claude-opus-5').get('s_dangling')!
    expect(loaded.messages).toHaveLength(3)
    expect((loaded.messages[2].content as Array<{ tool_use_id: string }>)[0].tool_use_id).toBe('toolu_x')
  })

  it('skips corrupt files and titles empty text', async () => {
    const dir = path.join(ws.tmp, 'sessions-bad')
    await fsp.mkdir(dir, { recursive: true })
    await fsp.writeFile(path.join(dir, 'x.json'), '{not json')
    expect(createSessionStore(dir, 'm').list()).toEqual([])
    expect(titleFromText('   ', 'ko')).toBe('새 대화')
  })
})

describe('projection', () => {
  it('projects assistant content, tool cards, diffs and images', () => {
    const call: ToolCallRecord = { toolUseId: 'tu1', name: 'case_edit', input: { path: 'a' }, policy: 'ask', status: 'ok', summary: 'Edited case (a)', resultPreview: '{}', error: null, runId: null, startedAt: 1, endedAt: 2 }
    const ui = projectAssistant(
      'm1',
      [
        { type: 'thinking', thinking: 'hmm', signature: 's' },
        { type: 'redacted_thinking', data: 'x' },
        { type: 'text', text: 'Done.' },
        { type: 'tool_use', id: 'tu1', name: 'case_edit', input: { path: 'a' } },
        { type: 'tool_use', id: 'tu2', name: 'gpu_info', input: {} },
      ],
      new Map([['tu1', call]]),
      { createdAt: 5, stopReason: 'tool_use', model: 'claude-opus-5', suggestions: ['next'], extras: { diffs: [{ path: 'a', before: 'x', after: 'y', applied: true, toolUseId: 'tu1' }], images: [{ base64: 'AA', mime: 'image/png', alt: 'shot' }] } },
    )
    expect(ui.blocks.map((b) => b.kind)).toEqual(['thinking', 'thinking', 'text', 'tool', 'tool', 'diff', 'image'])
    expect(ui.blocks[3]).toEqual({ kind: 'tool', call })
    const pending = ui.blocks[4]
    expect(pending.kind === 'tool' ? pending.call.status : null).toBe('pending')
    expect(ui.model).toBe('claude-opus-5')
    expect(ui.suggestions).toEqual(['next'])
    expect(projectUser('u1', { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'x', content: '{}' }] }, { createdAt: 1, synthetic: false })).toBeNull()
    const user = projectUser('u2', { role: 'user', content: [{ type: 'text', text: 'hi' }, { type: 'text', text: '<file path="a">..</file>' }] }, { createdAt: 1, synthetic: true, notices: ['@a (3 bytes)'] })
    expect(user?.blocks.map((b) => b.kind)).toEqual(['text', 'text', 'notice'])
    expect(user?.synthetic).toBe(true)
  })

  it('streams block_start / delta / tool events and tracks completeness', () => {
    const sent: ServerMsg[] = []
    const p = createStreamProjector((m) => sent.push(m), 's', 'm1')
    const events: BetaRawMessageStreamEvent[] = [
      { type: 'content_block_start', index: 0, content_block: { type: 'thinking', thinking: '', signature: '' } },
      { type: 'content_block_delta', index: 0, delta: { type: 'thinking_delta', thinking: 'plan', estimated_tokens: null } },
      { type: 'content_block_delta', index: 0, delta: { type: 'signature_delta', signature: 'SIG' } },
      { type: 'content_block_stop', index: 0 },
      { type: 'content_block_start', index: 1, content_block: { type: 'text', text: '', citations: null } },
      { type: 'content_block_delta', index: 1, delta: { type: 'text_delta', text: 'Hel' } },
      { type: 'content_block_delta', index: 1, delta: { type: 'text_delta', text: 'lo' } },
      { type: 'content_block_stop', index: 1 },
      { type: 'content_block_start', index: 2, content_block: { type: 'tool_use', id: 'tu1', name: 'gpu_info', input: {} } },
      { type: 'content_block_delta', index: 2, delta: { type: 'input_json_delta', partial_json: '{' } },
      { type: 'content_block_delta', index: 2, delta: { type: 'input_json_delta', partial_json: '}' } },
      { type: 'content_block_stop', index: 2 },
      { type: 'content_block_start', index: 3, content_block: { type: 'tool_use', id: 'tu2', name: 'run_wait', input: {} } },
      { type: 'content_block_delta', index: 3, delta: { type: 'input_json_delta', partial_json: '{"runId": "r' } },
    ]
    for (const ev of events) p.onEvent(ev)
    expect(sent.map((m) => m.t)).toEqual(['msg.block_start', 'msg.delta', 'msg.block_start', 'msg.delta', 'msg.delta', 'tool.start', 'tool.input_delta', 'tool.input_delta', 'tool.start', 'tool.input_delta'])
    expect(sent[0]).toMatchObject({ t: 'msg.block_start', blockIndex: 0, kind: 'thinking' })
    expect(sent[5]).toMatchObject({ t: 'tool.start', blockIndex: 2, toolUseId: 'tu1', name: 'gpu_info' })
    expect(p.completeContent()).toEqual([
      { type: 'thinking', thinking: 'plan', signature: 'SIG' },
      { type: 'text', text: 'Hello' },
      { type: 'tool_use', id: 'tu1', name: 'gpu_info', input: {} },
    ])
  })
})

describe('policy', () => {
  const settings = { ...DEFAULT_SESSION_SETTINGS }
  it('applies defaults, settings, allowlist and overrides', () => {
    expect(classifyTool('case_read', {}, { settings, allowedTools: [], overrides: {} })).toBe('auto')
    expect(classifyTool('case_edit', { dryRun: false }, { settings, allowedTools: [], overrides: {} })).toBe('ask')
    expect(classifyTool('case_edit', { dryRun: true }, { settings, allowedTools: [], overrides: {} })).toBe('auto')
    expect(classifyTool('case_edit', { dryRun: false }, { settings, allowedTools: ['case_edit'], overrides: {} })).toBe('auto')
    expect(classifyTool('run_start', {}, { settings: { ...settings, autoApprove: 'all' }, allowedTools: [], overrides: {} })).toBe('auto')
    expect(classifyTool('shell_exec', {}, { settings: { ...settings, autoApprove: 'all' }, allowedTools: ['shell_exec'], overrides: {} })).toBe('never')
    expect(classifyTool('shell_exec', {}, { settings, allowedTools: [], overrides: { shell_exec: 'ask' } })).toBe('ask')
    expect(classifyTool('viewer_command', {}, { settings: { ...settings, autoApprove: 'none' }, allowedTools: [], overrides: {} })).toBe('auto')
    expect(classifyTool('run_wait', {}, { settings: { ...settings, autoApprove: 'none' }, allowedTools: [], overrides: {} })).toBe('auto')
    expect(classifyTool('my_custom', {}, { settings, allowedTools: [], overrides: {} })).toBe('ask')
    expect(parsePolicyOverrides('{"tools":{"shell_exec":"ask","x":"bogus"}}')).toEqual({ shell_exec: 'ask' })
    expect(parsePolicyOverrides('{"file_write":"auto"}')).toEqual({ file_write: 'auto' })
    expect(parsePolicyOverrides('nope')).toEqual({})
  })
})

describe('approval manager', () => {
  it('resolves per tool_use id, batches, expires and cancels', async () => {
    const resolved: Array<[string[], string]> = []
    const mgr = createApprovalManager((ids, d) => resolved.push([ids, d]))
    const req = mgr.request('t1', [
      { toolUseId: 'a', name: 'run_start', input: {}, summary: 's', preview: null },
      { toolUseId: 'b', name: 'file_write', input: {}, summary: 's', preview: 'p' },
    ])
    expect(req.approval.toolUseIds).toEqual(['a', 'b'])
    expect(mgr.pending()).toHaveLength(1)
    expect(mgr.resolve(['a'], 'approved')).toEqual(['a'])
    expect(await req.outcomes.get('a')).toEqual({ decision: 'approved', reason: null })
    expect(mgr.pending()[0].toolUseIds).toEqual(['b'])
    mgr.resolve(['b', 'zzz'], 'denied', 'no')
    expect(await req.outcomes.get('b')).toEqual({ decision: 'denied', reason: 'no' })
    expect(mgr.pending()).toHaveLength(0)
    expect(resolved).toEqual([
      [['a'], 'approved'],
      [['b'], 'denied'],
    ])
    const short = mgr.request('t2', [{ toolUseId: 'c', name: 'x', input: {}, summary: 's', preview: null }], 10)
    expect((await short.outcomes.get('c'))?.decision).toBe('expired')
    const cancelled = mgr.request('t3', [{ toolUseId: 'd', name: 'x', input: {}, summary: 's', preview: null }])
    mgr.cancelAll('bye')
    expect(await cancelled.outcomes.get('d')).toEqual({ decision: 'denied', reason: 'bye' })
  })
})

describe('prompt', () => {
  it('static system prompt is stable and mentions every binary and tool rule', () => {
    expect(STATIC_SYSTEM).toBe(STATIC_SYSTEM)
    expect(STATIC_SYSTEM).not.toMatch(/\d{4}-\d{2}-\d{2}T/)
    for (const b of ['ofgpu-k-epsilon', 'ofgpu-generate-mesh', 'ofgpu-vof', 'ofgpu-decompose']) expect(STATIC_SYSTEM).toContain(b)
    expect(STATIC_SYSTEM).toContain('kOmegaSST')
    expect(STATIC_SYSTEM).toContain('suggest_followups')
    expect(STATIC_SYSTEM).toContain('case_edit')
    expect(STATIC_SYSTEM).toContain('Korean')
    const sys = systemParam()
    expect(sys[0].cache_control).toEqual({ type: 'ephemeral' })
    const params = buildStreamParams('claude-opus-5', { system: sys, messages: [], tools: [], maxTokens: 64000, effort: 'high', signal: new AbortController().signal })
    expect(params).toMatchObject({ model: 'claude-opus-5', max_tokens: 64000, thinking: { type: 'adaptive', display: 'summarized' }, output_config: { effort: 'high' }, betas: ['server-side-fallback-2026-07-01'], fallbacks: 'default' })
    // Without this the whole conversation is re-processed uncached each round.
    expect(params.cache_control).toEqual({ type: 'ephemeral' })
  })

  it('volatile context lists runs in a parseable form and folds into the user turn', () => {
    const runs = fakeRuns()
    const text = buildVolatileContext({
      workspaceRoot: '/w',
      mode: 'demo',
      gpu: runs.gpu(),
      runs: [{ id: 'r_1', binary: 'ofgpu-k-epsilon', status: 'running', iter: 120, targetIter: 400, casePath: 'cases/plume.jsonc', written: [], error: null } as unknown as RunInfo],
      context: { activeFile: 'cases/plume.jsonc', activeRun: 'r_1', attachments: [], selection: null },
      customTools: ['t1'],
      locale: 'ko',
      now: new Date('2026-09-07T00:00:00Z'),
    })
    expect(text).toContain('- run r_1 | ofgpu-k-epsilon | running | iter 120/400 | case cases/plume.jsonc')
    expect(text).toContain("User's active file: cases/plume.jsonc")
    expect(text).toContain('Local time: 2026-09-07T00:00:00.000Z')
    const folded = foldContextIntoUser([{ role: 'user', content: 'hi' }, { role: 'assistant', content: 'yo' }, { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'x', content: '{}' }] }], 'CTX')
    const last = folded[2]
    expect(Array.isArray(last.content) && last.content.length).toBe(2)
    expect(Array.isArray(last.content) && last.content[1].type === 'text' ? last.content[1].text : '').toBe('[context]\nCTX')
    expect(folded[0].content).toBe('hi')
  })
})

describe('quick actions', () => {
  it('builds the messages, attaching the log tail for explain', async () => {
    const runs = fakeRuns({ endStatus: 'failed' })
    await runs.start({ binary: 'ofgpu-k-epsilon', casePath: 'cases/plume.jsonc', args: [], positionals: [], label: null, sessionId: null })
    await until(() => runs.get('r_1')?.status === 'failed')
    const base = { casePath: null, runId: null, activeFile: null, locale: 'en' as const, runs }
    expect(buildQuickMessage({ ...base, action: 'mesh', casePath: 'cases/channel' }).text).toContain('Generate a mesh for cases/channel')
    expect(buildQuickMessage({ ...base, action: 'run', activeFile: 'cases/plume.jsonc' }).text).toContain('Run the appropriate solver for cases/plume.jsonc')
    const explain = buildQuickMessage({ ...base, action: 'explain' })
    expect(explain.text).toContain('Explain the error in run r_1')
    expect(explain.text).toContain('Problems:')
    expect(explain.text).toContain('kOmegaSST')
    expect(explain.text).toContain('```')
    expect(buildQuickMessage({ ...base, action: 'postprocess', casePath: 'cases/plume.jsonc' }).text).toContain('cases/plume_jsonc')
    expect(buildQuickMessage({ ...base, action: 'validate' }).text).toContain('ofgpu-validate')
    expect(buildQuickMessage({ ...base, action: 'export' }).text).toContain('field_stats')
    expect(buildQuickMessage({ ...base, action: 'create_tool', locale: 'ko' }).text).toContain('custom_tool_create')
    expect(buildQuickMessage({ ...base, action: 'run', locale: 'ko' }).text).toContain('솔버')
  })
})

describe('mock script', () => {
  it('detects scenarios in both languages and reads facts from tool results', () => {
    expect(detectScenario('메쉬 만들어줘')).toBe('mesh')
    expect(detectScenario('Run the solver')).toBe('run')
    expect(detectScenario('3D 뷰어로 보여줘')).toBe('viewer')
    expect(detectScenario('explain the error')).toBe('explain')
    expect(detectScenario('endTime 바꿔줘')).toBe('edit')
    expect(detectScenario('error-test')).toBe('error')
    expect(detectScenario('what is this')).toBe('default')
    const facts = extractFacts([
      { role: 'user', content: 'run it' },
      { role: 'assistant', content: [{ type: 'tool_use', id: 't1', name: 'run_start', input: { binary: 'x' } }] },
      { role: 'user', content: [{ type: 'tool_result', tool_use_id: 't1', content: JSON.stringify({ runId: 'r_9', written: ['cases/plume_jsonc/1'] }) }] },
      { role: 'system', content: '- run r_2 | x | done | iter 1' },
    ])
    expect(facts.lastRunId).toBe('r_9')
    expect(facts.lastWrittenDir).toBe('cases/plume_jsonc/1')
    expect(facts.lastResult?.name).toBe('run_start')
    expect(facts.results).toHaveLength(1)
    expect(facts.korean).toBe(false)
    expect(TOOL_NAMES).toContain('suggest_followups')
  })
})
