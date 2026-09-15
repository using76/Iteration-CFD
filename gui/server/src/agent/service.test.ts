import fsp from 'node:fs/promises'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import type { ChatResponse } from '@cfd/shared'
import { setSchemaValidator, structuralValidate } from '../tools/case.js'
import { createMockLlm } from './mockLlm.js'
import { createAgentService } from './service.js'
import { fakeClient, fakeDatasets, fakeHub, fakeRuns, makeWorkspace, type FakeHub, type FakeRuns, type TempWorkspace } from './test-fakes.js'
import { until } from './test-util.js'
import type { AgentService } from './types.js'

let ws: TempWorkspace
let hub: FakeHub
let runs: FakeRuns
let agent: AgentService
beforeAll(async () => {
  ws = await makeWorkspace()
  setSchemaValidator(structuralValidate)
  hub = fakeHub()
  runs = fakeRuns()
  await fsp.mkdir(ws.config.configDir, { recursive: true })
  await fsp.writeFile(path.join(ws.config.configDir, 'policy.json'), JSON.stringify({ tools: { file_write: 'auto' } }))
  agent = createAgentService({ config: ws.config, hub, runs, datasets: fakeDatasets(), llm: createMockLlm({ delayMs: 0 }), retryDelayMs: 5 })
})
afterAll(async () => {
  await agent.shutdown()
  setSchemaValidator(null)
  await ws.cleanup()
})

const ctx = { activeFile: null, activeRun: null, attachments: [], selection: null }

describe('agent service', () => {
  it('opens, lists, renames and deletes sessions', async () => {
    const client = fakeClient()
    expect(await agent.handleClientMessage(client, { t: 'session.open', sessionId: null })).toBe(true)
    const state = client.of('session.state')[0].session
    expect(client.sessionId).toBe(state.id)
    expect(state.turnActive).toBe(false)
    expect(state.messages).toEqual([])
    expect(agent.listSessions().map((s) => s.id)).toContain(state.id)
    await agent.handleClientMessage(client, { t: 'session.rename', sessionId: state.id, title: 'Renamed' })
    expect(agent.getSessionState(state.id)?.title).toBe('Renamed')
    await agent.handleClientMessage(client, { t: 'session.list' })
    expect(client.of('session.list').at(-1)?.sessions.some((s) => s.title === 'Renamed')).toBe(true)
    await agent.handleClientMessage(client, { t: 'session.delete', sessionId: state.id })
    expect(agent.getSessionState(state.id)).toBeNull()
    expect(hub.of('session.deleted').at(-1)?.sessionId).toBe(state.id)
    expect(await agent.handleClientMessage(client, { t: 'ping', ts: 1 })).toBe(false)
  })

  it('runs a user message turn with attachments and remembers approvals for the session', async () => {
    const client = fakeClient()
    await agent.handleClientMessage(client, { t: 'session.new' })
    const id = client.of('session.state')[0].session.id
    client.sessionId = id
    await agent.handleClientMessage(client, { t: 'settings.set', sessionId: id, patch: { locale: 'en' } })
    hub.clear()
    await agent.handleClientMessage(client, { t: 'user.message', sessionId: id, text: 'change cases/plume.jsonc endTime to 7', context: { ...ctx, activeFile: 'cases/plume.jsonc', attachments: ['cases/plume.jsonc', 'missing.txt'] } })
    // the case the window had open is what the history list shows the conversation was about
    expect(agent.listSessions().find((s) => s.id === id)?.casePath).toBe('cases/plume.jsonc')
    const echoed = hub.of('msg.user')[0].message
    expect(echoed.blocks[0]).toEqual({ kind: 'text', text: 'change cases/plume.jsonc endTime to 7' })
    expect(echoed.blocks.filter((b) => b.kind === 'notice').map((b) => (b.kind === 'notice' ? b.text : ''))).toEqual([expect.stringMatching(/^@cases\/plume\.jsonc \(\d+ bytes\)$/), expect.stringMatching(/^@missing\.txt: /)])
    expect(agent.getSessionState(id)?.turnActive).toBe(true)
    await until(() => hub.of('tool.approval_request').length === 1)
    expect(agent.getSessionState(id)?.pendingApprovals).toHaveLength(1)
    const ids = hub.of('tool.approval_request')[0].approval.toolUseIds
    await agent.handleClientMessage(client, { t: 'tool.approve', sessionId: id, toolUseIds: ids, remember: 'session' })
    await until(() => hub.of('turn.done').length === 1)
    expect(hub.of('tool.approval_resolved')[0]).toMatchObject({ toolUseIds: ids, decision: 'approved' })
    const state = agent.getSessionState(id)!
    expect(state.turnActive).toBe(false)
    expect(state.pendingApprovals).toEqual([])
    expect(state.title).toBe('change cases/plume.jsonc endTime to 7')
    expect(await fsp.readFile(path.join(ws.root, 'cases/plume.jsonc'), 'utf8')).toContain('"endTime": 7')
    // Second edit is auto-approved because case_edit was remembered.
    hub.clear()
    await agent.handleClientMessage(client, { t: 'user.message', sessionId: id, text: 'change cases/plume.jsonc endTime to 8', context: ctx })
    await until(() => hub.of('turn.done').length === 1)
    expect(hub.of('tool.approval_request')).toHaveLength(0)
    expect(await fsp.readFile(path.join(ws.root, 'cases/plume.jsonc'), 'utf8')).toContain('"endTime": 8')
    const raw = JSON.parse(await fsp.readFile(path.join(ws.config.sessionsDir, `${id}.json`), 'utf8')) as { allowedTools: string[]; casePath: string | null }
    expect(raw.allowedTools).toEqual(['case_edit'])
    expect(raw.casePath).toBe('cases/plume.jsonc')
  })

  it('rejects a second turn while one is active and cancels on turn.cancel', async () => {
    const client = fakeClient()
    await agent.handleClientMessage(client, { t: 'session.new' })
    const id = client.of('session.state')[0].session.id
    hub.clear()
    await agent.handleClientMessage(client, { t: 'user.message', sessionId: id, text: 'edit cases/plume.jsonc endTime to 9', context: ctx })
    await until(() => hub.of('tool.approval_request').length === 1)
    await agent.handleClientMessage(client, { t: 'user.message', sessionId: id, text: 'again', context: ctx })
    expect(client.of('error').at(-1)?.message).toMatch(/already active/)
    await agent.handleClientMessage(client, { t: 'turn.cancel', sessionId: id })
    await until(() => hub.of('turn.done').length === 1)
    // the cancelled turn's session.state is sent from the turn promise's own
    // continuation, one microtask *after* turn.done: wait for it, do not race it
    await until(() => hub.of('session.state').length > 0)
    const state = agent.getSessionState(id)!
    expect(state.turnActive).toBe(false)
    const last = state.messages[state.messages.length - 1]
    expect(last.role).toBe('assistant')
    const card = last.blocks.find((b) => b.kind === 'tool')
    expect(card && card.kind === 'tool' ? card.call.status : null).toBe('cancelled')
  })

  it('two user.message frames in the same tick produce one turn', async () => {
    const client = fakeClient()
    await agent.handleClientMessage(client, { t: 'session.new' })
    const id = client.of('session.state')[0].session.id
    hub.clear()
    // The second frame arrives while the first is still reading its attachment.
    // An rt.active-only guard is false for both, so both messages were appended
    // and the second startTurn returned false without a word to the client.
    const first = agent.handleClientMessage(client, { t: 'user.message', sessionId: id, text: 'tell me about the registry', context: { ...ctx, attachments: ['cases/plume.jsonc'] } })
    const second = agent.handleClientMessage(client, { t: 'user.message', sessionId: id, text: 'and again', context: ctx })
    await Promise.all([first, second])
    expect(client.of('error').at(-1)?.message).toMatch(/already active/)
    await until(() => hub.of('turn.done').length === 1)
    expect(hub.of('msg.user')).toHaveLength(1)
    expect(agent.getSessionState(id)!.messages.filter((m) => m.role === 'user')).toHaveLength(1)
  })

  it('quick actions become synthetic user turns', async () => {
    const client = fakeClient()
    await agent.handleClientMessage(client, { t: 'session.new' })
    const id = client.of('session.state')[0].session.id
    await agent.handleClientMessage(client, { t: 'settings.set', sessionId: id, patch: { autoApprove: 'all', locale: 'en' } })
    hub.clear()
    await agent.handleClientMessage(client, { t: 'quick', sessionId: id, action: 'mesh', casePath: 'cases/channel', runId: null })
    expect(hub.of('msg.user')[0].message.synthetic).toBe(true)
    await until(() => hub.of('turn.done').length === 1)
    const state = agent.getSessionState(id)!
    expect(state.runs).toHaveLength(1)
    expect(state.messages.at(-1)?.blocks[0]).toMatchObject({ kind: 'text' })
  })

  it('notifyRunEnded appends a run notice and starts a turn when idle', async () => {
    const client = fakeClient()
    await agent.handleClientMessage(client, { t: 'session.new' })
    const id = client.of('session.state')[0].session.id
    await agent.handleClientMessage(client, { t: 'settings.set', sessionId: id, patch: { autoApprove: 'all', locale: 'en' } })
    hub.clear()
    runs.finishAfterMs = null
    await agent.handleClientMessage(client, { t: 'user.message', sessionId: id, text: 'run the solver on cases/plume.jsonc', context: ctx })
    await until(() => hub.of('tool.update').some((u) => u.call.name === 'run_wait' && u.call.status === 'running'))
    const runId = agent.getSessionState(id)!.runs[0]
    runs.finish(runId)
    await until(() => hub.of('turn.done').length === 1)
    hub.clear()
    agent.notifyRunEnded(runId)
    await until(() => hub.of('turn.done').length === 1)
    const state = agent.getSessionState(id)!
    // The notice arrives as a system line on an assistant message - never as
    // words in the operator's mouth.
    const notice = state.messages.find((m) => m.role === 'assistant' && m.synthetic)
    expect(notice?.blocks[0]).toMatchObject({ kind: 'notice', level: 'info', text: expect.stringContaining('run r_') })
    expect(state.messages.some((m) => m.role === 'user' && m.synthetic)).toBe(false)
    expect(state.messages.at(-1)?.role).toBe('assistant')
    runs.finishAfterMs = 5
  })

  it('honours policy.json overrides', async () => {
    const client = fakeClient()
    await agent.handleClientMessage(client, { t: 'session.new' })
    const id = client.of('session.state')[0].session.id
    const state = agent.getSessionState(id)!
    expect(state.customTools).toEqual([])
    expect(agent.createSession().id).toBeTruthy()
  })

  let r: ChatResponse
  it('chat() creates a session, runs a whole turn and returns the turn messages', async () => {
    r = await agent.chat({ sessionId: null, text: 'what is this repository?', attachments: ['cases/plume.jsonc', '../../etc/passwd'], attachmentIds: [], activeFile: null, autoApprove: 'all', locale: 'en', timeoutMs: 20000 })
    expect(r.status).toBe('done')
    expect(r.sessionId).toMatch(/^s_/)
    expect(r.turnId).toMatch(/^t_/)
    expect(r.messages[0].role).toBe('user')
    expect(r.messages[0].blocks[0]).toEqual({ kind: 'text', text: 'what is this repository?' })
    const last = r.messages.at(-1)!
    expect(last.role).toBe('assistant')
    expect(last.blocks.filter((b) => b.kind === 'text').map((b) => (b.kind === 'text' ? b.text : '')).join('\n')).toContain('meteor-cfd')
    expect(r.pendingApprovals).toEqual([])
    expect(r.runs).toEqual([])
    expect(agent.getSessionState(r.sessionId)!.turnActive).toBe(false)
    // the attachments reached the UserContext (the @ notice) and the settings patch landed
    expect(r.messages[0].blocks.some((b) => b.kind === 'notice' && b.text.includes('cases/plume.jsonc'))).toBe(true)
    expect(agent.getSessionState(r.sessionId)!.settings.autoApprove).toBe('all')
    expect(agent.getSessionState(r.sessionId)!.settings.locale).toBe('en')
    // the escape was refused by resolveInWorkspace before any read and became a notice, never content
    expect(r.messages[0].blocks.some((b) => b.kind === 'notice' && b.text.includes('outside the workspace'))).toBe(true)
    expect(JSON.stringify(r.messages).includes('root:')).toBe(false)
    expect(r.rounds).toBeGreaterThanOrEqual(2)
  })

  it('chat() continues an existing session and returns only the new turn', async () => {
    const r2 = await agent.chat({ sessionId: r.sessionId, text: 'what is this repository?', attachments: [], attachmentIds: [], activeFile: null, autoApprove: 'all', locale: 'en', timeoutMs: 20000 })
    expect(r2.sessionId).toBe(r.sessionId)
    expect(r2.messages.length).toBeGreaterThanOrEqual(2)
    expect(r2.messages[0].blocks[0]).toMatchObject({ kind: 'text', text: 'what is this repository?' })
    expect(agent.getSessionState(r.sessionId)!.messages.length).toBe(r.messages.length + r2.messages.length)
  })

  it('chat() refuses a second turn with 409, an unknown session with 404 and an empty message with 400', async () => {
    const id = agent.createSession().id
    const p = agent.chat({ sessionId: id, text: 'what is this repository?', attachments: [], attachmentIds: [], activeFile: null, autoApprove: 'all', locale: 'en', timeoutMs: 20000 })
    await expect(agent.chat({ sessionId: id, text: 'what is this repository?', attachments: [], attachmentIds: [], activeFile: null, autoApprove: 'all', locale: 'en', timeoutMs: 20000 })).rejects.toMatchObject({ status: 409 })
    await p
    await expect(agent.chat({ sessionId: 's_nope', text: 'hi', attachments: [], attachmentIds: [], activeFile: null, autoApprove: null, locale: null, timeoutMs: null })).rejects.toMatchObject({ status: 404 })
    const n = agent.listSessions().length
    await expect(agent.chat({ sessionId: null, text: '   ', attachments: [], attachmentIds: [], activeFile: null, autoApprove: null, locale: null, timeoutMs: null })).rejects.toMatchObject({ status: 400 })
    expect(agent.listSessions().length).toBe(n)
  })

  it('chat() answers timeout without cancelling the turn', async () => {
    const svc = createAgentService({ config: ws.config, hub: fakeHub(), runs: fakeRuns(), datasets: fakeDatasets(), llm: createMockLlm({ delayMs: 100 }), retryDelayMs: 5 })
    const rt = await svc.chat({ sessionId: null, text: 'what is this repository?', attachments: [], attachmentIds: [], activeFile: null, autoApprove: 'all', locale: 'en', timeoutMs: 1000 })
    expect(rt.status).toBe('timeout')
    expect(rt.messages[0].role).toBe('user')
    await until(() => svc.getSessionState(rt.sessionId)!.turnActive === false, 20000)
    await svc.shutdown()
  }, 20_000)
})
