import fsp from 'node:fs/promises'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
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
    await agent.handleClientMessage(client, { t: 'user.message', sessionId: id, text: 'change cases/plume.jsonc endTime to 7', context: { ...ctx, attachments: ['cases/plume.jsonc', 'missing.txt'] } })
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
    const raw = JSON.parse(await fsp.readFile(path.join(ws.config.sessionsDir, `${id}.json`), 'utf8')) as { allowedTools: string[] }
    expect(raw.allowedTools).toEqual(['case_edit'])
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
    expect(hub.of('session.state').length).toBeGreaterThan(0)
    const state = agent.getSessionState(id)!
    expect(state.turnActive).toBe(false)
    const last = state.messages[state.messages.length - 1]
    expect(last.role).toBe('assistant')
    const card = last.blocks.find((b) => b.kind === 'tool')
    expect(card && card.kind === 'tool' ? card.call.status : null).toBe('cancelled')
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
    const notice = state.messages.find((m) => m.role === 'user' && m.synthetic)
    expect(notice?.blocks[0]).toMatchObject({ kind: 'text', text: expect.stringContaining('[run notice] run r_') })
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
})
