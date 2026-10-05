// The shutdown warning's delivery path, proven without a server, a port or a
// browser: a real agent service holds one real turn active, warnActiveTurns
// puts the line on the log and on that turn's own session window, and stays
// silent the moment the turn is gone or the agent is null.
import fsp from 'node:fs/promises'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { ServerMsgSchema, type ServerMsg } from '@cfd/shared'
import { setSchemaValidator, structuralValidate } from '../tools/case.js'
import { closeOntologyHandles } from '../ontology/handle.js'
import { createMockLlm } from './mockLlm.js'
import { activeTurnWarning, createAgentService, warnActiveTurns } from './service.js'
import { fakeClient, fakeDatasets, fakeHub, fakeRuns, makeWorkspace, type FakeHub, type TempWorkspace } from './test-fakes.js'
import { until } from './test-util.js'
import type { AgentService } from './types.js'

let ws: TempWorkspace
let hub: FakeHub
let agent: AgentService
// One record for both sinks, in call order: log.warn first, sendToSession second.
const events: Array<{ kind: 'log' | 'send'; text: string }> = []
let sentTo: Array<{ id: string; msg: ServerMsg }> = []
const log = { warn: (msg: string) => { events.push({ kind: 'log', text: msg }) } }
const spyHub = {
  // fakeHub().sendToSession drops the id, so wrap it to prove WHICH session got the frame.
  sendToSession: (id: string, msg: ServerMsg) => {
    events.push({ kind: 'send', text: msg.t === 'error' ? msg.message : msg.t })
    sentTo.push({ id, msg })
    hub.sendToSession(id, msg)
  },
}
function resetRecorders(): void {
  events.splice(0, events.length)
  sentTo = []
}
beforeAll(async () => {
  ws = await makeWorkspace()
  setSchemaValidator(structuralValidate)
  hub = fakeHub()
  await fsp.mkdir(ws.config.configDir, { recursive: true })
  await fsp.writeFile(path.join(ws.config.configDir, 'policy.json'), JSON.stringify({ tools: { file_write: 'auto' } }))
  agent = createAgentService({ config: ws.config, hub, runs: fakeRuns(), datasets: fakeDatasets(), llm: createMockLlm({ delayMs: 0 }), retryDelayMs: 5 })
})
afterAll(async () => {
  await agent.shutdown()
  await closeOntologyHandles()
  setSchemaValidator(null)
  await ws.cleanup()
})

describe('the going-down server warns active turns', () => {
  it('warns the log and the session window of the active turn, log first', async () => {
    const client = fakeClient()
    await agent.handleClientMessage(client, { t: 'session.new' })
    const id = client.of('session.state')[0].session.id
    client.sessionId = id
    await agent.handleClientMessage(client, { t: 'settings.set', sessionId: id, patch: { locale: 'en' } })
    await agent.handleClientMessage(client, { t: 'user.message', sessionId: id, text: 'hello', context: { activeFile: null, activeRun: null, attachments: [], attachmentIds: [], selection: null } })
    const turns = agent.activeTurns()
    expect(turns).toEqual([{ sessionId: id, turnId: expect.stringMatching(/^t_/) }])
    const line = activeTurnWarning(turns[0])
    resetRecorders()
    const warned = warnActiveTurns(agent, spyHub, log)
    expect(warned).toEqual([line])
    expect(events).toEqual([{ kind: 'log', text: line }, { kind: 'send', text: line }])
    expect(sentTo).toEqual([{ id, msg: { t: 'error', message: line, fatal: false } }])
    expect(ServerMsgSchema.safeParse(sentTo[0].msg).success).toBe(true)
    await until(() => agent.activeTurns().length === 0)
  })

  it('warns and sends nothing once the turn has finished', () => {
    expect(agent.activeTurns()).toEqual([])
    resetRecorders()
    expect(warnActiveTurns(agent, spyHub, log)).toEqual([])
    expect(events).toEqual([])
    expect(sentTo).toEqual([])
  })

  it('warns and sends nothing for a null or undefined agent', () => {
    resetRecorders()
    expect(warnActiveTurns(null, spyHub, log)).toEqual([])
    expect(warnActiveTurns(undefined, spyHub, log)).toEqual([])
    expect(events).toEqual([])
    expect(sentTo).toEqual([])
  })
})
