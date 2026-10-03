// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
import { afterAll, beforeAll, beforeEach, describe, expect, it } from 'vitest'
import type { BetaMessageParam } from '@anthropic-ai/sdk/resources/beta/messages/messages'
import { createMockLlm } from '../src/agent/mockLlm.js'
import { languageInstruction } from '../src/agent/prompt.js'
import { createAgentService } from '../src/agent/service.js'
import { fakeDatasets, fakeRuns, makeWorkspace, type TempWorkspace } from '../src/agent/test-fakes.js'
import { spyLlm, textOf } from '../src/agent/test-util.js'
import { Router } from '../src/http/router.js'
import { createHttpServer, type HttpServerHandle } from '../src/http/server.js'
import { createHub, type HubHandle } from '../src/ws/hub.js'
import { drive, parseOptions } from './ai-drive.js'

// A real hub + agent service on an ephemeral port, exactly the way hub.test.ts
// stands one up: the drive under test is the script itself, imported for its
// parseOptions and drive (the entry guard keeps the import from connecting).
let ws: TempWorkspace
let srv: HttpServerHandle
let hub: HubHandle
let agent: ReturnType<typeof createAgentService>
let calls: BetaMessageParam[][]
let url: string

beforeAll(async () => {
  ws = await makeWorkspace()
  const runs = fakeRuns()
  hub = createHub({
    hello: () => ({
      version: 't', mode: 'demo', llm: 'mock', model: 'm',
      gpu: { state: 'demo', name: null, memUsedMB: null, memTotalMB: null, source: 'demo' },
      workspaceRoot: ws.root, availableBinaries: [], platform: 'win32',
    }),
    sessions: () => agent?.listSessions() ?? [],
    runs,
    heartbeatMs: 50,
    idleMs: 100_000,
  })
  srv = createHttpServer({ config: { host: '127.0.0.1', port: 0, allowRemote: false, authToken: null }, router: new Router(), hub, staticDir: null })
  url = `ws://127.0.0.1:${(await srv.listen()).port}/ws`
  calls = []
  agent = createAgentService({ config: ws.config, hub, runs, datasets: fakeDatasets(), llm: spyLlm(createMockLlm({ delayMs: 0 }), calls), retryDelayMs: 5 })
  hub.onClientMessage(async (c, m) => {
    await agent.handleClientMessage(c, m)
  })
})

beforeEach(() => {
  calls.splice(0, calls.length)
})

afterAll(async () => {
  await agent.shutdown()
  hub.close()
  await srv.close()
  await ws.cleanup()
})

describe('ai-drive --locale', () => {
  it('parseOptions defaults to en, takes exactly ko or en, and lists the flag in the usage', () => {
    expect(parseOptions([]).locale).toBe('en')
    expect(parseOptions(['--locale', 'ko']).locale).toBe('ko')
    expect(parseOptions(['--locale', 'en']).locale).toBe('en')
    expect(() => parseOptions(['--locale', 'fr'])).toThrow(/--locale wants ko or en/)
    expect(() => parseOptions(['--locale', 'KO'])).toThrow(/--locale wants ko or en/)
    expect(() => parseOptions(['--locale'])).toThrow(/needs a value/)
    expect(() => parseOptions(['--bogus'])).toThrow(/\[--locale ko\|en\]/)
  })

  it('a Korean drive carries the Korean language instruction on every LLM call of the turn', async () => {
    const opts = { ...parseOptions(['--locale', 'ko', '--prompt', 'What is this repository?', '--timeout', '30']), url }
    await expect(drive(opts)).resolves.toBe(0)
    expect(calls.length).toBeGreaterThanOrEqual(1)
    for (const messages of calls) {
      const text = messages.map((m) => textOf(m)).join('\n')
      expect(text).toContain(languageInstruction('ko'))
      expect(text).not.toContain(languageInstruction('en'))
    }
  }, 20_000)

  it('a drive with no --locale stays English on every LLM call of the turn', async () => {
    const opts = { ...parseOptions(['--prompt', 'What is this repository?', '--timeout', '30']), url }
    await expect(drive(opts)).resolves.toBe(0)
    expect(calls.length).toBeGreaterThanOrEqual(1)
    for (const messages of calls) {
      const text = messages.map((m) => textOf(m)).join('\n')
      expect(text).toContain(languageInstruction('en'))
      expect(text).not.toContain(languageInstruction('ko'))
    }
  }, 20_000)
})
