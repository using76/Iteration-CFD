// Shared helpers for the agent tests: turn deps over the fakes, a spying
// LLM wrapper and a small poll-until.
import type { BetaMessageParam } from '@anthropic-ai/sdk/resources/beta/messages/messages'
import type { ServerMsg } from '@cfd/shared'
import { createApprovalManager } from './approvals.js'
import type { LlmClient } from './llm.js'
import type { TurnDeps } from './loop.js'
import { createMockLlm } from './mockLlm.js'
import { createSessionStore } from './session.js'
import { fakeDatasets, fakeHub, fakeRuns, type FakeHub, type FakeRuns, type TempWorkspace } from './test-fakes.js'

export interface TestDeps extends TurnDeps {
  hub: FakeHub
  runs: FakeRuns
  calls: BetaMessageParam[][]
}

export function spyLlm(inner: LlmClient, calls: BetaMessageParam[][]): LlmClient {
  return {
    kind: inner.kind,
    model: inner.model,
    stream(params) {
      calls.push(params.messages)
      return inner.stream(params)
    },
  }
}

export function makeDeps(ws: TempWorkspace, over: Partial<TurnDeps> & { runs?: FakeRuns; hub?: FakeHub } = {}): TestDeps {
  const hub = over.hub ?? fakeHub()
  const runs = over.runs ?? fakeRuns()
  const calls: BetaMessageParam[][] = []
  const llm = spyLlm(over.llm ?? createMockLlm({ delayMs: 0 }), calls)
  return {
    config: ws.config,
    hub,
    runs,
    datasets: fakeDatasets(),
    approvals: createApprovalManager(),
    overrides: {},
    store: createSessionStore(ws.config.sessionsDir, ws.config.model),
    customTools: () => [],
    userContext: () => null,
    emit: (msg: ServerMsg) => hub.sendToSession('s', msg),
    retryDelayMs: 5,
    ...over,
    llm,
    calls,
  }
}

export async function until(cond: () => boolean, timeoutMs = 3000): Promise<void> {
  const start = Date.now()
  while (!cond()) {
    if (Date.now() - start > timeoutMs) throw new Error('timed out waiting for condition')
    await new Promise((r) => setTimeout(r, 5))
  }
}

export function toolResultsOf(m: BetaMessageParam): Array<{ tool_use_id: string; is_error: boolean; content: string }> {
  if (typeof m.content === 'string') return []
  return m.content
    .filter((b): b is Extract<typeof b, { type: 'tool_result' }> => b.type === 'tool_result')
    .map((b) => ({ tool_use_id: b.tool_use_id, is_error: b.is_error === true, content: typeof b.content === 'string' ? b.content : JSON.stringify(b.content) }))
}

export function toolUsesOf(m: BetaMessageParam): Array<{ id: string; name: string; input: unknown }> {
  if (typeof m.content === 'string') return []
  return m.content.filter((b): b is Extract<typeof b, { type: 'tool_use' }> => b.type === 'tool_use').map((b) => ({ id: b.id, name: b.name, input: b.input }))
}

export function textOf(m: BetaMessageParam): string {
  if (typeof m.content === 'string') return m.content
  return m.content
    .filter((b): b is Extract<typeof b, { type: 'text' }> => b.type === 'text')
    .map((b) => b.text)
    .join('')
}
