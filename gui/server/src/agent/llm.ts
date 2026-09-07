// The one interface the agent loop streams from. The Anthropic wrapper and
// the scripted mock both produce the SDK's raw stream events and a final
// BetaMessage, so loop.ts is identical in real and demo mode.
import type { BetaMessage, BetaMessageParam, BetaRawMessageStreamEvent, BetaTextBlockParam, BetaTool } from '@anthropic-ai/sdk/resources/beta/messages/messages'
import type { SessionSettings } from '@cfd/shared'

export interface LlmStreamParams {
  system: BetaTextBlockParam[]
  messages: BetaMessageParam[]
  tools: BetaTool[]
  maxTokens: number
  effort: SessionSettings['effort']
  signal: AbortSignal
}

export interface LlmStream {
  events: AsyncIterable<BetaRawMessageStreamEvent>
  finalMessage(): Promise<BetaMessage>
}

export interface LlmClient {
  kind: 'anthropic' | 'mock'
  model: string
  stream(params: LlmStreamParams): LlmStream
}

export class LlmAbortError extends Error {
  constructor(message = 'cancelled by user') {
    super(message)
    this.name = 'LlmAbortError'
  }
}

/** Empty usage record, the shape turn.done carries. */
export function emptyUsage(): { inputTokens: number; outputTokens: number; cacheReadTokens: number; cacheWriteTokens: number } {
  return { inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0 }
}
