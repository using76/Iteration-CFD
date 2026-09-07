// LlmClient over @anthropic-ai/sdk 0.124: client.beta.messages.stream with
// adaptive summarised thinking, effort, server-side fallbacks and the
// cache_control breakpoint on the static system prompt. No prefill.
import Anthropic from '@anthropic-ai/sdk'
import type { BetaMessageStreamParams } from '@anthropic-ai/sdk/resources/beta/messages/messages'
import type { ServerConfig } from '../config.js'
import type { LlmClient, LlmStream, LlmStreamParams } from './llm.js'

export const ANTHROPIC_BETAS = ['server-side-fallback-2026-07-01'] as const

export function buildStreamParams(model: string, p: LlmStreamParams): BetaMessageStreamParams {
  return {
    model,
    max_tokens: p.maxTokens,
    thinking: { type: 'adaptive', display: 'summarized' },
    output_config: { effort: p.effort },
    betas: [...ANTHROPIC_BETAS],
    fallbacks: 'default',
    system: p.system,
    tools: p.tools,
    messages: p.messages,
  }
}

export function createAnthropicClient(config: Pick<ServerConfig, 'model'>, client: Anthropic = new Anthropic()): LlmClient {
  return {
    kind: 'anthropic',
    model: config.model,
    stream(params: LlmStreamParams): LlmStream {
      const stream = client.beta.messages.stream(buildStreamParams(config.model, params), { signal: params.signal })
      return {
        events: stream,
        finalMessage: () => stream.finalMessage(),
      }
    },
  }
}

export function isRetryableError(err: unknown): boolean {
  return err instanceof Anthropic.RateLimitError || err instanceof Anthropic.APIConnectionError || err instanceof Anthropic.InternalServerError
}

export function isAbortError(err: unknown): boolean {
  if (err instanceof Anthropic.APIUserAbortError) return true
  return err instanceof Error && (err.name === 'AbortError' || err.name === 'LlmAbortError')
}

/** A 400 that complains about role "system" inside messages: fold the context into the user turn and retry once. */
export function isSystemRoleRejection(err: unknown): boolean {
  return err instanceof Anthropic.BadRequestError && /system/i.test(err.message) && /role/i.test(err.message)
}

export function describeError(err: unknown): string {
  if (err instanceof Anthropic.APIError) return `${err.name}${err.status ? ` (${err.status})` : ''}: ${err.message}`
  return err instanceof Error ? err.message : String(err)
}
