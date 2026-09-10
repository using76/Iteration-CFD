// LlmClient over z.ai's Anthropic-compatible endpoint (default
// https://api.z.ai/api/anthropic), which serves GLM models. It accepts the
// plain Messages shape and none of the Anthropic-only extras.
//
// Accepted request shape (verified against glm-5.3-flash by
// scripts/llm-probe.ts): model, max_tokens, system, messages, tools,
// thinking: { type: 'enabled', budget_tokens } - streamed back as the
// ordinary message_start / content_block_* / message_delta / message_stop
// events. Never sent, because GLM rejects or ignores them: betas, fallbacks,
// output_config (effort), top-level cache_control, adaptive thinking with
// display, temperature. Tool input comes back as ordinary input_json_delta
// and is passed through unchanged.
import Anthropic from '@anthropic-ai/sdk'
import type { BetaMessage, BetaMessageStreamParams, BetaRawMessageStreamEvent } from '@anthropic-ai/sdk/resources/beta/messages/messages'
import { DEFAULT_ZAI_BASE_URL } from '../config.js'
import type { ServerConfig } from '../config.js'
import type { LlmClient, LlmStream, LlmStreamParams } from './llm.js'

export const DEFAULT_ZAI_THINKING_TOKENS = 4096
export const DEFAULT_ZAI_MAX_TOKENS = 64000

export interface ZaiShapeOptions {
  thinkingTokens: number
  maxTokens: number
}

/** The plain Messages shape GLM accepts. max_tokens covers the thinking budget plus real output. */
export function buildZaiStreamParams(model: string, p: LlmStreamParams, opts: Partial<ZaiShapeOptions> = {}): BetaMessageStreamParams {
  const budget = Math.max(1024, opts.thinkingTokens ?? DEFAULT_ZAI_THINKING_TOKENS)
  const cap = Math.max(opts.maxTokens ?? DEFAULT_ZAI_MAX_TOKENS, budget + 1024)
  return {
    model,
    max_tokens: Math.min(cap, Math.max(p.maxTokens, budget + 1024)),
    thinking: { type: 'enabled', budget_tokens: budget },
    system: p.system,
    ...(p.tools.length ? { tools: p.tools } : {}),
    messages: p.messages,
  }
}

/** The z.ai LlmClient: the same SDK stream objects the loop already consumes, against the GLM endpoint. */
export function createZaiClient(config: Pick<ServerConfig, 'model' | 'zai'>, client?: Anthropic): LlmClient {
  const key = config.zai?.key ?? null
  if (!key) throw new Error('no z.ai API key: set ZAI_API_KEY, or put the key in the CFD_ZAI_KEY_FILE (default ~/.claude/zai-key)')
  const anthropic = client ?? new Anthropic({ baseURL: config.zai?.baseUrl ?? DEFAULT_ZAI_BASE_URL, apiKey: key })
  const shape = {
    thinkingTokens: config.zai?.thinkingTokens ?? DEFAULT_ZAI_THINKING_TOKENS,
    maxTokens: config.zai?.maxTokens ?? DEFAULT_ZAI_MAX_TOKENS,
  }
  return {
    kind: 'zai',
    model: config.model,
    stream(params: LlmStreamParams): LlmStream {
      const stream = anthropic.messages.stream(buildZaiStreamParams(config.model, params, shape) as never, { signal: params.signal })
      return {
        events: stream as unknown as AsyncIterable<BetaRawMessageStreamEvent>,
        finalMessage: () => stream.finalMessage() as unknown as Promise<BetaMessage>,
      }
    },
  }
}
