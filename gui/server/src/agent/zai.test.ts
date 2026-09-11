import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type Anthropic from '@anthropic-ai/sdk'
import type { BetaMessage, BetaRawMessageStreamEvent } from '@anthropic-ai/sdk/resources/beta/messages/messages'
import { DEFAULT_ZAI_BASE_URL, loadConfig, type ServerConfig } from '../config.js'
import type { LlmClient, LlmStreamParams } from './llm.js'
import { buildZaiStreamParams, createZaiClient } from './zai.js'

// Capture the SDK constructor args; the mocked class must never reach the network.
const h = vi.hoisted(() => ({ ctorArgs: [] as unknown[] }))
vi.mock('@anthropic-ai/sdk', () => {
  class Anthropic {
    messages = {
      stream: () => {
        throw new Error('unexpected stream on the constructed client')
      },
    }
    constructor(args: unknown) {
      h.ctorArgs.push(args)
    }
  }
  return { default: Anthropic }
})

interface CapturedCall {
  params: Record<string, unknown>
  options: Record<string, unknown>
}

/** A fake client.messages.stream that records the request and answers with two harmless events. */
function fakeStreamClient(calls: CapturedCall[]): Anthropic {
  return {
    messages: {
      stream(params: Record<string, unknown>, options: Record<string, unknown>) {
        calls.push({ params, options })
        // Like the real SDK MessageStream: one object, itself the async iterable, carrying finalMessage().
        async function* iterate(): AsyncGenerator<BetaRawMessageStreamEvent> {
          yield { type: 'message_start', message: { id: 'msg_probe', type: 'message', role: 'assistant', model: String(params.model), content: [], stop_reason: null } as unknown as BetaMessage }
          yield { type: 'message_stop' }
        }
        return Object.assign(iterate(), {
          finalMessage: (): Promise<BetaMessage> =>
            Promise.resolve({ id: 'msg_probe', type: 'message', role: 'assistant', model: String(params.model), content: [], stop_reason: 'end_turn', stop_sequence: null, usage: { input_tokens: 1, output_tokens: 1 } } as unknown as BetaMessage),
        })
      },
    },
  } as unknown as Anthropic
}

const baseParams = (over: Partial<LlmStreamParams> = {}): LlmStreamParams => ({
  system: [{ type: 'text', text: 'sys' }],
  messages: [{ role: 'user', content: 'hi' }],
  tools: [{ name: 'echo', description: 'd', input_schema: { type: 'object', properties: {} } }],
  maxTokens: 16000,
  effort: 'high',
  signal: new AbortController().signal,
  ...over,
})

const zaiConfig = (over: Partial<NonNullable<ServerConfig['zai']>> = {}): Pick<ServerConfig, 'model' | 'zai'> => ({
  model: 'glm-5.3-flash',
  zai: { baseUrl: 'https://z.example/api/anthropic', key: 'zk-test', keyFile: '/k', thinkingTokens: 4096, maxTokens: 64000, ...over },
})

describe('buildZaiStreamParams', () => {
  it('sends the plain shape GLM accepts and nothing Anthropic-only', () => {
    const p = baseParams()
    const params = buildZaiStreamParams('glm-5.3-flash', p, { thinkingTokens: 4096, maxTokens: 64000 }) as unknown as Record<string, unknown>
    expect(params).toMatchObject({ model: 'glm-5.3-flash', max_tokens: 16000, thinking: { type: 'enabled', budget_tokens: 4096 }, system: p.system, messages: p.messages })
    for (const absent of ['betas', 'fallbacks', 'output_config', 'cache_control', 'temperature']) expect(params[absent]).toBeUndefined()
  })

  it('caps max_tokens and keeps thinking plus output inside it', () => {
    const capped = buildZaiStreamParams('m', baseParams({ maxTokens: 200_000 }), { thinkingTokens: 4096, maxTokens: 64_000 })
    expect(capped.max_tokens).toBe(64_000)
    const tiny = buildZaiStreamParams('m', baseParams({ maxTokens: 1024 }), { thinkingTokens: 4096, maxTokens: 64_000 })
    expect(tiny.max_tokens).toBe(4096 + 1024)
  })

  it('omits tools when the loop sends none', () => {
    const params = buildZaiStreamParams('m', baseParams({ tools: [] }), { thinkingTokens: 4096, maxTokens: 64_000 }) as unknown as Record<string, unknown>
    expect(params.tools).toBeUndefined()
  })

  it('passes tool input back through unchanged', () => {
    const input = { path: 'cases/a.jsonc', edits: [{ pointer: '/run/endTime', op: 'set', valueJson: '2.0' }] }
    const messages: LlmStreamParams['messages'] = [{ role: 'assistant', content: [{ type: 'tool_use', id: 'tu_1', name: 'case_edit', input }] }]
    const params = buildZaiStreamParams('m', baseParams({ messages }), { thinkingTokens: 4096, maxTokens: 64_000 })
    const block = (params.messages[0].content as Array<{ type: string; input?: unknown }>)[0]
    expect(block.input).toBe(input)
  })
})

describe('createZaiClient', () => {
  it('streams through the injected client and forwards the abort signal', async () => {
    const calls: CapturedCall[] = []
    const client: LlmClient = createZaiClient(zaiConfig(), fakeStreamClient(calls))
    expect(client.kind).toBe('zai')
    expect(client.model).toBe('glm-5.3-flash')
    const signal = new AbortController().signal
    const stream = client.stream(baseParams({ signal }))
    await stream.finalMessage()
    expect(calls).toHaveLength(1)
    expect(calls[0].options).toEqual({ signal })
    expect((calls[0].params as Record<string, unknown>).model).toBe('glm-5.3-flash')
    for await (const _ of stream.events) void _
  })

  it('builds the SDK client from config: base URL and key', () => {
    createZaiClient(zaiConfig())
    expect(h.ctorArgs.at(-1)).toEqual({ baseURL: 'https://z.example/api/anthropic', apiKey: 'zk-test' })
  })

  it('falls back to the z.ai defaults when the config carries none', () => {
    createZaiClient({ model: 'glm-5.3-flash', zai: { baseUrl: DEFAULT_ZAI_BASE_URL, key: 'zk', keyFile: '/k', thinkingTokens: 4096, maxTokens: 64000 } })
    expect(h.ctorArgs.at(-1)).toEqual({ baseURL: DEFAULT_ZAI_BASE_URL, apiKey: 'zk' })
  })

  it('refuses to build with no key, before any request', () => {
    const before = h.ctorArgs.length
    expect(() => createZaiClient(zaiConfig({ key: null }))).toThrow(/z\.ai API key/)
    expect(h.ctorArgs.length).toBe(before)
  })
})

describe('loadConfig z.ai resolution', () => {
  let dir: string | null = null
  afterEach(() => {
    if (dir) {
      fs.rmSync(dir, { recursive: true, force: true })
      dir = null
    }
  })

  const env = (over: NodeJS.ProcessEnv = {}): NodeJS.ProcessEnv => ({
    ANTHROPIC_API_KEY: undefined,
    ANTHROPIC_AUTH_TOKEN: undefined,
    CFD_DEMO: undefined,
    CFD_LLM: undefined,
    ZAI_API_KEY: undefined,
    CFD_ZAI_KEY_FILE: undefined,
    CFD_LLM_BASE_URL: undefined,
    CFD_ZAI_THINKING_TOKENS: undefined,
    CFD_ZAI_MAX_TOKENS: undefined,
    CFD_MODEL: undefined,
    ...over,
  })

  it('takes ZAI_API_KEY over the key file and marks hasKey', () => {
    const c = loadConfig(env({ CFD_LLM: 'zai', ZAI_API_KEY: ' zk-env \n' }))
    expect(c.zai?.key).toBe('zk-env')
    expect(c.hasKey).toBe(true)
    expect(c.llm).toBe('zai')
    expect(c.model).toBe('glm-5.3-flash')
  })

  it('reads the key file trimming CR/LF', () => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), 'zai-key-'))
    const file = path.join(dir, 'zai-key')
    fs.writeFileSync(file, ' zk-file \r\n')
    const c = loadConfig(env({ CFD_LLM: 'zai', CFD_ZAI_KEY_FILE: file }))
    expect(c.zai?.key).toBe('zk-file')
    expect(c.zai?.keyFile).toBe(file)
  })

  it('handles ~ in the key file path and leaves the key null when nothing resolves', () => {
    // ~ expands to the home directory, so the probe file has to live there.
    const homeFile = path.join(os.homedir(), `.zai-key-test-${process.pid}`)
    try {
      fs.writeFileSync(homeFile, 'zk-tilde')
      const c = loadConfig(env({ CFD_ZAI_KEY_FILE: `~/${path.basename(homeFile)}` }))
      expect(c.zai?.key).toBe('zk-tilde')
      expect(c.hasKey).toBe(true)
    } finally {
      fs.rmSync(homeFile, { force: true })
    }
    const none = loadConfig(env({ CFD_ZAI_KEY_FILE: path.join(os.tmpdir(), `zai-key-absent-${process.pid}`) }))
    expect(none.zai?.key).toBeNull()
    expect(none.hasKey).toBe(false)
  })

  it('defaults the base URL and honours CFD_LLM_BASE_URL', () => {
    expect(loadConfig(env({ CFD_LLM: 'zai' })).zai?.baseUrl).toBe(DEFAULT_ZAI_BASE_URL)
    expect(loadConfig(env({ CFD_LLM: 'zai', CFD_LLM_BASE_URL: 'https://proxy.example/anthropic' })).zai?.baseUrl).toBe('https://proxy.example/anthropic')
  })

  it('lets the key that resolved pick the client, and keeps the scripted mock when none does', () => {
    // Point the key file at a missing path: this machine may carry a real ~/.claude/zai-key.
    const absent = path.join(os.tmpdir(), `zai-key-absent-${process.pid}`)
    // Only a z.ai key: 'zai', never 'anthropic' - otherwise main.ts refuses to boot for want of an
    // ANTHROPIC_API_KEY on a machine that has a perfectly good key.
    expect(loadConfig(env({ CFD_ZAI_KEY_FILE: absent, ZAI_API_KEY: 'zk' })).llm).toBe('zai')
    // An Anthropic key still wins the default, and the model default follows the client.
    const both = loadConfig(env({ CFD_ZAI_KEY_FILE: absent, ZAI_API_KEY: 'zk', ANTHROPIC_API_KEY: 'sk-a' }))
    expect(both.llm).toBe('anthropic')
    expect(both.model).toBe('claude-opus-5')
    expect(loadConfig(env({ CFD_DEMO: '1', CFD_ZAI_KEY_FILE: absent })).llm).toBe('mock')
  })

  it('keeps demo mode on the scripted mock whatever key is on the machine, unless CFD_LLM says otherwise', () => {
    const absent = path.join(os.tmpdir(), `zai-key-absent-${process.pid}`)
    // README line 29 and PLAN §0 both advertise `CFD_DEMO=1` as needing no key; picking
    // the client from whatever key happened to be on disk spent it behind that promise.
    const demoZai = loadConfig(env({ CFD_DEMO: '1', CFD_ZAI_KEY_FILE: absent, ZAI_API_KEY: 'zk' }))
    expect(demoZai.llm).toBe('mock')
    expect(demoZai.model).toBe('mock-assistant')
    expect(loadConfig(env({ CFD_DEMO: '1', CFD_ZAI_KEY_FILE: absent, ANTHROPIC_API_KEY: 'sk-a' })).llm).toBe('mock')
    // and the GLM suite's own command still reaches GLM
    expect(loadConfig(env({ CFD_DEMO: '1', CFD_LLM: 'zai', ZAI_API_KEY: 'zk' })).llm).toBe('zai')
  })

  it('falls back to the default token budgets when the env carries junk', () => {
    const absent = path.join(os.tmpdir(), `zai-key-absent-${process.pid}`)
    const bad = loadConfig(env({ CFD_ZAI_KEY_FILE: absent, ZAI_API_KEY: 'zk', CFD_ZAI_THINKING_TOKENS: '4k', CFD_ZAI_MAX_TOKENS: '' }))
    expect(bad.zai?.thinkingTokens).toBe(4096)
    expect(bad.zai?.maxTokens).toBe(64000)
    const good = loadConfig(env({ CFD_ZAI_KEY_FILE: absent, ZAI_API_KEY: 'zk', CFD_ZAI_THINKING_TOKENS: '2048', CFD_ZAI_MAX_TOKENS: '32000' }))
    expect(good.zai?.thinkingTokens).toBe(2048)
    expect(good.zai?.maxTokens).toBe(32000)
  })
})
