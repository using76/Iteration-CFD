// Live probe for the configured LLM (z.ai/GLM by default): one plain text
// call, then one tool call. Exit 0 only when both worked. Never prints the
// key. Run from gui/: CFD_LLM=zai npx tsx server/scripts/llm-probe.ts
import type { BetaTextBlockParam, BetaTool } from '@anthropic-ai/sdk/resources/beta/messages/messages'
import { createAnthropicClient } from '../src/agent/anthropic.js'
import type { LlmClient, LlmStreamParams } from '../src/agent/llm.js'
import { createZaiClient } from '../src/agent/zai.js'
import { loadConfig } from '../src/config.js'

const SYSTEM: BetaTextBlockParam[] = [{ type: 'text', text: 'You are a probe. Follow the user instruction exactly.' }]

async function call(client: LlmClient, messages: LlmStreamParams['messages'], tools: BetaTool[]) {
  const params: LlmStreamParams = { system: SYSTEM, messages, tools, maxTokens: 2048, effort: 'medium', signal: new AbortController().signal }
  return client.stream(params).finalMessage()
}

async function main(): Promise<number> {
  const config = loadConfig({ ...process.env, CFD_LLM: process.env.CFD_LLM ?? 'zai' })
  if (config.llm === 'mock') {
    console.error('CFD_LLM=mock: the scripted mock has nothing to probe. Unset it or set CFD_LLM=zai.')
    return 1
  }
  const client = config.llm === 'zai' ? createZaiClient(config) : createAnthropicClient(config)
  console.log(`endpoint client kind: ${client.kind} model: ${config.model} baseUrl: ${config.llm === 'zai' ? config.zai?.baseUrl : '(anthropic default)'}`)

  const final = await call(client, [{ role: 'user', content: 'Reply with the single word PROBE_OK and call no tool.' }], [])
  const text = final.content.filter((b) => b.type === 'text').map((b) => b.text).join('')
  const toolUses = final.content.filter((b) => b.type === 'tool_use')
  console.log(`text call -> model: ${final.model}`)
  console.log(`  text: ${JSON.stringify(text)}`)
  console.log(`  usage: input=${final.usage.input_tokens} output=${final.usage.output_tokens} cache_read=${final.usage.cache_read_input_tokens ?? 0} cache_write=${final.usage.cache_creation_input_tokens ?? 0}`)
  const textOk = /PROBE_OK/.test(text) && toolUses.length === 0
  console.log(`  text call ${textOk ? 'OK' : 'FAILED'} (tool calls: ${toolUses.length})`)

  const echo: BetaTool = {
    name: 'echo',
    description: 'Echo the given text back.',
    input_schema: { type: 'object', properties: { text: { type: 'string', description: 'The text to echo.' } }, required: ['text'] },
  }
  const final2 = await call(client, [{ role: 'user', content: 'Call the echo tool with text set to "probe ping". Do not answer in words first.' }], [echo])
  const toolCall = final2.content.find((b): b is Extract<(typeof b), { type: 'tool_use' }> => b.type === 'tool_use') ?? null
  console.log(`tool call -> ${toolCall ? `${toolCall.name} input: ${JSON.stringify(toolCall.input)}` : '(none)'} (stop_reason: ${final2.stop_reason})`)
  const toolOk = toolCall?.name === 'echo' && typeof (toolCall?.input as { text?: unknown } | undefined)?.text === 'string'
  console.log(`  tool call ${toolOk ? 'OK' : 'FAILED'}`)
  console.log(`PROBE ${textOk && toolOk ? 'PASSED' : 'FAILED'}`)
  return textOk && toolOk ? 0 : 1
}

main()
  .then((code) => process.exit(code))
  .catch((err: unknown) => {
    console.error(`PROBE errored: ${err instanceof Error ? err.message : String(err)}`)
    process.exit(1)
  })
