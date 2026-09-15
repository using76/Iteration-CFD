// Live probe for the configured LLM (z.ai/GLM by default): one plain text
// call, then one tool call. Exit 0 only when both worked. Never prints the
// key. Run from gui/: CFD_LLM=zai npx tsx server/scripts/llm-probe.ts
import fs from 'node:fs/promises'
import path from 'node:path'
import type { BetaImageBlockParam, BetaTextBlockParam, BetaTool } from '@anthropic-ai/sdk/resources/beta/messages/messages'
import { createAnthropicClient } from '../src/agent/anthropic.js'
import type { LlmClient, LlmStreamParams } from '../src/agent/llm.js'
import { createZaiClient } from '../src/agent/zai.js'
import { loadConfig } from '../src/config.js'

const SYSTEM: BetaTextBlockParam[] = [{ type: 'text', text: 'You are a probe. Follow the user instruction exactly.' }]

/** A 1x1 opaque-red PNG, hard-coded so the probe needs no file on disk (70 bytes decoded). */
const ONE_PIXEL_PNG =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=='

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

  // ---- image-block probe (N6): does this endpoint translate Anthropic image
  // blocks? The one-word verdict decides the Run 2 vision default; REFUSED is
  // a legitimate answer, so it never gates the exit code.
  const image: BetaImageBlockParam = { type: 'image', source: { type: 'base64', media_type: 'image/png', data: ONE_PIXEL_PNG } }
  let verdict: 'SUPPORTED' | 'REFUSED' | 'UNCLEAR' = 'REFUSED'
  let imageText = ''
  let imageStopReason: string | null = null
  let errorName: string | null = null
  let errorMessage: string | null = null
  let imageUsage = { input: 0, output: 0 }
  try {
    const final3 = await call(client, [{ role: 'user', content: [image, { type: 'text', text: 'Answer with one word: how many pixels does this image contain? If you cannot see an image at all, answer NO_IMAGE.' }] }], [])
    imageStopReason = final3.stop_reason
    imageText = final3.content.filter((b) => b.type === 'text').map((b) => b.text).join('')
    imageUsage = { input: final3.usage.input_tokens, output: final3.usage.output_tokens }
    if (final3.stop_reason === 'refusal' || !imageText.trim()) verdict = 'UNCLEAR'
    else verdict = /NO_IMAGE/i.test(imageText) ? 'REFUSED' : 'SUPPORTED'
  } catch (err) {
    // name and message only: the error object can carry request headers
    errorName = err instanceof Error ? err.name : null
    errorMessage = err instanceof Error ? err.message : String(err)
  }
  console.log(`VISION zai image-block: ${verdict}`)
  const record = {
    probedAt: new Date().toISOString(),
    client: client.kind,
    model: config.model,
    baseUrl: config.llm === 'zai' ? (config.zai?.baseUrl ?? null) : null,
    verdict,
    stopReason: imageStopReason,
    text: imageText,
    errorName,
    errorMessage,
    usage: imageUsage,
  }
  const probeDir = path.join(config.cacheDir, 'probe')
  await fs.mkdir(probeDir, { recursive: true })
  await fs.writeFile(path.join(probeDir, 'zai-image-block.json'), JSON.stringify(record, null, 2) + '\n')
  console.log(`PROBE ${textOk && toolOk ? 'PASSED' : 'FAILED'}`)
  return textOk && toolOk ? 0 : 1
}

main()
  .then((code) => process.exit(code))
  .catch((err: unknown) => {
    console.error(`PROBE errored: ${err instanceof Error ? err.message : String(err)}`)
    process.exit(1)
  })
