// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The real vision probe (AMG-11, docs/16a D.11): N fresh challenges whose
// answers live only in the PNG pixels, one arm with the image and one without,
// the input-token delta as proof of delivery. Writes the record under
// <cacheDir>/probe/vision/ and prints trial lines that never quote a response.
// Run from gui/: npx tsx server/scripts/vision-probe.ts --provider anthropic|zai [--model <id>] [--trials <n>]
import { buildLlmClient, loadLlmSettings, resolveEffectiveLlm } from '../src/agent/llmSettings.js'
import { runProbe, writeVisionRecord, type ArmResult } from '../src/attachments/visionProbe.js'
import { loadConfig } from '../src/config.js'

function usage(): void {
  console.log('usage: npx tsx server/scripts/vision-probe.ts --provider anthropic|zai [--model <id>] [--trials <n>]')
}

/** The image-arm half of the trial line: the parsed kind, never the response text. */
function imagePhrase(a: ArmResult['answer']): string {
  return a.kind === 'answer' ? ` ${a.value.number}/${a.value.D_i}/${a.value.L}` : ''
}

async function main(): Promise<number> {
  const argv = process.argv.slice(2)
  let provider: string | null = null
  let model: string | null = null
  let trials = 5
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (a === '--provider') provider = argv[++i] ?? null
    else if (a === '--model') model = argv[++i] ?? null
    else if (a === '--trials') trials = Number(argv[++i] ?? NaN)
    else {
      usage()
      return 3
    }
  }
  if ((provider !== 'anthropic' && provider !== 'zai') || !Number.isInteger(trials) || trials < 1) {
    usage()
    return 3
  }
  // resolveEffectiveLlm reads CFD_LLM out of the environment, so the requested provider wins.
  process.env.CFD_LLM = provider
  const config = loadConfig(process.env)
  const eff = resolveEffectiveLlm(config, loadLlmSettings(config.configDir))
  if (model) eff.model = model
  if ((provider === 'anthropic' && eff.anthropicKey === null) || (provider === 'zai' && eff.zaiKey === null)) {
    console.log(`VISION ${provider} ${eff.model} image-block: NOT RUN (no key)`)
    return 3
  }
  const client = buildLlmClient(config, eff)
  console.log(`probe client kind: ${client.kind} model: ${client.model}`)
  const { record, pngs } = await runProbe({
    client,
    trials,
    onTrial: (t) => {
      const image = t.withImage
      const noImage = t.withoutImage
      console.log(
        `trial ${t.index}: image ${image.answer.kind}${imagePhrase(image.answer)} correct=${t.imageCorrect} in=${image.inputTokens ?? 'null'}` +
          ` | no-image ${noImage === null ? 'never-ran' : noImage.answer.kind} in=${noImage?.inputTokens ?? 'null'}` +
          ` | delta=${t.delta ?? 'null'}`,
      )
    },
  })
  const recordPath = await writeVisionRecord(config.cacheDir, record, pngs)
  console.log(`record: ${recordPath}`)
  for (const reason of record.reasons) console.log(`  - ${reason}`)
  if (record.supersedes.length > 0) console.log(`superseded: ${record.supersedes.join(', ')}`)
  console.log(`VISION ${record.provider} ${record.model} image-block: ${record.status.toUpperCase()}`)
  const hit429 = record.trials.some((t) => t.withImage.errorStatus === 429 || t.withoutImage?.errorStatus === 429)
  if (hit429) {
    console.log('GLM quota / rate limit: stopped, no retry')
    return 4
  }
  return record.status === 'passed' ? 0 : record.status === 'failed' ? 1 : 2
}

main()
  .then((code) => process.exit(code))
  .catch((err: unknown) => {
    console.error(`PROBE errored: ${err instanceof Error ? err.name : 'Error'}`)
    process.exit(1)
  })
