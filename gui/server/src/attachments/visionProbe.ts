// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The real vision probe (AMG-11): every trial draws a fresh random challenge
// whose answer is only in the PNG pixels, sends one arm with the image and one
// without, and judges passed / failed / inconclusive on the parsed answers and
// the input-token delta. The old probe's "SUPPORTED" only tested for the
// absence of NO_IMAGE; this record decides whether z.ai gets image blocks.
// The idea (a fresh random challenge whose answer is only
// in the PNG pixels, a strict parse, a three-valued status) is reimplemented from reading Amagine3D
// (https://github.com/amagine-ai/Amagine3D, e608dc6, packages/a3d-runtime/src/vision-probe.ts, Apache-2.0); no code is
// copied. Extended per docs/16a D.11: printed digits and dimensions, a no-image arm, and the input-token delta as proof of delivery.
import { createHash, randomInt } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import type { BetaMessage, BetaMessageParam, BetaTextBlockParam } from '@anthropic-ai/sdk/resources/beta/messages/messages'
import type { LlmClient, LlmStreamParams } from '../agent/llm.js'
import { drawSketch } from './visionPng.js'

export interface SketchTruth {
  number: string
  D_i: number
  L: number
}
export interface Challenge {
  png: Buffer
  truth: SketchTruth
}

/** rand(min, max) returns an integer in [min, max) like crypto.randomInt. */
export function makeChallenge(rand: (min: number, max: number) => number = randomInt): Challenge {
  const truth = { number: String(rand(1000, 10000)), D_i: rand(40, 200), L: rand(100, 1000) }
  return { png: drawSketch(truth.number, truth.D_i, truth.L), truth }
}

export const SYSTEM_TEXT = 'You are a vision probe. Follow the user instruction exactly.'
export const PROMPT =
  'This message may contain one image: a four-digit number printed at the top, and below it a sketch of a nozzle ' +
  'with two printed dimensions, D (the inlet diameter) and L (the length), both whole millimetres. ' +
  'Do not call any tool. If you can see the image, reply with only a JSON object ' +
  '{"number": "<the four digits>", "D_i": <D as an integer>, "L": <L as an integer>}. ' +
  'If you cannot see an image, reply with only the word UNAVAILABLE. Do not guess.'

export type ParsedAnswer =
  | { kind: 'unavailable' }
  | { kind: 'answer'; value: SketchTruth }
  | { kind: 'malformed' }
  | { kind: 'error' }

/**
 * Strict: trim; if the whole text is one fence /^```(?:json)?\s*\n?([\s\S]*?)\n?\s*```$/ take its body and trim again;
 * exactly 'UNAVAILABLE' (case-sensitive) -> unavailable; else JSON.parse must give a plain object (not an array, not null)
 * whose Object.keys sorted equal ['D_i', 'L', 'number'], number a string matching /^\d{4}$/, D_i and L Number.isInteger;
 * -> answer. Anything else (including a JSON.parse throw) -> malformed.
 */
export function parseAnswer(text: string): ParsedAnswer {
  let t = text.trim()
  const fence = /^```(?:json)?\s*\n?([\s\S]*?)\n?\s*```$/.exec(t)
  if (fence) t = (fence[1] as string).trim()
  if (t === 'UNAVAILABLE') return { kind: 'unavailable' }
  let parsed: unknown
  try {
    parsed = JSON.parse(t)
  } catch {
    return { kind: 'malformed' }
  }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) return { kind: 'malformed' }
  const obj = parsed as Record<string, unknown>
  const keys = Object.keys(obj).sort()
  if (keys.length !== 3 || keys[0] !== 'D_i' || keys[1] !== 'L' || keys[2] !== 'number') return { kind: 'malformed' }
  const number = obj['number']
  const d = obj['D_i']
  const l = obj['L']
  if (typeof number !== 'string' || !/^\d{4}$/.test(number)) return { kind: 'malformed' }
  if (typeof d !== 'number' || !Number.isInteger(d)) return { kind: 'malformed' }
  if (typeof l !== 'number' || !Number.isInteger(l)) return { kind: 'malformed' }
  return { kind: 'answer', value: { number, D_i: d, L: l } }
}

export interface ArmResult {
  answer: ParsedAnswer
  inputTokens: number | null
  outputTokens: number | null
  stopReason: string | null
  /** The model id the response names (a server-side fallback can differ from the one asked for); null on error. */
  respondedModel: string | null
  toolUse: boolean
  /** sha256 hex of the concatenated text blocks; the text itself is never stored. null on error. */
  textSha256: string | null
  textLength: number
  /** On a thrown error: err.name and a numeric err.status if present. err.message is NEVER read into the record. */
  errorName: string | null
  errorStatus: number | null
}

export interface Trial {
  index: number // 1-based
  pngSha256: string
  truth: SketchTruth
  withImage: ArmResult
  withoutImage: ArmResult | null // null only when the probe stopped on a 429 during the image arm
  /** withImage.inputTokens - withoutImage.inputTokens, or null when either is null. */
  delta: number | null
  imageCorrect: boolean // withImage.answer is 'answer' and all three fields equal truth
  noImageUnavailable: boolean // withoutImage?.answer.kind === 'unavailable'
}

export type ProbeStatus = 'passed' | 'failed' | 'inconclusive'
export const REQUIRED_TRIALS = 5
export const MIN_DELTA = 100
export const DELIVERY_PATH = 'image-block'

/**
 * One isolated request: messages = [{ role: 'user', content: png ? [imageBlock, textBlock] : [textBlock] }] where
 * imageBlock = { type: 'image', source: { type: 'base64', media_type: 'image/png', data: png.toString('base64') } }
 * and textBlock = { type: 'text', text: PROMPT }; system = [{ type: 'text', text: SYSTEM_TEXT }]; tools = [];
 * maxTokens 2048; effort 'medium'; signal = new AbortController().signal. Calls client.stream(params).finalMessage().
 * toolUse = any content block of type 'tool_use' (or 'server_tool_use'). Text = the 'text' blocks joined with ''.
 * A throw becomes answer {kind:'error'} with errorName/errorStatus; it is never rethrown.
 */
export async function runArm(client: LlmClient, png: Buffer | null): Promise<ArmResult> {
  const textBlock: BetaTextBlockParam = { type: 'text', text: PROMPT }
  const content: BetaMessageParam['content'] = png
    ? [{ type: 'image', source: { type: 'base64', media_type: 'image/png', data: png.toString('base64') } }, textBlock]
    : [textBlock]
  const params: LlmStreamParams = {
    system: [{ type: 'text', text: SYSTEM_TEXT }],
    messages: [{ role: 'user', content }],
    tools: [],
    maxTokens: 2048,
    effort: 'medium',
    signal: new AbortController().signal,
  }
  try {
    const message: BetaMessage = await client.stream(params).finalMessage()
    const toolUse = message.content.some((b) => b.type === 'tool_use' || b.type === 'server_tool_use')
    const text = message.content
      .filter((b): b is Extract<(typeof b), { type: 'text' }> => b.type === 'text')
      .map((b) => b.text)
      .join('')
    return {
      answer: parseAnswer(text),
      inputTokens: message.usage?.input_tokens ?? null,
      outputTokens: message.usage?.output_tokens ?? null,
      stopReason: message.stop_reason ?? null,
      respondedModel: typeof message.model === 'string' ? message.model : null,
      toolUse,
      textSha256: createHash('sha256').update(text).digest('hex'),
      textLength: text.length,
      errorName: null,
      errorStatus: null,
    }
  } catch (err) {
    const e = err as { name?: unknown; status?: unknown }
    return {
      answer: { kind: 'error' },
      inputTokens: null,
      outputTokens: null,
      stopReason: null,
      respondedModel: null,
      toolUse: false,
      textSha256: null,
      textLength: 0,
      errorName: typeof e.name === 'string' ? e.name : null,
      errorStatus: typeof e.status === 'number' ? e.status : null,
    }
  }
}

function answerPhrase(a: ParsedAnswer): string {
  return a.kind === 'answer' ? `${a.value.number}/${a.value.D_i}/${a.value.L}` : a.kind
}

/**
 * Reasons are collected in this order, each naming its trial ("trial 3: ..."):
 *   inconclusive reasons: fewer than REQUIRED_TRIALS trials; any arm errorName !== null ("trial k: image arm errored
 *   RateLimitError 429"); any arm toolUse; any withoutImage === null; any delta === null or delta <= MIN_DELTA
 *   ("trial k: input-token delta 12 <= 100: delivery unproven").
 *   failed reasons: any !imageCorrect ("trial k: image arm answered malformed" or "... answered 1234/56/789,
 *   expected 4821/137/412"); any !noImageUnavailable ("trial k: no-image arm answered answer, expected UNAVAILABLE").
 * status = 'inconclusive' if any inconclusive reason, else 'failed' if any failed reason, else 'passed'.
 * All reasons of both kinds are returned.
 */
export function judge(trials: readonly Trial[]): { status: ProbeStatus; reasons: string[] } {
  const inconclusive: string[] = []
  const failed: string[] = []
  if (trials.length < REQUIRED_TRIALS) inconclusive.push(`fewer than ${REQUIRED_TRIALS} trials (${trials.length})`)
  for (const t of trials) {
    const k = t.index
    if (t.withImage.errorName !== null) {
      const status = t.withImage.errorStatus !== null ? ` ${t.withImage.errorStatus}` : ''
      inconclusive.push(`trial ${k}: image arm errored ${t.withImage.errorName}${status}`)
    }
    if (t.withImage.toolUse) inconclusive.push(`trial ${k}: image arm used a tool`)
    if (t.withoutImage === null) {
      inconclusive.push(`trial ${k}: no-image arm never ran`)
    } else {
      if (t.withoutImage.errorName !== null) {
        const status = t.withoutImage.errorStatus !== null ? ` ${t.withoutImage.errorStatus}` : ''
        inconclusive.push(`trial ${k}: no-image arm errored ${t.withoutImage.errorName}${status}`)
      }
      if (t.withoutImage.toolUse) inconclusive.push(`trial ${k}: no-image arm used a tool`)
      if (t.delta === null) inconclusive.push(`trial ${k}: input-token delta null: delivery unproven`)
      else if (t.delta <= MIN_DELTA) inconclusive.push(`trial ${k}: input-token delta ${t.delta} <= ${MIN_DELTA}: delivery unproven`)
      if (!t.noImageUnavailable) failed.push(`trial ${k}: no-image arm answered ${t.withoutImage.answer.kind}, expected UNAVAILABLE`)
    }
    if (!t.imageCorrect) failed.push(`trial ${k}: image arm answered ${answerPhrase(t.withImage.answer)}, expected ${t.truth.number}/${t.truth.D_i}/${t.truth.L}`)
  }
  const status: ProbeStatus = inconclusive.length > 0 ? 'inconclusive' : failed.length > 0 ? 'failed' : 'passed'
  return { status, reasons: [...inconclusive, ...failed] }
}

export interface VisionRecord {
  schema: 'vision-probe/1'
  provider: 'anthropic' | 'zai' | 'mock'
  model: string
  date: string // UTC YYYY-MM-DD of probedAt
  deliveryPath: 'image-block'
  probedAt: string // ISO
  status: ProbeStatus
  reasons: string[]
  summary: { trials: number; imageCorrect: number; noImageUnavailable: number; minDelta: number | null }
  trials: Trial[]
  /** Records this one marked superseded, relative to <cacheDir>/probe (e.g. 'zai-image-block.json'); filled by writeVisionRecord. */
  supersedes: string[]
}

export interface ProbeOptions {
  client: LlmClient
  trials?: number // default REQUIRED_TRIALS
  makeChallenge?: () => Challenge // default makeChallenge()
  now?: () => Date // default () => new Date()
  onTrial?: (t: Trial) => void
}

/**
 * For i = 1..trials: a fresh challenge; the image arm; then the no-image arm; push the Trial; call onTrial.
 * If either arm's errorStatus === 429, push that trial (withoutImage null if the image arm hit it) and STOP - no retry.
 * provider = client.kind, model = client.model. Returns the record (supersedes: []) and the PNGs in trial order.
 */
export async function runProbe(opts: ProbeOptions): Promise<{ record: VisionRecord; pngs: Buffer[] }> {
  const count = opts.trials ?? REQUIRED_TRIALS
  const challengeFor = opts.makeChallenge ?? (() => makeChallenge())
  const now = opts.now ?? (() => new Date())
  const probedAt = now()
  const trials: Trial[] = []
  const pngs: Buffer[] = []
  let stopped = false
  for (let i = 1; i <= count && !stopped; i++) {
    const challenge = challengeFor()
    const withImage = await runArm(opts.client, challenge.png)
    let withoutImage: ArmResult | null = null
    if (withImage.errorStatus === 429) {
      stopped = true
    } else {
      withoutImage = await runArm(opts.client, null)
      if (withoutImage.errorStatus === 429) stopped = true
    }
    const delta = withImage.inputTokens !== null && withoutImage !== null && withoutImage.inputTokens !== null ? withImage.inputTokens - withoutImage.inputTokens : null
    const trial: Trial = {
      index: i,
      pngSha256: createHash('sha256').update(challenge.png).digest('hex'),
      truth: challenge.truth,
      withImage,
      withoutImage,
      delta,
      imageCorrect: withImage.answer.kind === 'answer' && withImage.answer.value.number === challenge.truth.number && withImage.answer.value.D_i === challenge.truth.D_i && withImage.answer.value.L === challenge.truth.L,
      noImageUnavailable: withoutImage?.answer.kind === 'unavailable',
    }
    trials.push(trial)
    pngs.push(challenge.png)
    opts.onTrial?.(trial)
  }
  const verdict = judge(trials)
  const deltas = trials.map((t) => t.delta).filter((d): d is number => d !== null)
  const record: VisionRecord = {
    schema: 'vision-probe/1',
    provider: opts.client.kind,
    model: opts.client.model,
    date: probedAt.toISOString().slice(0, 10),
    deliveryPath: DELIVERY_PATH,
    probedAt: probedAt.toISOString(),
    status: verdict.status,
    reasons: verdict.reasons,
    summary: {
      trials: trials.length,
      imageCorrect: trials.filter((t) => t.imageCorrect).length,
      noImageUnavailable: trials.filter((t) => t.noImageUnavailable).length,
      minDelta: deltas.length ? Math.min(...deltas) : null,
    },
    trials,
    supersedes: [],
  }
  return { record, pngs }
}

/** model with every character outside [A-Za-z0-9._-] replaced by '_'. */
export function safeModel(model: string): string {
  return model.replace(/[^A-Za-z0-9._-]/g, '_')
}

/** `${provider}__${safeModel(model)}__${date}__image-block.json` */
export function recordName(provider: string, model: string, date: string): string {
  return `${provider}__${safeModel(model)}__${date}__image-block.json`
}

/**
 * mkdir -p <cacheDir>/probe/vision; write the record (JSON, 2-space indent, trailing newline) at recordName(...),
 * overwriting a same-key record; write each PNG as <recordName without .json>.trial<k>.png (k 1-based).
 * If provider === 'zai' and <cacheDir>/probe/zai-image-block.json exists, parses as a plain object and has no
 * 'supersededBy' key: write it back (2-space JSON + newline) with supersededBy = 'vision/<recordName>' and
 * supersededReason = 'AMG-11 (docs/16a D.11): the old probe tested only for the absence of NO_IMAGE; the vision-probe record decides'
 * and push 'zai-image-block.json' into record.supersedes BEFORE the record is written. Returns the record path.
 */
export async function writeVisionRecord(cacheDir: string, record: VisionRecord, pngs: readonly Buffer[]): Promise<string> {
  const visionDir = path.join(cacheDir, 'probe', 'vision')
  await fs.promises.mkdir(visionDir, { recursive: true })
  const name = recordName(record.provider, record.model, record.date)
  if (record.provider === 'zai') {
    const oldPath = path.join(cacheDir, 'probe', 'zai-image-block.json')
    try {
      const raw = JSON.parse(await fs.promises.readFile(oldPath, 'utf8')) as unknown
      if (typeof raw === 'object' && raw !== null && !Array.isArray(raw) && !('supersededBy' in raw)) {
        const marked = { ...(raw as Record<string, unknown>), supersededBy: `vision/${name}`, supersededReason: 'AMG-11 (docs/16a D.11): the old probe tested only for the absence of NO_IMAGE; the vision-probe record decides' }
        await fs.promises.writeFile(oldPath, JSON.stringify(marked, null, 2) + '\n')
        record.supersedes.push('zai-image-block.json')
      }
    } catch {
      // absent or unreadable: nothing to supersede
    }
  }
  const recordPath = path.join(visionDir, name)
  await fs.promises.writeFile(recordPath, JSON.stringify(record, null, 2) + '\n')
  const base = name.replace(/\.json$/, '')
  for (let k = 0; k < pngs.length; k++) await fs.promises.writeFile(path.join(visionDir, `${base}.trial${k + 1}.png`), pngs[k] as Buffer)
  return recordPath
}

/**
 * Synchronous. Lists <cacheDir>/probe/vision, keeps names starting `${provider}__${safeModel(model)}__` and ending
 * '__image-block.json', takes the lexicographically greatest (dates sort as strings), parses it, and returns its
 * status only if schema === 'vision-probe/1', provider and model match and status is one of the three; otherwise,
 * or on any error (missing dir included), null.
 */
export function readVisionVerdict(cacheDir: string, provider: string, model: string): ProbeStatus | null {
  try {
    const dir = path.join(cacheDir, 'probe', 'vision')
    const prefix = `${provider}__${safeModel(model)}__`
    const names = fs
      .readdirSync(dir)
      .filter((n) => n.startsWith(prefix) && n.endsWith('__image-block.json'))
      .sort()
    const latest = names.at(-1)
    if (!latest) return null
    const rec = JSON.parse(fs.readFileSync(path.join(dir, latest), 'utf8')) as Record<string, unknown>
    if (rec['schema'] !== 'vision-probe/1') return null
    if (rec['provider'] !== provider || rec['model'] !== model) return null
    const status = rec['status']
    if (status !== 'passed' && status !== 'failed' && status !== 'inconclusive') return null
    return status
  } catch {
    return null
  }
}
