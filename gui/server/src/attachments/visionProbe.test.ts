// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// Vision probe tests (AMG-11 R1-R12). The PNG is decoded by the test's OWN
// decoder — chunk walk, CRC check, inflate, filter-strip — never by anything
// imported from the module, so the pixels are read back exactly as a model
// would see them. Mock clients answer from what they saw, from the prompt, or
// not at all; a 429 must stop the probe and leak nothing.
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import zlib from 'node:zlib'
import { afterAll, describe, expect, it } from 'vitest'
import type { BetaMessage, BetaRawMessageStreamEvent } from '@anthropic-ai/sdk/resources/beta/messages/messages'
import type { LlmClient, LlmStreamParams } from '../agent/llm.js'
import { FONT, HEIGHT, WIDTH } from './visionPng.js'
import { makeChallenge, parseAnswer, PROMPT, readVisionVerdict, runProbe, SYSTEM_TEXT, writeVisionRecord, type Challenge, type SketchTruth } from './visionProbe.js'

// ---- temp dirs, removed in afterAll ----------------------------------------

const tempDirs: string[] = []
const newTempDir = (): string => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'vprobe-'))
  tempDirs.push(dir)
  return dir
}
afterAll(() => {
  for (const dir of tempDirs) fs.rmSync(dir, { recursive: true, force: true })
})

// ---- mock clients -----------------------------------------------------------

type MockReply = { text?: string; inputTokens: number; toolUse?: boolean; throws?: { name: string; status: number; message: string } }
type MockAnswer = (req: LlmStreamParams) => MockReply

function mockClientOf(kind: LlmClient['kind'], answer: MockAnswer): LlmClient {
  const model = 'mock-vision'
  return {
    kind,
    model,
    stream: (p: LlmStreamParams) => ({
      events: (async function* (): AsyncGenerator<BetaRawMessageStreamEvent> {})(),
      finalMessage: async () => {
        const a = answer(p)
        if (a.throws) {
          const err = new Error(a.throws.message)
          err.name = a.throws.name
          ;(err as Error & { status?: number }).status = a.throws.status
          throw err
        }
        const content: Array<Record<string, unknown>> = [{ type: 'text', text: a.text ?? '' }]
        if (a.toolUse) content.push({ type: 'tool_use', id: 't', name: 'x', input: {} })
        const message = { id: 'm', type: 'message', role: 'assistant', model, content, stop_reason: 'end_turn', stop_sequence: null, usage: { input_tokens: a.inputTokens, output_tokens: 5 } }
        return message as unknown as BetaMessage
      },
    }),
  }
}

/** The brief's helper: kind 'zai', model 'mock-vision'. */
function mockClient(answer: MockAnswer): LlmClient {
  return mockClientOf('zai', answer)
}

/** The base64 data of the image block in the request's first message, or null. */
function imageOf(req: LlmStreamParams): string | null {
  const content = req.messages[0]?.content
  if (!Array.isArray(content)) return null
  for (const b of content) {
    if ((b as { type?: string }).type === 'image') return (b as { source?: { data?: string } }).source?.data ?? null
  }
  return null
}

const seen = new Map<string, SketchTruth>()
/** A fresh challenge whose truth the seeing mocks can look up by the PNG's base64. */
const mk = (): Challenge => {
  const c = makeChallenge()
  seen.set(c.png.toString('base64'), c.truth)
  return c
}

/** Answers ONLY from seen.get(imageOf(req)); UNAVAILABLE when there is no image. */
const seeAnswer = (req: LlmStreamParams): MockReply => {
  const t = seen.get(imageOf(req) ?? '')
  return { text: t ? JSON.stringify({ number: t.number, D_i: t.D_i, L: t.L }) : 'UNAVAILABLE', inputTokens: 200 + (imageOf(req) ? 1500 : 0) }
}

/** Pattern-matches the ask out of the prompt text — a model that never sees the pixels. */
const promptAnswer = (req: LlmStreamParams): MockReply => {
  const content = req.messages[0]?.content
  const text = Array.isArray(content) ? content.filter((b) => (b as { type?: string }).type === 'text').map((b) => (b as { text?: string }).text ?? '').join('') : ''
  const number = /\d{4}/.exec(text)?.[0] ?? '0000'
  const afterD = text.slice(text.indexOf('D') + 1)
  const afterL = text.slice(text.indexOf('L') + 1)
  return {
    text: JSON.stringify({ number, D_i: Number(/(\d+)/.exec(afterD)?.[1] ?? 0), L: Number(/(\d+)/.exec(afterL)?.[1] ?? 0) }),
    inputTokens: 200 + (imageOf(req) ? 1500 : 0),
  }
}

// ---- the test's own PNG decoder (T1) ----------------------------------------

interface Chunk {
  type: string
  data: Buffer
}

/** Signature, every chunk's CRC against node:zlib crc32, the chunk-type list, the IHDR fields, and the filter-stripped pixels. */
function decode(png: Buffer): { types: string[]; pixels: Uint8Array } {
  expect(png.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))).toBe(true)
  const chunks: Chunk[] = []
  let off = 8
  while (off < png.length) {
    const len = png.readUInt32BE(off)
    const typeBuf = png.subarray(off + 4, off + 8)
    const data = Buffer.from(png.subarray(off + 8, off + 8 + len))
    expect(png.readUInt32BE(off + 8 + len)).toBe(zlib.crc32(Buffer.concat([typeBuf, data])))
    chunks.push({ type: typeBuf.toString('ascii'), data })
    off += 12 + len
  }
  const types = chunks.map((c) => c.type)
  expect(types[0]).toBe('IHDR')
  expect(types.at(-1)).toBe('IEND')
  expect(types.length).toBeGreaterThanOrEqual(3)
  for (let i = 1; i < types.length - 1; i++) expect(types[i]).toBe('IDAT') // one or more IDAT, nothing else between
  for (const t of ['tEXt', 'zTXt', 'iTXt']) expect(types).not.toContain(t)
  const ihdr = chunks[0]!.data
  expect(ihdr.length).toBe(13)
  expect(ihdr.readUInt32BE(0)).toBe(480) // the brief's literals, not the module's constants
  expect(ihdr.readUInt32BE(4)).toBe(340)
  expect(ihdr[8]).toBe(8) // bit depth
  expect(ihdr[9]).toBe(0) // colour type: grayscale
  const raw = zlib.inflateSync(Buffer.concat(chunks.filter((c) => c.type === 'IDAT').map((c) => c.data)))
  expect(raw.length).toBe(HEIGHT * (1 + WIDTH))
  const pixels = new Uint8Array(WIDTH * HEIGHT)
  for (let y = 0; y < HEIGHT; y++) {
    expect(raw[y * (1 + WIDTH)]).toBe(0) // filter byte: None
    pixels.set(raw.subarray(y * (1 + WIDTH) + 1, (y + 1) * (1 + WIDTH)), y * WIDTH)
  }
  return { types, pixels }
}

function inkAt(pixels: Uint8Array, x: number, y: number): boolean {
  return pixels[y * WIDTH + x] === 0
}

/** Reads n glyphs by matching every FONT key against the 5x7 cell centres; exactly one key must match per position. */
function readText(pixels: Uint8Array, x: number, y: number, n: number, scale: number): string {
  let out = ''
  for (let i = 0; i < n; i++) {
    const ox = x + i * 6 * scale
    let found: string | null = null
    for (const [ch, glyph] of Object.entries(FONT)) {
      let ok = true
      for (let row = 0; row < 7 && ok; row++) {
        for (let col = 0; col < 5 && ok; col++) {
          const want = glyph[row]!.charAt(col) === '#'
          if (inkAt(pixels, ox + col * scale + Math.floor(scale / 2), y + row * scale + Math.floor(scale / 2)) !== want) ok = false
        }
      }
      if (ok) {
        expect(found).toBeNull() // exactly one key must match
        found = ch
      }
    }
    expect(found).not.toBeNull()
    out += found
  }
  return out
}

// ---- the tests ---------------------------------------------------------------

describe('vision probe', () => {
  it('T1: the answer is only in the IDAT pixels, and the drawn digits read back', () => {
    for (let i = 0; i < 5; i++) {
      const c = makeChallenge()
      const { pixels } = decode(c.png)
      expect(readText(pixels, 24, 16, 4, 6)).toBe(c.truth.number)
      expect(readText(pixels, 12, 190, 2 + String(c.truth.D_i).length, 3)).toBe(`D=${c.truth.D_i}`)
      expect(readText(pixels, 250, 286, 2 + String(c.truth.L).length, 3)).toBe(`L=${c.truth.L}`)
      expect(c.png.includes(Buffer.from(c.truth.number))).toBe(false)
    }
  })

  it('T2: no digit reaches the text; the image arm is one image block then the prompt, the no-image arm text only', async () => {
    expect(/[0-9]/.test(PROMPT)).toBe(false)
    expect(/[0-9]/.test(SYSTEM_TEXT)).toBe(false)
    const captured: LlmStreamParams[] = []
    const client = mockClient((req) => {
      captured.push(req)
      return seeAnswer(req)
    })
    await runProbe({ client, trials: 1, makeChallenge: mk })
    expect(captured).toHaveLength(2)
    const withImage = captured[0]!
    const noImage = captured[1]!
    expect(withImage.messages).toHaveLength(1)
    expect(withImage.messages[0]!.role).toBe('user')
    const wc = withImage.messages[0]!.content as unknown as Array<Record<string, unknown>>
    expect(wc).toHaveLength(2)
    expect(wc[0]!.type).toBe('image')
    expect((wc[0]! as { source: Record<string, unknown> }).source).toEqual({ type: 'base64', media_type: 'image/png', data: expect.any(String) })
    expect(seen.has((wc[0]! as { source: { data: string } }).source.data)).toBe(true)
    expect(wc[1]).toEqual({ type: 'text', text: PROMPT })
    const nc = noImage.messages[0]!.content as unknown as Array<Record<string, unknown>>
    expect(nc).toEqual([{ type: 'text', text: PROMPT }])
    expect(withImage.tools).toEqual([])
    expect(noImage.tools).toEqual([])
  })

  it('T3: 20 default challenges are fresh and every value is in range', () => {
    const numbers = new Set<string>()
    for (let i = 0; i < 20; i++) {
      const c = makeChallenge()
      expect(c.truth.number).toMatch(/^\d{4}$/)
      expect(Number(c.truth.number)).toBeGreaterThanOrEqual(1000)
      expect(Number(c.truth.number)).toBeLessThanOrEqual(9999)
      expect(c.truth.D_i).toBeGreaterThanOrEqual(40)
      expect(c.truth.D_i).toBeLessThanOrEqual(199)
      expect(c.truth.L).toBeGreaterThanOrEqual(100)
      expect(c.truth.L).toBeLessThanOrEqual(999)
      numbers.add(c.truth.number)
    }
    expect(numbers.size).toBeGreaterThanOrEqual(15)
  })

  it('T4: the parse is strict', () => {
    expect(parseAnswer('UNAVAILABLE')).toEqual({ kind: 'unavailable' })
    expect(parseAnswer(' UNAVAILABLE\n')).toEqual({ kind: 'unavailable' })
    expect(parseAnswer('{"number":"4821","D_i":137,"L":412}')).toEqual({ kind: 'answer', value: { number: '4821', D_i: 137, L: 412 } })
    expect(parseAnswer('```json\n{"number":"4821","D_i":137,"L":412}\n```')).toEqual({ kind: 'answer', value: { number: '4821', D_i: 137, L: 412 } })
    for (const bad of [
      'unavailable',
      '',
      'The answer is {"number":"4821","D_i":137,"L":412}',
      '{"number":4821,"D_i":137,"L":412}',
      '{"number":"482","D_i":137,"L":412}',
      '{"number":"4821","D_i":137.5,"L":412}',
      '{"number":"4821","D_i":137,"L":412,"x":1}',
      '[1,2]',
      'null',
    ]) {
      expect(parseAnswer(bad)).toEqual({ kind: 'malformed' })
    }
  })

  it('T5: a mock that answers from the prompt fails', async () => {
    const { record } = await runProbe({ client: mockClient(promptAnswer), makeChallenge: mk })
    expect(record.status).toBe('failed')
    expect(record.summary.imageCorrect).toBe(0)
  })

  it('T6: a mock that always answers a number fails the no-image arm', async () => {
    const client = mockClient((req) => {
      if (imageOf(req)) return seeAnswer(req)
      return { text: '{"number":"1234","D_i":100,"L":500}', inputTokens: 200 }
    })
    const { record } = await runProbe({ client, makeChallenge: mk })
    expect(record.status).toBe('failed')
    expect(record.summary.imageCorrect).toBe(5)
    expect(record.summary.noImageUnavailable).toBe(0)
  })

  it('T7: a mock that sees passes', async () => {
    const { record } = await runProbe({ client: mockClient(seeAnswer), makeChallenge: mk })
    expect(record.status).toBe('passed')
    expect(record.reasons).toEqual([])
    expect(record.summary.imageCorrect).toBe(5)
    expect(record.summary.noImageUnavailable).toBe(5)
    expect(record.summary.minDelta).toBe(1500)
  })

  it('T8: no token delta is inconclusive, never passed', async () => {
    const client = mockClient((req) => {
      const reply = seeAnswer(req)
      return { ...reply, inputTokens: 200 }
    })
    const { record } = await runProbe({ client, makeChallenge: mk })
    expect(record.status).toBe('inconclusive')
    expect(record.summary.imageCorrect).toBe(5)
    expect(record.reasons).toHaveLength(5)
    for (const r of record.reasons) expect(r).toContain('delivery unproven')
  })

  it('T9: any tool use is inconclusive', async () => {
    let calls = 0
    const client = mockClient((req) => {
      const reply = seeAnswer(req)
      return { ...reply, toolUse: ++calls > 2 } // trial 1's arms stay clean; trial 2's image arm uses a tool
    })
    const { record } = await runProbe({ client, makeChallenge: mk })
    expect(record.status).toBe('inconclusive')
    expect(record.reasons.some((r) => r.includes('tool'))).toBe(true)
  })

  it('T10: a 429 stops the probe with no retry, and no provider body reaches the record', async () => {
    const quota = mockClient(() => ({ inputTokens: 200, throws: { name: 'RateLimitError', status: 429, message: 'SECRET-BODY' } }))
    const { record } = await runProbe({ client: quota, trials: 5, makeChallenge: mk })
    expect(record.trials).toHaveLength(1)
    expect(record.trials[0]!.withoutImage).toBeNull()
    expect(record.trials[0]!.withImage.errorStatus).toBe(429)
    expect(record.status).toBe('inconclusive')
    expect(JSON.stringify(record)).not.toContain('SECRET-BODY')
    const malformed = mockClient((req) => (imageOf(req) ? { text: 'BODYMARK not json', inputTokens: 1700 } : { text: 'UNAVAILABLE', inputTokens: 200 }))
    const second = await runProbe({ client: malformed, trials: 5, makeChallenge: mk })
    const json = JSON.stringify(second.record)
    expect(json).not.toContain('SECRET-BODY')
    expect(json).not.toContain('BODYMARK')
    expect(second.record.status).toBe('failed')
  })

  it("T11: one record per provider/model/date/path; readVisionVerdict returns the latest date's status", async () => {
    const dir = newTempDir()
    const passed = await runProbe({ client: mockClient(seeAnswer), makeChallenge: mk, now: () => new Date('2026-09-25T12:00:00Z') })
    await writeVisionRecord(dir, { ...passed.record, model: 'glm-5.3-flash' }, passed.pngs)
    const failed = await runProbe({ client: mockClient(promptAnswer), makeChallenge: mk, now: () => new Date('2026-09-26T12:00:00Z') })
    const recordPath = await writeVisionRecord(dir, { ...failed.record, model: 'glm-5.3-flash' }, failed.pngs)
    expect(path.basename(recordPath)).toBe('zai__glm-5.3-flash__2026-09-26__image-block.json')
    const visionDir = path.join(dir, 'probe', 'vision')
    for (let k = 1; k <= 5; k++) expect(fs.existsSync(path.join(visionDir, `zai__glm-5.3-flash__2026-09-26__image-block.trial${k}.png`))).toBe(true)
    expect(readVisionVerdict(dir, 'zai', 'glm-5.3-flash')).toBe('failed')
    expect(readVisionVerdict(dir, 'zai', 'other-model')).toBeNull()
    expect(readVisionVerdict(newTempDir(), 'zai', 'glm-5.3-flash')).toBeNull() // no probe folder at all
    const broken = JSON.parse(fs.readFileSync(recordPath, 'utf8')) as Record<string, unknown>
    broken['schema'] = 'vision-probe/0'
    fs.writeFileSync(recordPath, JSON.stringify(broken, null, 2) + '\n')
    expect(readVisionVerdict(dir, 'zai', 'glm-5.3-flash')).toBeNull()
  })

  it('T12: writing a zai record supersedes the old probe file; an anthropic record does not touch it', async () => {
    const dir = newTempDir()
    fs.mkdirSync(path.join(dir, 'probe'), { recursive: true })
    const oldPath = path.join(dir, 'probe', 'zai-image-block.json')
    fs.writeFileSync(oldPath, JSON.stringify({ verdict: 'SUPPORTED', text: '3000' }, null, 2) + '\n')
    const run = await runProbe({ client: mockClient(seeAnswer), trials: 1, makeChallenge: mk })
    const recordPath = await writeVisionRecord(dir, run.record, run.pngs)
    const old = JSON.parse(fs.readFileSync(oldPath, 'utf8')) as Record<string, unknown>
    expect(old['verdict']).toBe('SUPPORTED')
    expect(old['supersededBy']).toBe(`vision/${path.basename(recordPath)}`)
    expect(run.record.supersedes).toEqual(['zai-image-block.json'])
    const dir2 = newTempDir()
    fs.mkdirSync(path.join(dir2, 'probe'), { recursive: true })
    const old2 = path.join(dir2, 'probe', 'zai-image-block.json')
    const bytes = Buffer.from(JSON.stringify({ verdict: 'SUPPORTED', text: '3000' }, null, 2) + '\n')
    fs.writeFileSync(old2, bytes)
    const run2 = await runProbe({ client: mockClientOf('anthropic', seeAnswer), trials: 1, makeChallenge: mk })
    await writeVisionRecord(dir2, run2.record, run2.pngs)
    expect(fs.readFileSync(old2).equals(bytes)).toBe(true)
    expect(run2.record.supersedes).toEqual([])
  })
})
