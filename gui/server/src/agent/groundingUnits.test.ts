// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// GROUND-UNITS, pure tests: a number is grounded on a tool result, on the user's own earlier turns, or - written
// with a UNIT_TO_SI unit - when its SI value is within UNIT_REL of a source number; the repair round speaks the session locale.
import type { BetaMessage, BetaMessageParam, BetaRawMessageStreamEvent } from '@anthropic-ai/sdk/resources/beta/messages/messages'
import { describe, expect, it } from 'vitest'
import { extractNumbers, groundingPools, lintSession, lintText, UNIT_REL, UNIT_TO_SI, userTurnTexts } from './grounding.js'
import { groundReply, type Block, type GroundReplyResult } from './groundingRepair.js'
import type { LlmClient } from './llm.js'
import { makeMessage, mockEvents, type MockPlan } from './mockLlm.js'
import { spyLlm, textOf } from './test-util.js'

function planLlm(kind: LlmClient['kind'], model: string, plans: MockPlan[]): LlmClient {
  let n = 0
  return {
    kind,
    model,
    stream(params) {
      const p = plans[Math.min(n++, plans.length - 1)]
      let final: BetaMessage | null = null
      const gen = mockEvents(p, model, { signal: params.signal, delayMs: 0, inputTokens: 10 }, (m) => (final = m))
      return {
        events: gen as AsyncIterable<BetaRawMessageStreamEvent>,
        finalMessage: async () => final ?? makeMessage(model, [], 'end_turn', null, 10),
      }
    },
  }
}

const say = (text: string): MockPlan => ({ blocks: [{ type: 'text', text }], stopReason: 'end_turn' })
const ung = (r: { ungrounded: Array<{ raw: string }> }): string[] => r.ungrounded.map((s) => s.raw)

const H_KO: BetaMessageParam[] = [
  { role: 'user', content: '입구 직경 60 mm, 출구 35 mm, 유량 7.07 L/s로 해 주세요. 3개 안을 비교해 주세요.' },
  { role: 'assistant', content: [{ type: 'tool_use', id: 'c1', name: 'cad_study_status', input: { study_id: 'v1' } }] },
  { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'c1', content: JSON.stringify({ d_in_m: 0.06, d_out_m: 0.035, q_m3s: 0.00707, dp_pa: 1234.5 }) }] },
]
const TABLE_KO = '| 항목 | 값 |\n|---|---|\n| 입구 직경 | 60 mm |\n| 출구 직경 | 35 mm |\n| 유량 | 7.07 L/s |\n| 압력 손실 | 1234.5 Pa |\n3개 안 중 첫 안입니다.'
const PLANT_KO = TABLE_KO + ' 출구 속도는 12.34 m/s입니다.'
const FIXED_KO = TABLE_KO + ' 출구 속도는 도구 결과에 없습니다.'
const H_EN: BetaMessageParam[] = [
  { role: 'user', content: 'Inlet diameter 60 mm, outlet 35 mm, flow 7.07 L/s; compare 3 designs.' },
  H_KO[1],
  H_KO[2],
]
const TABLE_EN = '| Item | Value |\n|---|---|\n| Inlet diameter | 60 mm |\n| Outlet diameter | 35 mm |\n| Flow | 7.07 L/s |\n| Pressure loss | 1234.5 Pa |\nThe first of 3 designs.'
const PLANT_EN = TABLE_EN + ' The outlet speed is 12.34 m/s.'
const FIXED_EN = TABLE_EN + ' The outlet speed is in no tool result.'

const displayText = (blocks: readonly Block[]): string => blocks.map((b) => (b.type === 'text' ? b.text : '')).join('')
const noticeOf = (r: GroundReplyResult) => {
  if (r.notice.kind !== 'notice') throw new Error('the reply ended without a notice block')
  return r.notice
}

async function guReply(calls: BetaMessageParam[][], history: readonly BetaMessageParam[], content: Block[], locale: 'ko' | 'en', plans: MockPlan[]) {
  const llm = spyLlm(planLlm('zai', 'glm-5.3-flash', plans), calls)
  return groundReply({ llm, tools: [], maxTokens: 1024, effort: 'high', signal: new AbortController().signal, history, content, turnId: locale === 'ko' ? 'gu-ko' : 'gu-en', locale, now: () => new Date('2026-10-04T00:00:00Z') })
}

describe('grounding units (GROUND-UNITS)', () => {
  it('U0 the unit table and the unit rule', () => {
    expect(Object.keys(UNIT_TO_SI).sort()).toEqual(['L/s', 'Pa', 'deg', 'kPa', 'km/h', 'l/s', 'm', 'm/s', 'm3/s', 'm^3/s', 'mm', 'rad', '°'].sort())
    expect(UNIT_REL).toBe(1e-6)
    const mm = extractNumbers('35 mm')[0]
    expect(mm.unit).toBe('mm')
    expect(mm.raw).toBe('35')
    expect(mm.si).toBeCloseTo(0.035, 15)
    expect(extractNumbers('0.00707 m³/s')[0].unit).toBe('m3/s')
    expect(extractNumbers('12.34 m/s입니다')[0].unit).toBe('m/s')
    expect(extractNumbers('5 m')[0].unit).toBe('m')
    expect(extractNumbers('25 °C')[0].unit).toBeNull()
    expect(extractNumbers('45%')[0]).toMatchObject({ unit: null, si: null })
    expect(extractNumbers('60 cells')[0].unit).toBeNull()
  })

  it('U1 mm and m', () => {
    expect(ung(lintText('입구 60 mm, 출구 35 mm', [0.06, 0.035]))).toEqual([])
    expect(ung(lintText('the outlet is 0.035 m', [], [0.035]))).toEqual([])
    expect(ung(lintText('35.1 mm and 36 mm', [0.035, 0.06]))).toEqual(['35.1', '36'])
    expect(ung(lintText('35 mm', [0.035 * (1 + 5e-7)]))).toEqual([])
    expect(ung(lintText('35 mm', [0.035 * (1 + 2e-6)]))).toEqual(['35'])
  })

  it('U2 L/s and m3/s', () => {
    expect(ung(lintText('유량 7.07 L/s', [0.00707]))).toEqual([])
    expect(ung(lintText('0.00707 m³/s and 0.00707 m3/s', [], [0.00707]))).toEqual([])
    expect(ung(lintText('7.1 L/s', [0.00707]))).toEqual(['7.1'])
  })

  it('U3 deg and rad', () => {
    expect(ung(lintText('15° and 15 deg', [Math.PI / 12]))).toEqual([])
    expect(ung(lintText('0.2617994 rad', [], [Math.PI / 12]))).toEqual([])
    expect(ung(lintText('15.5° and 0.2618 rad', [], [Math.PI / 12]))).toEqual(['15.5', '0.2618'])
  })

  it('U4 km/h and m/s', () => {
    expect(ung(lintText('90 km/h', [25]))).toEqual([])
    expect(ung(lintText('25 m/s', [], [90 * 1000 / 3600]))).toEqual([])
    expect(ung(lintText('91 km/h', [25]))).toEqual(['91'])
  })

  it('U5 kPa and Pa', () => {
    expect(ung(lintText('101.325 kPa', [101325]))).toEqual([])
    expect(ung(lintText('5000 Pa', [], [5000]))).toEqual([])
    expect(ung(lintText('101.3 kPa', [101325]))).toEqual(['101.3'])
  })

  it("U6 lintSession takes the user's words, only before the reply", () => {
    const g = lintSession([...H_KO, { role: 'assistant', content: [{ type: 'text', text: PLANT_KO }] }])
    const last = g.messages[g.messages.length - 1]
    expect(last.campaign).toBe(true)
    expect(last.final).toBe(true)
    expect(last.checked).toBe(6)
    expect(last.ungrounded.map((u) => u.raw)).toEqual(['12.34'])
    expect(g.ungrounded).toBe(1)
    const g2 = lintSession([
      { role: 'user', content: 'Status?' },
      { role: 'assistant', content: [{ type: 'text', text: 'It is 4321.5 Pa.' }] },
      { role: 'user', content: '4321.5 Pa is what I measured.' },
      { role: 'assistant', content: [{ type: 'text', text: 'Then 4321.5 Pa it is.' }] },
    ] as BetaMessageParam[])
    expect(g2.messages.map((m) => m.ungrounded.map((u) => u.raw))).toEqual([['4321.5'], []])
    const pools = groundingPools(H_KO)
    expect(pools.values).toEqual([60, 35, 7.07, 3, 0.06, 0.035, 0.00707, 1234.5])
    expect(pools.si).toHaveLength(3)
    expect(pools.si[0]).toBeCloseTo(0.06, 15)
    expect(pools.si[1]).toBeCloseTo(0.035, 15)
    expect(pools.si[2]).toBeCloseTo(0.00707, 15)
    expect(userTurnTexts(H_KO)).toHaveLength(1)
  })

  it('U7 locale ko through groundReply', async () => {
    const calls: BetaMessageParam[][] = []
    const a = await guReply(calls, H_KO, [{ type: 'text', text: TABLE_KO }], 'ko', [say(FIXED_KO)])
    expect(a).toBeNull()
    expect(calls).toHaveLength(0)
    const b = await guReply(calls, H_KO, [{ type: 'text', text: PLANT_KO }], 'ko', [say(FIXED_KO)])
    expect(calls).toHaveLength(1)
    expect(b).not.toBeNull()
    const req = calls[0]
    expect(req.at(-1)!.role).toBe('user')
    const prompt = textOf(req.at(-1)!)
    expect(prompt.startsWith('[grounding check] 이 답변은 아직 표시되지 않았습니다.')).toBe(true)
    expect(prompt).toContain('"12.34"')
    expect(prompt).toContain('한국어')
    expect(prompt).not.toContain('Your reply')
    expect(req.at(-2)).toEqual({ role: 'assistant', content: [{ type: 'text', text: PLANT_KO }] })
    expect(b!.record.fixed).toEqual(['12.34'])
    expect(b!.record.remaining).toEqual([])
    expect(b!.record.replaced).toBe(true)
    expect(b!.record.after.text).toBe(FIXED_KO)
    for (const s of ['60 mm', '35 mm', '7.07 L/s']) expect(b!.record.after.text).toContain(s)
    expect(noticeOf(b!).level).toBe('info')
    expect(noticeOf(b!).text.startsWith('표시하기 전에')).toBe(true)
    const callsC: BetaMessageParam[][] = []
    const c = await guReply(callsC, H_KO, [{ type: 'text', text: PLANT_KO }], 'ko', [say(PLANT_KO)])
    expect(callsC).toHaveLength(1)
    expect(c!.record.remaining).toEqual(['12.34'])
    expect(c!.record.replaced).toBe(false)
    expect(displayText(c!.display)).toContain('12.34 [?] m/s')
    expect(noticeOf(c!).level).toBe('warning')
    expect(noticeOf(c!).text.startsWith('이 답변의 숫자 1개는')).toBe(true)
  })

  it('U8 locale en through groundReply', async () => {
    const calls: BetaMessageParam[][] = []
    const a = await guReply(calls, H_EN, [{ type: 'text', text: TABLE_EN }], 'en', [say(FIXED_EN)])
    expect(a).toBeNull()
    expect(calls).toHaveLength(0)
    const b = await guReply(calls, H_EN, [{ type: 'text', text: PLANT_EN }], 'en', [say(FIXED_EN)])
    expect(calls).toHaveLength(1)
    expect(b).not.toBeNull()
    const prompt = textOf(calls[0].at(-1)!)
    expect(prompt.startsWith('[grounding check] Your reply has not been shown yet. It states 1 number(s)')).toBe(true)
    expect(prompt).toContain('"12.34"')
    expect(prompt).toContain('in the language it was written in')
    expect(b!.record.fixed).toEqual(['12.34'])
    expect(b!.record.remaining).toEqual([])
    expect(b!.record.replaced).toBe(true)
    expect(noticeOf(b!).level).toBe('info')
    expect(noticeOf(b!).text.startsWith('Checked against the tool results')).toBe(true)
    const callsC: BetaMessageParam[][] = []
    const c = await guReply(callsC, H_EN, [{ type: 'text', text: PLANT_EN }], 'en', [say(PLANT_EN)])
    expect(callsC).toHaveLength(1)
    expect(c!.record.remaining).toEqual(['12.34'])
    expect(displayText(c!.display)).toContain('12.34 [?] m/s')
    expect(noticeOf(c!).text.startsWith('1 number(s) in this reply')).toBe(true)
  })
})
