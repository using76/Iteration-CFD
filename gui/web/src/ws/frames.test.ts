import { describe, expect, it } from 'vitest'
import type { ClientMsg } from '@cfd/shared'
import { ALL_FRAME_TYPES, FRAMES } from '../state/fixtures'
import { BACKOFF_MAX_MS, BACKOFF_MIN_MS, backoffDelay, encodeClientFrame, parseServerFrame, wsUrlFor } from './frames'

describe('parseServerFrame', () => {
  it('round-trips every recorded server frame', () => {
    for (const t of ALL_FRAME_TYPES) {
      const parsed = parseServerFrame(JSON.stringify(FRAMES[t]), () => {})
      expect(parsed).toEqual(FRAMES[t])
    }
  })

  it('drops invalid JSON, non-text frames and unknown or malformed messages with a warning', () => {
    const warnings: string[] = []
    const warn = (m: string) => warnings.push(m)
    expect(parseServerFrame('{not json', warn)).toBeNull()
    expect(parseServerFrame(new ArrayBuffer(4), warn)).toBeNull()
    expect(parseServerFrame(JSON.stringify({ t: 'nope' }), warn)).toBeNull()
    expect(parseServerFrame(JSON.stringify({ t: 'gpu', gpu: { state: 'weird' } }), warn)).toBeNull()
    expect(parseServerFrame(JSON.stringify({ t: 'run.log', runId: 'r', lines: [{ seq: 'x' }] }), warn)).toBeNull()
    expect(warnings).toHaveLength(5)
    expect(warnings[3]).toContain('gpu')
  })

  it('rejects frames that merely look right (extra discriminator, missing fields)', () => {
    expect(parseServerFrame(JSON.stringify({ t: 'hello' }), () => {})).toBeNull()
    expect(parseServerFrame(JSON.stringify({ t: 'msg.delta', sessionId: 's', messageId: 'm', blockIndex: 0 }), () => {})).toBeNull()
  })
})

describe('encodeClientFrame', () => {
  it('serialises valid client messages', () => {
    const msgs: ClientMsg[] = [
      { t: 'ping', ts: 1 },
      { t: 'session.open', sessionId: null },
      { t: 'user.message', sessionId: 's', text: 'hi', context: { activeFile: null, activeRun: null, attachments: [], selection: null } },
      { t: 'run.subscribe', runId: 'r', fromSeq: 12 },
      { t: 'viewer.result', requestId: 'q', result: { ok: false, state: null, error: { code: 'NO_VIEWER', message: 'x' }, image: null } },
      { t: 'quick', sessionId: 's', action: 'mesh', casePath: 'cases/plume.jsonc', runId: null },
      { t: 'settings.set', sessionId: 's', patch: { effort: 'max' } },
    ]
    for (const m of msgs) {
      const s = encodeClientFrame(m, () => {})
      expect(s).not.toBeNull()
      expect(JSON.parse(s!)).toEqual(m)
    }
  })

  it('refuses malformed client messages', () => {
    const warnings: string[] = []
    expect(encodeClientFrame({ t: 'run.subscribe', runId: 'r' } as unknown as ClientMsg, (m) => warnings.push(m))).toBeNull()
    expect(encodeClientFrame({ t: 'tool.approve', sessionId: 's', toolUseIds: ['a'], remember: 'forever' } as unknown as ClientMsg, (m) => warnings.push(m))).toBeNull()
    expect(warnings).toHaveLength(2)
    expect(warnings[0]).toContain('run.subscribe')
  })
})

describe('backoffDelay', () => {
  it('grows from 0.5 s to a 10 s cap with bounded jitter', () => {
    expect(backoffDelay(0, 0.5)).toBe(BACKOFF_MIN_MS)
    expect(backoffDelay(1, 0.5)).toBe(1000)
    expect(backoffDelay(4, 0.5)).toBe(8000)
    expect(backoffDelay(10, 0.5)).toBe(BACKOFF_MAX_MS)
    for (let a = 0; a < 12; a++) {
      for (const r of [0, 0.25, 0.999]) {
        const d = backoffDelay(a, r)
        expect(d).toBeGreaterThanOrEqual(BACKOFF_MIN_MS * 0.75)
        expect(d).toBeLessThanOrEqual(BACKOFF_MAX_MS * 1.25)
      }
    }
    expect(backoffDelay(3, 0)).toBeLessThan(backoffDelay(3, 0.999))
  })
})

describe('wsUrlFor', () => {
  it('uses the page origin and switches to wss on https', () => {
    expect(wsUrlFor({ protocol: 'http:', host: '127.0.0.1:5173' })).toBe('ws://127.0.0.1:5173/ws')
    expect(wsUrlFor({ protocol: 'https:', host: 'cfd.example' })).toBe('wss://cfd.example/ws')
  })
})
