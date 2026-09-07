// Wire-level helpers kept free of browser APIs so they can be unit-tested.
import { ClientMsgSchema, ServerMsgSchema, type ClientMsg, type ServerMsg } from '@cfd/shared'

export interface FrameWarn {
  (message: string): void
}

/** Parse one incoming frame; invalid JSON or schema failures return null after a warning. */
export function parseServerFrame(raw: unknown, warn: FrameWarn = (m) => console.warn(m)): ServerMsg | null {
  if (typeof raw !== 'string') {
    warn('ws: dropped non-text frame')
    return null
  }
  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    warn(`ws: dropped invalid JSON frame (${raw.slice(0, 80)})`)
    return null
  }
  const r = ServerMsgSchema.safeParse(parsed)
  if (!r.success) {
    const t = parsed && typeof parsed === 'object' ? (parsed as { t?: unknown }).t : undefined
    warn(`ws: dropped invalid ${typeof t === 'string' ? t : 'frame'}: ${r.error.issues.map((i) => `${i.path.join('.')}: ${i.message}`).join('; ')}`)
    return null
  }
  return r.data
}

/** Validate and serialise an outgoing frame; returns null (after a warning) when it does not match the schema. */
export function encodeClientFrame(msg: ClientMsg, warn: FrameWarn = (m) => console.warn(m)): string | null {
  const r = ClientMsgSchema.safeParse(msg)
  if (!r.success) {
    warn(`ws: refused to send invalid ${String((msg as { t?: unknown }).t)}: ${r.error.issues.map((i) => `${i.path.join('.')}: ${i.message}`).join('; ')}`)
    return null
  }
  return JSON.stringify(r.data)
}

export const BACKOFF_MIN_MS = 500
export const BACKOFF_MAX_MS = 10_000

/** Exponential backoff 0.5 s -> 10 s with +-25 % jitter; `random` in [0,1). */
export function backoffDelay(attempt: number, random: number = Math.random()): number {
  const base = Math.min(BACKOFF_MAX_MS, BACKOFF_MIN_MS * 2 ** Math.max(0, attempt))
  const jitter = (random - 0.5) * 0.5 * base
  return Math.round(Math.max(BACKOFF_MIN_MS * 0.75, Math.min(BACKOFF_MAX_MS * 1.25, base + jitter)))
}

/** Same-origin WebSocket URL for the /ws endpoint. */
export function wsUrlFor(location: { protocol: string; host: string }, path = '/ws'): string {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${location.host}${path}`
}
