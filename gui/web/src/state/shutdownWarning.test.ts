// The delivery path on the browser side: the exact frame a going-down server
// sends lands in the output ring at level error, origin server, and carries no
// fatal effect. Pure state work - no DOM, no mocks.
import { describe, expect, it } from 'vitest'
import { ServerMsgSchema, activeTurnWarning, type ServerMsg } from '@cfd/shared'
import { applyEvent } from './applyEvent'
import { initialSessionData } from './types'

// The SAME function the server calls builds the frame; not a copied literal.
const line = activeTurnWarning({ sessionId: 's_1', turnId: 't_1' })
const frame: ServerMsg = { t: 'error', message: line, fatal: false }

describe('the shutdown warning reaches the session window', () => {
  it('is a schema-valid non-fatal error frame', () => {
    expect(frame).toEqual({ t: 'error', message: line, fatal: false })
    expect(ServerMsgSchema.safeParse(frame).success).toBe(true)
  })

  it('lands last in the output ring at level error, origin server, with no effects', () => {
    const { state, effects } = applyEvent(initialSessionData(), frame, 1_700_000_000_000)
    expect(state.outputs.at(-1)).toMatchObject({ level: 'error', text: line, origin: 'server', ts: 1_700_000_000_000 })
    expect(state.serverErrors.at(-1)).toBe(line)
    expect(effects).toEqual([])
  })
})
