import { describe, expect, it } from 'vitest'
import { ClientMsgSchema, ServerMsgSchema, type ClientMsg, type ServerMsg, type UiState } from './protocol.js'

// A wire frame must survive JSON.stringify -> parse -> zod parse unchanged:
// that is exactly the path every message takes in the browser and the server.
function roundTripClient(msg: ClientMsg): ClientMsg {
  return ClientMsgSchema.parse(JSON.parse(JSON.stringify(msg)))
}
function roundTripServer(msg: ServerMsg): ServerMsg {
  return ServerMsgSchema.parse(JSON.parse(JSON.stringify(msg)))
}

const fullState: UiState = {
  activeTab: 'velocity',
  activeStep: 'mesh',
  rightTab: 'Properties',
  tool: 'select',
  frame: 12,
  projection: 'Perspective',
  showAxes: true,
  showColorBars: false,
  selection: { kind: 'cell', id: 4311, center: [0.5, 1.5, 2.5] },
  runId: 'r_1',
  sim: { status: 'running', iteration: 240, maxIterations: 4000 },
}

describe('ui bridge frames', () => {
  it('round-trips ui.command with every command type', () => {
    const cmds: ServerMsg[] = [
      { t: 'ui.command', requestId: 'u_1', cmd: { type: 'select_tab', tab: 'velocity' } },
      { t: 'ui.command', requestId: 'u_2', cmd: { type: 'show_field', field: 'Velocity' } },
      { t: 'ui.command', requestId: 'u_3', cmd: { type: 'select_step', step: 'mesh' } },
      { t: 'ui.command', requestId: 'u_4', cmd: { type: 'open_panel', panel: 'AI Assistant' } },
      { t: 'ui.command', requestId: 'u_5', cmd: { type: 'set_tool', tool: 'box' } },
      { t: 'ui.command', requestId: 'u_6', cmd: { type: 'set_projection', projection: 'Orthographic' } },
      { t: 'ui.command', requestId: 'u_7', cmd: { type: 'fit_view' } },
      { t: 'ui.command', requestId: 'u_8', cmd: { type: 'show_overlay', what: 'axes', on: false } },
      { t: 'ui.command', requestId: 'u_9', cmd: { type: 'set_centerline', quantity: 'Temperature' } },
      { t: 'ui.command', requestId: 'u_a', cmd: { type: 'run', action: 'stop' } },
      { t: 'ui.command', requestId: 'u_b', cmd: { type: 'notify', level: 'warning', text: 'mesh is coarse' } },
    ]
    for (const m of cmds) expect(roundTripServer(m)).toEqual(m)
  })

  it('round-trips ui.state with a selection and with everything unknown', () => {
    const withCell: ClientMsg = { t: 'ui.state', state: fullState }
    expect(roundTripClient(withCell)).toEqual(withCell)
    const bare: ClientMsg = {
      t: 'ui.state',
      state: { activeTab: null, activeStep: null, rightTab: null, tool: null, frame: null, projection: null, showAxes: null, showColorBars: null, selection: { kind: 'none' }, runId: null, sim: null },
    }
    expect(roundTripClient(bare)).toEqual(bare)
  })

  it('round-trips ui.result and host', () => {
    const result: ClientMsg = { t: 'ui.result', requestId: 'u_1', ok: false, error: 'no such tab' }
    expect(roundTripClient(result)).toEqual(result)
    const okResult: ClientMsg = { t: 'ui.result', requestId: 'u_1', ok: true, error: null }
    expect(roundTripClient(okResult)).toEqual(okResult)
    const host: ServerMsg = { t: 'host', host: { cpu: 12.5, memUsedGb: 9.2, memTotalGb: 32, ts: 1_760_000_000_000 } }
    expect(roundTripServer(host)).toEqual(host)
  })

  it('rejects unknown ui command types and unknown fields', () => {
    expect(ClientMsgSchema.safeParse({ t: 'ui.state', state: { activeTab: 'x' } }).success).toBe(false)
    expect(ClientMsgSchema.safeParse({ t: 'ui.result', requestId: 'u_1', ok: true }).success).toBe(false)
    expect(ServerMsgSchema.safeParse({ t: 'ui.command', requestId: 'u_1', cmd: { type: 'reboot' } }).success).toBe(false)
    expect(ServerMsgSchema.safeParse({ t: 'ui.command', requestId: 'u_1', cmd: { type: 'show_field', field: 'Salinity' } }).success).toBe(false)
  })
})
