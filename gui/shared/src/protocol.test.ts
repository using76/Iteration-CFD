import { describe, expect, it } from 'vitest'
import { CHAT_TIMEOUT_MAX_MS, ChatRequestSchema, ClientMsgSchema, REST, RunInfoSchema, ServerMsgSchema, type ClientMsg, type ServerMsg, type UiState } from './protocol.js'

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

  it('round-trips the six geometry commands', () => {
    const cmds: ServerMsg[] = [
      { t: 'ui.command', requestId: 'g_1', cmd: { type: 'geometry_open', path: 'cases/x.stl' } },
      { t: 'ui.command', requestId: 'g_2', cmd: { type: 'geometry_import_step', path: 'cases/x.step' } },
      { t: 'ui.command', requestId: 'g_3', cmd: { type: 'geometry_part', name: 'a', action: 'rename', newName: 'body' } },
      { t: 'ui.command', requestId: 'g_4', cmd: { type: 'geometry_transform', op: 'translate', value: [1, 2, 3], pivot: 'centre' } },
      { t: 'ui.command', requestId: 'g_5', cmd: { type: 'geometry_boolean', op: 'cut', a: 'a', b: 'b' } },
      { t: 'ui.command', requestId: 'g_6', cmd: { type: 'geometry_save', path: 'tmp/y.stl', binary: true, keepVisibleOnly: true, overwrite: true } },
    ]
    for (const m of cmds) expect(roundTripServer(m)).toEqual(m)
    const state: ClientMsg = {
      t: 'ui.state',
      state: {
        ...fullState,
        geometry: { id: 'g1', path: 'cases/x.stl', triangleCount: 8, closed: true, openEdges: 0, solids: [{ name: 'a', triangles: 4, visible: true }, { name: 'b', triangles: 4, visible: false }], selected: 'b', edits: 2, dirty: true },
      },
    }
    expect(roundTripClient(state)).toEqual(state)
  })

  it('rejects unknown ui command types and unknown fields', () => {
    expect(ClientMsgSchema.safeParse({ t: 'ui.state', state: { activeTab: 'x' } }).success).toBe(false)
    expect(ClientMsgSchema.safeParse({ t: 'ui.result', requestId: 'u_1', ok: true }).success).toBe(false)
    expect(ServerMsgSchema.safeParse({ t: 'ui.command', requestId: 'u_1', cmd: { type: 'reboot' } }).success).toBe(false)
    expect(ServerMsgSchema.safeParse({ t: 'ui.command', requestId: 'u_1', cmd: { type: 'show_field', field: 'Salinity' } }).success).toBe(false)
  })
})

// The 24 keys every run.json on disk carries, in their on-disk order (the
// literal at runs/manager.ts).
const legacyRunInfo = {
  id: 'r_1',
  binary: 'ofgpu-k-epsilon',
  argv: ['ofgpu-k-epsilon', 'cases/plume.jsonc'],
  cwd: '',
  casePath: 'cases/plume.jsonc',
  outputRoot: 'cases/plume_jsonc',
  status: 'done',
  pid: null,
  startedAt: '2026-09-15T00:00:00.000Z',
  endedAt: '2026-09-15T00:01:00.000Z',
  exitCode: 0,
  signal: null,
  iter: 10,
  targetIter: 100,
  time: null,
  endTime: null,
  lastResidual: null,
  written: [],
  error: null,
  converged: false,
  device: '',
  logLines: 0,
  mode: 'demo',
  label: null,
}

describe('run info provenance', () => {
  it('a legacy 24-key record still parses, and the five provenance keys survive a round trip', () => {
    const legacy = RunInfoSchema.parse(legacyRunInfo)
    expect(legacy.gitSha).toBeUndefined()
    expect(legacy.id).toBe('r_1')
    const full = RunInfoSchema.parse(
      JSON.parse(
        JSON.stringify({
          ...legacyRunInfo,
          gitSha: 'a'.repeat(40),
          gitDirty: false,
          caseId: 'cases/plume.jsonc',
          meshId: 'cases/box/constant/polyMesh/.meshSummary.json#box',
          machine: { hostname: 'H', gpu: '', platform: 'win32' },
        }),
      ),
    )
    expect(full.gitSha).toBe('a'.repeat(40))
    expect(full.gitDirty).toBe(false)
    expect(full.caseId).toBe('cases/plume.jsonc')
    expect(full.meshId).toBe('cases/box/constant/polyMesh/.meshSummary.json#box')
    expect(full.machine).toEqual({ hostname: 'H', gpu: '', platform: 'win32' })
  })
  it('refuses a machine that is a bare string, because N1 declares a struct', () => {
    expect(RunInfoSchema.safeParse({ ...legacyRunInfo, machine: 'H' }).success).toBe(false)
  })
})

describe('chat REST schemas', () => {
  it('ChatRequestSchema fills its defaults and bounds timeoutMs', () => {
    expect(ChatRequestSchema.parse({ text: 'hi' })).toEqual({
      sessionId: null,
      text: 'hi',
      attachments: [],
      attachmentIds: [],
      activeFile: null,
      autoApprove: null,
      locale: null,
      timeoutMs: null,
    })
    expect(ChatRequestSchema.safeParse({ text: 'hi', timeoutMs: CHAT_TIMEOUT_MAX_MS + 1 }).success).toBe(false)
    expect(ChatRequestSchema.safeParse({}).success).toBe(false)
    expect(REST.chat).toBe('/api/chat')
  })
})
