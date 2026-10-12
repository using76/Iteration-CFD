// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// STU-03: set_post, show_metric, set_log_filter and show_chart surface drive
// the REAL useUiStore through the bridge, and every sent frame stays schema-valid.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ClientMsgSchema, UiCommandSchema, type ClientMsg, type UiCommand } from '@cfd/shared'
import { tabScreenLabel, useUiStore } from '../state/uiStore'
import { createUiBridge } from './uiBridge'

const executed: unknown[] = []
const fake = { fieldNull: false, failSetField: false }
const fakeApi = {
  execute: async (cmd: unknown) => {
    executed.push(cmd)
    if (fake.failSetField && (cmd as { type: string }).type === 'setField')
      return { ok: false, state: null, error: { code: 'INVALID', message: 'range max must be greater than min' }, image: null }
    return { ok: true, state: null, error: null, image: null }
  },
  getState: () => ({
    camera: { position: [10, 10, 10], target: [0, 0, 0], projection: 'perspective' },
    datasetId: 'ds1',
    datasetName: 'channel',
    field: fake.fieldNull ? null : 'U',
    time: null,
    colormap: 'viridis',
    range: null,
    representation: 'surface',
  }),
  subscribe: () => () => {},
  isMounted: () => true,
  setBaseUrl: () => {},
  setTool: () => {},
  getTool: () => 'select' as const,
  viewport: () => ({ left: 0, top: 0, width: 800, height: 600 }),
  probePixel: () => ({ ok: true, hit: { cell: 0, value: 0, field: null, center: null } }),
  probePoint: () => ({ ok: false, code: 'NO_DATASET', message: 'no dataset' }),
}
vi.mock('../viewer', () => ({ getViewerApi: () => fakeApi, setViewerApi: () => {} }))
vi.mock('./actions', () => ({ actions: { setSettings: () => true, openSession: () => true }, activeCasePath: () => null }))
vi.mock('../api/rest', () => ({ api: { geometrySave: vi.fn(), geometryEdit: vi.fn() }, ApiError: class extends Error {} }))

const session = {
  problems: {},
  connection: 'online',
  addNote: vi.fn(),
  runs: {} as Record<string, unknown>,
  runData: {
    r1: { residuals: [], logs: [], metrics: [{ seq: 1, iter: 1, time: null, metrics: { 'T[max]': 300, 'T[min]': 290, dt: 0.1 }, raw: '' }] },
    r2: {
      residuals: [],
      logs: [],
      metrics: [
        { seq: 1, iter: 1, time: null, metrics: { 'fan_a.Q': 0.5, 'fan_a.dp': 20, Tmax: 301 }, raw: '' },
        { seq: 2, iter: 2, time: null, metrics: { 'fan_b.Q': 0.4 }, raw: '' },
      ],
    },
  },
}

let sent: ClientMsg[] = []
let n = 0
const subscribeRun = vi.fn()

function makeBridge() {
  return createUiBridge({
    send: (m) => {
      sent.push(m)
      return true
    },
    ui: useUiStore,
    session: { getState: () => session as never },
    subscribeRun,
    saveImage: vi.fn(),
  })
}

beforeEach(() => {
  sent = []
  n = 0
  executed.length = 0
  fake.fieldNull = false
  fake.failSetField = false
  subscribeRun.mockClear()
  useUiStore.setState({
    tabs: [],
    activeTabId: null,
    activeRunId: null,
    bottomVisible: false,
    bottomTab: 'terminal',
    logFilter: { text: '', streams: ['stdout', 'stderr', 'system'], follow: true },
    metricSlots: [null, null],
    metricMode: 'metrics',
  })
})

afterEach(() => {
  for (const m of sent) expect(ClientMsgSchema.safeParse(m).success).toBe(true)
})

async function drive(bridge: ReturnType<typeof createUiBridge>, command: UiCommand) {
  const id = `r${++n}`
  await bridge.handleCommand(id, command)
  const r = sent.find((m): m is Extract<ClientMsg, { t: 'ui.result' }> => m.t === 'ui.result' && m.requestId === id)
  expect(r).toBeTruthy()
  return r!
}

describe('set_post (STU-03)', () => {
  it('P1: the schema coerces strings and keeps the whole patch', () => {
    expect(UiCommandSchema.parse({ type: 'set_post', range: '[-2, 1]', opacity: '0.4', colormap: 'turbo', log: 'true', patches: 'all' })).toEqual({
      type: 'set_post',
      range: [-2, 1],
      opacity: 0.4,
      colormap: 'turbo',
      log: true,
      patches: 'all',
    })
    expect((UiCommandSchema.parse({ type: 'set_post', range: ['-2', '1'] }) as { range: number[] }).range).toEqual([-2, 1])
  })

  it('P2: one call drives setField then setRepresentation and opens the viewer tab', async () => {
    const bridge = makeBridge()
    const cmd = UiCommandSchema.parse({ type: 'set_post', range: '[-2, 1]', opacity: '0.4', colormap: 'turbo', log: 'true', patches: 'all' }) as UiCommand
    const r = await drive(bridge, cmd)
    expect(r.ok).toBe(true)
    expect(executed).toEqual([
      { type: 'setField', field: 'U', component: null, range: [-2, 1], colormap: 'turbo', log: true },
      { type: 'setRepresentation', mode: 'surface', opacity: 0.4, patches: 'all' },
    ])
    expect(useUiStore.getState().activeTabId).toBe('viewer')
  })

  it('P3: representation-only touches setRepresentation alone', async () => {
    const bridge = makeBridge()
    const r = await drive(bridge, { type: 'set_post', representation: 'wireframe' })
    expect(r.ok).toBe(true)
    expect(executed).toEqual([{ type: 'setRepresentation', mode: 'wireframe', opacity: null, patches: null }])
  })

  it('P4: colormap-only touches setField alone', async () => {
    const bridge = makeBridge()
    const r = await drive(bridge, { type: 'set_post', colormap: 'jet' })
    expect(r.ok).toBe(true)
    expect(executed).toEqual([{ type: 'setField', field: 'U', component: null, range: null, colormap: 'jet', log: null }])
  })

  it('P5/P6: nothing to change and no coloured field refuse', async () => {
    const bridge = makeBridge()
    const r = await drive(bridge, { type: 'set_post' })
    expect(r.ok).toBe(false)
    expect(r.error).toBe('INVALID (set_post): nothing to change; give colormap, range, component, representation, opacity, patches or log')
    expect(executed).toEqual([])
    fake.fieldNull = true
    const r2 = await drive(bridge, { type: 'set_post', colormap: 'jet' })
    expect(r2.ok).toBe(false)
    expect(r2.error).toBe('NO_FIELD (set_post): no field is coloured; open_result or post_field first')
    expect(executed).toEqual([])
  })

  it('P7: a viewer refusal stops after the failing command', async () => {
    fake.failSetField = true
    const bridge = makeBridge()
    const r = await drive(bridge, { type: 'set_post', range: [1, -2], opacity: 0.5 })
    expect(r.ok).toBe(false)
    expect(r.error).toBe('range max must be greater than min')
    expect(executed).toHaveLength(1)
  })
})

describe('show_metric (STU-03)', () => {
  it('M1: punctuation-blind pick lands on slot 1 and the state reports it', async () => {
    useUiStore.setState({ activeRunId: 'r1' })
    const bridge = makeBridge()
    const r = await drive(bridge, { type: 'show_metric', metric: 'Tmax' })
    expect(r.ok).toBe(true)
    expect(useUiStore.getState().metricSlots).toEqual(['T[max]', null])
    const tab = useUiStore.getState().tabs.find((t) => t.kind === 'residuals')
    expect(tab).toMatchObject({ kind: 'residuals', chart: 'metrics', runId: 'r1' })
    expect(r.state?.metricPick).toEqual({ left: 'T[max]', right: null, mode: 'metrics' })
  })

  it('M2: slot 2 adds the right axis pick; the schema coerces slot "2"', async () => {
    useUiStore.setState({ activeRunId: 'r1' })
    const bridge = makeBridge()
    await drive(bridge, { type: 'show_metric', metric: 'Tmax' })
    const r = await drive(bridge, { type: 'show_metric', metric: 'DT', slot: 2 })
    expect(r.ok).toBe(true)
    expect(useUiStore.getState().metricSlots).toEqual(['T[max]', 'dt'])
    expect((UiCommandSchema.parse({ type: 'show_metric', metric: 'DT', slot: '2' }) as { slot: number }).slot).toBe(2)
  })

  it('M3: an unknown metric refuses with the run\'s reported list', async () => {
    useUiStore.setState({ activeRunId: 'r1' })
    const bridge = makeBridge()
    const r = await drive(bridge, { type: 'show_metric', metric: 'Umax' })
    expect(r.ok).toBe(false)
    expect(r.error).toBe('NO_METRIC (show_metric): run "r1" reports no metric "Umax"; reported: T[max], T[min], dt')
  })

  it('M4: slot 3 refuses before anything is picked', async () => {
    useUiStore.setState({ activeRunId: 'r1' })
    const bridge = makeBridge()
    const r = await drive(bridge, { type: 'show_metric', metric: 'Tmax', slot: 3 })
    expect(r.ok).toBe(false)
    expect(r.error).toBe('INVALID (show_metric): slot must be 1 (left) or 2 (right), got 3')
  })

  it('M5: mode sweeps switches the chart; a metric with sweeps refuses', async () => {
    useUiStore.setState({ activeRunId: 'r1' })
    const bridge = makeBridge()
    const r = await drive(bridge, { type: 'show_metric', mode: 'sweeps' })
    expect(r.ok).toBe(true)
    expect(useUiStore.getState().metricMode).toBe('sweeps')
    expect(r.state?.metricPick).toMatchObject({ mode: 'sweeps' })
    const r2 = await drive(bridge, { type: 'show_metric', metric: 'Tmax', mode: 'sweeps' })
    expect(r2.ok).toBe(false)
    expect(r2.error).toBe('INVALID (show_metric): a metric pick applies to mode "metrics"; drop the metric or send mode "metrics"')
  })

  it('M6/M7: no run and no metric both refuse', async () => {
    const bridge = makeBridge()
    const r = await drive(bridge, { type: 'show_metric', metric: 'Tmax' })
    expect(r.ok).toBe(false)
    expect(r.error).toBe('NO_RUN (show_metric): no run is followed; follow_run first')
    const r2 = await drive(bridge, { type: 'show_metric' })
    expect(r2.ok).toBe(false)
    expect(r2.error).toBe('INVALID (show_metric): needs a metric or a mode')
  })
})

describe('set_log_filter (STU-03)', () => {
  it('L1/L2: one patch applies whole, a null text clears it', async () => {
    const bridge = makeBridge()
    const r = await drive(bridge, { type: 'set_log_filter', text: 'error', streams: ['stderr'], follow: false })
    expect(r.ok).toBe(true)
    expect(r.state?.logs).toEqual({ text: 'error', streams: ['stderr'], follow: false })
    expect(r.state?.bottomTab).toBe('logs')
    expect(useUiStore.getState().bottomTab).toBe('logs')
    const r2 = await drive(bridge, { type: 'set_log_filter', text: null })
    expect(r2.ok).toBe(true)
    expect(r2.state?.logs).toEqual({ text: '', streams: ['stderr'], follow: false })
  })

  it('L3: stream lists land de-duplicated in canonical order', async () => {
    const bridge = makeBridge()
    const r = await drive(bridge, { type: 'set_log_filter', streams: ['system', 'stdout', 'system'] })
    expect(r.ok).toBe(true)
    expect(r.state?.logs).toEqual({ text: '', streams: ['stdout', 'system'], follow: true })
  })

  it('L4: empty streams and a bare command refuse; Boolish coerces follow', async () => {
    const bridge = makeBridge()
    const r = await drive(bridge, { type: 'set_log_filter', streams: [] })
    expect(r.ok).toBe(false)
    expect(r.error).toBe('INVALID (set_log_filter): streams must name at least one of stdout, stderr, system')
    const r2 = await drive(bridge, { type: 'set_log_filter' })
    expect(r2.ok).toBe(false)
    expect(r2.error).toBe('INVALID (set_log_filter): nothing to change; give text, streams or follow')
    expect((UiCommandSchema.parse({ type: 'set_log_filter', follow: 'false' }) as { follow: boolean }).follow).toBe(false)
  })
})

describe('show_chart surface (STU-03)', () => {
  it('S1: a run with fan metrics opens the surface chart', async () => {
    const bridge = makeBridge()
    const r = await drive(bridge, { type: 'show_chart', chart: 'surface', runId: 'r2' })
    expect(r.ok).toBe(true)
    expect(subscribeRun).toHaveBeenCalledWith('r2')
    const tab = useUiStore.getState().tabs.find((t) => t.kind === 'residuals')
    expect(tab).toMatchObject({ kind: 'residuals', chart: 'surface', runId: 'r2' })
    expect(r.state?.tabs).toContainEqual({ id: 'residuals', kind: 'residuals', label: 'surface' })
  })

  it('S2: a run without surface series refuses with its metric list', async () => {
    const bridge = makeBridge()
    const r = await drive(bridge, { type: 'show_chart', chart: 'surface', runId: 'r1' })
    expect(r.ok).toBe(false)
    expect(r.error).toBe('NO_SURFACE_METRICS (show_chart): run "r1" reports no per-patch surface-integral series (<patch>.<quantity>, e.g. fan.Q); its metrics: T[max], T[min], dt')
  })

  it('S3: no runId and no followed run refuses', async () => {
    const bridge = makeBridge()
    const r = await drive(bridge, { type: 'show_chart', chart: 'surface' })
    expect(r.ok).toBe(false)
    expect(r.error).toBe('NO_SURFACE_METRICS (show_chart): no run is followed; follow_run or pass runId')
  })
})

describe('tab label (STU-03)', () => {
  it('U1: a surface tab reads Surface integrals / 표면 적분', () => {
    const tab = { id: 'residuals' as const, kind: 'residuals' as const, runId: null, chart: 'surface' as const }
    expect(tabScreenLabel(tab, 'en')).toBe('Surface integrals')
    expect(tabScreenLabel(tab, 'ko')).toBe('표면 적분')
  })
})

