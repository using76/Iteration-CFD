// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// STU-09: set_centerline reaches this window with the line the server resolved
// (server/src/tools/guiCenterline.test.ts holds the samples against the analytic
// fixture), and the bridge opens the one centreline tab on exactly that line;
// the chart asks GET /api/results/sample for the same keys and draws value
// against distance. run_custom_tool never reaches the window: the server runs
// it as custom_tool_run.
import { renderToStaticMarkup } from 'react-dom/server'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { ClientMsg } from '@cfd/shared'
import { tx } from '../i18n/extra'
import { useUiStore, type CenterlineLine } from '../state/uiStore'
import { createUiBridge } from '../ws/uiBridge'
import { CenterlineChartView, polylineRuns, valueRange } from './CenterlineChart'

const fakeViewer = {
  execute: async () => ({ ok: true, state: null, error: null, image: null }),
  getState: () => ({ camera: { position: [1, 1, 1], target: [0, 0, 0], projection: 'perspective' }, datasetId: null, datasetName: null, field: null, time: null, colormap: 'viridis', range: null, representation: 'surface' }),
  subscribe: () => () => {},
  isMounted: () => false,
  setBaseUrl: () => {},
  setTool: () => {},
  getTool: () => 'select' as const,
  viewport: () => null,
  probePixel: () => ({ ok: false, code: 'MISS', message: 'centreline test' }),
  probePoint: () => ({ ok: false, code: 'NO_DATASET', message: 'centreline test' }),
}
vi.mock('../viewer', () => ({ getViewerApi: () => fakeViewer, setViewerApi: () => {} }))
vi.mock('../ws/actions', () => ({ actions: { setSettings: () => true, openSession: () => true }, activeCasePath: () => null }))
vi.mock('../api/rest', () => ({ api: { geometrySave: vi.fn(), geometryEdit: vi.fn(), resultsSample: vi.fn() }, ApiError: class extends Error {} }))

let sent: ClientMsg[] = []
function bridge() {
  return createUiBridge({
    send: (m) => {
      sent.push(m)
      return true
    },
    ui: useUiStore,
    session: { getState: () => ({ problems: {}, connection: 'online', addNote: vi.fn(), runs: {} }) as never },
    subscribeRun: vi.fn(),
    saveImage: vi.fn(),
  })
}
const resultOf = (id: string) => sent.find((m): m is Extract<ClientMsg, { t: 'ui.result' }> => m.t === 'ui.result' && m.requestId === id)

/** The line the server forwards for the fixture's default centreline (guiCenterline.test.ts). */
const LINE: CenterlineLine = { path: 'cases/fixture/fixture_jsonc', field: 'U', component: 'magnitude', time: '0', p0: [0, 1, 0.5], p1: [4, 1, 0.5] }
const t = (key: Parameters<typeof tx>[1], vars?: Record<string, string | number>) => tx('en', key, vars)

beforeEach(() => {
  sent = []
  useUiStore.setState({ tabs: [], activeTabId: null })
})

describe('the centreline chart', () => {
  it('set_centerline opens the one centreline tab on the line the server resolved', async () => {
    const b = bridge()
    await b.handleCommand('c1', { type: 'set_centerline', quantity: 'U', component: 'magnitude', p0: [0, 1, 0.5], p1: [4, 1, 0.5], path: LINE.path, time: '0' })
    expect(resultOf('c1')).toMatchObject({ ok: true, error: null })
    const s = useUiStore.getState()
    expect(s.activeTabId).toBe('centerline')
    expect(s.tabs).toEqual([{ id: 'centerline', kind: 'centerline', line: LINE }])
    // A second line re-aims the same tab.
    await b.handleCommand('c2', { type: 'set_centerline', quantity: 'p', component: 'magnitude', p0: [0, 0, 0], p1: [1, 1, 1], path: LINE.path, time: '100' })
    expect(useUiStore.getState().tabs).toHaveLength(1)
    expect(useUiStore.getState().tabs[0]).toMatchObject({ line: { field: 'p', time: '100', p1: [1, 1, 1] } })
  })

  it('a set_centerline without the resolved line, and run_custom_tool, are named refusals', async () => {
    const b = bridge()
    await b.handleCommand('n1', { type: 'set_centerline', quantity: 'p' })
    expect(resultOf('n1')?.ok).toBe(false)
    expect(resultOf('n1')?.error).toMatch(/^NO_LINE \(set_centerline\): /)
    await b.handleCommand('n2', { type: 'run_custom_tool', name: 'field_stats' })
    expect(resultOf('n2')?.error).toBe('UNSUPPORTED (run_custom_tool): gui_control runs this on the server as custom_tool_run; this window should not receive it')
    expect(useUiStore.getState().tabs).toEqual([])
  })

  it('draws value against distance, breaks the line at a NaN, and says what it is waiting for', () => {
    expect(valueRange([3, 1, 2])).toEqual([1, 3])
    expect(valueRange([2, 2])).toEqual([1.9, 2.1])
    expect(valueRange([Number.NaN])).toBeNull()
    const runs = polylineRuns([0, 1, 2, 3], [0, 1, Number.NaN, 1], [0, 1])
    expect(runs).toEqual(['64.00,284.00 250.67,12.00', '624.00,12.00'])
    const data = { s: [0, 0.5, 1], x: [0, 2, 4], values: [0.25, 2, 4], field: 'U', time: '0', cells: 64 }
    const html = renderToStaticMarkup(<CenterlineChartView line={LINE} data={data} error={null} t={t} />)
    expect(html).toContain('Centreline: U')
    expect(html).toContain('U (magnitude), time 0, 64 cells')
    expect(html).toContain('Line (0, 1, 0.5) → (4, 1, 0.5)')
    expect(html.match(/<polyline/g)).toHaveLength(1)
    expect(renderToStaticMarkup(<CenterlineChartView line={LINE} data={null} error={null} t={t} />)).toContain('Reading the centreline samples...')
    expect(renderToStaticMarkup(<CenterlineChartView line={LINE} data={null} error="no field &quot;T&quot;" t={t} />)).toContain('role="alert"')
  })
})
