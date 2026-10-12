// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// STU-02: show_overlay, open_tab, select_tab and close_tab drive the REAL
// useUiStore through the bridge, and every sent frame stays schema-valid.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ClientMsgSchema, type ClientMsg, type UiCommand } from '@cfd/shared'
import { tabScreenLabel, tabStateLabel, useUiStore, type Tab } from '../state/uiStore'
import { createUiBridge } from './uiBridge'

const executed: unknown[] = []
const saveImage = vi.fn()
const fakeApi = {
  execute: async (cmd: unknown) => {
    executed.push(cmd)
    return { ok: true, state: null, error: null, image: { base64: 'AAAA', mime: 'image/png', width: 1, height: 1 } }
  },
  getState: () => ({
    camera: { position: [10, 10, 10], target: [0, 0, 0], projection: 'perspective' },
    datasetId: 'ds1',
    datasetName: 'channel',
    field: 'U',
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
  probePoint: () => ({ ok: false, code: 'NO_DATASET', message: 'tabs: no dataset' }),
}
vi.mock('../viewer', () => ({ getViewerApi: () => fakeApi, setViewerApi: () => {} }))
vi.mock('./actions', () => ({ actions: { setSettings: () => true, openSession: () => true }, activeCasePath: () => null }))
vi.mock('../api/rest', () => ({ api: { geometrySave: vi.fn(), geometryEdit: vi.fn() }, ApiError: class extends Error {} }))

const session = { problems: {}, connection: 'online', addNote: vi.fn(), runs: {} as Record<string, unknown> }

interface EditorDep {
  isDirty(path: string): boolean
  closeBuffer(path: string): void
  removeDiff(id: string): void
}
let sent: ClientMsg[] = []
let n = 0

function makeBridge(editor?: EditorDep) {
  return createUiBridge({
    send: (m) => {
      sent.push(m)
      return true
    },
    ui: useUiStore,
    session: { getState: () => session as never },
    subscribeRun: vi.fn(),
    saveImage,
    editor,
  })
}

beforeEach(() => {
  sent = []
  n = 0
  saveImage.mockClear()
  executed.length = 0
  useUiStore.setState({ tabs: [], activeTabId: null, showAxes: true, showColorBars: true, bottomVisible: false, bottomTab: 'terminal', activeRunId: null })
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

describe('uiBridge tabs (STU-02)', () => {
  it('show_overlay toggles the store and reports overlays in state', async () => {
    const bridge = makeBridge()
    const r1 = await drive(bridge, { type: 'show_overlay', what: 'axes', on: false })
    expect(r1.ok).toBe(true)
    expect(r1.state).toMatchObject({ showAxes: false, showColorBars: true, overlays: { axes: false, colorBars: true } })
    expect(useUiStore.getState().showAxes).toBe(false)
    await drive(bridge, { type: 'show_overlay', what: 'colorbars', on: false })
    expect(useUiStore.getState().showColorBars).toBe(false)
    const r3 = await drive(bridge, { type: 'show_overlay', what: 'axes', on: true })
    expect(r3.state?.overlays).toEqual({ axes: true, colorBars: false })
    expect(useUiStore.getState().showAxes).toBe(true)
  })

  it('post_screenshot honours showColorBars for includeLegend', async () => {
    const bridge = makeBridge()
    await drive(bridge, { type: 'post_screenshot' })
    expect(executed.find((c) => (c as { type: string }).type === 'screenshot')).toMatchObject({ includeLegend: true })
    expect(saveImage).toHaveBeenCalledTimes(1)
    await drive(bridge, { type: 'show_overlay', what: 'colorbars', on: false })
    await drive(bridge, { type: 'post_screenshot' })
    const shots = executed.filter((c) => (c as { type: string }).type === 'screenshot')
    expect(shots[1]).toMatchObject({ includeLegend: false })
  })

  it('open_tab opens every kind and refuses an unknown one', async () => {
    const bridge = makeBridge()
    const ids = () => useUiStore.getState().tabs.map((t) => t.id)
    const rv = await drive(bridge, { type: 'open_tab', kind: 'viewer' })
    expect(ids()).toContain('viewer')
    expect(rv.state?.tabs).toContainEqual({ id: 'viewer', kind: 'viewer', label: 'viewer' })
    expect(useUiStore.getState().activeTabId).toBe('viewer')
    const rr = await drive(bridge, { type: 'open_tab', kind: 'residuals' })
    expect(rr.state?.tabs).toContainEqual({ id: 'residuals', kind: 'residuals', label: 'residuals' })
    expect(useUiStore.getState().activeTabId).toBe('residuals')
    const rm = await drive(bridge, { type: 'open_tab', kind: 'metrics' })
    expect(rm.state?.tabs).toContainEqual({ id: 'residuals', kind: 'residuals', label: 'metrics' })
    expect(useUiStore.getState().activeTabId).toBe('residuals')
    await drive(bridge, { type: 'open_tab', kind: 'logs' })
    expect(useUiStore.getState().bottomTab).toBe('logs')
    expect(useUiStore.getState().bottomVisible).toBe(true)
    await drive(bridge, { type: 'open_tab', kind: 'problems' })
    expect(useUiStore.getState().bottomTab).toBe('problems')
    const rg = await drive(bridge, { type: 'open_tab', kind: 'geometry', path: 'cases/a.stl' })
    expect(useUiStore.getState().activeTabId).toBe('geometry:cases/a.stl')
    expect(rg.state?.tabs).toContainEqual({ id: 'geometry:cases/a.stl', kind: 'geometry', label: 'cases/a.stl' })
    const rc = await drive(bridge, { type: 'open_tab', kind: 'campaign', path: 'campaigns/c1' })
    expect(useUiStore.getState().activeTabId).toBe('campaign:campaigns/c1')
    expect(rc.state?.tabs).toContainEqual({ id: 'campaign:campaigns/c1', kind: 'campaign', label: 'campaigns/c1' })
  })

  it('open_tab geometry without a path and an unknown kind refuse', async () => {
    const bridge = makeBridge()
    const r1 = await drive(bridge, { type: 'open_tab', kind: 'geometry' })
    expect(r1.ok).toBe(false)
    expect(r1.error?.startsWith('INVALID (open_tab): kind "geometry" needs path')).toBe(true)
    const r2 = await drive(bridge, { type: 'open_tab', kind: 'spreadsheet' })
    expect(r2.ok).toBe(false)
    expect(r2.error).toBe('INVALID (open_tab): tab kind "spreadsheet" does not exist on this screen; kinds: viewer, residuals, metrics, logs, problems, geometry, campaign')
  })

  it('select_tab resolves id, ui.state label, screen label and heuristics', async () => {
    const bridge = makeBridge()
    const u = useUiStore.getState()
    u.openViewerTab()
    u.openResidualsTab(null)
    u.openFile('cases/x/case.jsonc')
    expect(useUiStore.getState().activeTabId).toBe('file:cases/x/case.jsonc')
    await drive(bridge, { type: 'select_tab', tab: 'viewer' })
    expect(useUiStore.getState().activeTabId).toBe('viewer')
    await drive(bridge, { type: 'select_tab', tab: 'file:cases/x/case.jsonc' })
    expect(useUiStore.getState().activeTabId).toBe('file:cases/x/case.jsonc')
    await drive(bridge, { type: 'select_tab', tab: 'Residuals' })
    expect(useUiStore.getState().activeTabId).toBe('residuals')
    await drive(bridge, { type: 'select_tab', tab: '잔차' })
    expect(useUiStore.getState().activeTabId).toBe('residuals')
    await drive(bridge, { type: 'select_tab', tab: 'case.jsonc' })
    expect(useUiStore.getState().activeTabId).toBe('file:cases/x/case.jsonc')
    await drive(bridge, { type: 'select_tab', tab: 'logs' })
    expect(useUiStore.getState().bottomTab).toBe('logs')
    expect(useUiStore.getState().bottomVisible).toBe(true)
    const bad = await drive(bridge, { type: 'select_tab', tab: 'nonexistent' })
    expect(bad.ok).toBe(false)
    expect(bad.error).toBe('NO_TAB (select_tab): no tab "nonexistent"; open tabs: viewer, residuals, file:cases/x/case.jsonc; kinds that open on select: viewer, residuals, logs, problems')
  })

  it('close_tab closes by id and by label and refuses unknown ids', async () => {
    const bridge = makeBridge()
    const u = useUiStore.getState()
    u.openViewerTab()
    u.openResidualsTab(null)
    const r1 = await drive(bridge, { type: 'close_tab', id: 'viewer' })
    expect(r1.ok).toBe(true)
    expect(useUiStore.getState().tabs.map((t) => t.id)).toEqual(['residuals'])
    expect(useUiStore.getState().activeTabId).toBe('residuals')
    const bad = await drive(bridge, { type: 'close_tab', id: 'tab-1' })
    expect(bad.ok).toBe(false)
    expect(bad.error).toBe('NO_TAB (close_tab): no tab "tab-1"; open tabs: residuals')
    const r2 = await drive(bridge, { type: 'close_tab', id: 'Residuals' })
    expect(r2.ok).toBe(true)
    expect(useUiStore.getState().tabs).toEqual([])
    const bad2 = await drive(bridge, { type: 'close_tab', id: 'tab-1' })
    expect(bad2.error).toBe('NO_TAB (close_tab): no tab "tab-1"; open tabs: none')
  })

  it('close_tab of a dirty file tab refuses and leaves the tab open', async () => {
    useUiStore.getState().openFile('cases/x/case.jsonc')
    const dirty: EditorDep = { isDirty: vi.fn(() => true), closeBuffer: vi.fn(), removeDiff: vi.fn() }
    const bridge = makeBridge(dirty)
    const r = await drive(bridge, { type: 'close_tab', id: 'cases/x/case.jsonc' })
    expect(r.ok).toBe(false)
    expect(r.error?.startsWith('DIRTY (close_tab): "cases/x/case.jsonc"')).toBe(true)
    expect(useUiStore.getState().tabs.map((t) => t.id)).toEqual(['file:cases/x/case.jsonc'])
    expect(dirty.closeBuffer).not.toHaveBeenCalled()
    dirty.isDirty = vi.fn(() => false)
    const r2 = await drive(bridge, { type: 'close_tab', id: 'cases/x/case.jsonc' })
    expect(r2.ok).toBe(true)
    expect(useUiStore.getState().tabs).toEqual([])
    expect(dirty.closeBuffer).toHaveBeenCalledTimes(1)
    expect(dirty.closeBuffer).toHaveBeenCalledWith('cases/x/case.jsonc')
  })

  it('tabScreenLabel matches what TabBar showed and tabStateLabel names the path or chart', () => {
    const residuals: Tab = { id: 'residuals', kind: 'residuals', runId: null }
    const metrics: Tab = { id: 'residuals', kind: 'residuals', runId: null, chart: 'metrics' }
    expect(tabScreenLabel({ id: 'viewer', kind: 'viewer' }, 'en')).toBe('3D Viewer')
    expect(tabScreenLabel(residuals, 'en')).toBe('Residuals')
    expect(tabScreenLabel(metrics, 'en')).toBe('Metrics')
    expect(tabScreenLabel({ id: 'file:cases/a/case.jsonc', kind: 'file', path: 'cases/a/case.jsonc' }, 'en')).toBe('case.jsonc')
    expect(tabScreenLabel({ id: 'geometry:cases/a.stl', kind: 'geometry', path: 'cases/a.stl' }, 'en')).toBe('Geometry: a.stl')
    expect(tabScreenLabel({ id: 'campaign:c', kind: 'campaign', path: 'campaigns/c1' }, 'en')).toBe('Campaign: c1')
    expect(tabScreenLabel({ id: 'd1', kind: 'diff', path: 'cases/a/x.ts', toolUseId: null }, 'en')).toBe('Diff: x.ts')
    expect(tabStateLabel({ id: 'file:cases/a/case.jsonc', kind: 'file', path: 'cases/a/case.jsonc' })).toBe('cases/a/case.jsonc')
    expect(tabStateLabel({ id: 'geometry:cases/a.stl', kind: 'geometry', path: 'cases/a.stl' })).toBe('cases/a.stl')
    expect(tabStateLabel({ id: 'campaign:c', kind: 'campaign', path: 'campaigns/c1' })).toBe('campaigns/c1')
    expect(tabStateLabel({ id: 'd1', kind: 'diff', path: 'cases/a/x.ts', toolUseId: null })).toBe('cases/a/x.ts')
    expect(tabStateLabel(metrics)).toBe('metrics')
    expect(tabStateLabel(residuals)).toBe('residuals')
    expect(tabStateLabel({ id: 'viewer', kind: 'viewer' })).toBe('viewer')
  })
})
