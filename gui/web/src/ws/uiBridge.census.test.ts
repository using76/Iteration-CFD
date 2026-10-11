// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The STU census: one test enumerates every UiCommandSchema type and drives one
// minimal valid sample of each through a fresh uiBridge, recording every type the
// studio screen answers "does not implement it" for (an explicit stub or the
// default fall-through of apply). OPEN_UNIMPLEMENTED is the census result; each
// later STU unit deletes its rows when its types are implemented, and the last
// test flips from an expected failure to a plain pass when the list is empty.
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { UI_COMMAND_TYPES, UiCommandSchema, type ClientMsg, type UiCommand, type UiCommandType } from '@cfd/shared'
import type { ViewerTool } from '../viewer'
import { createUiBridge } from './uiBridge'

interface FakeViewerState {
  camera: { position: [number, number, number]; target: [number, number, number]; projection: 'perspective' | 'orthographic' }
  datasetId: string | null
  datasetName: string | null
  field: string | null
  time: { index: number; value: number; count: number } | null
  colormap: string
  range: [number, number] | null
  representation: string
}

const fake = {
  mounted: true,
  toolNow: 'select' as ViewerTool,
  state: null as unknown as FakeViewerState,
}

const fakeApi = {
  execute: async (cmd: unknown) => {
    const c = cmd as { type: string }
    return {
      ok: true,
      state: fake.state,
      error: null,
      image: c.type === 'screenshot' ? { base64: 'AAAA', mime: 'image/png', width: 640, height: 480 } : null,
    }
  },
  getState: () => fake.state,
  subscribe: () => () => {},
  isMounted: () => fake.mounted,
  setBaseUrl: () => {},
  setTool: (tool: ViewerTool) => {
    fake.toolNow = tool
  },
  getTool: () => fake.toolNow,
  viewport: () => (fake.mounted ? { left: 100, top: 50, width: 800, height: 600 } : null),
  probePixel: () => ({ ok: true, hit: { cell: 1, value: 0, field: 'U', center: [0, 0, 0] } }),
  probePoint: () => ({ ok: false, code: 'NO_STRUCTURED_GRID', message: 'census: no structured grid' }),
}

vi.mock('../viewer', () => ({ getViewerApi: () => fakeApi, setViewerApi: () => {} }))
vi.mock('./actions', () => ({ actions: { setSettings: () => true, openSession: () => true }, activeCasePath: () => null }))
vi.mock('../api/rest', () => ({ api: { geometrySave: vi.fn(), geometryEdit: vi.fn() }, ApiError: class extends Error {} }))

/** A fresh fake ui store with every method uiBridge's apply can call. */
function makeUi() {
  return {
    tabs: [] as { id: string; kind: string }[],
    activeTabId: null as string | null,
    assistantVisible: false,
    activeRunId: null as string | null,
    locale: 'en' as 'ko' | 'en',
    viewSplit: false,
    focusedView: 'A' as 'A' | 'B',
    camerasLinked: false,
    showAxes: true,
    showColorBars: true,
    logFilter: { text: '', streams: ['stdout', 'stderr', 'system'], follow: true },
    metricSlots: [null, null] as [string | null, string | null],
    metricMode: 'metrics' as 'metrics' | 'sweeps',
    openViewerTab: vi.fn(),
    openResidualsTab: vi.fn(),
    setBottomVisible: vi.fn(),
    setBottomTab: vi.fn(),
    setLogFilter: vi.fn(),
    setMetricSlot: vi.fn(),
    setMetricMode: vi.fn(),
    openFile: vi.fn(),
    setLocale: vi.fn(),
    setActiveRun: vi.fn(),
    toggleAssistant: vi.fn(),
    setCompareRun: vi.fn(),
    setViewSplit: vi.fn(),
    setFocusedView: vi.fn(),
    setCamerasLinked: vi.fn(),
    openGeometryTab: vi.fn(),
    openCampaignTab: vi.fn(),
    setOverlay: vi.fn(),
    closeTab: vi.fn(),
    activateTab: vi.fn(),
  }
}

/** Exactly one minimal VALID command per UiCommandSchema type; the Record typing makes a missing type a compile error. */
const SAMPLES: Record<UiCommandType, UiCommand> = {
  select_tab: { type: 'select_tab', tab: 'viewer' },
  show_field: { type: 'show_field', field: 'Temperature' },
  select_step: { type: 'select_step', step: 'step-1' },
  open_panel: { type: 'open_panel', panel: 'AI Assistant' },
  set_tool: { type: 'set_tool', tool: 'select' },
  set_projection: { type: 'set_projection', projection: 'Perspective' },
  fit_view: { type: 'fit_view' },
  show_overlay: { type: 'show_overlay', what: 'axes', on: true },
  set_centerline: { type: 'set_centerline', quantity: 'p' },
  run: { type: 'run', action: 'run' },
  notify: { type: 'notify', level: 'info', text: 'census' },
  open_case: { type: 'open_case', path: 'cases/x' },
  save_case: { type: 'save_case' },
  validate_case: { type: 'validate_case' },
  new_case: { type: 'new_case', template: 'empty', name: 'census' },
  set_run_setting: { type: 'set_run_setting' },
  start_run: { type: 'start_run' },
  stop_run: { type: 'stop_run' },
  follow_run: { type: 'follow_run', runId: 'r1' },
  open_mesh_dialog: { type: 'open_mesh_dialog' },
  start_mesh: { type: 'start_mesh' },
  show_chart: { type: 'show_chart', chart: 'residuals' },
  show_metric: { type: 'show_metric' },
  set_log_filter: { type: 'set_log_filter' },
  open_result: { type: 'open_result', path: 'cases/x/output/100' },
  set_post: { type: 'set_post' },
  add_layer: { type: 'add_layer', kind: 'slice', args: {} },
  remove_layer: { type: 'remove_layer', id: 'layer-1' },
  set_camera: { type: 'set_camera', preset: 'iso' },
  probe: { type: 'probe', at: 'center' },
  open_tab: { type: 'open_tab', kind: 'viewer' },
  close_tab: { type: 'close_tab', id: 'tab-1' },
  set_locale: { type: 'set_locale', locale: 'en' },
  post_field: { type: 'post_field', field: 'U' },
  post_representation: { type: 'post_representation', mode: 'surface' },
  post_time: { type: 'post_time', index: 'last' },
  post_screenshot: { type: 'post_screenshot' },
  post_warp: { type: 'post_warp' },
  open_mesh_view: { type: 'open_mesh_view' },
  set_patch: { type: 'set_patch', patch: 'inlet' },
  open_boundary_editor: { type: 'open_boundary_editor' },
  open_session: { type: 'open_session', sessionId: 's1' },
  set_setting: { type: 'set_setting', notifyOnRunEnd: true },
  run_custom_tool: { type: 'run_custom_tool', name: 'tool' },
  split_view: { type: 'split_view', on: true },
  focus_view: { type: 'focus_view', view: 'A' },
  open_result_in_view: { type: 'open_result_in_view', view: 'A', path: 'cases/x/output/100' },
  link_cameras: { type: 'link_cameras', on: true },
  compare_run: { type: 'compare_run', runId: null },
  geometry_open: { type: 'geometry_open', path: 'cases/x.stl' },
  geometry_import_step: { type: 'geometry_import_step', path: 'cases/x.step' },
  geometry_part: { type: 'geometry_part', name: 'a', action: 'select' },
  geometry_transform: { type: 'geometry_transform', op: 'reset' },
  geometry_boolean: { type: 'geometry_boolean', op: 'fuse', a: 'a', b: 'b' },
  geometry_save: { type: 'geometry_save', path: 'cases/y.stl' },
}

/** type -> the later STU unit that implements it. Each later STU unit deletes
 *  its rows when its types are implemented; when the object is empty the last
 *  census test becomes a plain pass. */
const OPEN_UNIMPLEMENTED: Record<string, string> = {
  add_layer: 'STU-04',
  remove_layer: 'STU-04',
  new_case: 'STU-05',
  validate_case: 'STU-05',
  save_case: 'STU-05',
  open_mesh_dialog: 'STU-06a',
  start_mesh: 'STU-06a',
  open_mesh_view: 'STU-06b',
  set_patch: 'STU-07a',
  open_boundary_editor: 'STU-07b',
  select_step: 'STU-08b',
  set_centerline: 'STU-09',
  run_custom_tool: 'STU-09',
}

type ResultFrame = Extract<ClientMsg, { t: 'ui.result' }>

/** A fresh bridge over one sent-frame recorder; the census never lets two samples share one. */
function freshBridge() {
  const sent: ClientMsg[] = []
  const ui = makeUi()
  const session = { problems: {}, connection: 'online', addNote: vi.fn(), runs: {} as Record<string, unknown>, runData: {} as Record<string, unknown> }
  const bridge = createUiBridge({
    send: (m) => {
      sent.push(m)
      return true
    },
    ui: { getState: () => ui as never },
    session: { getState: () => session as never },
    subscribeRun: vi.fn(),
    saveImage: vi.fn(),
  })
  return { sent, bridge }
}

/** The sent ui.result frame for a requestId. */
function resultFor(sent: ClientMsg[], requestId: string): ResultFrame | undefined {
  return sent.find((m): m is ResultFrame => m.t === 'ui.result' && m.requestId === requestId)
}

/** Runs every sample through a fresh bridge and returns the sorted types whose
 *  answer is ok === false with an error containing "does not implement it". */
async function unimplementedTypes(): Promise<string[]> {
  const out: string[] = []
  for (const t of UI_COMMAND_TYPES) {
    const { sent, bridge } = freshBridge()
    await bridge.handleCommand(`census-${t}`, SAMPLES[t])
    const r = resultFor(sent, `census-${t}`)
    if (r && !r.ok && r.error != null && r.error.includes('does not implement it')) out.push(t)
  }
  return out.sort()
}

beforeEach(() => {
  fake.mounted = true
  fake.toolNow = 'select'
  fake.state = {
    camera: { position: [10, 10, 10], target: [0, 0, 0], projection: 'perspective' },
    datasetId: 'ds1',
    datasetName: 'channel (demo)',
    field: 'U',
    time: null,
    colormap: 'viridis',
    range: null,
    representation: 'surface',
  }
})

describe('uiBridge.census', () => {
  it('every UiCommandSchema type has one valid sample', () => {
    expect(UI_COMMAND_TYPES.length).toBe(55)
    expect(new Set(UI_COMMAND_TYPES).size).toBe(UI_COMMAND_TYPES.length)
    expect(Object.keys(SAMPLES).sort()).toEqual([...UI_COMMAND_TYPES].sort())
    for (const t of UI_COMMAND_TYPES) {
      expect(SAMPLES[t].type).toBe(t)
      expect(() => UiCommandSchema.parse(SAMPLES[t])).not.toThrow()
    }
  })

  it('every command answers in under 4,000 ms', async () => {
    vi.useFakeTimers()
    try {
      const { sent, bridge } = freshBridge()
      const pending = UI_COMMAND_TYPES.map((t) => bridge.handleCommand(`census-${t}`, SAMPLES[t]))
      await vi.advanceTimersByTimeAsync(3999)
      for (const t of UI_COMMAND_TYPES) expect(resultFor(sent, `census-${t}`)).toBeTruthy()
      await Promise.all(pending)
    } finally {
      vi.useRealTimers()
    }
  })

  it('the default branch answers "does not implement it"', async () => {
    const { sent, bridge } = freshBridge()
    await bridge.handleCommand('census-no_such_command', { type: 'no_such_command' } as unknown as UiCommand)
    const r = resultFor(sent, 'census-no_such_command')
    expect(r?.ok).toBe(false)
    expect(r?.error).toBe('UNSUPPORTED (no_such_command): this screen does not implement it')
  })

  it('the unimplemented set is exactly the open list', async () => {
    expect(await unimplementedTypes()).toEqual(Object.keys(OPEN_UNIMPLEMENTED).sort())
  })

  // RED by construction while OPEN_UNIMPLEMENTED is non-empty: it.fails reports
  // the failing assertion as an expected failure, and once the last row is
  // deleted the registration becomes a plain it that passes.
  ;(Object.keys(OPEN_UNIMPLEMENTED).length > 0 ? it.fails : it)('no command is unimplemented', async () => {
    expect(await unimplementedTypes()).toEqual([])
  })
})

