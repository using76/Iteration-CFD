// The bridge is driven against a faked viewer module and a faked store: no
// canvas, no real renderer, node environment. vi.mock hoists above the
// imports, so the factories close over the doubles but never read them at
// factory time - the arrows only run once a test makes the bridge call them.
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ClientMsgSchema, type ClientMsg, type GeometryBounds, type GeometryInfo, type UiCommand, type UiState } from '@cfd/shared'
import type { ProbeOutcome, ViewerTool } from '../viewer'
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
  calls: [] as unknown[],
  tools: [] as string[],
  pixels: [] as [number, number][],
  noImage: false,
  state: null as unknown as FakeViewerState,
  listeners: new Set<(s: FakeViewerState) => void>(),
  pixelOutcome: null as unknown as ProbeOutcome,
  pointOutcome: null as unknown as ProbeOutcome,
}

const fakeApi = {
  execute: async (cmd: unknown) => {
    fake.calls.push(cmd)
    const c = cmd as { type: string; projection?: 'perspective' | 'orthographic' }
    if (c.type === 'setCamera' && c.projection) fake.state.camera.projection = c.projection
    return {
      ok: true,
      state: fake.state,
      error: null,
      image: c.type === 'screenshot' && !fake.noImage ? { base64: 'AAAA', mime: 'image/png', width: 640, height: 480 } : null,
    }
  },
  getState: () => fake.state,
  subscribe: (fn: (s: FakeViewerState) => void) => {
    fake.listeners.add(fn)
    return () => fake.listeners.delete(fn)
  },
  emit: (next: FakeViewerState) => {
    fake.state = next
    for (const l of fake.listeners) l(next)
  },
  isMounted: () => fake.mounted,
  setBaseUrl: () => {},
  setTool: (tool: ViewerTool) => {
    fake.tools.push(tool)
    fake.toolNow = tool
  },
  getTool: () => fake.toolNow,
  viewport: () => (fake.mounted ? { left: 100, top: 50, width: 800, height: 600 } : null),
  probePixel: (x: number, y: number) => {
    fake.pixels.push([x, y])
    return fake.pixelOutcome
  },
  probePoint: () => fake.pointOutcome,
}

// View B: the second viewport half. Same shape as fakeApi, its own state,
// recorder and listener set, so the split-viewport commands can prove which
// half a command landed on.
const fakeB = {
  mounted: true,
  toolNow: 'select' as ViewerTool,
  calls: [] as unknown[],
  tools: [] as string[],
  pixels: [] as [number, number][],
  state: null as unknown as FakeViewerState,
  listeners: new Set<(s: FakeViewerState) => void>(),
}

const fakeApiB = {
  execute: async (cmd: unknown) => {
    fakeB.calls.push(cmd)
    const c = cmd as { type: string; projection?: 'perspective' | 'orthographic' }
    if (c.type === 'setCamera' && c.projection) fakeB.state.camera.projection = c.projection
    return { ok: true, state: fakeB.state, error: null, image: null }
  },
  getState: () => fakeB.state,
  subscribe: (fn: (s: FakeViewerState) => void) => {
    fakeB.listeners.add(fn)
    return () => fakeB.listeners.delete(fn)
  },
  isMounted: () => fakeB.mounted,
  setBaseUrl: () => {},
  setTool: (tool: ViewerTool) => {
    fakeB.tools.push(tool)
    fakeB.toolNow = tool
  },
  getTool: () => fakeB.toolNow,
  viewport: () => (fakeB.mounted ? { left: 100, top: 50, width: 800, height: 600 } : null),
  probePixel: (x: number, y: number) => {
    fakeB.pixels.push([x, y])
    return { ok: false, code: 'NO_VIEWER', message: 'view B has no probe outcome in this test' } as ProbeOutcome
  },
  probePoint: () => ({ ok: false, code: 'NO_VIEWER', message: 'view B has no probe outcome in this test' } as ProbeOutcome),
  emit: (next: FakeViewerState) => {
    fakeB.state = next
    for (const l of fakeB.listeners) l(next)
  },
}

vi.mock('../viewer', () => ({ getViewerApi: (view: 'A' | 'B' = 'A') => (view === 'B' ? fakeApiB : fakeApi), setViewerApi: () => {} }))
vi.mock('./actions', () => ({ actions: { setSettings: () => true, openSession: () => true }, activeCasePath: () => null }))
vi.mock('../api/rest', () => ({ api: { geometrySave: vi.fn(), geometryEdit: vi.fn() }, ApiError: class extends Error {} }))

import { api } from '../api/rest'
import { uiGeometryState, useGeometryStore } from '../state/geometryStore'

const B: GeometryBounds = { min: [0, 0, 0], max: [1, 1, 1] }

const stlSolid = (name: string, first: number) => ({ name, first, count: 2, bounds: B, area: 0.5, volume: 0.5, closed: true, openEdges: 0 })
const stepSolid = (name: string, first: number, tag: number) => ({ name, first, count: 2, bounds: B, area: 0.5, volume: 0.5, closed: true, openEdges: 0, tag })

const INFO: GeometryInfo = {
  id: 'g1',
  path: 'cases/x.stl',
  format: 'stl-ascii',
  triangleCount: 4,
  vertexCount: 6,
  bounds: { min: [0, 0, 0], max: [2, 2, 2] },
  area: 1,
  volume: 1,
  closed: true,
  openEdges: 0,
  solids: [stlSolid('a', 0), stlSolid('b', 2)],
  blobs: {} as never,
  readMs: 1,
  warnings: [],
  source: null,
}

const STEP_INFO: GeometryInfo = {
  ...INFO,
  path: 'cases/car.step',
  solids: [stepSolid('body', 0, 7), stepSolid('wing', 2, 9)],
  source: { kind: 'step', path: 'cases/car.step', tool: 'geom_tool' },
}

type GeoTab = { id: string; kind: 'geometry'; path: string }
type ResTab = { id: 'residuals'; kind: 'residuals'; runId: string | null; compareRunId?: string | null }
type ResultFrame = Extract<ClientMsg, { t: 'ui.result' }>
type RunResult = ResultFrame & { state: UiState }

const geoTab = (path: string): GeoTab => ({ id: `geometry:${path}`, kind: 'geometry', path })

const st = () => useGeometryStore.getState()

function load(path: string, info: GeometryInfo) {
  st().begin(path)
  st().setLoaded(path, { id: info.id, info })
}

function resetFake(): void {
  fake.mounted = true
  fake.toolNow = 'select'
  fake.calls = []
  fake.tools = []
  fake.pixels = []
  fake.noImage = false
  fake.state = {
    camera: { position: [10, 10, 10], target: [0, 0, 0], projection: 'perspective' },
    datasetId: 'ds1', datasetName: 'channel (demo)', field: 'U', time: null,
    colormap: 'viridis', range: null, representation: 'surface',
  }
  fake.pixelOutcome = { ok: true, hit: { cell: 42, value: 1.5, field: 'U', center: [1, 2, 3] } }
  fake.pointOutcome = { ok: false, code: 'NO_STRUCTURED_GRID', message: 'channel (demo) has no structured grid (vtu); a world point cannot be located - probe by fx,fy, x,y or at:"center"' }
  fake.listeners = new Set()
  fakeB.mounted = true
  fakeB.toolNow = 'select'
  fakeB.calls = []
  fakeB.tools = []
  fakeB.pixels = []
  fakeB.state = {
    camera: { position: [10, 10, 10], target: [0, 0, 0], projection: 'perspective' },
    datasetId: 'dsB', datasetName: 'channel (demo)', field: 'U', time: null,
    colormap: 'viridis', range: null, representation: 'surface',
  }
  fakeB.listeners = new Set()
}

function harness(tabs: (GeoTab | ResTab)[] = []) {
  const sent: ClientMsg[] = []
  const saveImage = vi.fn()
  const subscribeRun = vi.fn()
  const putResTab = (tab: ResTab) => {
    const i = ui.tabs.findIndex((t) => t.id === tab.id)
    if (i >= 0) ui.tabs[i] = tab
    else ui.tabs.push(tab)
  }
  const ui = {
    tabs,
    activeTabId: tabs[0]?.id ?? null,
    assistantVisible: false,
    activeRunId: null as string | null,
    locale: 'en' as const,
    viewSplit: false,
    focusedView: 'A' as 'A' | 'B',
    camerasLinked: false,
    openGeometryTab: vi.fn(),
    openViewerTab: vi.fn(),
    setActiveRun(runId: string | null) {
      ui.activeRunId = runId
    },
    openResidualsTab(runId: string | null) {
      const prev = ui.tabs.find((t): t is ResTab => t.kind === 'residuals')
      putResTab({ id: 'residuals', kind: 'residuals', runId: runId ?? (prev ? prev.runId : null) })
      ui.activeTabId = 'residuals'
      ui.activeRunId = runId ?? ui.activeRunId
    },
    setCompareRun(runId: string | null) {
      const prev = ui.tabs.find((t): t is ResTab => t.kind === 'residuals')
      putResTab({ id: 'residuals', kind: 'residuals', runId: prev ? prev.runId : ui.activeRunId, compareRunId: runId })
      ui.activeTabId = 'residuals'
    },
    setViewSplit(on: boolean) {
      if (on) ui.viewSplit = true
      else {
        ui.viewSplit = false
        ui.focusedView = 'A'
        ui.camerasLinked = false
      }
    },
    setFocusedView(view: 'A' | 'B') {
      ui.focusedView = view
    },
    setCamerasLinked(on: boolean) {
      ui.camerasLinked = on
    },
  }
  const session = { problems: {}, connection: 'online', addNote: vi.fn(), runs: {} as Record<string, unknown> }
  const bridge = createUiBridge({
    send: (m) => {
      sent.push(m)
      return true
    },
    ui: { getState: () => ui as never },
    session: { getState: () => session as never },
    subscribeRun,
    saveImage,
  })
  return {
    sent,
    ui,
    session,
    saveImage,
    subscribeRun,
    async run(cmd: UiCommand): Promise<RunResult> {
      await bridge.handleCommand('r1', cmd)
      const last = sent[sent.length - 1] as ResultFrame
      if (last.t !== 'ui.result' || last.state == null) throw new Error('no ui.result with state was sent')
      return last as RunResult
    },
  }
}

beforeEach(() => {
  resetFake()
  useGeometryStore.setState({ byPath: {} })
  vi.mocked(api.geometrySave).mockReset()
  vi.mocked(api.geometryEdit).mockReset()
  vi.mocked(api.geometrySave).mockResolvedValue({ id: 'g2', info: { ...INFO, id: 'g2', path: 'cases/y.stl' } })
  vi.mocked(api.geometryEdit).mockResolvedValue({ path: 'cases/car-common.step', solids: [], applied: [], stdout: [], toolMs: 1 })
})

describe('the four geometry edit commands', () => {
  it('an unlisted command is still refused by name', async () => {
    const h = harness([])
    const r = await h.run({ type: 'show_overlay', what: 'axes', on: true })
    expect(r.ok).toBe(false)
    expect(r.error).toBe('UNSUPPORTED (show_overlay): this screen does not implement it')
  })

  it('without a geometry tab the four commands refuse and the snapshot has no geometry', async () => {
    const h = harness([])
    const cmds: UiCommand[] = [
      { type: 'geometry_part', name: 'a', action: 'hide' },
      { type: 'geometry_transform', op: 'translate', value: [1, 0, 0] },
      { type: 'geometry_boolean', op: 'fuse', a: 'a', b: 'b' },
      { type: 'geometry_save', path: 'cases/y.stl' },
    ]
    for (const cmd of cmds) {
      const r = await h.run(cmd)
      expect(r.ok).toBe(false)
      expect(r.error).toBe('no geometry tab is open; geometry_open or geometry_import_step first')
      expect(r.state.geometry).toBeNull()
    }
  })

  it('geometry_part flips visibility, selects and renames, and gui_state sees it', async () => {
    const h = harness([geoTab('cases/x.stl')])
    load('cases/x.stl', INFO)
    const sol = (r: RunResult, name: string) => r.state.geometry!.solids.find((s) => s.name === name)
    let r = await h.run({ type: 'geometry_part', name: 'a', action: 'hide' })
    expect(r.ok).toBe(true)
    expect(sol(r, 'a')?.visible).toBe(false)
    r = await h.run({ type: 'geometry_part', name: 'b', action: 'keep_only' })
    expect(sol(r, 'a')?.visible).toBe(false)
    expect(sol(r, 'b')?.visible).toBe(true)
    r = await h.run({ type: 'geometry_part', name: 'a', action: 'show' })
    expect(sol(r, 'a')?.visible).toBe(true)
    expect(sol(r, 'b')?.visible).toBe(true)
    r = await h.run({ type: 'geometry_part', name: 'b', action: 'select' })
    expect(r.state.geometry!.selected).toBe('b')
    r = await h.run({ type: 'geometry_part', name: 'b', action: 'rename', newName: 'body' })
    expect(sol(r, 'body')).toBeTruthy()
    expect(r.state.geometry!.selected).toBe('body')
    expect(r.state.geometry!.dirty).toBe(true)
    r = await h.run({ type: 'geometry_part', name: 'zz', action: 'hide' })
    expect(r.ok).toBe(false)
    expect(r.error).toBe('no part "zz"; parts: a, body')
  })

  it('geometry_transform pushes, undoes, redoes, resets and refuses a mis-shaped value by name', async () => {
    const h = harness([geoTab('cases/x.stl')])
    load('cases/x.stl', INFO)
    let r = await h.run({ type: 'geometry_transform', op: 'translate', value: [1, 0, 0] })
    expect(r.ok).toBe(true)
    expect(r.state.geometry!.edits).toBe(1)
    expect(r.state.geometry!.dirty).toBe(true)
    r = await h.run({ type: 'geometry_transform', op: 'undo' })
    expect(r.state.geometry!.edits).toBe(0)
    expect(r.state.geometry!.dirty).toBe(false)
    r = await h.run({ type: 'geometry_transform', op: 'undo' })
    expect(r.ok).toBe(false)
    expect(r.error).toBe('nothing to undo')
    r = await h.run({ type: 'geometry_transform', op: 'redo' })
    expect(r.state.geometry!.edits).toBe(1)
    r = await h.run({ type: 'geometry_transform', op: 'reset' })
    expect(r.state.geometry!.edits).toBe(0)
    r = await h.run({ type: 'geometry_transform', op: 'translate', value: 3 })
    expect(r.ok).toBe(false)
    expect(r.error).toBe('translate needs value [x,y,z] in metres')
    r = await h.run({ type: 'geometry_transform', op: 'scale', value: 0 })
    expect(r.ok).toBe(false)
    expect(r.error).toBe('scale needs one non-zero factor or [sx,sy,sz]')
    r = await h.run({ type: 'geometry_transform', op: 'mirror', value: [0, 0, 0] })
    expect(r.ok).toBe(false)
    expect(r.error).toBe('mirror needs a non-zero plane normal [nx,ny,nz]')
    r = await h.run({ type: 'geometry_transform', op: 'rotate', value: [0, 0, 90], pivot: 'origin' })
    expect(r.ok).toBe(true)
    expect(r.state.geometry!.edits).toBe(1)
  })

  it('geometry_save posts the composed edit, opens the written file and reports the ASCII notice', async () => {
    const h = harness([geoTab('cases/x.stl')])
    load('cases/x.stl', INFO)
    await h.run({ type: 'geometry_part', name: 'a', action: 'hide' })
    await h.run({ type: 'geometry_part', name: 'b', action: 'rename', newName: 'body' })
    await h.run({ type: 'geometry_transform', op: 'translate', value: [1, 0, 0] })
    vi.mocked(api.geometrySave).mockClear()
    const r = await h.run({ type: 'geometry_save', path: 'cases/y.stl', keepVisibleOnly: true })
    expect(r.ok).toBe(true)
    expect(api.geometrySave).toHaveBeenCalledTimes(1)
    expect(api.geometrySave).toHaveBeenCalledWith('g1', {
      path: 'cases/y.stl',
      binary: false,
      transform: [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
      keepSolids: ['b'],
      names: { b: 'body' },
      overwrite: false,
    })
    expect(h.ui.openGeometryTab).toHaveBeenCalledWith('cases/y.stl')
    expect(h.session.addNote).toHaveBeenCalledWith('info', 'saved as ASCII STL so the part names survive')
    expect(r.state.geometry!.edits).toBe(0)
    expect(r.state.geometry!.dirty).toBe(false)
    vi.mocked(api.geometrySave).mockRejectedValue(new Error('cases/y.stl already exists; pass overwrite: true to replace it'))
    const r2 = await h.run({ type: 'geometry_save', path: 'cases/y.stl' })
    expect(r2.ok).toBe(false)
    expect(r2.error).toBe('cases/y.stl already exists; pass overwrite: true to replace it')
  })

  it('geometry_boolean runs only on a STEP and re-imports the result', async () => {
    const h = harness([geoTab('cases/x.stl')])
    load('cases/x.stl', INFO)
    const r = await h.run({ type: 'geometry_boolean', op: 'common', a: 'a', b: 'b' })
    expect(r.ok).toBe(false)
    expect(r.error?.startsWith('UNSUPPORTED (geometry_boolean): ')).toBe(true)
    expect(api.geometryEdit).not.toHaveBeenCalled()
    const h2 = harness([geoTab('cases/car.step')])
    load('cases/car.step', STEP_INFO)
    const r2 = await h2.run({ type: 'geometry_boolean', op: 'common', a: 'body', b: 'wing' })
    expect(r2.ok).toBe(true)
    expect(api.geometryEdit).toHaveBeenCalledTimes(1)
    expect(api.geometryEdit).toHaveBeenCalledWith({
      path: 'cases/car.step',
      ops: '[{"op":"intersect","object":[7],"tools":[9]}]',
      out: 'cases/car-common.step',
      overwrite: true,
    })
    expect(h2.ui.openGeometryTab).toHaveBeenCalledWith('cases/car-common.step')
  })

  it('every answer carries the store\'s geometry state; a loading tab says so', async () => {
    const h = harness([geoTab('cases/x.stl')])
    load('cases/x.stl', INFO)
    await h.run({ type: 'geometry_part', name: 'a', action: 'hide' })
    expect(h.sent).toHaveLength(2)
    expect(h.sent[0].t).toBe('ui.state')
    expect(h.sent[1].t).toBe('ui.result')
    const r = await h.run({ type: 'geometry_transform', op: 'translate', value: [1, 0, 0] })
    expect(h.sent).toHaveLength(4)
    expect(r.state.geometry).toEqual(uiGeometryState(st().byPath['cases/x.stl']))
    useGeometryStore.setState({ byPath: {} })
    st().begin('cases/x.stl')
    const r2 = await h.run({ type: 'geometry_part', name: 'a', action: 'hide' })
    expect(r2.ok).toBe(false)
    expect(r2.error).toBe('geometry "cases/x.stl" is still loading; try again')
  })
})

describe('the viewer commands the bridge drives', () => {
  it('set_projection issues setCamera with the lower-case projection and the snapshot reports the ui spelling', async () => {
    const h = harness()
    const r = await h.run({ type: 'set_projection', projection: 'Orthographic' })
    expect(fake.calls[0]).toEqual({ type: 'setCamera', preset: null, position: null, target: null, projection: 'orthographic' })
    expect(r.ok).toBe(true)
    expect(r.state.projection).toBe('Orthographic')
    const r2 = await h.run({ type: 'set_projection', projection: 'Perspective' })
    expect(fake.calls[1]).toEqual({ type: 'setCamera', preset: null, position: null, target: null, projection: 'perspective' })
    expect(r2.ok).toBe(true)
    expect(r2.state.projection).toBe('Perspective')
  })

  it('set_tool maps probe to select and box to section and the snapshot echoes the ui name', async () => {
    const h = harness()
    let r = await h.run({ type: 'set_tool', tool: 'probe' })
    expect(fake.tools[0]).toBe('select')
    expect(r.state.tool).toBe('select')
    r = await h.run({ type: 'set_tool', tool: 'box' })
    expect(fake.tools[1]).toBe('section')
    expect(r.state.tool).toBe('box')
    r = await h.run({ type: 'set_tool', tool: 'move' })
    expect(fake.tools[2]).toBe('rotate')
    expect(r.state.tool).toBe('move')
    fake.mounted = false
    r = await h.run({ type: 'set_tool', tool: 'pan' })
    expect(r.ok).toBe(true)
    expect(fake.tools[3]).toBe('pan')
  })

  it('probe at center picks the canvas middle and the selection carries the cell', async () => {
    const h = harness()
    const r = await h.run({ type: 'probe', at: 'center' })
    expect(fake.pixels[0]).toEqual([500, 350])
    expect(r.ok).toBe(true)
    expect(r.state.selection).toEqual({ kind: 'cell', id: 42, center: [1, 2, 3], value: 1.5, field: 'U', patch: null })
  })

  it('probe by fraction scales into the viewport, by pixel passes through, and bad fractions are refused by name', async () => {
    const h = harness()
    const r = await h.run({ type: 'probe', fx: 0.25, fy: 0.5 })
    expect(fake.pixels[0]).toEqual([300, 350])
    expect(r.ok).toBe(true)
    const r2 = await h.run({ type: 'probe', x: 7, y: 9 })
    expect(fake.pixels[1]).toEqual([7, 9])
    expect(r2.ok).toBe(true)
    const r3 = await h.run({ type: 'probe', fx: 1.5, fy: 0.5 })
    expect(r3.ok).toBe(false)
    expect(r3.error?.startsWith('INVALID (probe): fx,fy must lie in 0..1')).toBe(true)
    expect(fake.pixels).toHaveLength(2)
    const r4 = await h.run({ type: 'probe' })
    expect(r4.ok).toBe(false)
    expect(r4.error?.startsWith('INVALID (probe): give')).toBe(true)
    expect(fake.pixels).toHaveLength(2)
  })

  it('probe by point without a structured grid is UNSUPPORTED and a miss clears the selection', async () => {
    const h = harness()
    const r = await h.run({ type: 'probe', point: [0, 0, 0] })
    expect(r.ok).toBe(false)
    expect(r.error?.startsWith('UNSUPPORTED (probe)')).toBe(true)
    fake.pointOutcome = { ok: false, code: 'MISS', message: 'the point lies outside the domain' }
    const r2 = await h.run({ type: 'probe', point: [0, 0, 0] })
    expect(r2.ok).toBe(false)
    expect(r2.error?.startsWith('MISS (probe)')).toBe(true)
    expect(r2.state.selection).toEqual({ kind: 'none' })
    expect(fake.pixels).toHaveLength(0)
  })

  it('post_screenshot takes a viewer screenshot and hands the PNG to saveImage', async () => {
    const h = harness()
    const r = await h.run({ type: 'post_screenshot' })
    expect(fake.calls[0]).toEqual({ type: 'screenshot', width: null, height: null, includeLegend: true })
    expect(h.saveImage).toHaveBeenCalledWith('channel (demo).png', 'AAAA')
    expect(r.ok).toBe(true)
    fake.state.datasetName = null
    await h.run({ type: 'post_screenshot' })
    expect(h.saveImage).toHaveBeenLastCalledWith('viewer.png', 'AAAA')
    fake.noImage = true
    const r2 = await h.run({ type: 'post_screenshot' })
    expect(r2.ok).toBe(false)
    expect(r2.error).toBe('NO_IMAGE (post_screenshot): the viewer returned no image')
  })

  it('without a mounted viewer set_projection, probe and post_screenshot answer NO_VIEWER at once and open the viewer tab', async () => {
    const h = harness()
    fake.mounted = false
    const r1 = await h.run({ type: 'set_projection', projection: 'Orthographic' })
    expect(r1.error?.startsWith('NO_VIEWER (set_projection)')).toBe(true)
    const r2 = await h.run({ type: 'probe', at: 'center' })
    expect(r2.error?.startsWith('NO_VIEWER (probe)')).toBe(true)
    const r3 = await h.run({ type: 'post_screenshot' })
    expect(r3.error?.startsWith('NO_VIEWER (post_screenshot)')).toBe(true)
    expect(fake.calls).toHaveLength(0)
    expect(fake.pixels).toHaveLength(0)
    expect(h.ui.openViewerTab).toHaveBeenCalledTimes(3)
  })

  it('every frame the bridge sends parses with ClientMsgSchema and viewer.viewport is reported', async () => {
    const h = harness()
    await h.run({ type: 'set_tool', tool: 'probe' })
    const r = await h.run({ type: 'probe', at: 'center' })
    expect(h.sent.length).toBe(4)
    for (const msg of h.sent) expect(ClientMsgSchema.safeParse(msg).success).toBe(true)
    expect(h.sent.filter((m) => m.t === 'ui.state')).toHaveLength(2)
    expect(r.state.viewer!.viewport).toEqual({ width: 800, height: 600 })
    expect(r.state.selection).toEqual({ kind: 'cell', id: 42, center: [1, 2, 3], value: 1.5, field: 'U', patch: null })
    fake.mounted = false
    const r2 = await h.run({ type: 'post_screenshot' })
    for (const msg of h.sent) expect(ClientMsgSchema.safeParse(msg).success).toBe(true)
    expect(r2.state.viewer!.viewport).toBeNull()
  })
})

describe('compare_run', () => {
  const resTab = (runId: string | null, compareRunId?: string | null): ResTab => ({ id: 'residuals', kind: 'residuals', runId, compareRunId })

  it('compare_run r_2 overlays and subscribes', async () => {
    const h = harness()
    h.session.runs = { r_1: {} as never, r_2: {} as never }
    h.ui.activeRunId = 'r_1'
    const r = await h.run({ type: 'compare_run', runId: 'r_2' })
    expect(r.ok).toBe(true)
    expect(r.state.compareRunId).toBe('r_2')
    expect(h.subscribeRun).toHaveBeenCalledWith('r_2')
    expect(h.ui.activeTabId).toBe('residuals')
    expect(h.ui.tabs).toEqual([resTab('r_1', 'r_2')])
  })

  it('compare_run null removes the overlay', async () => {
    const h = harness([resTab('r_1')])
    h.session.runs = { r_1: {} as never, r_2: {} as never }
    h.ui.activeRunId = 'r_1'
    expect((await h.run({ type: 'compare_run', runId: 'r_2' })).ok).toBe(true)
    const r = await h.run({ type: 'compare_run', runId: null })
    expect(r.ok).toBe(true)
    expect(r.state.compareRunId).toBeNull()
    expect(h.ui.tabs).toEqual([resTab('r_1', null)])
    expect(h.subscribeRun).toHaveBeenCalledTimes(1)
  })

  it('compare_run refuses an unknown run by name', async () => {
    const h = harness()
    h.session.runs = { r_1: {} as never }
    h.ui.activeRunId = 'r_1'
    const r = await h.run({ type: 'compare_run', runId: 'r_9' })
    expect(r.ok).toBe(false)
    expect(r.error).toBe('UNSUPPORTED (compare_run): no run "r_9" on this screen; known: r_1')
    expect(h.ui.tabs).toEqual([])
  })

  it('compare_run refuses the followed run', async () => {
    const h = harness([resTab('r_1')])
    h.session.runs = { r_1: {} as never, r_2: {} as never }
    h.ui.activeRunId = 'r_1'
    const r = await h.run({ type: 'compare_run', runId: 'r_1' })
    expect(r.ok).toBe(false)
    expect(r.error).toBe('UNSUPPORTED (compare_run): run "r_1" is the run the chart already follows; pick another or follow_run first')
  })
})

describe('the split viewport', () => {
  it('split_view true opens the viewer tab and reports views', async () => {
    const h = harness([])
    const r = await h.run({ type: 'split_view', on: true })
    expect(r.ok).toBe(true)
    expect(h.ui.openViewerTab).toHaveBeenCalled()
    expect(r.state.views).toEqual({ split: true, focused: 'A', camerasLinked: false, datasetA: 'ds1', datasetB: 'dsB' })
    const off = await h.run({ type: 'split_view', on: false })
    expect(off.ok).toBe(true)
    expect(off.state.views).toBeNull()
  })

  it('open_result_in_view B loads on the B api only', async () => {
    const h = harness([])
    const r = await h.run({ type: 'open_result_in_view', view: 'B', path: 'cases/plume.vtu' })
    expect(r.ok).toBe(true)
    expect(fakeB.calls).toEqual([{ type: 'load', path: 'cases/plume.vtu', timeIndex: 'last', field: null }])
    expect(fake.calls.some((c) => (c as { type: string }).type === 'load')).toBe(false)
    expect(h.ui.viewSplit).toBe(true)
    expect(h.ui.openViewerTab).toHaveBeenCalled()
    expect(r.state.views!.split).toBe(true)
  })

  it('link_cameras true mirrors A\'s camera into B', async () => {
    const h = harness([])
    await h.run({ type: 'split_view', on: true })
    const r = await h.run({ type: 'link_cameras', on: true })
    expect(r.ok).toBe(true)
    expect(h.ui.camerasLinked).toBe(true)
    fakeApi.emit({ ...fake.state, camera: { position: [1, 2, 3], target: [0, 0, 0], projection: 'orthographic' } })
    expect(fakeB.calls).toContainEqual({ type: 'setCamera', preset: null, position: [1, 2, 3], target: [0, 0, 0], projection: 'orthographic' })
    expect(fake.calls.some((c) => (c as { type: string }).type === 'setCamera')).toBe(false)
  })

  it('focus_view B re-links with B as leader', async () => {
    const h = harness([])
    await h.run({ type: 'split_view', on: true })
    await h.run({ type: 'link_cameras', on: true })
    const r = await h.run({ type: 'focus_view', view: 'B' })
    expect(r.ok).toBe(true)
    expect(h.ui.focusedView).toBe('B')
    // the re-link makes B the leader: B's current pose is pushed into A once
    const pushed = fake.calls.filter((c) => (c as { type: string }).type === 'setCamera')
    expect(pushed).toEqual([{ type: 'setCamera', preset: null, position: [10, 10, 10], target: [0, 0, 0], projection: 'perspective' }])
    fakeApiB.emit({ ...fakeB.state, camera: { position: [4, 5, 6], target: [0, 1, 0], projection: 'perspective' } })
    expect(fake.calls).toContainEqual({ type: 'setCamera', preset: null, position: [4, 5, 6], target: [0, 1, 0], projection: 'perspective' })
  })

  it('focus_view and link_cameras refuse when not split', async () => {
    const h = harness([])
    const f = await h.run({ type: 'focus_view', view: 'B' })
    expect(f.ok).toBe(false)
    expect(f.error).toBe('UNSUPPORTED (focus_view): the viewport is not split; send split_view true first')
    const l = await h.run({ type: 'link_cameras', on: true })
    expect(l.ok).toBe(false)
    expect(l.error).toBe('UNSUPPORTED (link_cameras): the viewport is not split; send split_view true first')
  })

  it('split_view false unlinks and clears views', async () => {
    const h = harness([])
    await h.run({ type: 'split_view', on: true })
    await h.run({ type: 'link_cameras', on: true })
    const r = await h.run({ type: 'split_view', on: false })
    expect(r.ok).toBe(true)
    expect(r.state.views).toBeNull()
    expect(h.ui.viewSplit).toBe(false)
    expect(h.ui.focusedView).toBe('A')
    expect(h.ui.camerasLinked).toBe(false)
    const before = fakeB.calls.length
    fakeApi.emit({ ...fake.state, camera: { position: [9, 9, 9], target: [0, 0, 0], projection: 'perspective' } })
    expect(fakeB.calls).toHaveLength(before)
  })
})
