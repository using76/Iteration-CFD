// Unit tests for the bridge's four geometry edit commands (E1 Run 2): the
// refusals, the store round-trip and the exact ui.state / ui.result frames.
import { beforeEach, describe, expect, it, vi } from 'vitest'
vi.mock('../viewer', () => ({ getViewerApi: () => ({ execute: async () => ({ ok: true }) }) }))
vi.mock('./actions', () => ({ actions: { setSettings: () => true, openSession: () => true }, activeCasePath: () => null }))
vi.mock('../api/rest', () => ({ api: { geometrySave: vi.fn(), geometryEdit: vi.fn() }, ApiError: class extends Error {} }))
import type { ClientMsg, UiCommand, UiState } from '@cfd/shared'
import type { GeometryBounds, GeometryInfo } from '@cfd/shared'
import { api } from '../api/rest'
import { uiGeometryState, useGeometryStore } from '../state/geometryStore'
import { createUiBridge } from './uiBridge'

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
type ResultFrame = Extract<ClientMsg, { t: 'ui.result' }>
type RunResult = ResultFrame & { state: UiState }

const geoTab = (path: string): GeoTab => ({ id: `geometry:${path}`, kind: 'geometry', path })

function harness(tabs: GeoTab[]) {
  const sent: ClientMsg[] = []
  const ui = { tabs, activeTabId: tabs[0]?.id ?? null, assistantVisible: false, activeRunId: null, locale: 'en' as const, openGeometryTab: vi.fn(), openViewerTab: vi.fn() }
  const session = { problems: {}, connection: 'online', addNote: vi.fn() }
  const bridge = createUiBridge({
    send: (m) => {
      sent.push(m)
      return true
    },
    ui: { getState: () => ui as never },
    session: { getState: () => session as never },
    subscribeRun: () => {},
  })
  return {
    sent,
    ui,
    session,
    async run(cmd: UiCommand): Promise<RunResult> {
      await bridge.handleCommand('r1', cmd)
      const last = sent[sent.length - 1] as ResultFrame
      if (last.t !== 'ui.result' || last.state == null) throw new Error('no ui.result with state was sent')
      return last as RunResult
    },
  }
}

const st = () => useGeometryStore.getState()

function load(path: string, info: GeometryInfo) {
  st().begin(path)
  st().setLoaded(path, { id: info.id, info })
}

beforeEach(() => {
  useGeometryStore.setState({ byPath: {} })
  vi.mocked(api.geometrySave).mockReset()
  vi.mocked(api.geometryEdit).mockReset()
  vi.mocked(api.geometrySave).mockResolvedValue({ id: 'g2', info: { ...INFO, id: 'g2', path: 'cases/y.stl' } })
  vi.mocked(api.geometryEdit).mockResolvedValue({ path: 'cases/car-common.step', solids: [], applied: [], stdout: [], toolMs: 1 })
})

describe('the four geometry edit commands', () => {
  it('an unlisted command is still refused by name', async () => {
    const h = harness([])
    const r = await h.run({ type: 'split_view', on: true })
    expect(r.ok).toBe(false)
    expect(r.error).toBe('UNSUPPORTED (split_view): this screen does not implement it')
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
