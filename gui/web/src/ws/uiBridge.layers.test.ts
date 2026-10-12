// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// STU-04: add_layer / remove_layer drive a REAL headless viewer controller through
// the bridge; every kind lands as a layer, refusals name the field, and every
// sent frame stays schema-valid.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ClientMsgSchema, type ClientMsg, type UiCommand } from '@cfd/shared'
import { SYNTHETIC_CHANNEL_PATH, buildSyntheticChannel } from '../viewer/data/syntheticDataset'
import { SyntheticTransport } from '../viewer/data/transport'
import { InlineCompute } from '../viewer/worker/Compute'
import { ViewerController } from '../viewer/controller/ViewerController'
import { controllerViewerApi, type ViewerApi } from '../viewer/api/viewerApi'
import { useUiStore } from '../state/uiStore'
import { createUiBridge, layerCommand, layerKind } from './uiBridge'

const holder = vi.hoisted(() => ({ api: null as unknown }))
vi.mock('../viewer', () => ({ getViewerApi: () => holder.api, setViewerApi: () => {} }))
vi.mock('./actions', () => ({ actions: { setSettings: () => true, openSession: () => true }, activeCasePath: () => null }))
vi.mock('../api/rest', () => ({ api: { geometrySave: vi.fn(), geometryEdit: vi.fn() }, ApiError: class extends Error {} }))

const session = { problems: {}, connection: 'online', addNote: vi.fn(), runs: {} as Record<string, unknown>, runData: {} as Record<string, unknown> }

let sent: ClientMsg[] = []
let n = 0
const subscribeRun = vi.fn()

function makeBridge(cameraBudgetMs?: number) {
  return createUiBridge({
    send: (m) => {
      sent.push(m)
      return true
    },
    ui: useUiStore,
    session: { getState: () => session as never },
    subscribeRun,
    saveImage: vi.fn(),
    cameraBudgetMs,
  })
}

beforeEach(() => {
  sent = []
  n = 0
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

/** A headless controller over the synthetic channel, exactly like ViewerController.test's headless(). */
function headlessApi(): ViewerApi {
  const transport = new SyntheticTransport([buildSyntheticChannel({ dims: [24, 12, 8] }), buildSyntheticChannel({ dims: [24, 12, 1] })])
  const controller = new ViewerController({
    transport,
    createCompute: (provider) => new InlineCompute(provider),
    requireMount: false,
    loader: { pollMs: 1, sleep: async () => {} },
  })
  return controllerViewerApi(controller)
}

/** A headless controller with the synthetic channel already loaded; the bridge talks to it. */
async function loadedApi(): Promise<ViewerApi> {
  const api = headlessApi()
  const r = await api.execute({ type: 'load', path: SYNTHETIC_CHANNEL_PATH, timeIndex: null, field: null })
  expect(r.ok).toBe(true)
  holder.api = api
  return api
}

type Drive = Awaited<ReturnType<typeof drive>>
const layersOf = (r: Drive) => r.state?.viewer?.layers ?? []

describe('add_layer / remove_layer (STU-04)', () => {
  it('L1: the five kinds land as viewer layers, in order', async () => {
    await loadedApi()
    const bridge = makeBridge()
    const r1 = await drive(bridge, { type: 'add_layer', kind: 'slice', args: { axis: 'Z', position: '{"fraction":0.5}' } })
    expect(r1.ok).toBe(true)
    expect(layersOf(r1)).toHaveLength(1)
    expect(layersOf(r1)[0]).toEqual({ id: 'slice-1', type: 'slice', summary: 'slice z=0.5' })
    const r2 = await drive(bridge, { type: 'add_layer', kind: 'Plane', args: { normal: 'x' } })
    expect(r2.ok).toBe(true)
    expect(layersOf(r2)).toHaveLength(2)
    expect(layersOf(r2)[1]).toEqual({ id: 'plane-2', type: 'plane', summary: 'plane o=(1.5,0.75,0.5) n=(1,0,0)' })
    const r3 = await drive(bridge, { type: 'add_layer', kind: 'iso-surface', args: { value: '1.2' } })
    expect(r3.ok).toBe(true)
    expect(layersOf(r3)).toHaveLength(3)
    expect(layersOf(r3)[2].id).toBe('iso-3')
    expect(layersOf(r3)[2].type).toBe('iso')
    const m3 = /^iso U=1\.2 \((\d+) tris\)$/.exec(layersOf(r3)[2].summary)
    expect(m3).toBeTruthy()
    expect(Number(m3![1])).toBeGreaterThan(0)
    const r4 = await drive(bridge, { type: 'add_layer', kind: 'Streamlines', args: { seed: '{"plane":"x","position":{"fraction":0.05},"grid":[4,3]}' } })
    expect(r4.ok).toBe(true)
    expect(layersOf(r4)).toHaveLength(4)
    expect(layersOf(r4)[3]).toEqual({ id: 'streamlines-4', type: 'streamlines', summary: 'streamlines U n=12' })
    const r5 = await drive(bridge, { type: 'add_layer', kind: 'glyphs', args: { id: 'g', stride: '3', onSlice: 'slice-1' } })
    expect(r5.ok).toBe(true)
    expect(layersOf(r5)).toHaveLength(5)
    expect(layersOf(r5)[4].id).toBe('g')
    expect(layersOf(r5)[4].type).toBe('glyphs')
    expect(layersOf(r5)[4].summary).toMatch(/^glyphs U stride=3 n=\d+ on slice-1$/)
    expect(layersOf(r5).map((l) => l.id)).toEqual(['slice-1', 'plane-2', 'iso-3', 'streamlines-4', 'g'])
    const lastState = [...sent].reverse().find((m): m is Extract<ClientMsg, { t: 'ui.state' }> => m.t === 'ui.state')
    expect(lastState?.state.viewer?.layers).toEqual(layersOf(r5))
    expect(useUiStore.getState().tabs.some((t) => t.kind === 'viewer')).toBe(true)
  })

  it('L2: remove_layer removes by id and names the survivors', async () => {
    await loadedApi()
    const bridge = makeBridge()
    await drive(bridge, { type: 'add_layer', kind: 'slice', args: { axis: 'x' } })
    await drive(bridge, { type: 'add_layer', kind: 'plane', args: { normal: 'y' } })
    const r = await drive(bridge, { type: 'remove_layer', id: 'plane-2' })
    expect(r.ok).toBe(true)
    expect(layersOf(r).map((l) => l.id)).toEqual(['slice-1'])
    const bad = await drive(bridge, { type: 'remove_layer', id: 'nope' })
    expect(bad.ok).toBe(false)
    expect(bad.error).toBe('NO_SUCH_LAYER (remove_layer): no layer "nope"; layers: slice-1')
    const r2 = await drive(bridge, { type: 'remove_layer', id: 'slice-1' })
    expect(r2.ok).toBe(true)
    expect(layersOf(r2)).toEqual([])
    const bad2 = await drive(bridge, { type: 'remove_layer', id: 'slice-1' })
    expect(bad2.ok).toBe(false)
    expect(bad2.error).toBe('NO_SUCH_LAYER (remove_layer): no layer "slice-1"; layers: none')
  })

  it('L3: an unknown kind refuses with the valid list', async () => {
    await loadedApi()
    const bridge = makeBridge()
    const r = await drive(bridge, { type: 'add_layer', kind: 'tube', args: {} })
    expect(r.ok).toBe(false)
    expect(r.error).toBe('INVALID (add_layer): unknown kind "tube"; kinds: slice, plane, isoSurface, streamlines, glyphs')
    expect(layersOf(r)).toEqual([])
  })

  it('L4: bad args refuse naming the field, layers unchanged', async () => {
    await loadedApi()
    const bridge = makeBridge()
    expect(layersOf(await drive(bridge, { type: 'add_layer', kind: 'slice', args: { axis: 'x' } }))).toHaveLength(1)
    const r1 = await drive(bridge, { type: 'add_layer', kind: 'slice', args: {} })
    expect(r1.ok).toBe(false)
    expect(r1.error?.startsWith('INVALID (add_layer): ')).toBe(true)
    expect(r1.error).toContain('args.axis')
    const r2 = await drive(bridge, { type: 'add_layer', kind: 'slice', args: { axis: 'w' } })
    expect(r2.ok).toBe(false)
    expect(r2.error).toContain('args.axis')
    const r3 = await drive(bridge, { type: 'add_layer', kind: 'slice', args: { axis: 'x', postion: 0.5 } })
    expect(r3.ok).toBe(false)
    expect(r3.error).toBe('INVALID (add_layer): args.postion is not an option of slice; options: id, axis, position, fraction')
    const r4 = await drive(bridge, { type: 'add_layer', kind: 'plane', args: { normal: [0, 0, 0] } })
    expect(r4.ok).toBe(false)
    expect(r4.error).toBe('INVALID (add_layer): plane normal must be non-zero')
    const r5 = await drive(bridge, { type: 'add_layer', kind: 'streamlines', args: {} })
    expect(r5.ok).toBe(false)
    expect(r5.error).toContain('args.seed')
    const r6 = await drive(bridge, { type: 'add_layer', kind: 'isoSurface', args: { values: 'abc' } })
    expect(r6.ok).toBe(false)
    expect(r6.error).toContain('args.values')
    expect(layersOf(await drive(bridge, { type: 'add_layer', kind: 'slice', args: { axis: 'y' } }))).toHaveLength(2)
  })

  it('L5: no dataset refuses NO_DATASET', async () => {
    holder.api = headlessApi()
    const bridge = makeBridge()
    const r = await drive(bridge, { type: 'add_layer', kind: 'slice', args: { axis: 'x' } })
    expect(r.ok).toBe(false)
    expect(r.error).toBe('NO_DATASET (add_layer): no dataset is loaded; run load first')
  })

  it('L6: a silent viewer answers BUSY inside the budget', async () => {
    holder.api = {
      execute: () => new Promise<never>(() => {}),
      getState: () => ({
        camera: { position: [10, 10, 10], target: [0, 0, 0], projection: 'perspective' },
        datasetId: 'ds1',
        datasetName: null,
        field: 'U',
        time: null,
        bounds: null,
        layers: [],
        colormap: 'viridis',
        range: null,
        representation: 'surface',
      }),
      subscribe: () => () => {},
      isMounted: () => true,
      setBaseUrl: () => {},
      setTool: () => {},
      getTool: () => 'select',
      viewport: () => null,
      probePixel: () => ({ ok: false, code: 'NO_VIEWER', message: 'no' }),
      probePoint: () => ({ ok: false, code: 'NO_VIEWER', message: 'no' }),
    }
    const bridge = makeBridge(20)
    const t0 = Date.now()
    const r = await drive(bridge, { type: 'add_layer', kind: 'slice', args: { axis: 'x' } })
    expect(Date.now() - t0).toBeLessThan(1000)
    expect(r.ok).toBe(false)
    expect(r.error?.startsWith('BUSY (add_layer): ')).toBe(true)
  })
})

describe('layer helpers (STU-04)', () => {
  it('L7: layerKind and layerCommand coerce the loose forms', () => {
    expect(layerKind('Iso-Surface')).toBe('isoSurface')
    expect(layerKind('ARROWS')).toBe('glyphs')
    expect(layerKind('stream traces')).toBe('streamlines')
    expect(layerKind('cut plane')).toBe('plane')
    expect(layerKind('tube')).toBe(null)
    expect(layerCommand('plane', { normal: 'Y', origin: '[1, 0.5, 0.25]' }, { field: null, bounds: null })).toEqual({
      ok: true,
      cmd: { type: 'addPlane', id: null, origin: [1, 0.5, 0.25], normal: [0, 1, 0] },
    })
    const noOrigin = layerCommand('plane', { normal: 'z' }, { field: null, bounds: null })
    expect(noOrigin.ok).toBe(false)
    if (noOrigin.ok) throw new Error('expected a refusal')
    expect(noOrigin.error).toContain('args.origin')
    expect(layerCommand('isoSurface', { values: ['1', '2'] }, { field: 'p', bounds: null })).toEqual({
      ok: true,
      cmd: { type: 'addIsoSurface', id: null, field: 'p', values: [1, 2] },
    })
    expect(layerCommand('slice', { axis: 'X', fraction: '0.25', id: 7 }, { field: null, bounds: null })).toEqual({
      ok: true,
      cmd: { type: 'addSlice', id: '7', axis: 'x', position: { fraction: 0.25 } },
    })
  })
})

