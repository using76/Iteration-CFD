// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// STU-12b: on a dataset with no lattice that the server cuts by a plane (manifest.planeCut), add_layer
// slice and plane succeed, the layer's triangles carry exactly the cells the server's cut named, and
// each triangle's value is its cell's field value bit for bit. The lattice path does not change.
import { describe, expect, it, vi } from 'vitest'
import type { ClientMsg, CutPlaneSpec, DatasetOpenResponse, PlaneCutInfo, ViewerDataset } from '@cfd/shared'
import { controllerViewerApi } from '../api/viewerApi'
import { SYNTHETIC_CHANNEL_PATH, buildSyntheticChannel } from '../data/syntheticDataset'
import { SyntheticTransport, type DatasetTransport, type SyntheticDataset } from '../data/transport'
import { InlineCompute } from '../worker/Compute'
import { scalarOf } from '../worker/sampling'
import { useUiStore } from '../../state/uiStore'
import { createUiBridge, layerCommand } from '../../ws/uiBridge'
import { ViewerController } from './ViewerController'

// The bridge's own imports reach the window; the same three stand-ins as uiBridge.layers.test.
const holder = vi.hoisted(() => ({ api: null as unknown }))
vi.mock('../index', () => ({ getViewerApi: () => holder.api, setViewerApi: () => {} }))
vi.mock('../../ws/actions', () => ({ actions: { setSettings: () => true, openSession: () => true }, activeCasePath: () => null }))
vi.mock('../../api/rest', () => ({ api: { geometrySave: vi.fn(), geometryEdit: vi.fn() }, ApiError: class extends Error {} }))

const POLY_PATH = 'cases/warped-sphere'
const BOUNDS = { min: [-1, -1, -1] as [number, number, number], max: [1, 1, 1] as [number, number, number] }
/** The cells of the cut's triangles: some cells give several triangles (a polygon fanned). */
const CUT_CELLS = [3, 3, 7, 11, 11, 11, 20, 23]

/** A non-lattice polyMesh as the window sees it: no grid, planeCut true, and a server that cuts. */
class CutTransport implements DatasetTransport {
  readonly calls: CutPlaneSpec[] = []
  private readonly inner: SyntheticTransport
  private seq = 0

  constructor(private readonly data: SyntheticDataset[]) {
    this.inner = new SyntheticTransport(data)
  }

  open(path: string): Promise<DatasetOpenResponse> {
    return this.inner.open(path)
  }

  manifest(id: string): Promise<ViewerDataset | null> {
    return this.inner.manifest(id)
  }

  blob(id: string, key: string): Promise<ArrayBuffer> {
    return this.inner.blob(id, key)
  }

  async cut(id: string, plane: CutPlaneSpec): Promise<PlaneCutInfo> {
    this.calls.push(plane)
    const d = this.data.find((x) => x.manifest.id === id)!
    // A plane outside the box cuts nothing; any other plane cuts CUT_CELLS.
    const outside = plane.origin.some((v) => Math.abs(v) > 1)
    const cells = Uint32Array.from(outside ? [] : CUT_CELLS)
    const positions = new Float32Array(9 * cells.length)
    for (let t = 0; t < cells.length; t++) positions.set([plane.origin[0], t, 0, plane.origin[0], t + 1, 0, plane.origin[0], t, 1], 9 * t)
    const key = `cut.t${++this.seq}`
    d.blobs.set(`${key}.positions`, positions)
    d.blobs.set(`${key}.cells`, cells)
    return {
      key,
      origin: plane.origin,
      normal: plane.normal,
      triangleCount: cells.length,
      positions: { key: `${key}.positions`, dtype: 'f32', count: 3 * cells.length, components: 3, bytes: positions.byteLength },
      cells: { key: `${key}.cells`, dtype: 'u32', count: cells.length, components: 1, bytes: cells.byteLength },
      cellCount: new Set(cells).size,
      area: 0.5 * cells.length,
      openCells: 0,
    }
  }
}

function polyDataset(planeCut: boolean, path = POLY_PATH): SyntheticDataset {
  const d = buildSyntheticChannel({ id: `poly-${planeCut}`, path, dims: [4, 3, 2], bounds: BOUNDS })
  d.manifest.source = 'polymesh'
  d.manifest.grid = null
  d.manifest.planeCut = planeCut
  return d
}

function headless(transport: DatasetTransport) {
  return new ViewerController({ transport, createCompute: (provider) => new InlineCompute(provider), requireMount: false, loader: { pollMs: 1, sleep: async () => {} } })
}

function fieldData(d: SyntheticDataset, name: string): Float32Array {
  const ref = d.manifest.fields.find((f) => f.name === name)!.perTime.at(-1)!.blob
  return d.blobs.get(ref.key) as Float32Array
}

/** gui_control's add_layer through the real bridge, answered as its ui.result frame. */
async function bridgeAddLayer(c: ViewerController, kind: string, args: Record<string, unknown>) {
  holder.api = controllerViewerApi(c)
  const sent: ClientMsg[] = []
  const bridge = createUiBridge({
    send: (m) => {
      sent.push(m)
      return true
    },
    ui: useUiStore,
    session: { getState: () => ({ problems: {}, connection: 'online', addNote: vi.fn(), runs: {}, runData: {} }) as never },
    subscribeRun: vi.fn(),
    saveImage: vi.fn(),
  })
  await bridge.handleCommand('r1', { type: 'add_layer', kind, args })
  const r = sent.find((m): m is Extract<ClientMsg, { t: 'ui.result' }> => m.t === 'ui.result' && m.requestId === 'r1')
  if (!r) throw new Error('the bridge sent no ui.result')
  return r
}

function addLayer(c: ViewerController, kind: 'slice' | 'plane', args: Record<string, unknown>) {
  const s = c.getState()
  const built = layerCommand(kind, args, { field: s.field ?? null, bounds: s.bounds ?? null })
  if (!built.ok) throw new Error(built.error)
  return c.execute(built.cmd)
}

describe('slices and planes on a non-lattice polyMesh', () => {
  it('add_layer slice {axis:"x", position:0} cuts on the server and colours every triangle by its cell', async () => {
    const d = polyDataset(true)
    const t = new CutTransport([d])
    const c = headless(t)
    expect((await c.execute({ type: 'load', path: POLY_PATH, timeIndex: null, field: null })).ok).toBe(true)
    expect(c.getState().field).toBe('U')

    const r = await bridgeAddLayer(c, 'slice', { id: 's', axis: 'x', position: 0 })
    expect(r.ok).toBe(true)
    expect(t.calls).toEqual([{ origin: [0, 0, 0], normal: [1, 0, 0] }])
    expect(c.getState().layers).toEqual([{ id: 's', type: 'slice', summary: 'slice x=0 (cut, 8 tris)' }])
    const res = c.layers.get('s')!.result!
    if (res.op !== 'cut') throw new Error(`expected a cut, got ${res.op}`)
    expect([...res.cells]).toEqual(CUT_CELLS)
    expect(new Set(res.cells)).toEqual(new Set(CUT_CELLS))
    expect(res.positions.length).toBe(9 * CUT_CELLS.length)
    const speed = scalarOf(fieldData(d, 'U'), 3, 'magnitude')
    for (let k = 0; k < CUT_CELLS.length; k++) expect(Object.is(res.values[k], speed[CUT_CELLS[k]])).toBe(true)
    // Flat per cell: the three vertices of a triangle share one opaque colour.
    for (let k = 0; k < CUT_CELLS.length; k++) {
      const o = 12 * k
      expect([...res.colors.subarray(o + 4, o + 8)]).toEqual([...res.colors.subarray(o, o + 4)])
      expect([...res.colors.subarray(o + 8, o + 12)]).toEqual([...res.colors.subarray(o, o + 4)])
      expect(res.colors[o + 3]).toBe(255)
    }

    // A new field recolours the same cut: no second request, the values are p's own.
    expect((await c.execute({ type: 'setField', field: 'p', component: null, range: null, colormap: null, log: null })).ok).toBe(true)
    expect(t.calls.length).toBe(1)
    const again = c.layers.get('s')!.result!
    if (again.op !== 'cut') throw new Error('expected a cut')
    const p = fieldData(d, 'p')
    for (let k = 0; k < CUT_CELLS.length; k++) expect(Object.is(again.values[k], p[CUT_CELLS[k]])).toBe(true)
  })

  it('add_layer plane cuts with its own origin and normal; a slice on the min wall faces away from the domain', async () => {
    const t = new CutTransport([polyDataset(true)])
    const c = headless(t)
    await c.execute({ type: 'load', path: POLY_PATH, timeIndex: null, field: 'p' })
    expect((await addLayer(c, 'plane', { id: 'pl', origin: [0.25, 0, 0.5], normal: 'y' })).ok).toBe(true)
    expect(t.calls.at(-1)).toEqual({ origin: [0.25, 0, 0.5], normal: [0, 1, 0] })
    expect(c.layers.get('pl')!.result!.op).toBe('cut')
    expect((await addLayer(c, 'slice', { id: 'lo', axis: 'z', fraction: 0 })).ok).toBe(true)
    expect(t.calls.at(-1)).toEqual({ origin: [0, 0, -1], normal: [0, 0, -1] })
    expect((await addLayer(c, 'slice', { id: 'hi', axis: 'z', fraction: 1 })).ok).toBe(true)
    expect(t.calls.at(-1)).toEqual({ origin: [0, 0, 1], normal: [0, 0, 1] })
    const miss = await addLayer(c, 'plane', { id: 'out', origin: [0, 3, 0], normal: 'y' })
    expect(miss.ok).toBe(false)
    expect(miss.error?.code).toBe('INVALID')
    expect(miss.error?.message).toContain('does not intersect')
  })

  it('iso-surfaces and streamlines still need a lattice, and a mesh the server cannot cut has no slices', async () => {
    const t = new CutTransport([polyDataset(true), polyDataset(false, 'cases/vtu-cloud')])
    const c = headless(t)
    await c.execute({ type: 'load', path: POLY_PATH, timeIndex: null, field: 'p' })
    const iso = await c.execute({ type: 'addIsoSurface', field: 'p', values: [0] })
    expect(iso.error?.code).toBe('NO_STRUCTURED_GRID')
    const sl = await c.execute({ type: 'addStreamlines', field: 'U', seed: { line: [[0, 0, 0], [0, 1, 0]], count: 4 } })
    expect(sl.error?.code).toBe('NO_STRUCTURED_GRID')
    await c.execute({ type: 'load', path: 'cases/vtu-cloud', timeIndex: null, field: 'p' })
    const none = await addLayer(c, 'slice', { axis: 'x', position: 0 })
    expect(none.error?.code).toBe('NO_STRUCTURED_GRID')
    expect(t.calls).toEqual([])
  })

  it('a lattice dataset keeps its image slices and never asks for a cut', async () => {
    const lattice = buildSyntheticChannel({ dims: [8, 4, 4] })
    lattice.manifest.planeCut = true
    const t = new CutTransport([lattice])
    const c = headless(t)
    await c.execute({ type: 'load', path: SYNTHETIC_CHANNEL_PATH, timeIndex: null, field: null })
    expect((await addLayer(c, 'slice', { id: 'img', axis: 'x', position: 1 })).ok).toBe(true)
    expect(c.layers.get('img')!.result!.op).toBe('slice')
    expect(c.getState().layers[0].summary).toBe('slice x=1')
    expect(t.calls).toEqual([])
  })
})
