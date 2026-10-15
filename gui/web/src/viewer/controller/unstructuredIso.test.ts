// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// STU-14: on a dataset with no lattice that the server contours (manifest.planeCut and a transport
// with iso), add_layer iso asks the server for the field at the current time and draws exactly the
// served triangles and normals. The lattice keeps its marching cubes, and the cut-cell and 2-D
// refusals do not change.
import { describe, expect, it, vi } from 'vitest'
import type { ClientMsg, DatasetOpenResponse, IsoSurfaceInfo, IsoSurfaceSpec, ViewerDataset } from '@cfd/shared'
import { controllerViewerApi } from '../api/viewerApi'
import { SYNTHETIC_CHANNEL_PATH, buildSyntheticChannel } from '../data/syntheticDataset'
import { SyntheticTransport, type DatasetTransport, type SyntheticDataset } from '../data/transport'
import { InlineCompute } from '../worker/Compute'
import { useUiStore } from '../../state/uiStore'
import { createUiBridge } from '../../ws/uiBridge'
import { ViewerController } from './ViewerController'

const holder = vi.hoisted(() => ({ api: null as unknown }))
vi.mock('../index', () => ({ getViewerApi: () => holder.api, setViewerApi: () => {} }))
vi.mock('../../ws/actions', () => ({ actions: { setSettings: () => true, openSession: () => true }, activeCasePath: () => null }))
vi.mock('../../api/rest', () => ({ api: { geometrySave: vi.fn(), geometryEdit: vi.fn() }, ApiError: class extends Error {} }))

const POLY_PATH = 'cases/warped-sphere'
const BOUNDS = { min: [-1, -1, -1] as [number, number, number], max: [1, 1, 1] as [number, number, number] }

/** A non-lattice polyMesh whose server contours: each request gets 1 + its value count triangles. */
class IsoTransport implements DatasetTransport {
  readonly calls: IsoSurfaceSpec[] = []
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

  async iso(id: string, spec: IsoSurfaceSpec): Promise<IsoSurfaceInfo> {
    this.calls.push(spec)
    const d = this.data.find((x) => x.manifest.id === id)!
    const n = 1 + spec.values.length
    const positions = new Float32Array(9 * n)
    const normals = new Float32Array(9 * n)
    for (let t = 0; t < n; t++) {
      positions.set([spec.timeIndex, t, 0, spec.timeIndex, t + 1, 0, spec.timeIndex, t, 1], 9 * t)
      normals.set([1, 0, 0, 1, 0, 0, 1, 0, 0], 9 * t)
    }
    const cells = Uint32Array.from({ length: n }, (_, t) => 2 * t)
    const key = `iso.t${++this.seq}`
    d.blobs.set(`${key}.positions`, positions)
    d.blobs.set(`${key}.normals`, normals)
    d.blobs.set(`${key}.cells`, cells)
    const ref = (k: string, dtype: 'f32' | 'u32', count: number, components: number, bytes: number) => ({ key: `${key}.${k}`, dtype, count, components, bytes })
    return {
      key,
      field: spec.field,
      timeIndex: spec.timeIndex,
      values: spec.values,
      triangleCount: n,
      positions: ref('positions', 'f32', 3 * n, 3, positions.byteLength),
      normals: ref('normals', 'f32', 3 * n, 3, normals.byteLength),
      cells: ref('cells', 'u32', n, 1, cells.byteLength),
      valueCounts: spec.values.map((_, k) => (k === 0 ? 2 : 1)),
      area: n,
      fallbackPoints: 0,
      invertedTets: 0,
    }
  }
}

function polyDataset(timeCount = 1): SyntheticDataset {
  const d = buildSyntheticChannel({ id: 'poly-iso', path: POLY_PATH, dims: [4, 3, 2], bounds: BOUNDS, times: timeCount })
  d.manifest.source = 'polymesh'
  d.manifest.grid = null
  d.manifest.planeCut = true
  return d
}

function headless(transport: DatasetTransport) {
  return new ViewerController({ transport, createCompute: (provider) => new InlineCompute(provider), requireMount: false, loader: { pollMs: 1, sleep: async () => {} } })
}

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

describe('iso-surfaces on a non-lattice polyMesh', () => {
  it('add_layer iso asks the server and draws exactly its triangles and normals', async () => {
    const d = polyDataset(2)
    const t = new IsoTransport([d])
    const c = headless(t)
    expect((await c.execute({ type: 'load', path: POLY_PATH, timeIndex: null, field: 'p' })).ok).toBe(true)
    const r = await bridgeAddLayer(c, 'iso', { id: 'iso', field: 'p', values: [0.25, 0.5] })
    expect(r.ok).toBe(true)
    expect(t.calls).toEqual([{ field: 'p', timeIndex: 1, values: [0.25, 0.5] }])
    expect(c.getState().layers).toEqual([{ id: 'iso', type: 'iso', summary: 'iso p=0.25,0.5 (3 tris)' }])
    const res = c.layers.get('iso')!.result!
    if (res.op !== 'iso') throw new Error(`expected an iso result, got ${res.op}`)
    expect(res.triangleCount).toBe(3)
    expect([...res.positions]).toEqual([...(d.blobs.get('iso.t1.positions') as Float32Array)])
    expect([...res.normals]).toEqual([...(d.blobs.get('iso.t1.normals') as Float32Array)])

    // A new time contours again at that time; the same time again is the stored surface.
    expect((await c.execute({ type: 'setTime', index: 0 })).ok).toBe(true)
    expect(t.calls.at(-1)).toEqual({ field: 'p', timeIndex: 0, values: [0.25, 0.5] })
    const atZero = c.layers.get('iso')!.result!
    if (atZero.op !== 'iso') throw new Error('expected an iso result')
    expect(atZero.positions[0]).toBe(0)
    expect((await c.execute({ type: 'setTime', index: 1 })).ok).toBe(true)
    expect(t.calls.length).toBe(2)
  })

  it('a server refusal answers LOAD_FAILED naming it, and an unknown field is refused before any request', async () => {
    const d = polyDataset()
    const t = new IsoTransport([d])
    t.iso = async () => {
      throw new Error('INVALID: p has 6 components')
    }
    const c = headless(t)
    await c.execute({ type: 'load', path: POLY_PATH, timeIndex: null, field: 'p' })
    const bad = await c.execute({ type: 'addIsoSurface', field: 'p', values: [0] })
    expect(bad.error?.code).toBe('LOAD_FAILED')
    expect(bad.error?.message).toContain('INVALID: p has 6 components')
    const none = await c.execute({ type: 'addIsoSurface', field: 'nope', values: [0] })
    expect(none.ok).toBe(false)
    expect(c.getState().layers).toEqual([])
  })

  it('the lattice keeps marching cubes, the cut-cell and 2-D refusals stay, and streamlines still need a lattice', async () => {
    const lattice = buildSyntheticChannel({ dims: [8, 4, 4] })
    lattice.manifest.planeCut = true
    const t = new IsoTransport([lattice, polyDataset()])
    const c = headless(t)
    await c.execute({ type: 'load', path: SYNTHETIC_CHANNEL_PATH, timeIndex: null, field: 'p' })
    expect((await c.execute({ type: 'addIsoSurface', id: 'mc', field: 'p', values: [0] })).ok).toBe(true)
    expect(c.layers.get('mc')!.result!.op).toBe('iso')
    expect(t.calls).toEqual([])

    const cutCell = buildSyntheticChannel({ id: 'cutcell', path: 'cases/cutcell', dims: [4, 3, 2] })
    const n = 4 * 3 * 2
    cutCell.blobs.set('grid.index', Uint32Array.from({ length: n }, (_, i) => i))
    cutCell.manifest.grid!.index = { key: 'grid.index', dtype: 'u32', count: n, components: 1, bytes: 4 * n }
    const c2 = headless(new IsoTransport([cutCell]))
    await c2.execute({ type: 'load', path: 'cases/cutcell', timeIndex: null, field: 'p' })
    const cc = await c2.execute({ type: 'addIsoSurface', field: 'p', values: [0] })
    expect(cc.error?.code).toBe('NO_STRUCTURED_GRID')
    expect(cc.error?.message).toContain('ISO_CUTCELL')

    await c.execute({ type: 'load', path: POLY_PATH, timeIndex: null, field: 'p' })
    const sl = await c.execute({ type: 'addStreamlines', field: 'U', seed: { line: [[0, 0, 0], [0, 1, 0]], count: 4 } })
    expect(sl.error?.code).toBe('NO_STRUCTURED_GRID')
    expect(sl.error?.message).toContain('iso-surfaces are computed on its cells here')
  })
})
