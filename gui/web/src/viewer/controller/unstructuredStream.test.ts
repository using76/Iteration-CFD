// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// STU-15: on a dataset with no lattice that the server traces (manifest.planeCut and a transport
// with streamlines), add_layer streamlines asks the server with the layer's seeds at the current
// time and draws exactly the served points, speeds and offsets. Every seed outside answers
// INVALID; any other server refusal answers LOAD_FAILED. A lattice keeps its own tracer.
import { describe, expect, it, vi } from 'vitest'
import type { ClientMsg, DatasetOpenResponse, StreamlineInfo, StreamlineSpec, ViewerDataset } from '@cfd/shared'
import { controllerViewerApi } from '../api/viewerApi'
import { SYNTHETIC_CHANNEL_PATH, buildSyntheticChannel } from '../data/syntheticDataset'
import { SyntheticTransport, type DatasetTransport, type SyntheticDataset } from '../data/transport'
import { InlineCompute } from '../worker/Compute'
import { lineSeeds } from '../worker/streamlines'
import { useUiStore } from '../../state/uiStore'
import { createUiBridge } from '../../ws/uiBridge'
import { ViewerController } from './ViewerController'

const holder = vi.hoisted(() => ({ api: null as unknown }))
vi.mock('../index', () => ({ getViewerApi: () => holder.api, setViewerApi: () => {} }))
vi.mock('../../ws/actions', () => ({ actions: { setSettings: () => true, openSession: () => true }, activeCasePath: () => null }))
vi.mock('../../api/rest', () => ({ api: { geometrySave: vi.fn(), geometryEdit: vi.fn() }, ApiError: class extends Error {} }))

const POLY_PATH = 'cases/warped-sphere'
const BOUNDS = { min: [-1, -1, -1] as [number, number, number], max: [1, 1, 1] as [number, number, number] }

/** A non-lattice polyMesh whose server traces: each seed gets a two-point line from it along +x. */
class StreamTransport implements DatasetTransport {
  readonly calls: StreamlineSpec[] = []
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

  async streamlines(id: string, spec: StreamlineSpec): Promise<StreamlineInfo> {
    this.calls.push(spec)
    const d = this.data.find((x) => x.manifest.id === id)!
    const n = spec.seeds.length / 3
    const points = new Float32Array(6 * n)
    const speeds = new Float32Array(2 * n)
    for (let s = 0; s < n; s++) {
      points.set([spec.seeds[3 * s], spec.seeds[3 * s + 1], spec.seeds[3 * s + 2], spec.seeds[3 * s] + 0.1, spec.seeds[3 * s + 1], spec.timeIndex], 6 * s)
      speeds.set([1, 2], 2 * s)
    }
    const offsets = Uint32Array.from({ length: n + 1 }, (_, i) => 2 * i)
    const key = `stream.t${++this.seq}`
    d.blobs.set(`${key}.points`, points)
    d.blobs.set(`${key}.speeds`, speeds)
    d.blobs.set(`${key}.offsets`, offsets)
    const ref = (k: string, dtype: 'f32' | 'u32', count: number, components: number, bytes: number) => ({ key: `${key}.${k}`, dtype, count, components, bytes })
    return {
      key,
      field: spec.field,
      timeIndex: spec.timeIndex,
      lineCount: n,
      pointCount: 2 * n,
      points: ref('points', 'f32', 2 * n, 3, points.byteLength),
      speeds: ref('speeds', 'f32', 2 * n, 1, speeds.byteLength),
      offsets: ref('offsets', 'u32', n + 1, 1, offsets.byteLength),
      refusedSeeds: [],
      ends: { boundary: 2 * n, length: 0, steps: 0, stall: 0, lost: 0 },
      fallbackPoints: 0,
      invertedTets: 0,
    }
  }
}

function polyDataset(timeCount = 1): SyntheticDataset {
  const d = buildSyntheticChannel({ id: 'poly-stream', path: POLY_PATH, dims: [4, 3, 2], bounds: BOUNDS, times: timeCount })
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

describe('streamlines on a non-lattice polyMesh', () => {
  it('add_layer streamlines asks the server with the seeds and draws exactly its lines', async () => {
    const d = polyDataset(2)
    const t = new StreamTransport([d])
    const c = headless(t)
    expect((await c.execute({ type: 'load', path: POLY_PATH, timeIndex: null, field: 'p' })).ok).toBe(true)
    const r = await bridgeAddLayer(c, 'streamlines', { id: 'sl', field: 'U', seed: { line: [[0, 0, 0], [0, 0.5, 0]], count: 3 }, direction: 'forward' })
    expect(r.ok).toBe(true)
    expect(t.calls).toEqual([{ field: 'U', timeIndex: 1, seeds: [...lineSeeds([0, 0, 0], [0, 0.5, 0], 3)], direction: 'forward', maxLength: 4 * Math.hypot(2, 2, 2) }])
    expect(c.getState().layers).toEqual([{ id: 'sl', type: 'streamlines', summary: 'streamlines U n=3' }])
    const res = c.layers.get('sl')!.result!
    if (res.op !== 'streamlines') throw new Error(`expected a streamlines result, got ${res.op}`)
    expect([...res.points]).toEqual([...(d.blobs.get('stream.t1.points') as Float32Array)])
    expect([...res.speeds]).toEqual([...(d.blobs.get('stream.t1.speeds') as Float32Array)])
    expect([...res.offsets]).toEqual([0, 2, 4, 6])

    // A new time traces again at that time; the same time again is the stored set.
    expect((await c.execute({ type: 'setTime', index: 0 })).ok).toBe(true)
    expect(t.calls.at(-1)).toMatchObject({ timeIndex: 0 })
    expect((await c.execute({ type: 'setTime', index: 1 })).ok).toBe(true)
    expect(t.calls.length).toBe(2)
  })

  it('every seed outside answers INVALID, another refusal LOAD_FAILED, and a scalar field is refused first', async () => {
    const t = new StreamTransport([polyDataset()])
    const c = headless(t)
    await c.execute({ type: 'load', path: POLY_PATH, timeIndex: null, field: 'p' })
    t.streamlines = async () => {
      throw new Error('OUTSIDE: every seed (4) is outside the domain of poly-stream')
    }
    const out = await c.execute({ type: 'addStreamlines', field: 'U', seed: { line: [[5, 0, 0], [5, 1, 0]], count: 4 } })
    expect(out.error?.code).toBe('INVALID')
    expect(out.error?.message).toContain('OUTSIDE')
    t.streamlines = async () => {
      throw new Error('NO_POLY_STREAM: not a polyMesh')
    }
    const bad = await c.execute({ type: 'addStreamlines', field: 'U', seed: { line: [[0, 0, 0], [0, 1, 0]], count: 4 } })
    expect(bad.error?.code).toBe('LOAD_FAILED')
    expect(bad.error?.message).toContain('NO_POLY_STREAM')
    const scalar = await c.execute({ type: 'addStreamlines', field: 'p', seed: { line: [[0, 0, 0], [0, 1, 0]], count: 4 } })
    expect(scalar.error?.code).toBe('NOT_A_VECTOR_FIELD')
    expect(c.getState().layers).toEqual([])
  })

  it('a lattice keeps its own tracer and never asks the server; glyphs still need a lattice', async () => {
    const lattice = buildSyntheticChannel({ dims: [8, 4, 4] })
    lattice.manifest.planeCut = true
    const t = new StreamTransport([lattice, polyDataset()])
    const c = headless(t)
    await c.execute({ type: 'load', path: SYNTHETIC_CHANNEL_PATH, timeIndex: null, field: 'p' })
    expect((await c.execute({ type: 'addStreamlines', id: 'own', field: 'U', seed: { line: [[0.1, 0.1, 0.1], [0.1, 0.4, 0.1]], count: 4 } })).ok).toBe(true)
    expect(c.layers.get('own')!.result!.op).toBe('streamlines')
    expect(t.calls).toEqual([])
    await c.execute({ type: 'load', path: POLY_PATH, timeIndex: null, field: 'p' })
    const g = await c.execute({ type: 'addGlyphs', field: 'U' })
    expect(g.error?.code).toBe('NO_STRUCTURED_GRID')
    expect(g.error?.message).toContain('iso-surfaces and glyphs need one (streamlines are computed on its cells here)')
  })
})
