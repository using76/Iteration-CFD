// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// STU-13: on a polyMesh with no lattice (manifest.pointLocate), probe {point} asks the server which
// cell holds the point, reads the coloured value there, and says how the cell was found; the census
// row "probe with point" is green through the real bridge. A point in no cell is MISS (probe), a
// lattice dataset never asks the server, and the Inspector shows the label.
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import type { ClientMsg, DatasetOpenResponse, PointLocateInfo, UiCommand, ViewerDataset } from '@cfd/shared'
import { InspectorView } from '../../components/right/InspectorPanel'
import { controllerViewerApi } from '../api/viewerApi'
import { SYNTHETIC_CHANNEL_PATH, buildSyntheticChannel } from '../data/syntheticDataset'
import { SyntheticTransport, type DatasetTransport, type SyntheticDataset } from '../data/transport'
import { InlineCompute } from '../worker/Compute'
import { scalarOf } from '../worker/sampling'
import { useUiStore } from '../../state/uiStore'
import { createUiBridge } from '../../ws/uiBridge'
import { ViewerController } from './ViewerController'

// The bridge's own imports reach the window; the same three stand-ins as unstructuredSlice.test.
const holder = vi.hoisted(() => ({ api: null as unknown }))
vi.mock('../index', () => ({ getViewerApi: () => holder.api, setViewerApi: () => {} }))
vi.mock('../../ws/actions', () => ({ actions: { setSettings: () => true, openSession: () => true }, activeCasePath: () => null }))
vi.mock('../../api/rest', () => ({ api: { geometrySave: vi.fn(), geometryEdit: vi.fn() }, ApiError: class extends Error {} }))

const POLY_PATH = 'cases/warped-sphere'
const BOUNDS = { min: [-1, -1, -1] as [number, number, number], max: [1, 1, 1] as [number, number, number] }

/** A server that locates in [-1, 1]^3: x >= 0 is convex cell 2, x < 0 the nearest centre of warped cell 5, outside a miss. */
class LocateTransport implements DatasetTransport {
  readonly calls: Array<[number, number, number]> = []
  fail: string | null = null
  private readonly inner: SyntheticTransport

  constructor(data: SyntheticDataset[]) {
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

  async locate(_id: string, point: [number, number, number]): Promise<PointLocateInfo> {
    this.calls.push(point)
    if (this.fail) throw new Error(this.fail)
    if (point.some((v) => Math.abs(v) > 1)) return { point, cell: null, method: null, center: null }
    return point[0] >= 0
      ? { point, cell: 2, method: 'contains', center: [0.25, 0.5, 0.75] }
      : { point, cell: 5, method: 'nearest-centre', center: [-0.25, 0.5, 0.75] }
  }
}

function polyDataset(): SyntheticDataset {
  const d = buildSyntheticChannel({ id: 'poly-locate', path: POLY_PATH, dims: [4, 3, 2], bounds: BOUNDS })
  d.manifest.source = 'polymesh'
  d.manifest.grid = null
  d.manifest.planeCut = true
  d.manifest.pointLocate = true
  return d
}

function headless(transport: DatasetTransport) {
  return new ViewerController({ transport, createCompute: (provider) => new InlineCompute(provider), requireMount: false, loader: { pollMs: 1, sleep: async () => {} } })
}

function speed(d: SyntheticDataset): Float32Array {
  const ref = d.manifest.fields.find((f) => f.name === 'U')!.perTime.at(-1)!.blob
  return scalarOf(d.blobs.get(ref.key) as Float32Array, 3, 'magnitude')
}

/** gui_control's probe through the real bridge over this controller, on a canvas that is mounted. */
async function bridgeProbe(c: ViewerController, cmd: UiCommand) {
  holder.api = { ...controllerViewerApi(c), isMounted: () => true, viewport: () => ({ left: 0, top: 0, width: 800, height: 600 }) }
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
  await bridge.handleCommand('r1', cmd)
  const r = sent.find((m): m is Extract<ClientMsg, { t: 'ui.result' }> => m.t === 'ui.result' && m.requestId === 'r1')
  if (!r) throw new Error('the bridge sent no ui.result')
  return r
}

describe('probe {point} on a polyMesh with no lattice', () => {
  it('the server locates the cell; the value is that cell\'s, and the label says how it was found', async () => {
    const d = polyDataset()
    const t = new LocateTransport([d])
    const c = headless(t)
    expect((await c.execute({ type: 'load', path: POLY_PATH, timeIndex: null, field: null })).ok).toBe(true)
    expect(c.getState().field).toBe('U')
    // The synchronous lattice lookup still has no grid to use.
    expect(c.probePoint([0.5, 0, 0])).toMatchObject({ ok: false, code: 'NO_STRUCTURED_GRID' })

    const exact = await c.locatePoint([0.5, 0, 0])
    if (!exact.ok) throw new Error(exact.message)
    expect(exact.hit).toEqual({ cell: 2, value: exact.hit.value, field: 'U', center: [0.25, 0.5, 0.75], located: 'contains' })
    expect(Object.is(exact.hit.value, speed(d)[2])).toBe(true)

    const near = await c.locatePoint([-0.5, 0, 0])
    if (!near.ok) throw new Error(near.message)
    expect(near.hit).toMatchObject({ cell: 5, center: [-0.25, 0.5, 0.75], located: 'nearest-centre' })
    expect(Object.is(near.hit.value, speed(d)[5])).toBe(true)

    expect(await c.locatePoint([3, 0, 0])).toMatchObject({ ok: false, code: 'MISS' })
    t.fail = 'NO_POINT_LOCATE: refused by the fake'
    expect(await c.locatePoint([0.5, 0, 0])).toEqual({ ok: false, code: 'LOCATE_FAILED', message: 'NO_POINT_LOCATE: refused by the fake' })
    expect(t.calls).toEqual([[0.5, 0, 0], [-0.5, 0, 0], [3, 0, 0], [0.5, 0, 0]])
  })

  it('census row: probe with point is green on a polyMesh through the bridge, and ui.state carries the label', async () => {
    const t = new LocateTransport([polyDataset()])
    const c = headless(t)
    await c.execute({ type: 'load', path: POLY_PATH, timeIndex: null, field: null })

    const ok = await bridgeProbe(c, { type: 'probe', point: [0.5, 0, 0] })
    expect(ok.ok).toBe(true)
    expect(ok.error).toBeNull()
    expect(ok.state?.selection).toMatchObject({ kind: 'cell', id: 2, located: 'contains', field: 'U' })

    const near = await bridgeProbe(c, { type: 'probe', point: [-0.5, 0, 0] })
    expect(near.ok).toBe(true)
    expect(near.state?.selection).toMatchObject({ kind: 'cell', id: 5, located: 'nearest-centre' })
    expect(useUiStore.getState().selection).toMatchObject({ kind: 'cell', id: 5, located: 'nearest-centre' })

    const miss = await bridgeProbe(c, { type: 'probe', point: [3, 0, 0] })
    expect(miss.ok).toBe(false)
    expect(miss.error).toMatch(/^MISS \(probe\): /)
    expect(miss.state?.selection).toEqual({ kind: 'none' })

    t.fail = 'NO_POINT_LOCATE: refused by the fake'
    const failed = await bridgeProbe(c, { type: 'probe', point: [0.5, 0, 0] })
    expect(failed.error).toBe('LOCATE_FAILED (probe): NO_POINT_LOCATE: refused by the fake')
  })

  it('a lattice dataset keeps its own lookup and never asks the server', async () => {
    const lattice = buildSyntheticChannel({ dims: [8, 4, 4] })
    lattice.manifest.pointLocate = true
    const t = new LocateTransport([lattice])
    const c = headless(t)
    await c.execute({ type: 'load', path: SYNTHETIC_CHANNEL_PATH, timeIndex: null, field: null })
    const p: [number, number, number] = [0.5, 0.5, 0.5]
    expect(await c.locatePoint(p)).toEqual(c.probePoint(p))
    expect(t.calls).toEqual([])
  })

  it('the Inspector names how the cell was found, and says nothing of it for a pick', () => {
    const base = { kind: 'cell' as const, id: 5, center: [-0.25, 0.5, 0.75] as [number, number, number], value: 1.5, field: 'U', patch: null }
    const near = renderToStaticMarkup(InspectorView({ locale: 'en', selection: { ...base, located: 'nearest-centre' } }))
    expect(near).toContain('data-testid="inspector-located"')
    expect(near).toContain('nearest centre (warped or non-convex cell)')
    const exact = renderToStaticMarkup(InspectorView({ locale: 'en', selection: { ...base, located: 'contains' } }))
    expect(exact).toContain('in the cell (convex, exact)')
    expect(renderToStaticMarkup(InspectorView({ locale: 'en', selection: base }))).not.toContain('inspector-located')
  })
})
