// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// A 3-D polyMesh opens on its body walls and a fit frames them; patches:'all' brings the box back.
import { describe, expect, it } from 'vitest'
import type { GeometryBounds } from '@cfd/shared'
import { buildSyntheticChannel, SYNTHETIC_CHANNEL_2D_PATH, SYNTHETIC_CHANNEL_PATH } from '../data/syntheticDataset'
import { buildSyntheticSphereInBox, SYNTHETIC_SPHERE_IN_BOX_PATH } from '../data/syntheticSphereInBox'
import { SyntheticTransport } from '../data/transport'
import { InlineCompute } from '../worker/Compute'
import type { SceneView } from './SceneView'
import { ViewerController } from './ViewerController'

type CameraOpts = Parameters<SceneView['setCamera']>[0]

function headless() {
  const transport = new SyntheticTransport([buildSyntheticSphereInBox(), buildSyntheticChannel({ dims: [24, 12, 8] }), buildSyntheticChannel({ dims: [24, 12, 1] })])
  return new ViewerController({ transport, createCompute: (provider) => new InlineCompute(provider), requireMount: false, loader: { pollMs: 1, sleep: async () => {} } })
}

/** A SceneView that only records setCamera calls. */
function fakeView() {
  const cameras: CameraOpts[] = []
  const view: SceneView = {
    backend: 'webgl2',
    setDataset: () => {},
    setSurfaceColoring: () => {},
    setColorTable: () => {},
    setColoredField: () => {},
    setDisplay: () => {},
    upsertLayer: () => {},
    removeLayer: () => {},
    setLayerVisible: () => {},
    clearLayers: () => {},
    setClipBox: () => {},
    setCamera: (opts) => {
      cameras.push(opts)
    },
    getCamera: () => ({ position: [10, 10, 10], target: [0, 0, 0], projection: 'perspective' }),
    onCameraChange: () => () => {},
    setQuality: () => {},
    setTool: () => {},
    screenshot: () => Promise.reject(new Error('no screenshots headless')),
    pick: () => -1,
    viewport: () => ({ left: 0, top: 0, width: 100, height: 100 }),
    dispose: () => {},
  }
  return { view, cameras }
}

const S = Math.sqrt(3) * 1e-6
const SPHERE = { min: [-0.5, -0.5, -0.5], max: [0.5, 0.5, 0.5] }

/** Every one of the 6 numbers differs from the expected by at most S. */
function expectNearBox(b: GeometryBounds | null | undefined, box: { min: number[]; max: number[] }): void {
  const six = [...(b?.min ?? []), ...(b?.max ?? [])]
  expect(six).toHaveLength(6)
  const want = [...box.min, ...box.max]
  six.forEach((v, i) => expect(Math.abs(v - want[i])).toBeLessThanOrEqual(S))
}

async function loadSphere() {
  const c = headless()
  const r = await c.execute({ type: 'load', path: SYNTHETIC_SPHERE_IN_BOX_PATH, timeIndex: null, field: null })
  expect(r.ok).toBe(true)
  return c
}

describe('ViewerController patches', () => {
  it('the sphere-in-box opens on its wall', async () => {
    const c = await loadSphere()
    const m = c.dataset!.manifest
    expect(m.surface.triangleCount).toBe(960)
    expect(m.surface.patches.map((p) => p.triCount)).toEqual([128, 128, 128, 128, 128, 128, 192])
    expect(c.display.patches).toEqual(['sphere'])
    const r = await c.execute({ type: 'getState' })
    expect(r.state?.patches).toEqual({ visible: ['sphere'], hidden: ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'] })
  })

  it('fit frames the sphere', async () => {
    const c = await loadSphere()
    expectNearBox(c.fitBounds(), SPHERE)
    const { view, cameras } = fakeView()
    c.attachView(view)
    const fit = cameras.find((o) => o.preset === 'fit')
    expect(fit).toBeDefined()
    expectNearBox(fit?.fitBounds, SPHERE)
    await c.execute({ type: 'setCamera', preset: 'fit', position: null, target: null, projection: null })
    const last = cameras[cameras.length - 1]
    expect(last.preset).toBe('fit')
    expectNearBox(last.fitBounds, SPHERE)
  })

  it('patches all restores the box', async () => {
    const c = await loadSphere()
    const r = await c.execute({ type: 'setRepresentation', mode: 'surface', opacity: null, patches: 'all', shading: null })
    expect(r.state?.patches).toEqual({ visible: 'all', hidden: [] })
    expect(c.fitBounds()).toEqual({ min: [-1, -1, -1], max: [1, 1, 1] })
    const { view, cameras } = fakeView()
    c.attachView(view)
    await c.execute({ type: 'setCamera', preset: 'fit', position: null, target: null, projection: null })
    expect(cameras[cameras.length - 1].fitBounds).toEqual({ min: [-1, -1, -1], max: [1, 1, 1] })
  })

  it('a Cartesian box keeps every patch', async () => {
    const c = headless()
    const r = await c.execute({ type: 'load', path: SYNTHETIC_CHANNEL_PATH, timeIndex: null, field: null })
    expect(r.ok).toBe(true)
    expect(c.display.patches).toBe('all')
    expect(r.state?.patches).toEqual({ visible: 'all', hidden: [] })
    expect(c.fitBounds()).toEqual(c.dataset!.manifest.bounds)
  })

  it('a 2-D case keeps its empties', async () => {
    const c = headless()
    const r = await c.execute({ type: 'load', path: SYNTHETIC_CHANNEL_2D_PATH, timeIndex: null, field: null })
    expect(r.ok).toBe(true)
    expect(c.display.patches).toEqual(['front', 'back'])
    expect(r.state?.patches?.hidden).toEqual(['inlet', 'outlet', 'bottomWall', 'topWall'])
  })
})

