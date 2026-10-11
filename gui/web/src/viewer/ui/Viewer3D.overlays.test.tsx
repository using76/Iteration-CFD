// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// STU-02: show_overlay drives the viewer's axes triad and colour-bar legend
// through the ui store; the rendered DOM carries both as data attributes and
// the legend drops out when colour bars are off.
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import { Viewer3D } from './Viewer3D'

// The real store cannot drive a static server render: zustand v5's hook reads
// getInitialState() on the server, so any setState would be invisible. The
// component tests' precedent (GpuBadge.test.tsx) mocks the store instead, with
// the two overlay fields and the one action Viewer3D touches.
const { useUiStore } = vi.hoisted(() => {
  const state = {
    locale: 'en' as const,
    showAxes: true,
    showColorBars: true,
    setOverlay(what: 'axes' | 'colorbars', on: boolean): void {
      if (what === 'axes') state.showAxes = on
      else state.showColorBars = on
    },
  }
  const useUiStore = ((selector: (s: typeof state) => unknown) => selector(state)) as {
    (selector: (s: typeof state) => unknown): unknown
    getState(): typeof state
    getInitialState(): typeof state
    setState(patch: Partial<typeof state>): void
    subscribe(listener: () => void): () => void
  }
  useUiStore.getState = () => state
  useUiStore.getInitialState = () => state
  useUiStore.setState = (patch: Partial<typeof state>) => Object.assign(state, patch)
  useUiStore.subscribe = () => () => {}
  return { useUiStore }
})
vi.mock('../../state/uiStore', () => ({ useUiStore }))

const { controller } = vi.hoisted(() => {
  const fakeState = {
    backend: 'webgl2',
    loading: false,
    message: null,
    representation: 'surface',
    range: null,
    field: null,
    time: null,
    camera: { position: [10, 10, 10], target: [0, 0, 0], projection: 'perspective' },
    patches: { hidden: [] },
    layers: 0,
  }
  // LUT_SIZE 256 x 4 RGBA bytes; zeros are fine for static markup.
  const table = new Uint8Array(1024)
  const controller = {
    getState: () => fakeState,
    subscribe: () => () => {},
    dataset: null,
    layers: new Map(),
    field: null,
    tool: 'select',
    display: { colormap: 'turbo', representation: 'surface', opacity: 1, patches: 'all', shading: 'pbr', quality: 'medium', interpolate: true },
    legendSpec: () => ({ table, min: 0, max: 1, log: false, title: 'p' }),
    getView: () => null,
    setOverlayTextProvider: () => {},
    execute: async () => ({ ok: true, state: null, error: null, image: null }),
  }
  return { controller }
})

vi.mock('../api/viewerApi', () => ({ getViewerController: () => controller, demoDatasetPath: () => null }))
vi.mock('../engine/ThreeSceneView', () => ({ ThreeSceneView: { create: vi.fn() } }))

const render = (): string => renderToStaticMarkup(<Viewer3D overlay={null} locale="en" />)

describe('Viewer3D overlays (STU-02)', () => {
  beforeEach(() => {
    useUiStore.setState({ showAxes: true, showColorBars: true })
  })

  it('both overlays on: legend present, both data attributes on', () => {
    useUiStore.getState().setOverlay('axes', true)
    useUiStore.getState().setOverlay('colorbars', true)
    const html = render()
    expect(html).toContain('data-testid="viewer-legend"')
    expect(html).toContain('data-axes="on"')
    expect(html).toContain('data-colorbars="on"')
  })

  it('colour bars off: the legend is gone, the attribute says off', () => {
    useUiStore.getState().setOverlay('colorbars', false)
    const html = render()
    expect(html).not.toContain('viewer-legend')
    expect(html).toContain('data-colorbars="off"')
    expect(html).toContain('data-axes="on"')
  })

  it('axes off: the triad attribute says off, the legend stays', () => {
    useUiStore.getState().setOverlay('axes', false)
    const html = render()
    expect(html).toContain('data-axes="off"')
    expect(html).toContain('data-testid="viewer-legend"')
    expect(html).toContain('data-colorbars="on"')
  })
})
