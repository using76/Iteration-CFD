import { describe, expect, it } from 'vitest'
import { SYNTHETIC_CHANNEL_2D_PATH, SYNTHETIC_CHANNEL_PATH, buildSyntheticChannel } from '../data/syntheticDataset'
import { SyntheticTransport } from '../data/transport'
import { InlineCompute } from '../worker/Compute'
import { ViewerController } from './ViewerController'

function headless() {
  const transport = new SyntheticTransport([buildSyntheticChannel({ dims: [24, 12, 8] }), buildSyntheticChannel({ dims: [24, 12, 1] })])
  return new ViewerController({ transport, createCompute: (provider) => new InlineCompute(provider), requireMount: false, loader: { pollMs: 1, sleep: async () => {} } })
}

describe('ViewerController (headless)', () => {
  it('rejects malformed commands and commands without a dataset', async () => {
    const c = headless()
    const bad = await c.execute({ type: 'addSlice', axis: 'w', position: 0 })
    expect(bad.ok).toBe(false)
    expect(bad.error?.code).toBe('INVALID')
    const none = await c.execute({ type: 'setField', field: 'U', component: null, range: null, colormap: null, log: null })
    expect(none.error?.code).toBe('NO_DATASET')
    const state = await c.execute({ type: 'getState' })
    expect(state.ok).toBe(true)
    expect(state.state?.backend).toBe('none')
    const shot = await c.execute({ type: 'screenshot', width: null, height: null, includeLegend: null })
    expect(shot.error?.code).toBe('NO_VIEWER')
  })

  it('loads a dataset, shows the first field and reports errors with the available fields', async () => {
    const c = headless()
    const states: string[] = []
    c.subscribe((s) => states.push(s.loading ? 'loading' : 'idle'))
    const r = await c.execute({ type: 'load', path: SYNTHETIC_CHANNEL_PATH, timeIndex: null, field: null })
    expect(r.ok).toBe(true)
    expect(r.state?.datasetName).toBe('channel (demo)')
    expect(r.state?.field).toBe('U')
    expect(r.state?.component).toBe('magnitude')
    expect(r.state?.time).toEqual({ index: 2, value: 4000, count: 3 })
    expect(r.state?.range?.[0]).toBeGreaterThanOrEqual(0)
    expect(r.state?.range?.[1]).toBeGreaterThan(1)
    expect(r.state?.loading).toBe(false)
    expect(states).toContain('loading')

    const missing = await c.execute({ type: 'setField', field: 'nope', component: null, range: null, colormap: null, log: null })
    expect(missing.ok).toBe(false)
    expect(missing.error?.code).toBe('NO_SUCH_FIELD')
    expect(missing.error?.message).toContain('U (vector)')
    expect(missing.error?.message).toContain('p')

    const bad = await c.execute({ type: 'load', path: 'demo://missing', timeIndex: null, field: null })
    expect(bad.error?.code).toBe('LOAD_FAILED')

    const fallback = await c.execute({ type: 'load', path: SYNTHETIC_CHANNEL_PATH, timeIndex: 0, field: 'T' })
    expect(fallback.ok).toBe(true)
    expect(fallback.state?.field).toBe('U')
    expect(fallback.state?.message).toContain('"T" not found')
    expect(fallback.state?.time?.index).toBe(0)
  })

  it('applies field, range, colormap and time changes', async () => {
    const c = headless()
    await c.execute({ type: 'load', path: SYNTHETIC_CHANNEL_PATH, timeIndex: 'last', field: 'p' })
    let r = await c.execute({ type: 'setField', field: 'U', component: 'x', range: [0, 2], colormap: 'coolwarm', log: null })
    expect(r.ok).toBe(true)
    expect(r.state?.component).toBe('x')
    expect(r.state?.range).toEqual([0, 2])
    expect(r.state?.colormap).toBe('coolwarm')
    r = await c.execute({ type: 'setField', field: 'U', component: 'y', range: 'auto', colormap: null, log: null })
    // Uy straddles zero: the diverging map is centred
    expect(r.state?.range?.[0]).toBeCloseTo(-(r.state?.range?.[1] ?? 0), 6)
    r = await c.execute({ type: 'setField', field: 'p', component: 'z', range: 'global', colormap: 'viridis', log: true })
    expect(r.state?.component).toBeNull()
    expect(c.mapper?.transform.kind).toBe('log10')
    r = await c.execute({ type: 'setTime', index: 0 })
    expect(r.state?.time?.index).toBe(0)
    r = await c.execute({ type: 'setTime', index: 7 })
    expect(r.error?.code).toBe('INVALID')
  })

  it('creates every layer type and validates their inputs', async () => {
    const c = headless()
    await c.execute({ type: 'load', path: SYNTHETIC_CHANNEL_PATH, timeIndex: null, field: null })
    let r = await c.execute({ type: 'addSlice', id: null, axis: 'z', position: { fraction: 0.5 } })
    expect(r.ok).toBe(true)
    expect(r.state?.layers[0]).toEqual({ id: 'slice-1', type: 'slice', summary: 'slice z=0.5' })
    r = await c.execute({ type: 'addSlice', id: 'mid', axis: 'x', position: 99 })
    expect(r.state?.layers[1].summary).toBe('slice x=3')
    r = await c.execute({ type: 'addPlane', id: null, origin: [1.5, 0.75, 0.5], normal: [1, 1, 0] })
    expect(r.ok).toBe(true)
    expect(r.state?.layers[2].type).toBe('plane')
    r = await c.execute({ type: 'addPlane', id: null, origin: [50, 50, 50], normal: [0, 0, 1] })
    expect(r.error?.code).toBe('INVALID')
    r = await c.execute({ type: 'addIsoSurface', id: 'iso', field: 'U', values: [1.2] })
    expect(r.ok).toBe(true)
    const iso = c.layers.get('iso')!
    expect(iso.result?.op === 'iso' && iso.result.triangleCount).toBeGreaterThan(0)
    r = await c.execute({ type: 'addStreamlines', id: 'sl', field: null, seed: { plane: 'x', position: { fraction: 0.05 }, grid: [4, 3] }, style: null, maxLength: null, direction: null })
    expect(r.ok).toBe(true)
    expect(r.state?.layers.find((l) => l.id === 'sl')?.summary).toBe('streamlines U n=12')
    r = await c.execute({ type: 'addStreamlines', id: null, field: 'p', seed: { line: [[0, 0, 0], [1, 1, 1]], count: 3 }, style: null, maxLength: null, direction: null })
    expect(r.error?.code).toBe('NOT_A_VECTOR_FIELD')
    r = await c.execute({ type: 'addGlyphs', id: 'g', field: 'U', stride: 3, scale: null, onSlice: 'mid' })
    expect(r.ok).toBe(true)
    expect(r.state?.layers.find((l) => l.id === 'g')?.summary).toMatch(/^glyphs U stride=3 n=\d+ on mid$/)
    r = await c.execute({ type: 'addGlyphs', id: null, field: 'k', stride: null, scale: null, onSlice: null })
    expect(r.error?.code).toBe('NOT_A_VECTOR_FIELD')
    r = await c.execute({ type: 'addGlyphs', id: null, field: null, stride: null, scale: null, onSlice: 'nope' })
    expect(r.error?.code).toBe('NO_SUCH_LAYER')
    // time change recomputes the layers
    r = await c.execute({ type: 'setTime', index: 0 })
    expect(r.ok).toBe(true)
    expect(r.state?.layers.length).toBe(6)
    r = await c.execute({ type: 'remove', id: 'mid' })
    expect(r.ok).toBe(true)
    expect(r.state?.layers.map((l) => l.id)).not.toContain('mid')
    r = await c.execute({ type: 'remove', id: 'mid' })
    expect(r.error?.code).toBe('NO_SUCH_LAYER')
    r = await c.execute({ type: 'setClipBox', enabled: true, min: null, max: [1.5, 1.5, 1] })
    expect(r.ok).toBe(true)
    expect(c.clip).toEqual({ min: [0, 0, 0], max: [1.5, 1.5, 1] })
    r = await c.execute({ type: 'clear' })
    expect(r.state?.layers).toEqual([])
  })

  it('handles 2-D cases: empty faces by default, no iso-surfaces, planar streamlines', async () => {
    const c = headless()
    const r = await c.execute({ type: 'load', path: SYNTHETIC_CHANNEL_2D_PATH, timeIndex: null, field: null })
    expect(r.ok).toBe(true)
    expect(c.display.patches).toEqual(['front', 'back'])
    const iso = await c.execute({ type: 'addIsoSurface', id: null, field: 'U', values: [1] })
    expect(iso.error?.code).toBe('NO_STRUCTURED_GRID')
    expect(iso.error?.message).toContain('ISO_2D')
    const sl = await c.execute({ type: 'addStreamlines', id: 'sl', field: 'U', seed: { line: [[0.1, 0.1, 0], [0.1, 1.4, 0]], count: 5 }, style: null, maxLength: null, direction: 'forward' })
    expect(sl.ok).toBe(true)
    const res = c.layers.get('sl')!.result
    expect(res?.op).toBe('streamlines')
    if (res?.op === 'streamlines') {
      expect(res.offsets.length).toBe(6)
      for (let i = 2; i < res.points.length; i += 3) expect(res.points[i]).toBeCloseTo(0.5, 6)
    }
  })

  it('validates representation patches and camera commands without a renderer', async () => {
    const c = headless()
    await c.execute({ type: 'load', path: SYNTHETIC_CHANNEL_PATH, timeIndex: null, field: null })
    let r = await c.execute({ type: 'setRepresentation', mode: 'surfaceEdges', opacity: 0.5, patches: ['inlet', 'nope'], shading: 'flat' })
    expect(r.error?.code).toBe('INVALID')
    r = await c.execute({ type: 'setRepresentation', mode: 'wireframe', opacity: 2, patches: ['inlet'], shading: null })
    expect(r.ok).toBe(true)
    expect(r.state?.representation).toBe('wireframe')
    expect(c.display.opacity).toBe(1)
    r = await c.execute({ type: 'setCamera', preset: null, position: null, target: null, projection: 'orthographic' })
    expect(r.state?.camera.projection).toBe('orthographic')
    r = await c.execute({ type: 'setQuality', level: 'high' })
    expect(r.ok).toBe(true)
    expect(c.display.quality).toBe('high')
  })

  it('serialises commands in order', async () => {
    const c = headless()
    const order: string[] = []
    const p1 = c.execute({ type: 'load', path: SYNTHETIC_CHANNEL_PATH, timeIndex: null, field: null }).then(() => order.push('load'))
    const p2 = c.execute({ type: 'addSlice', id: 's', axis: 'y', position: 0.5 }).then((r) => order.push(r.ok ? 'slice' : `slice:${r.error?.code}`))
    await Promise.all([p1, p2])
    expect(order).toEqual(['load', 'slice'])
  })
})
