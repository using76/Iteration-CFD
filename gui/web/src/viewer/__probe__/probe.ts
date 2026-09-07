// Browser probe: mounts the engine on the synthetic dataset, adds a slice,
// streamlines, an iso-surface and glyphs, renders, takes a screenshot and
// writes `PROBE_OK <backend>` into document.title (or `PROBE_FAIL <msg>`).
import { SYNTHETIC_CHANNEL_PATH } from '../data/syntheticDataset'
import { getViewerController } from '../api/viewerApi'
import { ThreeSceneView } from '../engine/ThreeSceneView'
import { assertLittleEndian } from '../data/DatasetLoader'
import type { ViewerCommand } from '@cfd/shared'

declare global {
  interface Window {
    __probe: { backend: string; state: unknown; image: { width: number; height: number; base64: string } | null; errors: string[] }
  }
}

async function main() {
  const errors: string[] = []
  window.addEventListener('error', (e) => errors.push(String(e.message)))
  window.addEventListener('unhandledrejection', (e) => errors.push(String(e.reason)))
  document.title = 'PROBE_RUNNING'
  try {
    assertLittleEndian()
    const host = document.getElementById('host')!
    const view = await ThreeSceneView.create(host)
    view.resize(host.clientWidth, host.clientHeight, 1)
    const controller = getViewerController()
    controller.attachView(view)
    let step = 0
    const check = async (cmd: ViewerCommand) => {
      document.title = `PROBE_STEP ${++step} ${cmd.type}`
      const r = await controller.execute(cmd)
      if (!r.ok) throw new Error(`${JSON.stringify(cmd)} -> ${r.error?.code}: ${r.error?.message}`)
      return r
    }
    await check({ type: 'load', path: SYNTHETIC_CHANNEL_PATH, timeIndex: null, field: null })
    await check({ type: 'setRepresentation', mode: 'surfaceEdges', opacity: 0.6, patches: null, shading: null })
    await check({ type: 'addSlice', id: 'slice', axis: 'z', position: { fraction: 0.5 } })
    await check({ type: 'addSlice', id: 'section', axis: 'x', position: { fraction: 0.7 } })
    await check({ type: 'addStreamlines', id: 'sl', field: null, seed: { plane: 'x', position: { fraction: 0.03 }, grid: [6, 4] }, style: null, maxLength: null, direction: null })
    await check({ type: 'addIsoSurface', id: 'iso', field: 'U', values: [1.3] })
    await check({ type: 'addGlyphs', id: 'glyphs', field: null, stride: 4, scale: 1, onSlice: 'section' })
    await check({ type: 'setCamera', preset: 'iso', position: null, target: null, projection: null })
    await new Promise((r) => setTimeout(r, 300))
    const shot = await check({ type: 'screenshot', width: 960, height: 600, includeLegend: true })
    // Exercise the remaining paths for browser-only failures (shader compilation, geometry kinds).
    for (const mode of ['wireframe', 'points', 'outline', 'surface'] as const) await check({ type: 'setRepresentation', mode, opacity: 1, patches: mode === 'surface' ? ['inlet', 'bottomWall', 'front'] : null, shading: mode === 'points' ? 'flat' : 'pbr' })
    await check({ type: 'setClipBox', enabled: true, min: null, max: [2, 1.5, 0.6] })
    await check({ type: 'setCamera', preset: '+z', position: null, target: null, projection: 'orthographic' })
    await check({ type: 'addStreamlines', id: 'tubes', field: 'U', seed: { line: [[0.1, 0.2, 0.5], [0.1, 1.3, 0.5]], count: 6 }, style: 'tube', maxLength: 2, direction: 'forward' })
    await check({ type: 'setField', field: 'U', component: 'y', range: 'auto', colormap: 'coolwarm', log: null })
    await check({ type: 'setField', field: 'k', component: null, range: 'global', colormap: 'viridis', log: true })
    await check({ type: 'setTime', index: 0 })
    await check({ type: 'addPlane', id: 'plane', origin: [1.5, 0.75, 0.5], normal: [1, 0.5, 0.3] })
    await check({ type: 'addGlyphs', id: 'vol', field: 'U', stride: 6, scale: 1.5, onSlice: null })
    await check({ type: 'setQuality', level: 'high' })
    await check({ type: 'remove', id: 'tubes' })
    await check({ type: 'setClipBox', enabled: false, min: null, max: null })
    await check({ type: 'setCamera', preset: 'iso', position: null, target: null, projection: 'perspective' })
    await new Promise((r) => setTimeout(r, 200))
    const shot2 = await check({ type: 'screenshot', width: 640, height: 400, includeLegend: true })
    if (!shot2.image || shot2.image.width !== 640) throw new Error('second screenshot missing')
    const bad = await controller.execute({ type: 'addIsoSurface', id: null, field: 'nope', values: [1] })
    if (bad.ok || bad.error?.code !== 'NO_SUCH_FIELD') throw new Error('expected NO_SUCH_FIELD')
    await check({ type: 'clear' })
    const state = (await check({ type: 'getState' })).state
    window.__probe = { backend: view.backend, state, image: shot.image, errors }
    document.title = errors.length ? `PROBE_FAIL ${errors[0]}` : `PROBE_OK ${view.backend}`
  } catch (err) {
    window.__probe = { backend: 'none', state: null, image: null, errors: [...errors, err instanceof Error ? err.message : String(err)] }
    document.title = `PROBE_FAIL ${err instanceof Error ? err.message : String(err)}`
  }
}

void main()
