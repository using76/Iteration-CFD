// The viewer's public API. The shell (WS bridge, explorer clicks, run cards)
// only talks to the viewer through this object; the viewer never imports
// shell state. The real implementation installs itself at module import, so
// commands issued before the canvas mounts queue until it exists (or fail
// NO_VIEWER after 30 s).
import type { ViewerCommand, ViewerResult, ViewerState, ViewId } from '@cfd/shared'
import type { ProbeOutcome, ViewportRect } from '../controller/ViewerController'
import type { ViewerTool } from '../controller/model'
import { HttpTransport, RoutingTransport, SyntheticTransport } from '../data/transport'
import { SYNTHETIC_CHANNEL_2D_PATH, SYNTHETIC_CHANNEL_PATH, buildSyntheticChannel } from '../data/syntheticDataset'
import { ViewerController } from '../controller/ViewerController'
import { WorkerCompute } from '../worker/Compute'

export interface ViewerOverlayInfo {
  caseName: string | null
  solver: string | null
  model: string | null
  iter: number | null
  targetIter: number | null
  residual: { field: string; value: number } | null
}

export interface ViewerApi {
  /** Execute one command; never throws — errors come back as {ok:false}. Commands are serialised in order. */
  execute(cmd: ViewerCommand): Promise<ViewerResult>
  getState(): ViewerState
  subscribe(listener: (state: ViewerState) => void): () => void
  /** True once a canvas is mounted and the renderer initialised. */
  isMounted(): boolean
  /** Fetch base URL for dataset blobs (defaults to same origin). */
  setBaseUrl(url: string): void
  /** The interaction tool, in the viewer's vocabulary (controller/model.ts ViewerTool). Works before the canvas mounts. */
  setTool(tool: ViewerTool): void
  getTool(): ViewerTool
  /** The 3-D canvas rect in CSS pixels (window coordinates, as getBoundingClientRect gives it); null with no canvas mounted. */
  viewport(): ViewportRect | null
  /** Cell under a window pixel and the coloured field's value there. */
  probePixel(clientX: number, clientY: number): ProbeOutcome
  /** Cell containing a world point (structured grids only) and the coloured field's value there. */
  probePoint(point: [number, number, number]): ProbeOutcome
}

const controllers = new Map<ViewId, ViewerController>()
let http: HttpTransport | null = null

function createWorker(): Worker {
  return new Worker(new URL('../worker/viewer.worker.ts', import.meta.url), { type: 'module' })
}

/** The browser-wired controller per viewport half (lazy: creating it needs no
 *  DOM, but the worker is created on first compute). One HttpTransport is
 *  shared so its base URL covers both halves; each half gets its own router,
 *  synthetic fallback, worker and GPU context. */
export function getViewerController(view: ViewId = 'A'): ViewerController {
  let controller = controllers.get(view)
  if (!controller) {
    http ??= new HttpTransport()
    const synthetic = new SyntheticTransport([buildSyntheticChannel(), buildSyntheticChannel({ dims: [60, 30, 1] })])
    controller = new ViewerController({
      transport: new RoutingTransport(http, synthetic),
      createCompute: (provider) => new WorkerCompute(provider, createWorker),
      requireMount: true,
      mountTimeoutMs: 30_000,
    })
    controllers.set(view, controller)
  }
  return controller
}

class RealViewerApi implements ViewerApi {
  constructor(private readonly view: ViewId = 'A') {}
  execute(cmd: ViewerCommand): Promise<ViewerResult> {
    return getViewerController(this.view).execute(cmd)
  }
  getState(): ViewerState {
    return getViewerController(this.view).getState()
  }
  subscribe(listener: (state: ViewerState) => void): () => void {
    return getViewerController(this.view).subscribe(listener)
  }
  isMounted(): boolean {
    return getViewerController(this.view).isMounted()
  }
  setBaseUrl(url: string): void {
    getViewerController(this.view)
    http?.setBaseUrl(url)
  }
  setTool(tool: ViewerTool): void {
    getViewerController(this.view).setTool(tool)
  }
  getTool(): ViewerTool {
    return getViewerController(this.view).tool
  }
  viewport(): ViewportRect | null {
    return getViewerController(this.view).viewport()
  }
  probePixel(clientX: number, clientY: number): ProbeOutcome {
    return getViewerController(this.view).probePixel(clientX, clientY)
  }
  probePoint(point: [number, number, number]): ProbeOutcome {
    return getViewerController(this.view).probePoint(point)
  }
}

const apis = new Map<ViewId, ViewerApi>()
export function getViewerApi(view: ViewId = 'A'): ViewerApi {
  let api = apis.get(view)
  if (!api) {
    api = new RealViewerApi(view)
    apis.set(view, api)
  }
  return api
}
/** Used by the real viewer module to install itself (or by tests to inject a fake). */
export function setViewerApi(impl: ViewerApi, view: ViewId = 'A'): void {
  apis.set(view, impl)
}

/** `?demoDataset=1` (3-D) or `?demoDataset=2d` loads the synthetic channel without a server. */
export function demoDatasetPath(search: string = typeof location !== 'undefined' ? location.search : ''): string | null {
  const v = new URLSearchParams(search).get('demoDataset')
  if (!v || v === '0' || v === 'false') return null
  return v === '2d' ? SYNTHETIC_CHANNEL_2D_PATH : SYNTHETIC_CHANNEL_PATH
}

setViewerApi(new RealViewerApi('A'), 'A')
