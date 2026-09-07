// The viewer's public API. The shell (WS bridge, explorer clicks, run cards)
// only talks to the viewer through this object; the viewer never imports
// shell state. STUB until the viewer module lands.
import type { ViewerCommand, ViewerResult, ViewerState } from '@cfd/shared'
import { DEFAULT_VIEWER_STATE } from '@cfd/shared'

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
}

class StubViewerApi implements ViewerApi {
  private state: ViewerState = { ...DEFAULT_VIEWER_STATE }
  private listeners = new Set<(s: ViewerState) => void>()
  async execute(cmd: ViewerCommand): Promise<ViewerResult> {
    if (cmd.type === 'getState') return { ok: true, state: this.state, error: null, image: null }
    return { ok: false, state: this.state, error: { code: 'NO_VIEWER', message: 'viewer not implemented yet' }, image: null }
  }
  getState() {
    return this.state
  }
  subscribe(l: (s: ViewerState) => void) {
    this.listeners.add(l)
    return () => this.listeners.delete(l)
  }
  isMounted() {
    return false
  }
  setBaseUrl() {}
}

let api: ViewerApi | null = null
export function getViewerApi(): ViewerApi {
  if (!api) api = new StubViewerApi()
  return api
}
/** Used by the real viewer module to install itself. */
export function setViewerApi(impl: ViewerApi): void {
  api = impl
}
