// Glue between the WS protocol and the 3D viewer's public API: executes
// `viewer.command` frames, replies with `viewer.result`, and forwards
// throttled `viewer.state` snapshots.
import type { ClientMsg, ViewerCommand, ViewerResult, ViewerState } from '@cfd/shared'
import { getViewerApi } from '../viewer'
import type { UiStore } from '../state/uiStore'

export const VIEWER_STATE_THROTTLE_MS = 250

export interface ViewerBridge {
  handleCommand(requestId: string, cmd: ViewerCommand): Promise<void>
  openPath(path: string, runId: string | null): Promise<ViewerResult>
  /** (Re)subscribe to the current viewer API instance; safe to call repeatedly. */
  attach(): void
  detach(): void
}

interface StoreLike {
  getState(): Pick<UiStore, 'openViewerTab' | 'setActiveRun'>
}

export function createViewerBridge(send: (msg: ClientMsg) => boolean, ui: StoreLike, throttleMs = VIEWER_STATE_THROTTLE_MS): ViewerBridge {
  let unsubscribe: (() => void) | null = null
  let subscribedTo: unknown = null
  let pending: ViewerState | null = null
  let timer: ReturnType<typeof setTimeout> | null = null
  let lastSent = 0

  function flush() {
    timer = null
    if (!pending) return
    const state = pending
    pending = null
    lastSent = Date.now()
    send({ t: 'viewer.state', state })
  }

  function onState(state: ViewerState) {
    pending = state
    if (timer) return
    const wait = Math.max(0, throttleMs - (Date.now() - lastSent))
    timer = setTimeout(flush, wait)
  }

  return {
    async handleCommand(requestId, cmd) {
      if (cmd.type === 'load') ui.getState().openViewerTab()
      let result: ViewerResult
      try {
        result = await getViewerApi().execute(cmd)
      } catch (err) {
        result = { ok: false, state: null, error: { code: 'INTERNAL', message: err instanceof Error ? err.message : String(err) }, image: null }
      }
      send({ t: 'viewer.result', requestId, result })
      if (result.state) onState(result.state)
    },
    async openPath(path, runId) {
      const u = ui.getState()
      u.openViewerTab()
      if (runId) u.setActiveRun(runId)
      const result = await getViewerApi().execute({ type: 'load', path, timeIndex: 'last', field: null })
      if (result.state) onState(result.state)
      return result
    },
    attach() {
      const apiInstance = getViewerApi()
      if (subscribedTo === apiInstance) return
      unsubscribe?.()
      subscribedTo = apiInstance
      unsubscribe = apiInstance.subscribe(onState)
    },
    detach() {
      unsubscribe?.()
      unsubscribe = null
      subscribedTo = null
      if (timer) clearTimeout(timer)
      timer = null
      pending = null
    },
  }
}
