// The camera link against two in-memory fakes: the leader's pose reaches the
// follower once up front, then on every change, and never after unlink.
import { describe, expect, it } from 'vitest'
import { DEFAULT_VIEWER_STATE, type ViewerCommand, type ViewerResult, type ViewerState } from '@cfd/shared'
import type { ViewerApi } from './viewerApi'
import { linkCameras } from './cameraLink'

function fakeView(): { api: ViewerApi; calls: ViewerCommand[]; emit(next: ViewerState): void } {
  const calls: ViewerCommand[] = []
  const listeners = new Set<(s: ViewerState) => void>()
  let state: ViewerState = { ...DEFAULT_VIEWER_STATE }
  const api: ViewerApi = {
    execute: (cmd) => {
      calls.push(cmd)
      return Promise.resolve({ ok: true, state, error: null, image: null } satisfies ViewerResult)
    },
    getState: () => state,
    subscribe: (fn) => {
      listeners.add(fn)
      return () => listeners.delete(fn)
    },
    isMounted: () => true,
    setBaseUrl: () => {},
    setTool: () => {},
    getTool: () => 'select',
    viewport: () => null,
    probePixel: () => ({ ok: false, code: 'NO_VIEWER', message: 'no canvas in this test' }),
    probePoint: () => ({ ok: false, code: 'NO_VIEWER', message: 'no canvas in this test' }),
  }
  return { api, calls, emit: (next) => { state = next; for (const l of listeners) l(next) } }
}

const cam = (position: [number, number, number], projection: 'perspective' | 'orthographic'): ViewerState => ({
  ...DEFAULT_VIEWER_STATE,
  camera: { position, target: [0, 0, 0], projection },
})

describe('linkCameras', () => {
  it('pushes the leader camera once and on change', async () => {
    const leader = fakeView()
    const follower = fakeView()
    linkCameras(leader.api, follower.api)
    expect(follower.calls).toEqual([{ type: 'setCamera', preset: null, position: [10, 10, 10], target: [0, 0, 0], projection: 'perspective' }])
    leader.emit(cam([1, 2, 3], 'orthographic'))
    await Promise.resolve()
    expect(follower.calls[1]).toEqual({ type: 'setCamera', preset: null, position: [1, 2, 3], target: [0, 0, 0], projection: 'orthographic' })
    leader.emit(cam([1, 2, 3], 'orthographic'))
    await Promise.resolve()
    expect(follower.calls).toHaveLength(2)
  })

  it('stops after unlink', async () => {
    const leader = fakeView()
    const follower = fakeView()
    const unlink = linkCameras(leader.api, follower.api)
    expect(follower.calls).toHaveLength(1)
    unlink()
    leader.emit(cam([4, 5, 6], 'perspective'))
    await Promise.resolve()
    expect(follower.calls).toHaveLength(1)
  })
})
