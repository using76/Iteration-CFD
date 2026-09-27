// One-way camera link between two viewer halves: the leader's camera pose is
// pushed into the follower whenever it changes; unlink with the returned function.
import type { ViewerState } from '@cfd/shared'
import type { ViewerApi } from './viewerApi'
export function linkCameras(leader: ViewerApi, follower: ViewerApi): () => void {
  let last = ''
  const push = (s: ViewerState): void => {
    const key = JSON.stringify(s.camera)
    if (key === last) return
    last = key
    void follower.execute({ type: 'setCamera', preset: null, position: s.camera.position, target: s.camera.target, projection: s.camera.projection })
  }
  push(leader.getState())
  return leader.subscribe(push)
}
