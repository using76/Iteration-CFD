// Compute worker: holds dataset blobs in its own LRU and answers compute
// requests with transferable typed arrays.
import { BlobCache } from '../data/BlobCache'
import { MissingBlobs, runCompute } from './ops'
import { requestKeys, transferablesOf, type WorkerInbound, type WorkerOutbound } from './protocol'

const cache = new BlobCache(256 * 1024 * 1024)

function post(msg: WorkerOutbound, transfer: ArrayBuffer[] = []): void {
  ;(self as unknown as Worker).postMessage(msg, transfer)
}

self.onmessage = (ev: MessageEvent<WorkerInbound>) => {
  const msg = ev.data
  if (msg.kind === 'put') {
    for (const b of msg.blobs) cache.set(b.key, b.data)
    post({ id: msg.id, kind: 'ok', result: null })
    return
  }
  if (msg.kind === 'evict') {
    cache.deletePrefix(msg.prefix)
    post({ id: msg.id, kind: 'ok', result: null })
    return
  }
  const missing = requestKeys(msg.req).filter((k) => !cache.has(k))
  if (missing.length) {
    post({ id: msg.id, kind: 'missing', keys: missing })
    return
  }
  try {
    const result = runCompute(msg.req, (key) => cache.get(key))
    post({ id: msg.id, kind: 'ok', result }, transferablesOf(result))
  } catch (err) {
    if (err instanceof MissingBlobs) post({ id: msg.id, kind: 'missing', keys: err.keys })
    else post({ id: msg.id, kind: 'error', message: err instanceof Error ? err.message : String(err) })
  }
}
