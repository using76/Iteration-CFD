// The compute service used by layers: either a Web Worker (browser) or an
// inline implementation (tests, headless api). Both resolve missing blobs
// through the provider callback given at construction.
import type { TypedArray } from '../data/BlobCache'
import { MissingBlobs, runCompute } from './ops'
import { requestKeys, type ComputeRequest, type ResultOf, type WorkerInbound, type WorkerOutbound } from './protocol'

export type BlobProvider = (key: string) => Promise<TypedArray>

export interface Compute {
  run<R extends ComputeRequest>(req: R): Promise<ResultOf<R>>
  /** Forget every blob whose key starts with `prefix` (a dataset id). */
  evict(prefix: string): Promise<void>
  dispose(): void
}

export class InlineCompute implements Compute {
  private readonly blobs = new Map<string, TypedArray>()

  constructor(private readonly provider: BlobProvider) {}

  async run<R extends ComputeRequest>(req: R): Promise<ResultOf<R>> {
    for (const key of requestKeys(req)) if (!this.blobs.has(key)) this.blobs.set(key, await this.provider(key))
    for (;;) {
      try {
        return runCompute(req, (k) => this.blobs.get(k)) as ResultOf<R>
      } catch (err) {
        if (!(err instanceof MissingBlobs)) throw err
        for (const key of err.keys) this.blobs.set(key, await this.provider(key))
      }
    }
  }

  async evict(prefix: string): Promise<void> {
    for (const key of [...this.blobs.keys()]) if (key.startsWith(prefix)) this.blobs.delete(key)
  }

  dispose(): void {
    this.blobs.clear()
  }
}

type DistributiveOmit<T, K extends PropertyKey> = T extends unknown ? Omit<T, K> : never

interface Pending {
  resolve: (msg: WorkerOutbound) => void
  reject: (err: Error) => void
}

export class WorkerCompute implements Compute {
  private readonly pending = new Map<number, Pending>()
  private nextId = 1
  private worker: Worker

  constructor(
    private readonly provider: BlobProvider,
    factory: () => Worker,
  ) {
    this.worker = factory()
    this.worker.onmessage = (ev: MessageEvent<WorkerOutbound>) => {
      const p = this.pending.get(ev.data.id)
      if (!p) return
      this.pending.delete(ev.data.id)
      p.resolve(ev.data)
    }
    this.worker.onerror = (ev) => {
      const err = new Error(ev.message || 'viewer worker crashed')
      for (const p of this.pending.values()) p.reject(err)
      this.pending.clear()
    }
  }

  private send(msg: DistributiveOmit<WorkerInbound, 'id'>, transfer: ArrayBuffer[] = []): Promise<WorkerOutbound> {
    const id = this.nextId++
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject })
      this.worker.postMessage({ ...msg, id } as WorkerInbound, transfer)
    })
  }

  private async upload(keys: string[]): Promise<void> {
    const blobs = await Promise.all(keys.map(async (key) => ({ key, data: await this.provider(key) })))
    const reply = await this.send({ kind: 'put', blobs })
    if (reply.kind === 'error') throw new Error(reply.message)
  }

  async run<R extends ComputeRequest>(req: R): Promise<ResultOf<R>> {
    for (let attempt = 0; attempt < 4; attempt++) {
      const reply = await this.send({ kind: 'run', req })
      if (reply.kind === 'ok' && reply.result) return reply.result as ResultOf<R>
      if (reply.kind === 'error') throw new Error(reply.message)
      if (reply.kind === 'missing') await this.upload(reply.keys)
    }
    throw new Error('worker kept reporting missing blobs')
  }

  async evict(prefix: string): Promise<void> {
    await this.send({ kind: 'evict', prefix })
  }

  dispose(): void {
    this.worker.terminate()
    const err = new Error('viewer worker disposed')
    for (const p of this.pending.values()) p.reject(err)
    this.pending.clear()
  }
}
