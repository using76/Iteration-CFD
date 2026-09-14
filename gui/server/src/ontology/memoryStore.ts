// gui/server/src/ontology/memoryStore.ts — the in-memory ActionStore every engine test runs
// against (N4 D-F: the engine never talks to N2's SQLite store directly; the whole dependency
// sits at one reviewable seam, storeAdapter.ts in Run 2). It counts writes and records every
// mutating call so a test can prove propose() wrote NOTHING and apply() wrote it all in one tx.
import type { ActionStore } from './engine.js'
import type { EditLogEntry, LinkEdit } from '@cfd/shared'

export interface MemoryActionStore extends ActionStore {
  objects: Map<string, { objectType: string; props: Record<string, unknown>; sourcePath: string }>   // key `${objectType} ${id}`
  links: Array<{ edit: LinkEdit; sourcePath: string }>
  log: EditLogEntry[]
  /** Every mutating call, in order, prefixed `tx:` while inside transaction(). Run 2 asserts on it. */
  calls: string[]
  writes: number
  /** Set to make putLink throw once; Run 2 uses it for the rollback test. */
  failNextLink: boolean
}

export function createMemoryActionStore(): MemoryActionStore {
  const objects = new Map<string, { objectType: string; props: Record<string, unknown>; sourcePath: string }>()
  const links: Array<{ edit: LinkEdit; sourcePath: string }> = []
  const log: EditLogEntry[] = []
  const calls: string[] = []
  let writes = 0
  let failNextLink = false
  let inTx = false
  const rec = (name: string): string => { const c = (inTx ? 'tx:' : '') + name; calls.push(c); return c }
  const store: MemoryActionStore = {
    objects,
    links,
    log,
    calls,
    get writes() { return writes },
    set writes(n: number) { writes = n },
    get failNextLink() { return failNextLink },
    set failNextLink(v: boolean) { failNextLink = v },
    getObject(objectType, id) {
      const row = objects.get(`${objectType} ${id}`)
      return row ? { ...row.props } : null
    },
    putObject(objectType, id, props, sourcePath) {
      objects.set(`${objectType} ${id}`, { objectType, props: { ...props }, sourcePath })
      writes++
      rec('putObject')
    },
    deleteObject(objectType, id) {
      objects.delete(`${objectType} ${id}`)
      writes++
      rec('deleteObject')
    },
    putLink(edit, sourcePath) {
      if (failNextLink) { failNextLink = false; throw new Error('putLink failed (failNextLink)') }
      links.push({ edit: { ...edit, props: { ...edit.props } }, sourcePath })
      writes++
      rec('putLink')
    },
    appendEditLog(entry) {
      log.push(JSON.parse(JSON.stringify(entry)) as EditLogEntry)
      writes++
      rec('appendEditLog')
    },
    listEditLog() { return log.map((e) => JSON.parse(JSON.stringify(e)) as EditLogEntry) },
    transaction<T>(fn: () => T): T {
      // N2's tx semantics, emulated: BEGIN, and on ANY throw ROLLBACK — the whole snapshot is
      // restored so "nothing was written" holds exactly as it does on the real store.
      const objSnap = [...objects.entries()].map(([k, v]) =>
        [k, { objectType: v.objectType, props: { ...v.props }, sourcePath: v.sourcePath }] as [string, { objectType: string; props: Record<string, unknown>; sourcePath: string }])
      const linkSnap = links.map((l) => ({ edit: { ...l.edit, props: { ...l.edit.props } }, sourcePath: l.sourcePath }))
      const logSnap = log.map((e) => JSON.parse(JSON.stringify(e)) as EditLogEntry)
      const callsAt = calls.length
      const writesAt = writes
      inTx = true
      try {
        const r = fn()
        if (r !== null && typeof r === 'object' && typeof (r as { then?: unknown }).then === 'function')
          throw new Error('tx(body) is synchronous: body returned a promise (ASYNC_TX)')
        return r
      } catch (e) {
        objects.clear()
        for (const [k, v] of objSnap) objects.set(k, v)
        links.length = 0
        links.push(...linkSnap)
        log.length = 0
        log.push(...logSnap)
        calls.length = callsAt
        writes = writesAt
        throw e
      } finally {
        inTx = false
      }
    },
  }
  return store
}
