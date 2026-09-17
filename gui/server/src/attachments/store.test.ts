import { createHash } from 'node:crypto'
import fs from 'node:fs/promises'
import path from 'node:path'
import { describe, expect, it } from 'vitest'
import { HttpError } from '../http/router.js'
import { makeTempWorkspace, testConfig, type TempWorkspace } from '../runs/test-helpers.js'
import { WorkspaceError } from '../workspace/paths.js'
import { ATTACHMENT_MAX_BYTES, ATTACHMENT_TEXT_EXTRACT_CAP, createAttachmentStore, isAttachmentId } from './store.js'

// The 70-byte 1x1 opaque-red PNG the probe also carries (C8).
const ONE_PIXEL_PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=='

/** Every test overrides cacheDir INTO the workspace (D3): the helper's own <tmp>/cache is a sibling of <tmp>/ws. */
async function workspace() {
  const ws: TempWorkspace = await makeTempWorkspace()
  const config = testConfig(ws, { cacheDir: path.join(ws.root, 'gui', '.cache') })
  return { ws, config, store: createAttachmentStore(config) }
}

describe('attachment store', () => {
  it('round-trips the bytes and names them by content', async () => {
    const { ws, config, store } = await workspace()
    const buf = Buffer.from(ONE_PIXEL_PNG, 'base64')
    const stored = await store.put(buf, { filename: 'pixel.png', declaredType: 'image/png' })
    const id = createHash('sha256').update(buf).digest('hex')
    expect(stored.attachmentId).toMatch(/^[0-9a-f]{64}$/)
    expect(stored.attachmentId).toBe(id)
    expect(isAttachmentId(stored.attachmentId)).toBe(true)
    expect(stored.bytes).toBe(70)
    expect(stored.mediaType).toBe('image/png')
    expect(stored.width).toBe(1)
    expect(stored.height).toBe(1)
    expect(stored.fresh).toBe(true)
    const back = await store.get(stored.attachmentId)
    expect(back).not.toBeNull()
    expect(back!.equals(buf)).toBe(true)
    await fs.access(store.pathOf(stored.attachmentId))
    expect(store.pathOf(stored.attachmentId)).toBe(path.join(config.cacheDir, 'attachments', id.slice(0, 2), `${id}.bin`))
    expect(stored.storedPath).toBe(`gui/.cache/attachments/${id.slice(0, 2)}/${id}.bin`)
    await ws.cleanup()
  })

  it('refuses a cache directory outside the workspace', async () => {
    const ws = await makeTempWorkspace()
    let err: unknown = null
    try {
      createAttachmentStore(testConfig(ws))
    } catch (e) {
      err = e
    }
    expect(err).toBeInstanceOf(WorkspaceError)
    expect((err as WorkspaceError).code).toBe('OUTSIDE_WORKSPACE')
    await ws.cleanup()
  })

  it('the same bytes are one attachment', async () => {
    const { ws, config, store } = await workspace()
    const buf = Buffer.from('the same bytes twice', 'utf8')
    const first = await store.put(buf, { filename: 'same.txt', declaredType: 'text/plain' })
    const second = await store.put(buf, { filename: 'same.txt', declaredType: 'text/plain' })
    expect(second.attachmentId).toBe(first.attachmentId)
    expect(first.fresh).toBe(true)
    expect(second.fresh).toBe(false)
    const shardDir = path.join(config.cacheDir, 'attachments', first.attachmentId.slice(0, 2))
    expect((await fs.readdir(shardDir)).length).toBe(1)
    await ws.cleanup()
  })

  it('refuses a media type outside the eight', async () => {
    const { ws, store } = await workspace()
    const err = await store.put(Buffer.from('<html></html>', 'utf8'), { filename: 'x.html', declaredType: 'text/html' }).then(
      () => null,
      (e: unknown) => e,
    )
    expect(err).toBeInstanceOf(HttpError)
    expect((err as HttpError).status).toBe(415)
    expect((err as HttpError).message).toContain('text/html')
    await ws.cleanup()
  })

  it('refuses bytes that contradict the declared type', async () => {
    const { ws, store } = await workspace()
    const err = await store.put(Buffer.from('not a png', 'utf8'), { filename: 'x.png', declaredType: 'image/png' }).then(
      () => null,
      (e: unknown) => e,
    )
    expect((err as Error).message).toContain('magic')
    await ws.cleanup()
  })

  it('refuses bytes over the cap', async () => {
    const { ws, store } = await workspace()
    const big = Buffer.alloc(ATTACHMENT_MAX_BYTES + 1)
    const err = await store.put(big, { filename: 'big.bin', declaredType: 'application/octet-stream' }).then(
      () => null,
      (e: unknown) => e,
    )
    expect(err).toBeInstanceOf(HttpError)
    expect((err as HttpError).status).toBe(413)
    await ws.cleanup()
  })

  it('keeps a text extract for text types, and none for binary', async () => {
    const { ws, store } = await workspace()
    const text = await store.put(Buffer.from('a'.repeat(20 * 1024), 'utf8'), { filename: 'notes.txt', declaredType: 'text/plain' })
    expect(text.textExtract?.length).toBe(ATTACHMENT_TEXT_EXTRACT_CAP)
    const png = await store.put(Buffer.from(ONE_PIXEL_PNG, 'base64'), { filename: 'pixel.png', declaredType: 'image/png' })
    expect(png.textExtract).toBeNull()
    await ws.cleanup()
  })
})
