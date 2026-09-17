// Blocks-side tests (Run 2 requirements 2-5 and 10): the vision-mode switch,
// the model-facing shapes, the four-id cap and the dehydrator. The mirror is
// seeded directly (test seeding, not an extractor): N2's put validates the
// Attachment properties and throws MISSING_PROPERTY naming the first gap.
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import type { SessionRecord } from '../agent/session.js'
import { fakeRuns } from '../agent/test-fakes.js'
import { closeOntologyHandles, ontologyHandle } from '../ontology/handle.js'
import type { OntologyStore } from '../ontology/store.js'
import { makeTempWorkspace, testConfig, type TempWorkspace } from '../runs/test-helpers.js'
import { attachmentObjectBlocks, dehydrateAttachmentImages, visionMode } from './blocks.js'
import { createAttachmentStore, type AttachmentStore } from './store.js'

// The 70-byte 1x1 opaque-red PNG the probe also carries (C8).
const ONE_PIXEL_PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=='

let ws: TempWorkspace
let store: AttachmentStore
let mirror: OntologyStore

beforeAll(async () => {
  ws = await makeTempWorkspace()
  // cacheDir must sit INSIDE the workspace (D3); ontologyDir is a plain-spread override (CONTRACT §6).
  const config = testConfig(ws, { cacheDir: path.join(ws.root, 'gui', '.cache'), ontologyDir: path.join(ws.tmp, 'ontology') })
  store = createAttachmentStore(config)
  mirror = (await ontologyHandle({ config, runs: fakeRuns(), hub: undefined })).store
})

afterAll(async () => {
  await closeOntologyHandles()
  await ws.cleanup()
})

/** Store bytes and seed the mirror row the lookup reads. */
async function seed(filename: string, declaredType: string): Promise<{ id: string; buf: Buffer }> {
  const buf = declaredType === 'image/png' ? Buffer.from(ONE_PIXEL_PNG, 'base64') : Buffer.from('%PDF-1.4\n1 0 obj\n<</Type/Catalog>>\nendobj\n')
  const stored = await store.put(buf, { filename, declaredType })
  mirror.put({
    type: 'Attachment', id: stored.attachmentId, sourcePath: 'test',
    props: { attachmentId: stored.attachmentId, filename, mediaType: stored.mediaType, kind: 'other', bytes: buf.length, storedPath: stored.storedPath, tags: [], addedBy: 'test', addedAt: new Date().toISOString(), width: stored.width, height: stored.height, caption: null, sessionId: null, textExtract: null },
  })
  return { id: stored.attachmentId, buf }
}

describe('attachment blocks', () => {
  it('visionMode follows the provider and CFD_VISION', () => {
    expect(visionMode({ llm: 'anthropic' })).toBe('blocks')
    expect(visionMode({ llm: 'mock' })).toBe('blocks')
    expect(visionMode({ llm: 'zai' })).toBe('blocks') // the SUPPORTED literal the probe answered
    expect(visionMode({ llm: 'anthropic', vision: 'describe' })).toBe('describe')
    expect(visionMode({ llm: 'zai', vision: 'blocks' })).toBe('blocks')
  })

  it('an image becomes an image block', async () => {
    const { id, buf } = await seed('pixel.png', 'image/png')
    const { blocks, warnings } = await attachmentObjectBlocks([id], { mode: 'blocks', store, mirror })
    expect(blocks).toHaveLength(2)
    expect(blocks[0]).toEqual({ type: 'image', source: { type: 'base64', media_type: 'image/png', data: buf.toString('base64') } })
    expect(blocks[1]!.type).toBe('text')
    expect((blocks[1] as { text: string }).text).toContain(`id="${id}"`)
    expect((blocks[1] as { text: string }).text).toContain('width="1"')
    expect(warnings).toEqual([])
  })

  it('an image becomes a description', async () => {
    const { id } = await seed('pixel.png', 'image/png')
    const { blocks, warnings } = await attachmentObjectBlocks([id], { mode: 'describe', store, mirror })
    expect(blocks).toHaveLength(1)
    expect(blocks.every((b) => b.type !== 'image')).toBe(true)
    expect((blocks[0] as { text: string }).text).toContain('shown="false"')
    expect(warnings).toHaveLength(1)
    expect(warnings[0]).toContain(id)
    expect(warnings[0]).toContain('pixel.png')
  })

  it('a non-image attachment is a reference', async () => {
    const { id } = await seed('report.pdf', 'application/pdf')
    const { blocks } = await attachmentObjectBlocks([id], { mode: 'blocks', store, mirror })
    expect(blocks).toHaveLength(1)
    expect(blocks[0]!.type).toBe('text')
    expect((blocks[0] as { text: string }).text).toContain('mediaType="application/pdf"')
    expect(blocks.every((b) => b.type !== 'image')).toBe(true)
  })

  it('an unknown id is named, not thrown', async () => {
    const unknown = '0'.repeat(64)
    const { blocks, notices } = await attachmentObjectBlocks([unknown], { mode: 'blocks', store, mirror })
    expect(blocks).toHaveLength(1)
    expect((blocks[0] as { text: string }).text).toBe(`<attachment id="${unknown}" error="no such attachment"/>`)
    expect(notices).toHaveLength(1)
    const malformed = await attachmentObjectBlocks(['nope'], { mode: 'blocks', store, mirror })
    expect((malformed.blocks[0] as { text: string }).text).toBe('<attachment id="nope" error="no such attachment"/>')
  })

  it('never sends more than four', async () => {
    const ids = ['1', '2', '3', '4', '5', '6'].map((c) => c.repeat(64))
    const { blocks, notices } = await attachmentObjectBlocks(ids, { mode: 'blocks', store, mirror })
    expect(blocks).toHaveLength(4)
    expect(notices.some((n) => /2 of 6/.test(n))).toBe(true)
  })

  it('dehydrates a user image block', () => {
    const id = 'a'.repeat(64)
    const image = { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'AAAA' } }
    const reference = { type: 'text', text: `<attachment id="${id}" filename="mesh.png" mediaType="image/png" kind="other" bytes="70"/>` }
    const tail = { type: 'text', text: 'hello' }
    const assistantImage = { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'BBBB' } }
    const rec = { messages: [
      { role: 'user', content: [image, reference, tail] },
      { role: 'assistant', content: [assistantImage] },
    ] } as unknown as SessionRecord
    expect(dehydrateAttachmentImages(rec)).toBe(1)
    const content = (rec.messages[0] as { content: Array<{ type: string; text?: string }> }).content
    expect(content.some((b) => b.type === 'image')).toBe(false)
    expect(content[0]!.text).toContain(`id="${id}"`)
    expect(content[0]!.text).toContain('already shown on an earlier turn')
    expect(content[1]).toBe(reference)
    expect(content[2]).toBe(tail)
    expect(dehydrateAttachmentImages(rec)).toBe(0)
    expect((rec.messages[1] as { content: unknown[] }).content[0]).toBe(assistantImage)
  })
})
