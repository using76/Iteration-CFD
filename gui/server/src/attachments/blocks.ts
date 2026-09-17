// The model side of attachments (Run 2): which vision mode the provider gets,
// the per-attachment shapes the model reads (C12), and the once-only replay
// rule — an image block lives in rec.messages for the turn it arrives on and
// is dehydrated back to its <attachment/> reference at turn end (D12), because
// sessions have no compaction and one base64 blob would be re-sent and
// re-written to disk on every round of every later turn.
import type { BetaContentBlockParam, BetaImageBlockParam } from '@anthropic-ai/sdk/resources/beta/messages/messages'
import type { SessionRecord } from '../agent/session.js'
import type { ServerConfig } from '../config.js'
import type { OntologyStore } from '../ontology/store.js'
import { isAttachmentId, type AttachmentStore } from './store.js'

export type VisionMode = 'blocks' | 'describe'

/**
 * CFD_VISION wins; otherwise the provider decides. anthropic and mock have
 * always been sent image blocks, and z.ai's Anthropic-compatible endpoint
 * accepted one on 2026-09-15 — probe verdict SUPPORTED
 * (gui/.cache/probe/zai-image-block.json, stop_reason end_turn) — so its
 * default is the literal below. CFD_VISION=describe is how a user turns
 * images off for every provider at once.
 */
export function visionMode(config: Pick<ServerConfig, 'llm' | 'vision'>): VisionMode {
  if (config.vision === 'blocks' || config.vision === 'describe') return config.vision
  if (config.llm === 'zai') return 'blocks' // the probe's answer; flip to 'describe' if the endpoint regresses
  return 'blocks'
}

export interface AttachmentFacts {
  attachmentId: string
  filename: string
  mediaType: string
  kind: string
  bytes: number
  width: number | null
  height: number | null
  caption: string | null
}

/** The reference every mode emits (C12); width/height/caption are omitted when null. */
export function attachmentReferenceText(a: AttachmentFacts): string {
  const attrs = [
    `id="${a.attachmentId}"`,
    `filename="${a.filename}"`,
    `mediaType="${a.mediaType}"`,
    `kind="${a.kind}"`,
    `bytes="${a.bytes}"`,
    a.width === null ? null : `width="${a.width}"`,
    a.height === null ? null : `height="${a.height}"`,
    a.caption === null ? null : `caption="${a.caption}"`,
  ].filter((s): s is string => s !== null)
  return `<attachment ${attrs.join(' ')}/>`
}

/** describe mode only: the same reference with shown="false" and the refusal named (C12). */
export function attachmentDescription(a: AttachmentFacts): string {
  return attachmentReferenceText(a).replace(/\/>$/, ' shown="false" note="this endpoint does not accept image input; the image was stored but not shown to the model"/>')
}

/** The reference carries the id; the dehydrator recovers it from here. */
const REFERENCE_ID = /id="([0-9a-f]{64})"/

/** An image block is worth thousands of tokens and the round already carries 79,410 bytes of tool definitions, so ids cap below MAX_ATTACHMENTS. */
export const MAX_ATTACHMENT_IDS = 4

export interface AttachmentBlocksResult {
  blocks: BetaContentBlockParam[]
  notices: string[]
  /** Shown in the bubble as level 'warning'; one per image the model was not shown. */
  warnings: string[]
}

/** Resolve attachment ids into model-facing blocks; an unknown id becomes an error reference, never a throw. */
export async function attachmentObjectBlocks(ids: readonly string[], deps: { mode: VisionMode; store: AttachmentStore; mirror: OntologyStore }): Promise<AttachmentBlocksResult> {
  const blocks: BetaContentBlockParam[] = []
  const notices: string[] = []
  const warnings: string[] = []
  const sent = ids.slice(0, MAX_ATTACHMENT_IDS)
  const dropped = ids.length - sent.length
  for (const id of sent) {
    try {
      if (!isAttachmentId(id)) throw new Error('no such attachment')
      const row = deps.mirror.get('Attachment', id)
      if (!row) throw new Error('no such attachment')
      const p = row.props
      const facts: AttachmentFacts = {
        attachmentId: id,
        filename: typeof p.filename === 'string' ? p.filename : '',
        mediaType: typeof p.mediaType === 'string' ? p.mediaType : 'application/octet-stream',
        kind: typeof p.kind === 'string' ? p.kind : 'other',
        bytes: typeof p.bytes === 'number' ? p.bytes : 0,
        width: typeof p.width === 'number' ? p.width : null,
        height: typeof p.height === 'number' ? p.height : null,
        caption: typeof p.caption === 'string' ? p.caption : null,
      }
      const image = facts.mediaType.startsWith('image/')
      const shown = image && deps.mode === 'blocks'
      if (shown) {
        const buf = await deps.store.get(id)
        if (!buf) throw new Error('no such attachment')
        // The reference is emitted immediately AFTER the image: the dehydrator pairs them by position (D12).
        blocks.push({ type: 'image', source: { type: 'base64', media_type: facts.mediaType as 'image/png', data: buf.toString('base64') } })
      }
      blocks.push({ type: 'text', text: shown || !image ? attachmentReferenceText(facts) : attachmentDescription(facts) })
      if (image && !shown) warnings.push(`${id} (${facts.filename}): the model could not be shown this image — the endpoint refuses image input`)
      notices.push(`@${facts.filename} (${facts.bytes} bytes, attachment ${id})`)
    } catch (err) {
      blocks.push({ type: 'text', text: `<attachment id="${id}" error="${(err as Error).message}"/>` })
      notices.push(`attachment ${id}: ${(err as Error).message}`)
    }
  }
  if (dropped > 0) notices.push(`${dropped} of ${ids.length} attachment ids dropped (at most ${MAX_ATTACHMENT_IDS} are sent)`)
  return { blocks, notices, warnings }
}

/**
 * Replaces every user-role image block with its <attachment/> reference (D12).
 * Returns how many it replaced. The id is recovered from the adjacent
 * reference block — the image is always emitted immediately before it — and a
 * tool_result's own screenshot images (nested inside that block type, and on
 * assistant turns) are never touched.
 */
export function dehydrateAttachmentImages(rec: SessionRecord): number {
  let replaced = 0
  for (const msg of rec.messages) {
    if (msg.role !== 'user' || typeof msg.content === 'string') continue
    const content = msg.content as unknown as Array<Record<string, unknown>>
    for (let i = 0; i < content.length; i++) {
      const block = content[i]
      if (!block || block.type !== 'image') continue
      const id = idFrom(content[i + 1]) ?? idFrom(content[i - 1])
      content[i] = { type: 'text', text: id ? `<attachment id="${id}" shown="false" note="already shown on an earlier turn"/>` : '<attachment shown="false" note="an image shown on an earlier turn"/>' }
      replaced++
    }
  }
  return replaced
}

function idFrom(block: unknown): string | null {
  const text = (block as { text?: unknown } | undefined)?.text
  return typeof text === 'string' ? (REFERENCE_ID.exec(text)?.[1] ?? null) : null
}
