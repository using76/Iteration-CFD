// Content-addressed attachment bytes over the dataset BlobStore: the id IS
// the sha256 of the bytes, so the same bytes are one object everywhere and
// the shard is derivable from the id alone. There is no metadata sidecar:
// the mirror row is the metadata (N6 D4), only the bytes live here.
import { createHash } from 'node:crypto'
import path from 'node:path'
import type { ServerConfig } from '../config.js'
import { BlobStore } from '../datasets/blobs.js'
import { HttpError } from '../http/router.js'
import { resolveInWorkspace } from '../workspace/paths.js'
import { imageSize, magicBytesAgree, normaliseMediaType, type AttachmentMediaType } from './sniff.js'

/** The route's 413 and the store's own cap agree: 20 MB of bytes, not the router's 32 MB wall (D8). */
export const ATTACHMENT_MAX_BYTES = 20 * 1024 * 1024
export const ATTACHMENT_MEMORY_BYTES = 64 * 1024 * 1024
export const ATTACHMENT_TEXT_EXTRACT_CAP = 16 * 1024

export interface StoredAttachment {
  /** The bytes' sha256, 64 lowercase hex characters. This IS the primary key (D5); there is no second field for it. */
  attachmentId: string
  filename: string
  mediaType: AttachmentMediaType
  bytes: number
  width: number | null
  height: number | null
  /** Workspace-relative: gui/.cache/attachments/<shard>/<id>.bin */
  storedPath: string
  /** ≤16 KB of UTF-8 for text/plain and application/json; null otherwise. */
  textExtract: string | null
  /** False when these exact bytes were already stored. */
  fresh: boolean
}

export interface AttachmentStore {
  put(buf: Buffer, meta: { filename: string; declaredType: string | null }): Promise<StoredAttachment>
  get(attachmentId: string): Promise<Buffer | null>
  /** The absolute path the bytes live at; for a test and for `storedPath`. */
  pathOf(attachmentId: string): string
  /** Workspace-relative, forward slashes: `gui/.cache/attachments/<shard>/<id>.bin`. */
  storedPathOf(attachmentId: string): string
}

/** `/^[0-9a-f]{64}$/` — the only shape an attachment id ever has (D5, D16). */
export function isAttachmentId(id: string): boolean {
  return /^[0-9a-f]{64}$/.test(id)
}

const TEXT_TYPES: readonly AttachmentMediaType[] = ['text/plain', 'application/json']

/**
 * A second BlobStore instance whose directory lives INSIDE the workspace
 * (D3): the mirror's attachFile proposal hands `storedPath` to the
 * pathsInsideWorkspace criterion, which an outside directory fails before
 * the row is ever rendered — so a store that cannot sit in the workspace is
 * built to refuse construction rather than 400 every upload later.
 */
export function createAttachmentStore(config: Pick<ServerConfig, 'cacheDir' | 'workspaceRoot'>, blobs?: BlobStore): AttachmentStore {
  const dir = path.join(config.cacheDir, 'attachments')
  const resolved = resolveInWorkspace(config.workspaceRoot, dir)
  // resolveInWorkspace already returns the workspace-relative path, computed against the
  // realpath-resolved root; recomputing it from the raw root breaks when the root is a symlink.
  const relDir = resolved.rel
  const blobs_ = blobs ?? new BlobStore({ dir: resolved.abs, maxMemoryBytes: ATTACHMENT_MEMORY_BYTES })
  const textExtractOf = (mediaType: AttachmentMediaType, buf: Buffer): string | null =>
    TEXT_TYPES.includes(mediaType) ? buf.subarray(0, ATTACHMENT_TEXT_EXTRACT_CAP).toString('utf8') : null
  return {
    async put(buf, meta) {
      const attachmentId = createHash('sha256').update(buf).digest('hex')
      const mediaType = normaliseMediaType(meta.declaredType, meta.filename)
      if (!mediaType) throw new HttpError(415, `media type ${meta.declaredType ?? '(none)'} is not an allowed attachment type`)
      if (!magicBytesAgree(mediaType, buf)) throw new HttpError(415, `declared ${mediaType} but the magic bytes disagree with it`)
      if (buf.length > ATTACHMENT_MAX_BYTES) throw new HttpError(413, `attachment larger than ${ATTACHMENT_MAX_BYTES} bytes`)
      // The shard is derivable from the id alone, which is the whole reason the id is the hash.
      const shard = attachmentId.slice(0, 2)
      const fresh = !(await blobs_.exists(shard, attachmentId))
      if (fresh) await blobs_.put(shard, attachmentId, buf)
      const size = imageSize(mediaType, buf)
      return {
        attachmentId,
        filename: meta.filename,
        mediaType,
        bytes: buf.length,
        width: size?.width ?? null,
        height: size?.height ?? null,
        storedPath: path.posix.join(relDir, shard, `${attachmentId}.bin`),
        textExtract: textExtractOf(mediaType, buf),
        fresh,
      }
    },
    async get(attachmentId) {
      if (!isAttachmentId(attachmentId)) return null
      return blobs_.get(attachmentId.slice(0, 2), attachmentId)
    },
    pathOf(attachmentId) {
      // BlobStore.fileKey appends '.bin' after an encodeURIComponent pass;
      // for a 64-hex id that encoding is the identity, spelled out here
      // because fileKey is private.
      return path.join(blobs_.datasetDir(attachmentId.slice(0, 2)), `${encodeURIComponent(attachmentId)}.bin`)
    },
    storedPathOf(attachmentId) {
      return path.posix.join(relDir, attachmentId.slice(0, 2), `${attachmentId}.bin`)
    },
  }
}
