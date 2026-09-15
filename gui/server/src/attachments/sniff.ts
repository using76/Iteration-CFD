// Attachment sniffing: the allow-list of eight media types, the declared-type
// vs magic-bytes agreement check (png/jpeg/pdf carry a signature worth
// checking), and PNG IHDR / JPEG SOF dimension parsing only (D10: every
// other type stores width = height = null; WebP's three container variants
// are not worth another 40 lines in a unit that already carries a probe).
export const ATTACHMENT_MEDIA_TYPES = ['image/png', 'image/jpeg', 'image/webp', 'application/pdf', 'text/plain', 'application/json', 'model/step', 'application/octet-stream'] as const
export type AttachmentMediaType = (typeof ATTACHMENT_MEDIA_TYPES)[number]
export const ATTACHMENT_KINDS = ['screenshot', 'photo', 'drawing', 'geometry', 'report', 'log', 'other'] as const
export type AttachmentKind = (typeof ATTACHMENT_KINDS)[number]

// Fallback when no declared type arrived: the upload's own extension. Types
// outside the eight are refused, so the map only names the eight.
const EXTENSION_TYPES: Record<string, AttachmentMediaType> = {
  png: 'image/png',
  jpg: 'image/jpeg',
  jpeg: 'image/jpeg',
  webp: 'image/webp',
  pdf: 'application/pdf',
  txt: 'text/plain',
  log: 'text/plain',
  json: 'application/json',
  jsonc: 'application/json',
  step: 'model/step',
  stp: 'model/step',
  bin: 'application/octet-stream',
}

/** The declared type, normalised and checked against the allow-list; null when it is not one of the eight. */
export function normaliseMediaType(declared: string | null, filename: string): AttachmentMediaType | null {
  const mime = (declared ?? '').split(';')[0].trim().toLowerCase()
  if (mime) return (ATTACHMENT_MEDIA_TYPES as readonly string[]).includes(mime) ? (mime as AttachmentMediaType) : null
  const ext = filename.includes('.') ? (filename.split('.').pop() ?? '').toLowerCase() : ''
  return EXTENSION_TYPES[ext] ?? 'application/octet-stream'
}

const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
const JPEG_SIGNATURE = Buffer.from([0xff, 0xd8, 0xff])
const PDF_SIGNATURE = Buffer.from('%PDF-', 'ascii')

/** True when the bytes agree with the declared type. Only png/jpeg/pdf have a signature worth checking; everything else passes. */
export function magicBytesAgree(mediaType: AttachmentMediaType, buf: Buffer): boolean {
  const head = buf.subarray(0, 8)
  if (mediaType === 'image/png') return head.length >= 8 && head.equals(PNG_SIGNATURE)
  if (mediaType === 'image/jpeg') return head.subarray(0, 3).equals(JPEG_SIGNATURE)
  if (mediaType === 'application/pdf') return head.subarray(0, 5).equals(PDF_SIGNATURE)
  return true
}

const JPEG_SOF_MARKERS = new Set([0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf])

/** PNG IHDR and JPEG SOF only (D10); null for every other type or a truncated header. */
export function imageSize(mediaType: AttachmentMediaType, buf: Buffer): { width: number; height: number } | null {
  if (mediaType === 'image/png') {
    // Signature is bytes 0-7, then a 4-byte length and 'IHDR'; the payload's
    // first eight bytes are the big-endian width and height.
    if (buf.length < 24) return null
    return { width: buf.readUInt32BE(16), height: buf.readUInt32BE(20) }
  }
  if (mediaType !== 'image/jpeg') return null
  // Walk the marker segments from offset 2: FF <marker> <len:2>; skip the
  // standalone markers (FF01, FFD0-FFD7, embedded FFD8), stop at SOS (FFDA).
  let i = 2
  while (i + 4 <= buf.length) {
    if (buf[i] !== 0xff) return null
    const marker = buf[i + 1]
    if (marker === 0x01 || (marker >= 0xd0 && marker <= 0xd8)) { i += 2; continue }
    if (marker === 0xd9 || marker === 0xda) return null
    const len = buf.readUInt16BE(i + 2)
    if (len < 2) return null
    if (JPEG_SOF_MARKERS.has(marker)) {
      if (i + 9 > buf.length) return null
      return { width: buf.readUInt16BE(i + 7), height: buf.readUInt16BE(i + 5) }
    }
    i += 2 + len
  }
  return null
}

/** A NUL byte in the first 8 KB means "do not decode this as text" (used by Run 2, exported now). */
export function looksBinary(buf: Buffer): boolean {
  return buf.includes(0)
}
