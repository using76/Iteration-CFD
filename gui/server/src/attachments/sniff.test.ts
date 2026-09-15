import { describe, expect, it } from 'vitest'
import { imageSize, looksBinary, magicBytesAgree, normaliseMediaType } from './sniff.js'

// The 70-byte 1x1 opaque-red PNG the probe also carries (C8).
const ONE_PIXEL_PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=='

/** A hand-built minimal JPEG: SOI, SOF0 declaring height 11 width 7, SOS, EOI. */
function minimalJpeg(): Buffer {
  return Buffer.concat([
    Buffer.from([0xff, 0xd8]),
    Buffer.from([0xff, 0xc0, 0x00, 0x11, 0x08]),
    Buffer.from([0x00, 0x0b, 0x00, 0x07]),
    Buffer.from([0x03, 0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01]),
    Buffer.from([0xff, 0xda, 0x00, 0x02]),
    Buffer.from([0xff, 0xd9]),
  ])
}

describe('attachment sniffing', () => {
  it('normalises the declared type and falls back to the extension', () => {
    expect(normaliseMediaType(null, 'a.png')).toBe('image/png')
    expect(normaliseMediaType('image/png; charset=binary', 'a.png')).toBe('image/png')
    expect(normaliseMediaType('APPLICATION/PDF', 'a.pdf')).toBe('application/pdf')
    expect(normaliseMediaType(null, 'a.stp')).toBe('model/step')
    expect(normaliseMediaType(null, 'a.bin')).toBe('application/octet-stream')
    expect(normaliseMediaType('text/html', 'a.html')).toBeNull()
  })

  it('reads PNG and JPEG dimensions', () => {
    const png = Buffer.from(ONE_PIXEL_PNG, 'base64')
    expect(png.length).toBe(70)
    expect(imageSize('image/png', png)).toEqual({ width: 1, height: 1 })
    expect(imageSize('image/jpeg', minimalJpeg())).toEqual({ width: 7, height: 11 })
    expect(imageSize('image/png', Buffer.from([0x89, 0x50, 0x4e]))).toBeNull()
    expect(imageSize('application/pdf', Buffer.from('%PDF-1.7 rest of a pdf', 'ascii'))).toBeNull()
  })

  it('calls a buffer with a NUL byte binary', () => {
    expect(looksBinary(Buffer.from([0x61, 0x00, 0x62]))).toBe(true)
    expect(looksBinary(Buffer.from('plain ascii', 'ascii'))).toBe(false)
  })

  it('checks magic bytes for the three types that carry a signature', () => {
    const png = Buffer.from(ONE_PIXEL_PNG, 'base64')
    expect(magicBytesAgree('image/png', png)).toBe(true)
    expect(magicBytesAgree('image/png', Buffer.from('not a png', 'ascii'))).toBe(false)
    expect(magicBytesAgree('image/jpeg', Buffer.from([0xff, 0xd8, 0xff, 0xe0]))).toBe(true)
    expect(magicBytesAgree('application/pdf', Buffer.from('%PDF-1.7', 'ascii'))).toBe(true)
    expect(magicBytesAgree('application/pdf', Buffer.from('not pdf', 'ascii'))).toBe(false)
    expect(magicBytesAgree('text/plain', Buffer.from('anything', 'ascii'))).toBe(true)
  })
})
