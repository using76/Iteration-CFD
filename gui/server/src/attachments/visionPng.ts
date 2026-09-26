// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The vision probe's drawing kit: a hand-rolled 5x7 glyph font, a two-pixel
// Bresenham line, and a PNG encoder that emits exactly IHDR + one IDAT + IEND,
// so a challenge's answer can only live in the IDAT pixels — never in a text
// chunk (AMG-11 R2).
import { crc32, deflateSync } from 'node:zlib'

export const WIDTH = 480
export const HEIGHT = 340

/** 5 columns x 7 rows per glyph; '#' is ink. Exported so the test can read the pixels back. */
export const FONT: Record<string, readonly string[]> = {
  '0': ['.###.', '#...#', '#..##', '#.#.#', '##..#', '#...#', '.###.'],
  '1': ['..#..', '.##..', '..#..', '..#..', '..#..', '..#..', '.###.'],
  '2': ['.###.', '#...#', '....#', '...#.', '..#..', '.#...', '#####'],
  '3': ['#####', '...#.', '..#..', '...#.', '....#', '#...#', '.###.'],
  '4': ['...#.', '..##.', '.#.#.', '#..#.', '#####', '...#.', '...#.'],
  '5': ['#####', '#....', '####.', '....#', '....#', '#...#', '.###.'],
  '6': ['..##.', '.#...', '#....', '####.', '#...#', '#...#', '.###.'],
  '7': ['#####', '....#', '...#.', '..#..', '.#...', '.#...', '.#...'],
  '8': ['.###.', '#...#', '#...#', '.###.', '#...#', '#...#', '.###.'],
  '9': ['.###.', '#...#', '#...#', '.####', '....#', '...#.', '.##..'],
  D: ['###..', '#..#.', '#...#', '#...#', '#...#', '#..#.', '###..'],
  L: ['#....', '#....', '#....', '#....', '#....', '#....', '#####'],
  '=': ['.....', '.....', '#####', '.....', '#####', '.....', '.....'],
}

/** A grayscale canvas: 255 background, 0 ink, row-major, WIDTH x HEIGHT bytes. */
export function blankCanvas(): Uint8Array {
  return new Uint8Array(WIDTH * HEIGHT).fill(255)
}

/** Ink one pixel; silently ignores coordinates outside the canvas. */
export function setPixel(c: Uint8Array, x: number, y: number): void {
  if (x < 0 || y < 0 || x >= WIDTH || y >= HEIGHT) return
  c[y * WIDTH + x] = 0
}

/** Glyph cell (col,row) '#' fills the square [x+col*scale, x+col*scale+scale) x [y+row*scale, ...). Advance 6*scale per character. Throws on a character not in FONT. */
export function drawText(c: Uint8Array, x: number, y: number, text: string, scale: number): void {
  let cx = x
  for (const ch of text) {
    const glyph = FONT[ch]
    if (!glyph) throw new Error(`no glyph for ${JSON.stringify(ch)}`)
    for (let row = 0; row < glyph.length; row++) {
      const cells = glyph[row] as string
      for (let col = 0; col < cells.length; col++) {
        if (cells[col] !== '#') continue
        for (let dy = 0; dy < scale; dy++) {
          for (let dx = 0; dx < scale; dx++) setPixel(c, cx + col * scale + dx, y + row * scale + dy)
        }
      }
    }
    cx += 6 * scale
  }
}

/** Bresenham from (x0,y0) to (x1,y1) inclusive, 2 px thick: each point p also inks p+(1,0) and p+(0,1). */
export function drawLine(c: Uint8Array, x0: number, y0: number, x1: number, y1: number): void {
  const dx = Math.abs(x1 - x0)
  const dy = -Math.abs(y1 - y0)
  const sx = x0 < x1 ? 1 : -1
  const sy = y0 < y1 ? 1 : -1
  let err = dx + dy
  let x = x0
  let y = y0
  for (;;) {
    setPixel(c, x, y)
    setPixel(c, x + 1, y)
    setPixel(c, x, y + 1)
    if (x === x1 && y === y1) break
    const e2 = 2 * err
    if (e2 >= dy) {
      err += dy
      x += sx
    }
    if (e2 <= dx) {
      err += dx
      y += sy
    }
  }
}

/** chunk = length (uint32 BE of data) + type (4 ASCII bytes) + data + crc32(type + data) (uint32 BE, from node:zlib crc32). */
function chunk(type: string, data: Buffer): Buffer {
  const len = Buffer.alloc(4)
  len.writeUInt32BE(data.length, 0)
  const typeBuf = Buffer.from(type, 'ascii')
  const crc = Buffer.alloc(4)
  crc.writeUInt32BE(crc32(Buffer.concat([typeBuf, data])), 0)
  return Buffer.concat([len, typeBuf, data, crc])
}

/** PNG: 8-byte signature, IHDR(WIDTH, HEIGHT, depth 8, colour type 0, compression 0, filter 0, interlace 0), ONE IDAT = deflateSync(rows, each prefixed by filter byte 0), IEND. No other chunk. */
export function encodePng(c: Uint8Array): Buffer {
  const ihdr = Buffer.alloc(13)
  ihdr.writeUInt32BE(WIDTH, 0)
  ihdr.writeUInt32BE(HEIGHT, 4)
  ihdr[8] = 8 // bit depth
  ihdr[9] = 0 // colour type: grayscale
  ihdr[10] = 0 // compression
  ihdr[11] = 0 // filter
  ihdr[12] = 0 // interlace
  const stride = 1 + WIDTH
  const raw = Buffer.alloc(HEIGHT * stride)
  for (let y = 0; y < HEIGHT; y++) {
    raw[y * stride] = 0 // filter byte: None
    Buffer.from(c.buffer, c.byteOffset + y * WIDTH, WIDTH).copy(raw, y * stride + 1)
  }
  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw)),
    chunk('IEND', Buffer.alloc(0)),
  ])
}

/** Draws the challenge picture and returns its PNG. */
export function drawSketch(number: string, D_i: number, L: number): Buffer {
  const c = blankCanvas()
  drawText(c, 24, 16, number, 6) // the four-digit number, top left
  drawLine(c, 150, 130, 440, 170) // upper wall
  drawLine(c, 150, 270, 440, 230) // lower wall
  drawLine(c, 150, 130, 150, 270) // inlet plane
  drawLine(c, 440, 170, 440, 230) // exit plane
  drawLine(c, 130, 130, 130, 270) // D dimension line
  drawLine(c, 124, 130, 136, 130)
  drawLine(c, 124, 270, 136, 270)
  drawText(c, 12, 190, `D=${D_i}`, 3)
  drawLine(c, 150, 320, 440, 320) // L dimension line
  drawLine(c, 150, 314, 150, 326)
  drawLine(c, 440, 314, 440, 326)
  drawText(c, 250, 286, `L=${L}`, 3)
  return encodePng(c)
}
