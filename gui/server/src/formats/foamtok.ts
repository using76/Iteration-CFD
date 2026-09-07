// Streaming tokenizer for the OpenFOAM ASCII dictionary format, shared by
// the field and polyMesh readers. Bytes arrive in chunks; structural tokens
// (words, strings, punctuation) are pushed to a callback, while the body of a
// numeric list is parsed on a fast path that never allocates a JS string per
// value for the common decimal spellings. Comments (`//`, `/* */`) are
// stripped quote-aware, exactly like rust/src/io/tokenizer.rs.

export type FoamToken =
  | { kind: 'word'; text: string }
  | { kind: 'string'; text: string }
  | { kind: 'punct'; text: string }

/** Receives every number of a list body; `depth` is the paren depth relative to the list's own `(` (1 = direct child). */
export interface NumberSink {
  push(value: number, depth: number): void
}

export class FoamSyntaxError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'FoamSyntaxError'
  }
}

/** Thrown by a token handler to stop scanning early (header-only reads). */
export class StopScan extends Error {
  constructor() {
    super('stop')
    this.name = 'StopScan'
  }
}

const WS = new Uint8Array(256)
for (const c of [9, 10, 11, 12, 13, 32]) WS[c] = 1
const PUNCT = new Uint8Array(256)
for (const c of '{}()[];') PUNCT[c.charCodeAt(0)] = 1

const POW10 = new Float64Array(23)
for (let i = 0; i < 23; i++) POW10[i] = Number(`1e${i}`)

/** Parse the number spelled by buf[i..j). Exact (single rounding) for <=15 mantissa digits and |exp|<=22; otherwise Number(). */
export function parseFoamNumber(buf: Buffer, i: number, j: number): number {
  let k = i
  let neg = false
  let c = buf[k]
  if (c === 45) {
    neg = true
    k++
  } else if (c === 43) {
    k++
  }
  let mant = 0
  let digits = 0
  let scale = 0
  let sawDigit = false
  for (; k < j; k++) {
    c = buf[k]
    if (c >= 48 && c <= 57) {
      sawDigit = true
      if (mant !== 0 || c !== 48) digits++
      if (digits <= 15) mant = mant * 10 + (c - 48)
    } else break
  }
  if (k < j && buf[k] === 46) {
    k++
    for (; k < j; k++) {
      c = buf[k]
      if (c >= 48 && c <= 57) {
        sawDigit = true
        if (mant !== 0 || c !== 48) digits++
        if (digits <= 15) {
          mant = mant * 10 + (c - 48)
          scale--
        }
      } else break
    }
  }
  if (!sawDigit || digits > 15) return slowNumber(buf, i, j)
  if (k < j) {
    c = buf[k]
    if (c !== 101 && c !== 69) return slowNumber(buf, i, j)
    k++
    let eneg = false
    if (k < j && (buf[k] === 45 || buf[k] === 43)) {
      eneg = buf[k] === 45
      k++
    }
    let e = 0
    let eDigits = 0
    for (; k < j; k++) {
      c = buf[k]
      if (c < 48 || c > 57) return slowNumber(buf, i, j)
      e = e * 10 + (c - 48)
      eDigits++
      if (e > 100000) return slowNumber(buf, i, j)
    }
    if (eDigits === 0) return slowNumber(buf, i, j)
    scale += eneg ? -e : e
  }
  if (mant === 0) return neg ? -0 : 0
  let v: number
  if (scale === 0) v = mant
  else if (scale > 0 && scale <= 22) v = mant * POW10[scale]
  else if (scale < 0 && scale >= -22) v = mant / POW10[-scale]
  else return slowNumber(buf, i, j)
  return neg ? -v : v
}

function slowNumber(buf: Buffer, i: number, j: number): number {
  const s = buf.toString('latin1', i, j)
  const v = Number(s)
  if (v === v) return v
  const t = s.toLowerCase()
  if (t === 'inf' || t === '+inf' || t === 'infinity') return Infinity
  if (t === '-inf' || t === '-infinity') return -Infinity
  if (t.endsWith('nan')) return NaN
  throw new FoamSyntaxError(`not a number: "${s}"`)
}

export class FoamScanner {
  private carry: Buffer | null = null
  private inLineComment = false
  private inBlockComment = false
  private inString = false
  private str = ''
  private listSink: NumberSink | null = null
  private listDepth = 0
  private inList = false

  constructor(private readonly onToken: (tok: FoamToken) => void) {}

  /** Switch to list mode right after the parser consumed the list's `(`; `null` skips the numbers without parsing them. */
  beginList(sink: NumberSink | null): void {
    this.inList = true
    this.listSink = sink
    this.listDepth = 1
  }

  feed(chunk: Buffer): void {
    const buf = this.carry ? Buffer.concat([this.carry, chunk]) : chunk
    this.carry = null
    this.scan(buf, false)
  }

  finish(): void {
    const buf = this.carry ?? Buffer.alloc(0)
    this.carry = null
    this.scan(buf, true)
    if (this.inString) throw new FoamSyntaxError('unterminated string')
    if (this.inList) throw new FoamSyntaxError('unterminated list')
  }

  private scan(buf: Buffer, atEof: boolean): void {
    const end = buf.length
    let i = 0
    while (i < end) {
      if (this.inLineComment) {
        const nl = buf.indexOf(10, i)
        if (nl < 0) return
        i = nl + 1
        this.inLineComment = false
        continue
      }
      if (this.inBlockComment) {
        const close = buf.indexOf('*/', i, 'latin1')
        if (close < 0) {
          // A `*` at the very end may pair with a `/` in the next chunk.
          if (!atEof && buf[end - 1] === 42) this.carry = Buffer.from(buf.subarray(end - 1))
          return
        }
        i = close + 2
        this.inBlockComment = false
        continue
      }
      if (this.inString) {
        const q = buf.indexOf(34, i)
        if (q < 0) {
          this.str += buf.toString('latin1', i, end)
          return
        }
        this.str += buf.toString('latin1', i, q)
        this.inString = false
        this.onToken({ kind: 'string', text: this.str })
        this.str = ''
        i = q + 1
        continue
      }
      i = this.inList ? this.scanList(buf, i, end, atEof) : this.scanTokens(buf, i, end, atEof)
      if (i < 0) return
    }
  }

  /** Returns the next index to scan from, or -1 when the chunk is exhausted with a carry saved. */
  private scanTokens(buf: Buffer, i: number, end: number, atEof: boolean): number {
    while (i < end) {
      const c = buf[i]
      if (WS[c]) {
        i++
        continue
      }
      if (c === 34) {
        this.inString = true
        return i + 1
      }
      if (c === 47) {
        if (i + 1 >= end) {
          if (!atEof) {
            this.carry = Buffer.from(buf.subarray(i))
            return -1
          }
        } else if (buf[i + 1] === 47) {
          this.inLineComment = true
          return i + 2
        } else if (buf[i + 1] === 42) {
          this.inBlockComment = true
          return i + 2
        }
      }
      if (PUNCT[c]) {
        this.onToken({ kind: 'punct', text: String.fromCharCode(c) })
        i++
        if (this.inList || this.inString) return i
        continue
      }
      let j = i + 1
      while (j < end) {
        const d = buf[j]
        if (WS[d] || PUNCT[d] || d === 34) break
        if (d === 47 && j + 1 < end && (buf[j + 1] === 47 || buf[j + 1] === 42)) break
        j++
      }
      if (j >= end && !atEof) {
        this.carry = Buffer.from(buf.subarray(i))
        return -1
      }
      this.onToken({ kind: 'word', text: buf.toString('latin1', i, j) })
      i = j
      if (this.inList || this.inString) return i
    }
    return i
  }

  private scanList(buf: Buffer, i: number, end: number, atEof: boolean): number {
    const sink = this.listSink
    let depth = this.listDepth
    while (i < end) {
      const c = buf[i]
      if (WS[c]) {
        i++
        continue
      }
      if (c === 40) {
        depth++
        i++
        continue
      }
      if (c === 41) {
        depth--
        i++
        if (depth === 0) {
          this.inList = false
          this.listSink = null
          this.listDepth = 0
          this.onToken({ kind: 'punct', text: ')' })
          return i
        }
        continue
      }
      if (c === 47) {
        if (i + 1 >= end) {
          if (!atEof) {
            this.listDepth = depth
            this.carry = Buffer.from(buf.subarray(i))
            return -1
          }
        } else if (buf[i + 1] === 47) {
          this.inLineComment = true
          this.listDepth = depth
          return i + 2
        } else if (buf[i + 1] === 42) {
          this.inBlockComment = true
          this.listDepth = depth
          return i + 2
        }
      }
      let j = i + 1
      while (j < end) {
        const d = buf[j]
        if (WS[d] || d === 40 || d === 41) break
        j++
      }
      if (j >= end && !atEof) {
        this.listDepth = depth
        this.carry = Buffer.from(buf.subarray(i))
        return -1
      }
      if (sink) sink.push(parseFoamNumber(buf, i, j), depth)
      i = j
    }
    this.listDepth = depth
    return i
  }
}

/** Tokenise a whole (small) text into an array; list bodies are tokenised as ordinary words. */
export function tokenizeAll(text: string): FoamToken[] {
  const out: FoamToken[] = []
  const sc = new FoamScanner((t) => out.push(t))
  sc.feed(Buffer.from(text, 'latin1'))
  sc.finish()
  return out
}

/** Cursor over a token array for the small dictionary-style files (boundary, headers). */
export class TokenCursor {
  pos = 0
  constructor(readonly toks: FoamToken[]) {}
  done(): boolean {
    return this.pos >= this.toks.length
  }
  peek(offset = 0): FoamToken | undefined {
    return this.toks[this.pos + offset]
  }
  next(): FoamToken {
    const t = this.toks[this.pos]
    if (!t) throw new FoamSyntaxError('unexpected end of file')
    this.pos++
    return t
  }
  isPunct(ch: string, offset = 0): boolean {
    const t = this.peek(offset)
    return t !== undefined && t.kind === 'punct' && t.text === ch
  }
  isWord(text: string): boolean {
    const t = this.peek()
    return t !== undefined && t.kind === 'word' && t.text === text
  }
  expectPunct(ch: string): void {
    const t = this.next()
    if (t.kind !== 'punct' || t.text !== ch) throw new FoamSyntaxError(`expected '${ch}', got '${t.text}'`)
  }
  expectWord(): string {
    const t = this.next()
    if (t.kind === 'punct') throw new FoamSyntaxError(`expected a word, got '${t.text}'`)
    return t.text
  }
  expectInt(): number {
    const t = this.next()
    if (t.kind !== 'word') throw new FoamSyntaxError(`expected a number, got '${t.text}'`)
    const v = Number(t.text)
    if (!Number.isInteger(v)) throw new FoamSyntaxError(`expected an integer, got '${t.text}'`)
    return v
  }
  expectNumber(): number {
    const t = this.next()
    if (t.kind !== 'word') throw new FoamSyntaxError(`expected a number, got '${t.text}'`)
    return parseFoamNumber(Buffer.from(t.text, 'latin1'), 0, t.text.length)
  }
  /** Raw text of an entry value up to its `;` (consumed), OpenFOAM spacing. */
  gatherRaw(): string {
    const parts: string[] = []
    let depth = 0
    for (;;) {
      const t = this.next()
      if (t.kind === 'punct') {
        if (t.text === ';' && depth === 0) break
        if (t.text === '(' || t.text === '[' || t.text === '{') depth++
        if (t.text === ')' || t.text === ']' || t.text === '}') depth--
      }
      parts.push(t.kind === 'string' ? `"${t.text}"` : t.text)
    }
    return joinRaw(parts)
  }
  /** Skip a `{ ... }` sub-dictionary (the `{` is the next token). */
  skipDict(): void {
    this.expectPunct('{')
    let depth = 1
    while (depth > 0) {
      const t = this.next()
      if (t.kind === 'punct') {
        if (t.text === '{') depth++
        else if (t.text === '}') depth--
      }
    }
  }
}

/** Join tokens back into OpenFOAM's own spacing: no space after `(`/`[` or before `)`/`]`. */
export function joinRaw(parts: string[]): string {
  let out = ''
  for (let k = 0; k < parts.length; k++) {
    const p = parts[k]
    if (k > 0) {
      const prev = parts[k - 1]
      const tight = prev === '(' || prev === '[' || p === ')' || p === ']'
      if (!tight) out += ' '
    }
    out += p
  }
  return out
}

/** Parse a `FoamFile { k v; ... }` dictionary at the cursor (the `FoamFile` word is the next token). Missing header -> empty map. */
export function parseFoamFileDict(cur: TokenCursor): Map<string, string> {
  const out = new Map<string, string>()
  if (!cur.isWord('FoamFile')) return out
  cur.next()
  cur.expectPunct('{')
  while (!cur.done() && !cur.isPunct('}')) {
    if (cur.isPunct(';')) {
      cur.next()
      continue
    }
    const k = cur.expectWord()
    if (cur.isPunct('{')) {
      cur.skipDict()
      continue
    }
    out.set(k, cur.gatherRaw())
  }
  cur.expectPunct('}')
  return out
}

/** Strip one layer of double quotes from a header value like `"constant/polyMesh"`. */
export function unquote(s: string): string {
  return s.length >= 2 && s.startsWith('"') && s.endsWith('"') ? s.slice(1, -1) : s
}
