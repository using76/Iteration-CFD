// OpenFOAM ASCII vol*Field reader/writer, matching rust/src/io/fields.rs.
// The reader streams the file through FoamScanner and fills a Float32Array
// directly; the writer reproduces fields.rs' layout (FoamFile header,
// `dimensions`, `internalField`, `boundaryField`) with `%g`-style numbers.
import fs from 'node:fs/promises'
import path from 'node:path'
import { once } from 'node:events'
import { createWriteStream } from 'node:fs'
import { FoamScanner, FoamSyntaxError, StopScan, joinRaw, unquote, type FoamToken, type NumberSink } from './foamtok.js'

export interface FoamFieldHeader {
  version: string
  format: string
  /** volScalarField | volVectorField | surfaceScalarField | ... ('' when the file has no FoamFile header). */
  class: string
  location: string
  object: string
  dimensions: string | null
  /** The `note` header entry blockgen writes on owner/neighbour ("nPoints:.. nCells:.."), if any. */
  note: string | null
}

export interface FoamField {
  name: string
  header: FoamFieldHeader
  components: 1 | 3
  /** Number of tuples in `data`. */
  count: number
  data: Float32Array
  uniform: boolean
  uniformValue: number[] | null
  /** Patch names and types found in boundaryField (type may be null when absent). */
  patches: Array<{ name: string; type: string | null }>
}

export interface FoamPatchOut {
  name: string
  /** e.g. fixedValue, zeroGradient, empty, kqRWallFunction ... */
  type: string
  /** Extra entries written verbatim inside the patch block, e.g. { value: 'uniform (0 0 0)' }. */
  entries?: Record<string, string>
}

export interface FoamWriteOptions {
  name: string
  class: 'volScalarField' | 'volVectorField'
  /** e.g. "[0 1 -1 0 0 0 0]" */
  dimensions: string
  /** Time directory name, written into the header `location`. */
  time: string
  data: Float32Array | Float64Array
  components: 1 | 3
  patches: FoamPatchOut[]
  /** Significant digits (default 12). */
  precision?: number
  /** Write an all-equal list as `uniform x` like fields.rs' collapse_uniform (default true). */
  collapseUniform?: boolean
}

const CHUNK = 1 << 20
const HEADER_SCAN_LIMIT = 256 * 1024

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/** Collects a flat `{ key value; ... }` dictionary token by token, skipping nested dictionaries. */
export class DictCollector {
  readonly map = new Map<string, string>()
  private state: 'open' | 'key' | 'value' | 'nested' = 'open'
  private key = ''
  private parts: string[] = []
  private depth = 0

  /** Returns true once the closing `}` has been consumed. */
  feed(tok: FoamToken): boolean {
    switch (this.state) {
      case 'open':
        if (tok.kind !== 'punct' || tok.text !== '{') throw new FoamSyntaxError(`expected '{', got '${tok.text}'`)
        this.state = 'key'
        return false
      case 'key':
        if (tok.kind === 'punct') {
          if (tok.text === '}') return true
          if (tok.text === ';') return false
          throw new FoamSyntaxError(`expected a keyword, got '${tok.text}'`)
        }
        this.key = tok.text
        this.parts = []
        this.depth = 0
        this.state = 'value'
        return false
      case 'value':
        if (tok.kind === 'punct') {
          if (tok.text === '{' && this.parts.length === 0) {
            this.state = 'nested'
            this.depth = 1
            return false
          }
          if (tok.text === ';' && this.depth === 0) {
            this.map.set(this.key, joinRaw(this.parts))
            this.state = 'key'
            return false
          }
          if (tok.text === '}' && this.depth === 0) throw new FoamSyntaxError(`missing ';' after '${this.key}'`)
          if (tok.text === '(' || tok.text === '[') this.depth++
          if (tok.text === ')' || tok.text === ']') this.depth--
        }
        this.parts.push(tok.kind === 'string' ? `"${tok.text}"` : tok.text)
        return false
      case 'nested':
        if (tok.kind === 'punct') {
          if (tok.text === '{') this.depth++
          else if (tok.text === '}' && --this.depth === 0) this.state = 'key'
        }
        return false
    }
  }
}

export function headerFromDict(map: Map<string, string>, dimensions: string | null): FoamFieldHeader {
  return {
    version: map.get('version') ?? '',
    format: map.get('format') ?? '',
    class: map.get('class') ?? '',
    location: unquote(map.get('location') ?? ''),
    object: map.get('object') ?? '',
    dimensions,
    note: map.has('note') ? unquote(map.get('note')!) : null,
  }
}

/** Fills a Float32Array from a list body; learns the component count from the paren depth of the first value. */
class FieldSink implements NumberSink {
  data: Float32Array
  n = 0
  components: 0 | 1 | 3
  constructor(
    private readonly expectedTuples: number | null,
    componentsHint: 0 | 1 | 3,
  ) {
    this.components = componentsHint
    const tuples = expectedTuples ?? 1024
    this.data = new Float32Array(Math.max(1, tuples * (componentsHint || 1)))
  }
  push(value: number, depth: number): void {
    if (this.components === 0) {
      this.components = depth >= 2 ? 3 : 1
      if (this.expectedTuples !== null) this.data = new Float32Array(Math.max(1, this.expectedTuples * this.components))
    }
    if (this.n >= this.data.length) {
      const bigger = new Float32Array(this.data.length * 2)
      bigger.set(this.data)
      this.data = bigger
    }
    this.data[this.n++] = value
  }
  finish(): { data: Float32Array; components: 1 | 3; count: number } {
    const comps = this.components === 0 ? 1 : this.components
    if (this.n % comps !== 0) throw new FoamSyntaxError(`list holds ${this.n} numbers, not a multiple of ${comps}`)
    const count = this.n / comps
    if (this.expectedTuples !== null && count !== this.expectedTuples) {
      throw new FoamSyntaxError(`list declares ${this.expectedTuples} entries but holds ${count}`)
    }
    const data = this.data.length === this.n ? this.data : this.data.slice(0, this.n)
    return { data, components: comps, count }
  }
}

type ParserState =
  | 'top'
  | 'header'
  | 'raw'
  | 'skip'
  | 'skipDict'
  | 'entry'
  | 'entryUniform'
  | 'entryUniformVec'
  | 'entryNonuniform'
  | 'entryCount'
  | 'entryCompact'
  | 'entryNumber'
  | 'entryList'
  | 'entryEnd'
  | 'bfOpen'
  | 'bfName'
  | 'bfBodyOpen'
  | 'bfKey'

/** Token-driven parser for one vol*Field file. */
class FieldParser {
  scanner!: FoamScanner
  headerDict = new Map<string, string>()
  dimensions: string | null = null
  patches: Array<{ name: string; type: string | null }> = []
  sink: FieldSink | null = null
  uniformValue: number[] | null = null
  compactCount = 0
  declaredCount: number | null = null
  componentsHint: 0 | 1 | 3 = 0
  seenInternal = false

  private state: ParserState = 'top'
  private returnState: ParserState = 'top'
  private rawTarget: 'dimensions' | 'patchType' = 'dimensions'
  private rawParts: string[] = []
  private depth = 0
  private header: DictCollector | null = null
  private currentPatch: { name: string; type: string | null } | null = null
  private compactValues: number[] = []

  constructor(
    private readonly headerOnly: boolean,
    private readonly nCells: number | null,
  ) {}

  onToken = (tok: FoamToken): void => {
    switch (this.state) {
      case 'top':
        return this.top(tok)
      case 'header':
        if (this.header!.feed(tok)) {
          this.headerDict = this.header!.map
          this.header = null
          this.state = 'top'
        }
        return
      case 'raw':
        return this.raw(tok)
      case 'skip':
        return this.skip(tok)
      case 'skipDict':
        if (tok.kind === 'punct') {
          if (tok.text === '{') this.depth++
          else if (tok.text === '}' && --this.depth === 0) this.state = this.returnState
        }
        return
      case 'entry':
        return this.entry(tok)
      case 'entryUniform':
        if (tok.kind === 'punct' && tok.text === '(') {
          this.compactValues = []
          this.state = 'entryUniformVec'
          return
        }
        this.uniformValue = [numberOf(tok)]
        this.state = 'entryEnd'
        return
      case 'entryUniformVec':
        if (tok.kind === 'punct' && tok.text === ')') {
          this.uniformValue = this.compactValues
          this.state = 'entryEnd'
          return
        }
        this.compactValues.push(numberOf(tok))
        return
      case 'entryNonuniform':
        if (tok.kind === 'word' && isNumeric(tok.text)) {
          this.declaredCount = intOf(tok.text)
          this.state = 'entryCount'
          return
        }
        if (tok.kind === 'word') {
          if (tok.text.includes('vector')) this.componentsHint = 3
          else if (tok.text.includes('scalar')) this.componentsHint = 1
          return
        }
        throw new FoamSyntaxError(`unexpected '${tok.text}' after nonuniform`)
      case 'entryCount':
        return this.entryCount(tok)
      case 'entryCompact':
        if (tok.kind === 'punct' && tok.text === '}') {
          this.finishCompact()
          this.state = 'entryEnd'
          return
        }
        if (tok.kind === 'punct') return
        this.compactValues.push(numberOf(tok))
        return
      case 'entryNumber':
        if (tok.kind === 'punct' && tok.text === ';') {
          this.uniformValue = [this.compactCount]
          this.state = 'top'
          return
        }
        this.declaredCount = this.compactCount
        return this.entryCount(tok)
      case 'entryList':
        if (tok.kind === 'punct' && tok.text === ')') {
          this.state = 'entryEnd'
          return
        }
        throw new FoamSyntaxError(`unexpected '${tok.text}' after a list`)
      case 'entryEnd':
        if (tok.kind !== 'punct' || tok.text !== ';') throw new FoamSyntaxError(`expected ';' after internalField, got '${tok.text}'`)
        this.state = 'top'
        return
      case 'bfOpen':
        if (tok.kind !== 'punct' || tok.text !== '{') throw new FoamSyntaxError(`expected '{' after boundaryField`)
        this.state = 'bfName'
        return
      case 'bfName':
        if (tok.kind === 'punct') {
          if (tok.text === '}') {
            this.state = 'top'
            return
          }
          if (tok.text === ';') return
          throw new FoamSyntaxError(`expected a patch name, got '${tok.text}'`)
        }
        this.currentPatch = { name: tok.text, type: null }
        this.patches.push(this.currentPatch)
        this.state = 'bfBodyOpen'
        return
      case 'bfBodyOpen':
        if (tok.kind !== 'punct' || tok.text !== '{') throw new FoamSyntaxError(`expected '{' after patch '${this.currentPatch?.name}'`)
        this.state = 'bfKey'
        return
      case 'bfKey':
        if (tok.kind === 'punct') {
          if (tok.text === '}') {
            this.state = 'bfName'
            return
          }
          if (tok.text === ';') return
          throw new FoamSyntaxError(`expected a keyword in patch '${this.currentPatch?.name}', got '${tok.text}'`)
        }
        if (tok.text === 'type') {
          this.beginRaw('patchType', 'bfKey')
        } else {
          this.beginSkip('bfKey')
        }
        return
    }
  }

  private top(tok: FoamToken): void {
    if (tok.kind === 'punct') {
      if (tok.text === ';') return
      throw new FoamSyntaxError(`unexpected '${tok.text}' at top level`)
    }
    switch (tok.text) {
      case 'FoamFile':
        this.header = new DictCollector()
        this.state = 'header'
        return
      case 'dimensions':
        this.beginRaw('dimensions', 'top')
        return
      case 'internalField':
        if (this.headerOnly) throw new StopScan()
        this.seenInternal = true
        this.state = 'entry'
        return
      case 'boundaryField':
        if (this.headerOnly) throw new StopScan()
        this.state = 'bfOpen'
        return
      default:
        this.beginSkip('top')
    }
  }

  private beginRaw(target: 'dimensions' | 'patchType', ret: ParserState): void {
    this.rawTarget = target
    this.rawParts = []
    this.depth = 0
    this.returnState = ret
    this.state = 'raw'
  }

  private raw(tok: FoamToken): void {
    if (tok.kind === 'punct') {
      if (tok.text === ';' && this.depth === 0) {
        const text = joinRaw(this.rawParts)
        if (this.rawTarget === 'dimensions') this.dimensions = text
        else if (this.currentPatch) this.currentPatch.type = text
        this.state = this.returnState
        return
      }
      if (tok.text === '(' || tok.text === '[') this.depth++
      if (tok.text === ')' || tok.text === ']') this.depth--
    }
    this.rawParts.push(tok.kind === 'string' ? `"${tok.text}"` : tok.text)
  }

  private beginSkip(ret: ParserState): void {
    this.returnState = ret
    this.depth = 0
    this.state = 'skip'
  }

  private skip(tok: FoamToken): void {
    if (tok.kind !== 'punct') return
    switch (tok.text) {
      case ';':
        if (this.depth === 0) this.state = this.returnState
        return
      case '{':
        this.depth = 1
        this.state = 'skipDict'
        return
      case '(':
        // A patch value list: let the scanner skip its numbers without tokenising them.
        this.scanner.beginList(null)
        return
      case '[':
        this.depth++
        return
      case ']':
        this.depth--
        return
      case ')':
        return
      case '}':
        throw new FoamSyntaxError(`unexpected '}' inside an entry`)
    }
  }

  private entry(tok: FoamToken): void {
    if (tok.kind === 'word') {
      if (tok.text === 'uniform') {
        this.state = 'entryUniform'
        return
      }
      if (tok.text === 'nonuniform') {
        this.state = 'entryNonuniform'
        return
      }
      if (isNumeric(tok.text)) {
        // `4;` is a value, `4 (...)` a sized list, `4{v}` a compact list.
        this.compactCount = Number(tok.text)
        this.state = 'entryNumber'
        return
      }
      throw new FoamSyntaxError(`expected 'uniform' or 'nonuniform', got '${tok.text}'`)
    }
    if (tok.kind === 'punct' && tok.text === '(') {
      this.declaredCount = null
      this.startList()
      return
    }
    throw new FoamSyntaxError(`expected 'uniform' or 'nonuniform', got '${tok.text}'`)
  }

  private entryCount(tok: FoamToken): void {
    if (tok.kind === 'punct' && tok.text === '(') {
      this.startList()
      return
    }
    if (tok.kind === 'punct' && tok.text === '{') {
      this.compactValues = []
      this.state = 'entryCompact'
      return
    }
    throw new FoamSyntaxError(`expected '(' after the list size, got '${tok.text}'`)
  }

  private startList(): void {
    this.sink = new FieldSink(this.declaredCount, this.componentsHint)
    this.scanner.beginList(this.sink)
    this.state = 'entryList'
  }

  private finishCompact(): void {
    const n = this.declaredCount ?? 0
    const vals = this.compactValues
    const comps: 1 | 3 = vals.length === 3 ? 3 : 1
    const sink = new FieldSink(n, comps)
    for (let i = 0; i < n; i++) for (let c = 0; c < comps; c++) sink.push(vals[c], comps === 3 ? 2 : 1)
    this.sink = sink
  }
}

function isNumeric(s: string): boolean {
  return /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(s)
}

function intOf(s: string): number {
  const v = Number(s)
  if (!Number.isInteger(v) || v < 0) throw new FoamSyntaxError(`bad list size '${s}'`)
  return v
}

function numberOf(tok: FoamToken): number {
  if (tok.kind !== 'word') throw new FoamSyntaxError(`expected a number, got '${tok.text}'`)
  const v = Number(tok.text)
  if (v === v) return v
  const t = tok.text.toLowerCase()
  if (t === 'inf' || t === '+inf') return Infinity
  if (t === '-inf') return -Infinity
  if (t.endsWith('nan')) return NaN
  throw new FoamSyntaxError(`not a number: '${tok.text}'`)
}

async function scanFile(filePath: string, scanner: FoamScanner, limit: number | null): Promise<void> {
  const fh = await fs.open(filePath, 'r')
  try {
    const buf = Buffer.allocUnsafe(CHUNK)
    let total = 0
    for (;;) {
      const { bytesRead } = await fh.read(buf, 0, CHUNK, null)
      if (bytesRead === 0) break
      scanner.feed(bytesRead === CHUNK ? buf : buf.subarray(0, bytesRead))
      total += bytesRead
      if (limit !== null && total >= limit) return
    }
    scanner.finish()
  } catch (e) {
    if (!(e instanceof StopScan)) throw e
  } finally {
    await fh.close()
  }
}

function scanText(text: string, scanner: FoamScanner): void {
  try {
    scanner.feed(Buffer.from(text, 'latin1'))
    scanner.finish()
  } catch (e) {
    if (!(e instanceof StopScan)) throw e
  }
}

function decorate(e: unknown, filePath: string): never {
  if (e instanceof FoamSyntaxError) throw new FoamSyntaxError(`${filePath}: ${e.message}`)
  throw e
}

export async function readFoamFieldHeader(filePath: string): Promise<FoamFieldHeader> {
  const parser = new FieldParser(true, null)
  const scanner = new FoamScanner(parser.onToken)
  parser.scanner = scanner
  try {
    await scanFile(filePath, scanner, HEADER_SCAN_LIMIT)
  } catch (e) {
    decorate(e, filePath)
  }
  return headerFromDict(parser.headerDict, parser.dimensions)
}

function assemble(parser: FieldParser, name: string, nCells: number | null): FoamField {
  const header = headerFromDict(parser.headerDict, parser.dimensions)
  const fieldName = header.object || name
  const patches = parser.patches
  if (!parser.seenInternal) throw new FoamSyntaxError('no internalField entry')
  if (parser.uniformValue) {
    const v = parser.uniformValue
    const comps: 1 | 3 = v.length === 3 ? 3 : 1
    if (v.length !== comps) throw new FoamSyntaxError(`uniform value has ${v.length} components`)
    const count = nCells ?? 1
    const data = new Float32Array(count * comps)
    for (let i = 0; i < count; i++) for (let c = 0; c < comps; c++) data[i * comps + c] = v[c]
    return { name: fieldName, header, components: comps, count, data, uniform: true, uniformValue: v, patches }
  }
  if (!parser.sink) throw new FoamSyntaxError('internalField has no value')
  const list = parser.sink.finish()
  let comps = list.components
  if (comps === 1 && header.class === 'volVectorField' && list.count === 0) comps = 3
  return { name: fieldName, header, components: comps, count: list.count, data: list.data, uniform: false, uniformValue: null, patches }
}

/**
 * Read a field. `uniform x;` internalFields are expanded to `nCells` tuples
 * when given (otherwise count = 1 and uniform = true). Handles
 * `nonuniform 0()`, newlines between `List<scalar>`, N and `(`, and
 * double-quoted patch names.
 */
export async function readFoamField(filePath: string, opts: { nCells?: number | null } = {}): Promise<FoamField> {
  const nCells = opts.nCells ?? null
  const parser = new FieldParser(false, nCells)
  const scanner = new FoamScanner(parser.onToken)
  parser.scanner = scanner
  try {
    await scanFile(filePath, scanner, null)
    return assemble(parser, path.basename(filePath), nCells)
  } catch (e) {
    decorate(e, filePath)
  }
}

/** Same as readFoamField, from an in-memory text (tests, small files). */
export function parseFoamFieldText(text: string, name: string, opts: { nCells?: number | null } = {}): FoamField {
  const nCells = opts.nCells ?? null
  const parser = new FieldParser(false, nCells)
  const scanner = new FoamScanner(parser.onToken)
  parser.scanner = scanner
  scanText(text, scanner)
  return assemble(parser, name, nCells)
}

// ---------------------------------------------------------------------------
// Number formatting
// ---------------------------------------------------------------------------

function trimZeros(s: string): string {
  if (!s.includes('.')) return s
  let end = s.length
  while (end > 0 && s[end - 1] === '0') end--
  if (end > 0 && s[end - 1] === '.') end--
  return s.slice(0, end)
}

function pad2(n: number): string {
  return n < 10 ? `0${n}` : String(n)
}

/** printf("%.<p>g"): p significant digits, exponent form outside [1e-4, 1e<p>), trailing zeros stripped (fields.rs fmt_g). */
export function fmtG(v: number, precision = 12): string {
  if (Number.isNaN(v)) return 'nan'
  if (!Number.isFinite(v)) return v < 0 ? '-inf' : 'inf'
  if (v === 0) return Object.is(v, -0) ? '-0' : '0'
  const p = Math.max(1, Math.min(100, precision))
  const sci = v.toExponential(p - 1)
  const ePos = sci.indexOf('e')
  const mant = sci.slice(0, ePos)
  const exp = Number(sci.slice(ePos + 1))
  if (exp < -4 || exp >= p) return `${trimZeros(mant)}e${exp < 0 ? '-' : '+'}${pad2(Math.abs(exp))}`
  const decimals = Math.max(p - 1 - exp, 0)
  return trimZeros(v.toFixed(decimals))
}

/** Port of rust/src/io/case.rs format_time_name: 6 significant digits, exponent form when needed, "0" for zero. */
export function formatTimeName(v: number): string {
  const PREC = 6
  if (v === 0) return '0'
  if (Number.isNaN(v)) return 'NaN'
  if (!Number.isFinite(v)) return v < 0 ? '-inf' : 'inf'
  let exp = Math.floor(Math.log10(Math.abs(v)))
  if ((Math.abs(v) / 10 ** exp).toFixed(PREC - 1).startsWith('10')) exp += 1
  if (exp < -4 || exp >= PREC) {
    const mantissa = trimZeros((v / 10 ** exp).toFixed(PREC - 1))
    return `${mantissa}e${exp < 0 ? '-' : '+'}${pad2(Math.abs(exp))}`
  }
  return trimZeros(v.toFixed(Math.max(PREC - 1 - exp, 0)))
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

const SEPARATOR = '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n'
const FOOTER = '\n\n// ************************************************************************* //\n'

function keyword(indent: string, kw: string, width: number): string {
  return indent + kw + (kw.length < width ? ' '.repeat(width - kw.length) : ' ')
}

export function foamFileHeader(cls: string, location: string, object: string): string {
  return (
    'FoamFile\n{\n' +
    keyword('    ', 'format', 12) + 'ascii;\n' +
    keyword('    ', 'class', 12) + cls + ';\n' +
    keyword('    ', 'location', 12) + `"${location}";\n` +
    keyword('    ', 'object', 12) + object + ';\n' +
    '}\n' +
    SEPARATOR
  )
}

function vec3(data: ArrayLike<number>, i: number, p: number): string {
  return `(${fmtG(data[3 * i], p)} ${fmtG(data[3 * i + 1], p)} ${fmtG(data[3 * i + 2], p)})`
}

function allEqual(data: ArrayLike<number>, comps: number): boolean {
  for (let i = comps; i < data.length; i++) if (data[i] !== data[i % comps]) return false
  return true
}

function needsQuoting(s: string): boolean {
  return s.length === 0 || !/^[A-Za-z0-9_.\-]+$/.test(s)
}

export class TextWriter {
  private readonly out: ReturnType<typeof createWriteStream>
  private pending: string[] = []
  private pendingBytes = 0
  constructor(filePath: string) {
    this.out = createWriteStream(filePath)
  }
  async write(s: string): Promise<void> {
    this.pending.push(s)
    this.pendingBytes += s.length
    if (this.pendingBytes >= 1 << 20) await this.flush()
  }
  private async flush(): Promise<void> {
    if (this.pending.length === 0) return
    const chunk = this.pending.join('')
    this.pending = []
    this.pendingBytes = 0
    if (!this.out.write(chunk)) await once(this.out, 'drain')
  }
  async close(): Promise<void> {
    await this.flush()
    this.out.end()
    await once(this.out, 'finish')
  }
}

export async function writeFoamField(filePath: string, opts: FoamWriteOptions): Promise<void> {
  const p = opts.precision ?? 12
  const comps = opts.components
  const collapse = opts.collapseUniform ?? true
  const data = opts.data
  if (data.length % comps !== 0) throw new Error(`writeFoamField(${opts.name}): ${data.length} values is not a multiple of ${comps}`)
  const n = data.length / comps
  await fs.mkdir(path.dirname(filePath), { recursive: true })
  const w = new TextWriter(filePath)
  await w.write(foamFileHeader(opts.class, opts.time, opts.name))
  await w.write(keyword('', 'dimensions', 16) + (opts.dimensions || '[0 0 0 0 0 0 0]') + ';\n\n')
  await w.write(keyword('', 'internalField', 16))
  if (n === 0) {
    await w.write('nonuniform 0();\n')
  } else if (n === 1 || (collapse && allEqual(data, comps))) {
    await w.write(`uniform ${comps === 3 ? vec3(data, 0, p) : fmtG(data[0], p)};\n`)
  } else {
    await w.write(`nonuniform List<${comps === 3 ? 'vector' : 'scalar'}> \n${n}\n(\n`)
    const lines: string[] = []
    for (let i = 0; i < n; i++) {
      lines.push(comps === 3 ? vec3(data, i, p) : fmtG(data[i], p))
      if (lines.length === 4096) {
        await w.write(lines.join('\n') + '\n')
        lines.length = 0
      }
    }
    if (lines.length) await w.write(lines.join('\n') + '\n')
    await w.write(')\n;\n')
  }
  await w.write('\nboundaryField\n{\n')
  for (const patch of opts.patches) {
    await w.write(needsQuoting(patch.name) ? `    "${patch.name}"\n` : `    ${patch.name}\n`)
    await w.write('    {\n' + keyword('        ', 'type', 16) + (patch.type || 'calculated') + ';\n')
    for (const [k, v] of Object.entries(patch.entries ?? {})) await w.write(keyword('        ', k, 16) + v + ';\n')
    await w.write('    }\n')
  }
  await w.write('}\n' + FOOTER)
  await w.close()
}
