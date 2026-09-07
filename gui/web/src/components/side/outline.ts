// Tolerant top-level key scan for JSONC: skips comments and strings, tracks
// nesting, and reports the keys of the root object with their positions.
export type OutlineValueKind = 'object' | 'array' | 'string' | 'number' | 'boolean' | 'null' | 'unknown'

export interface OutlineEntry {
  key: string
  /** 1-based */
  line: number
  /** 1-based */
  col: number
  valueKind: OutlineValueKind
}

function valueKindAt(text: string, i: number): OutlineValueKind {
  const ch = text[i]
  if (ch === undefined) return 'unknown'
  if (ch === '{') return 'object'
  if (ch === '[') return 'array'
  if (ch === '"') return 'string'
  if (ch === '-' || (ch >= '0' && ch <= '9')) return 'number'
  if (text.startsWith('true', i) || text.startsWith('false', i)) return 'boolean'
  if (text.startsWith('null', i)) return 'null'
  return 'unknown'
}

/** Index of the first non-space, non-comment character at or after `i`. */
function skipTrivia(text: string, i: number): number {
  for (;;) {
    while (i < text.length && /\s/.test(text[i])) i++
    if (text.startsWith('//', i)) {
      const nl = text.indexOf('\n', i)
      i = nl < 0 ? text.length : nl + 1
      continue
    }
    if (text.startsWith('/*', i)) {
      const end = text.indexOf('*/', i + 2)
      i = end < 0 ? text.length : end + 2
      continue
    }
    return i
  }
}

export function scanTopLevelKeys(text: string): OutlineEntry[] {
  const out: OutlineEntry[] = []
  let depth = 0
  let rootIsObject: boolean | null = null
  let line = 1
  let lineStart = 0
  let i = 0
  const n = text.length

  const advanceTo = (j: number) => {
    for (let k = i; k < j && k < n; k++) {
      if (text[k] === '\n') {
        line++
        lineStart = k + 1
      }
    }
    i = j
  }

  while (i < n) {
    const j = skipTrivia(text, i)
    if (j !== i) {
      advanceTo(j)
      continue
    }
    const ch = text[i]
    if (ch === '"') {
      let k = i + 1
      while (k < n && text[k] !== '"') {
        if (text[k] === '\\') k++
        if (text[k] === '\n') {
          line++
          lineStart = k + 1
        }
        k++
      }
      const raw = text.slice(i + 1, k)
      const keyLine = line
      const keyCol = i - lineStart + 1
      const after = skipTrivia(text, k + 1)
      if (depth === 1 && rootIsObject && text[after] === ':') {
        const valueAt = skipTrivia(text, after + 1)
        out.push({ key: raw.replace(/\\(.)/g, '$1'), line: keyLine, col: keyCol, valueKind: valueKindAt(text, valueAt) })
      }
      advanceTo(k + 1)
      continue
    }
    if (ch === '{' || ch === '[') {
      if (depth === 0) {
        if (rootIsObject !== null) break
        rootIsObject = ch === '{'
      }
      depth++
    } else if (ch === '}' || ch === ']') {
      depth--
      if (depth <= 0) break
    }
    advanceTo(i + 1)
  }
  return out
}
