// Unified diff of two texts (line based, LCS). Case files are small, so an
// O(n*m) table is fine; anything above the cap falls back to a whole-file
// replacement hunk.

const MAX_LCS_LINES = 4000

type Op = { kind: ' ' | '-' | '+'; text: string }

function lcsOps(a: string[], b: string[]): Op[] {
  const n = a.length
  const m = b.length
  if (n * m > MAX_LCS_LINES * MAX_LCS_LINES) {
    return [...a.map((text): Op => ({ kind: '-', text })), ...b.map((text): Op => ({ kind: '+', text }))]
  }
  const table = new Uint32Array((n + 1) * (m + 1))
  const at = (i: number, j: number) => i * (m + 1) + j
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      table[at(i, j)] = a[i] === b[j] ? table[at(i + 1, j + 1)] + 1 : Math.max(table[at(i + 1, j)], table[at(i, j + 1)])
    }
  }
  const ops: Op[] = []
  let i = 0
  let j = 0
  while (i < n && j < m) {
    if (a[i] === b[j]) {
      ops.push({ kind: ' ', text: a[i] })
      i++
      j++
    } else if (table[at(i + 1, j)] >= table[at(i, j + 1)]) {
      ops.push({ kind: '-', text: a[i] })
      i++
    } else {
      ops.push({ kind: '+', text: b[j] })
      j++
    }
  }
  while (i < n) ops.push({ kind: '-', text: a[i++] })
  while (j < m) ops.push({ kind: '+', text: b[j++] })
  return ops
}

function splitLines(text: string): string[] {
  if (text === '') return []
  const lines = text.split('\n')
  if (lines[lines.length - 1] === '') lines.pop()
  return lines
}

export function unifiedDiff(before: string, after: string, path: string, context = 3): string {
  if (before === after) return ''
  const ops = lcsOps(splitLines(before), splitLines(after))
  const out: string[] = [`--- a/${path}`, `+++ b/${path}`]
  let oldLine = 1
  let newLine = 1
  let idx = 0
  while (idx < ops.length) {
    if (ops[idx].kind === ' ') {
      oldLine++
      newLine++
      idx++
      continue
    }
    const start = Math.max(0, idx - context)
    let end = idx
    let sinceChange = 0
    while (end < ops.length && sinceChange <= context * 2) {
      if (ops[end].kind === ' ') sinceChange++
      else sinceChange = 0
      end++
    }
    // Trim trailing context beyond `context` lines.
    let trailing = 0
    while (trailing < end - idx && ops[end - 1 - trailing].kind === ' ') trailing++
    end -= Math.max(0, trailing - context)
    const leading = idx - start
    const hunkOldStart = oldLine - leading
    const hunkNewStart = newLine - leading
    let oldCount = 0
    let newCount = 0
    const body: string[] = []
    for (let k = start; k < end; k++) {
      const op = ops[k]
      body.push(`${op.kind}${op.text}`)
      if (op.kind !== '+') oldCount++
      if (op.kind !== '-') newCount++
    }
    out.push(`@@ -${hunkOldStart},${oldCount} +${hunkNewStart},${newCount} @@`, ...body)
    for (let k = idx; k < end; k++) {
      if (ops[k].kind !== '+') oldLine++
      if (ops[k].kind !== '-') newLine++
    }
    idx = end
  }
  return `${out.join('\n')}\n`
}
