// `@path` mentions in the composer: extraction for the attachments list and
// caret-aware detection for the autocomplete popup.
const TRAILING = /[,.;:!?)\]}]+$/

export interface MentionQuery {
  /** Index of the '@' character. */
  start: number
  /** Text typed after the '@' up to the caret. */
  query: string
}

export function parseMentions(text: string): string[] {
  const out: string[] = []
  const re = /(^|[\s(\[{])@([^\s@]+)/g
  for (const m of text.matchAll(re)) {
    const path = m[2].replace(TRAILING, '')
    if (path && !out.includes(path)) out.push(path)
  }
  return out
}

/** The mention being typed at `caret`, or null when the caret is not inside one. */
export function activeMention(text: string, caret: number): MentionQuery | null {
  const before = text.slice(0, caret)
  const at = before.lastIndexOf('@')
  if (at < 0) return null
  if (at > 0 && !/[\s(\[{]/.test(before[at - 1])) return null
  const query = before.slice(at + 1)
  if (/\s/.test(query)) return null
  return { start: at, query }
}

/** Replace the mention at `start` with `@path ` and return the new text and caret. */
export function completeMention(text: string, caret: number, start: number, path: string): { text: string; caret: number } {
  const insert = `@${path} `
  const next = text.slice(0, start) + insert + text.slice(caret)
  return { text: next, caret: start + insert.length }
}

/** Case-insensitive fuzzy-ish filter for the autocomplete list. */
export function matchPaths(paths: string[], query: string, limit = 8): string[] {
  const q = query.toLowerCase()
  if (!q) return paths.slice(0, limit)
  const scored: Array<{ p: string; score: number }> = []
  for (const p of paths) {
    const lp = p.toLowerCase()
    const base = lp.slice(lp.lastIndexOf('/') + 1)
    let score = -1
    if (base.startsWith(q)) score = 0
    else if (lp.startsWith(q)) score = 1
    else if (base.includes(q)) score = 2
    else if (lp.includes(q)) score = 3
    if (score >= 0) scored.push({ p, score })
  }
  scored.sort((a, b) => a.score - b.score || a.p.length - b.p.length || a.p.localeCompare(b.p))
  return scored.slice(0, limit).map((s) => s.p)
}
