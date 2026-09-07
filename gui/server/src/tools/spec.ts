// spec_lookup: section and full-text lookup over rust/SPEC-LIT.md. The file
// is 1.3 MB, so it is read and indexed once per process, on first use.
import fsp from 'node:fs/promises'
import path from 'node:path'
import { z } from 'zod'
import { fail, okResult, type ToolDef } from './context.js'

export interface SpecSection {
  level: 2 | 3
  number: string
  title: string
  /** Line index of the heading. */
  start: number
  /** Exclusive line index where the section (with its subsections) ends. */
  end: number
  /** Exclusive end of the section's own text (before its first subsection). */
  ownEnd: number
}

export interface SpecIndex {
  lines: string[]
  sections: SpecSection[]
}

const HEADING = /^(#{2,3})\s+(\d+(?:\.\d+)?)\.?\s+(.+?)\s*$/

export function indexSpec(text: string): SpecIndex {
  const lines = text.split('\n')
  const sections: SpecSection[] = []
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(HEADING)
    if (!m) continue
    sections.push({ level: m[1].length as 2 | 3, number: m[2], title: m[3], start: i, end: lines.length, ownEnd: lines.length })
  }
  for (let s = 0; s < sections.length; s++) {
    const cur = sections[s]
    for (let t = s + 1; t < sections.length; t++) {
      if (sections[t].level <= cur.level) {
        cur.end = sections[t].start
        break
      }
    }
    cur.ownEnd = s + 1 < sections.length ? Math.min(cur.end, sections[s + 1].start) : cur.end
  }
  return { lines, sections }
}

const cache = new Map<string, Promise<SpecIndex | null>>()

export function loadSpec(absPath: string): Promise<SpecIndex | null> {
  let p = cache.get(absPath)
  if (!p) {
    p = fsp.readFile(absPath, 'utf8').then(indexSpec, () => null)
    cache.set(absPath, p)
  }
  return p
}

export function clearSpecCache(): void {
  cache.clear()
}

export function sectionText(index: SpecIndex, number: string, maxChars: number): { section: SpecSection; text: string; truncated: boolean } | null {
  const section = index.sections.find((s) => s.number === number)
  if (!section) return null
  const full = index.lines.slice(section.start, section.end).join('\n')
  return { section, text: full.slice(0, maxChars), truncated: full.length > maxChars }
}

function owner(index: SpecIndex, line: number): SpecSection | null {
  let best: SpecSection | null = null
  for (const s of index.sections) {
    if (s.start <= line && line < s.ownEnd) best = s
  }
  return best
}

export function searchSpec(index: SpecIndex, query: string, maxHits = 30): Array<{ section: string | null; title: string | null; line: number; text: string }> {
  const q = query.toLowerCase()
  const hits: Array<{ section: string | null; title: string | null; line: number; text: string }> = []
  for (let i = 0; i < index.lines.length && hits.length < maxHits; i++) {
    if (!index.lines[i].toLowerCase().includes(q)) continue
    const s = owner(index, i)
    hits.push({ section: s?.number ?? null, title: s?.title ?? null, line: i + 1, text: index.lines[i].trim().slice(0, 300) })
  }
  return hits
}

const SpecSchema = z.object({
  section: z.string().nullable().describe('Section number like "31" or "31.1"; null to search instead'),
  query: z.string().nullable().describe('Case-insensitive text to search for when no section is given'),
  maxChars: z.number().int().min(200).max(60000).nullable().describe('Cap on returned section text (default 8000)'),
})

export const specLookup: ToolDef<typeof SpecSchema> = {
  name: 'spec_lookup',
  description: 'Look up rust/SPEC-LIT.md, the numerics specification of the solver: return one numbered section ("31.1") or search it for a phrase and get the matching lines with their section numbers. Use it before explaining a model, scheme or refusal message.',
  schema: SpecSchema,
  async run(input, ctx) {
    const index = await loadSpec(path.join(ctx.workspaceRoot, 'rust', 'SPEC-LIT.md'))
    if (!index) return fail('NOT_FOUND', 'rust/SPEC-LIT.md is not in the workspace')
    const maxChars = input.maxChars ?? 8000
    if (input.section) {
      const found = sectionText(index, input.section.replace(/^§/, '').trim(), maxChars)
      if (!found) {
        const near = index.sections.filter((s) => s.number.startsWith(input.section!.split('.')[0])).map((s) => `${s.number} ${s.title}`)
        return fail('NOT_FOUND', `no section ${input.section}; nearby: ${near.slice(0, 12).join('; ') || 'none'}`)
      }
      return okResult({ section: found.section.number, title: found.section.title, text: found.text, truncated: found.truncated, subsections: index.sections.filter((s) => s.number.startsWith(`${found.section.number}.`)).map((s) => `${s.number} ${s.title}`) })
    }
    if (input.query && input.query.trim()) {
      const q = input.query.trim()
      const headings = index.sections.filter((s) => s.title.toLowerCase().includes(q.toLowerCase())).map((s) => ({ section: s.number, title: s.title }))
      return okResult({ query: q, headings: headings.slice(0, 20), hits: searchSpec(index, q) })
    }
    return okResult({ sections: index.sections.filter((s) => s.level === 2).map((s) => `${s.number} ${s.title}`) })
  },
}
