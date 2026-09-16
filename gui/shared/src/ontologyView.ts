// gui/shared/src/ontologyView.ts — the shapes the action card renders, and the three
// parsers that project N5's wire shapes into them (docs/12 §C row U2). The producer is
// N5: the approval preview is plain text in a five-line grammar (N5 §C10), the act
// result carries counts and {id,message} criteria (server/src/tools/ontology.ts), and
// the query result is OntologyQueryResult (server/src/ontology/query.ts). Every parser
// returns null rather than a partial and never throws: anything that does not parse
// renders exactly as it did before this module existed.

import { z } from 'zod'

// ---- the view shapes the window renders ------------------------------------

/** One line of an edit summary: an object created, modified or deleted. */
export interface EditLineView {
  op: 'create' | 'modify' | 'delete'
  objectType: string
  id: string
  /** The parenthetical N4 appends, e.g. 'plume refine, ofgpu-k-epsilon, cases/plume.jsonc, queued'. */
  note: string | null
}

/** One line of an edit summary: a link added ('+') or removed ('-'). */
export interface LinkLineView {
  op: 'create' | 'delete'
  linkType: string
  toType: string
  toId: string
  /** null on every link line N4 writes today; a link-property parenthetical is possible. */
  note: string | null
}

/** What a summary parses into: the lines, the blocking reasons, the trailing count. */
export interface EditSummaryView {
  objects: EditLineView[]
  links: LinkLineView[]
  blocked: string[]
  /** 0 when there is no '… and N more' line. */
  more: number
  /** Lines the grammar did not claim; the card shows these verbatim, at most ten. */
  unparsed: string[]
}

/** One criterion as the card shows it; the severity is decided by which array carried it. */
export interface CriterionView {
  id: string
  severity: 'block' | 'warn'
  message: string
}

/** Both proposal channels project into this one shape. The approval channel is plain
 *  text, so its header fields are null; the act-result channel fills them in. `state`
 *  stays a plain string on purpose (decision D10): an enum over ProposalState would
 *  drop the whole card into a <pre> the day an eighth state word appears. */
export interface ProposalView {
  proposalId: string | null
  action: string | null
  state: string | null
  applied: boolean | null
  /** N5 sends counts, not arrays. */
  objectCount: number | null
  linkCount: number | null
  edits: EditSummaryView
  blocking: CriterionView[]
  warnings: CriterionView[]
}

/** One end of a link row, as the object card's chip shows it. */
export interface OntoLinkView {
  linkType: string
  /** Which end of the link row this object sits on. */
  direction: 'from' | 'to'
  /** The far side's type and id. */
  type: string
  id: string
  /** Resolved from `objects`, then from `linked`, else null. */
  title: string | null
  /** True iff the far side is in `objects` — the chip's two tiers. */
  present: boolean
}

/** One object card of a query result. */
export interface OntoObjectView {
  type: string
  id: string
  /** The card shows the id when this is null. */
  title: string | null
  props: Array<{ property: string; value: string }>
  links: OntoLinkView[]
}

export interface QueryResultView {
  objects: OntoObjectView[]
  nextCursor: string | null
  trimmed: boolean
}

// ---- N5's wire shapes, mirrored so a stranger's preview cannot parse as ours ----

/** TREE WINS (server/src/tools/ontology.ts): the act result maps each criterion to
 *  { id, message } — no severity, no params on the wire. The view's severity is set by
 *  the projection: 'block' for a blocking[] entry, 'warn' for a warnings[] entry. */
export const CriterionWireSchema: z.ZodType<{ id: string; message: string }> = z.object({
  id: z.string(),
  message: z.string(),
})

/** The ontology_act tool result: requires the identity fields, the summary and the two
 *  COUNTS; applied/blocking/warnings are tolerated with .catch() because a refused
 *  proposal may arrive without them. */
export const ActResultWireSchema = z.object({
  kind: z.literal('ontologyProposal'),
  proposalId: z.string(),
  action: z.string(),
  state: z.string(),
  summary: z.string(),
  objects: z.number(),
  links: z.number(),
  applied: z.boolean().catch(false),
  blocking: z.array(CriterionWireSchema).catch([]),
  warnings: z.array(CriterionWireSchema).catch([]),
})

const ObjectRowWireSchema = z.object({
  type: z.string(),
  id: z.string(),
  /** null when the title-key property is null; '' is a real title; never coerced. */
  title: z.string().nullable(),
  props: z.record(z.string(), z.unknown()),
})
const LinkRowWireSchema = z.object({
  linkType: z.string(),
  fromType: z.string(),
  fromId: z.string(),
  toType: z.string(),
  toId: z.string(),
  props: z.record(z.string(), z.unknown()),
})

/** The ontology_query tool result: OntologyQueryResult, verbatim (N5 §C11). */
export const QueryResultWireSchema = z.object({
  kind: z.literal('ontologyObjects'),
  objects: z.array(ObjectRowWireSchema),
  links: z.array(LinkRowWireSchema),
  linked: z.array(ObjectRowWireSchema),
  nextCursor: z.string().nullable(),
  trimmed: z.boolean(),
})

// ---- the summary-line grammar (the whole specification of parseEditSummary) ----
// Five line forms, matched after trimEnd(); nothing else is a line of an edit summary.

const OBJECT_LINE  = /^(create|modify|delete) (\S+) (\S+?)(?: \((.*)\))?$/
const LINK_LINE    = /^([+-]) link (\S+) -> (\S+) (\S+?)(?: \((.*)\))?$/
const BLOCKED_LINE = /^blocked: (.+)$/
const MORE_LINE    = /^(?:…|\.\.\.) and (\d+) more$/

const EMPTY_EDITS: EditSummaryView = { objects: [], links: [], blocked: [], more: 0, unparsed: [] }

/** The line grammar of N5 §C10. Returns null iff objects, links and blocked are all
 *  empty — the single rule that keeps every diff, command and JSON preview in the tree
 *  rendering exactly as it did. Never throws. */
export function parseEditSummary(text: string | null | undefined): EditSummaryView | null {
  const view: EditSummaryView = { objects: [], links: [], blocked: [], more: 0, unparsed: [] }
  for (const raw of (text ?? '').split('\n')) {
    const line = raw.trimEnd()
    if (!line.trim()) continue
    let m = OBJECT_LINE.exec(line)
    if (m) {
      view.objects.push({ op: m[1] as EditLineView['op'], objectType: m[2], id: m[3], note: m[4] ?? null })
      continue
    }
    m = LINK_LINE.exec(line)
    if (m) {
      view.links.push({ op: m[1] === '+' ? 'create' : 'delete', linkType: m[2], toType: m[3], toId: m[4], note: m[5] ?? null })
      continue
    }
    m = BLOCKED_LINE.exec(line)
    if (m) {
      view.blocked.push(m[1])
      continue
    }
    m = MORE_LINE.exec(line)
    if (m) {
      view.more = Number(m[1]) // the last such line wins
      continue
    }
    view.unparsed.push(line)
  }
  if (!view.objects.length && !view.links.length && !view.blocked.length) return null
  return view
}

/** The window stringifies a props value and formats nothing: no rounding, no units, no
 *  locale (decision D9). `412` reads `412`. */
function valueText(v: unknown): string {
  if (typeof v === 'string') return v
  if (v === null || v === undefined) return ''
  if (typeof v === 'number' || typeof v === 'boolean') return String(v)
  try {
    return JSON.stringify(v) ?? ''
  } catch {
    return ''
  }
}

/** Channel A — PendingApproval.calls[].preview, which is PLAIN TEXT. Returns a
 *  ProposalView whose header fields are all null and whose `blocking` is one
 *  { id: 'blocked', severity: 'block', message } per `blocked:` line. Null for anything
 *  parseEditSummary refuses. Never throws. */
export function parseApprovalProposal(preview: string | null | undefined): ProposalView | null {
  const edits = parseEditSummary(preview)
  if (!edits) return null
  return {
    proposalId: null,
    action: null,
    state: null,
    applied: null,
    objectCount: null,
    linkCount: null,
    edits,
    blocking: edits.blocked.map((message) => ({ id: 'blocked', severity: 'block' as const, message })),
    warnings: [],
  }
}

/** Channel B — ToolCallRecord.resultPreview for ontology_act: the JSON of result.data
 *  alone, no {ok,data} envelope. Rows come from parseEditSummary(summary); if the
 *  summary parses to null the view is still returned with an empty EditSummaryView, so
 *  a proposal with counts and criteria but no lines is still a card. Never throws. */
export function parseActResultView(resultPreview: string | null | undefined): ProposalView | null {
  let parsed: unknown
  try {
    parsed = JSON.parse(resultPreview ?? '')
  } catch {
    return null
  }
  const wire = ActResultWireSchema.safeParse(parsed)
  if (!wire.success) return null
  const w = wire.data
  return {
    proposalId: w.proposalId,
    action: w.action,
    state: w.state,
    applied: w.applied,
    objectCount: w.objects,
    linkCount: w.links,
    edits: parseEditSummary(w.summary) ?? EMPTY_EDITS,
    blocking: w.blocking.map((c) => ({ id: c.id, severity: 'block' as const, message: c.message })),
    warnings: w.warnings.map((c) => ({ id: c.id, severity: 'warn' as const, message: c.message })),
  }
}

/** Channel B — ToolCallRecord.resultPreview for ontology_query: N5's
 *  {objects,links,linked,nextCursor,trimmed}, projected. Each link row is attached to
 *  exactly ONE object: the `from` end when that object is in `objects`, otherwise the
 *  `to` end; a row matching neither is dropped — so a link whose both ends are in the
 *  answer never grows a second, pointing-back chip. The far side's title is looked up
 *  in `objects`, then in `linked`, else null, and `present` is true only for a far side
 *  found in `objects`. Key order of props is the server's — never sorted. Never throws. */
export function parseQueryResultView(resultPreview: string | null | undefined): QueryResultView | null {
  let parsed: unknown
  try {
    parsed = JSON.parse(resultPreview ?? '')
  } catch {
    return null
  }
  const wire = QueryResultWireSchema.safeParse(parsed)
  if (!wire.success) return null
  const w = wire.data
  // far-side titles: objects first (they are the answer), linked only fills the rest
  const far = new Map<string, { title: string | null; present: boolean }>()
  for (const o of w.objects) far.set(o.type + o.id, { title: o.title, present: true })
  for (const l of w.linked) if (!far.has(l.type + l.id)) far.set(l.type + l.id, { title: l.title, present: false })
  const inObjects = new Set(w.objects.map((o) => o.type + o.id))
  const attached = new Map<string, OntoLinkView[]>()
  for (const l of w.links) {
    const fromKey = l.fromType + l.fromId
    const toKey = l.toType + l.toId
    let key: string
    let link: OntoLinkView
    if (inObjects.has(fromKey)) {
      const t = far.get(toKey)
      key = fromKey
      link = { linkType: l.linkType, direction: 'from', type: l.toType, id: l.toId, title: t?.title ?? null, present: t?.present ?? false }
    } else if (inObjects.has(toKey)) {
      const t = far.get(fromKey)
      key = toKey
      link = { linkType: l.linkType, direction: 'to', type: l.fromType, id: l.fromId, title: t?.title ?? null, present: t?.present ?? false }
    } else {
      continue
    }
    const list = attached.get(key)
    if (list) list.push(link)
    else attached.set(key, [link])
  }
  const objects = w.objects.map((o) => ({
    type: o.type,
    id: o.id,
    title: o.title,
    props: Object.entries(o.props).map(([property, value]) => ({ property, value: valueText(value) })),
    links: attached.get(o.type + o.id) ?? [],
  }))
  return { objects, nextCursor: w.nextCursor, trimmed: w.trimmed }
}
