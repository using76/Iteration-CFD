// gui/server/src/tools/ontology.ts — the agent's three ontology tools (docs/12 §B.3): query is a
// read, act only proposes (policy ask: it runs through the approval card), and apply applies a
// proposal this process saw an EXECUTED ontology_act produce. Every enum is generated from N1's
// registry at module load — no hand-written list of type names lives here.
//
// Symbols taken from the sibling units (CONTRACT.md): N1's ONTOLOGY (objectTypeNames /
// actionTypeNames / linkSideNames / property); N2's openOntologyStore, ontologyDbPath,
// resolveLinkSide (reached through ontology/handle.ts); N4's createActionEngine,
// createStoreActionStore (wired in ontology/handle.ts) and its preview exports
// ONTOLOGY_ACT_TOOL / proposalByToolUseId (ontology/preview.ts); N0's gitHead (read once per
// process in ontology/handle.ts).
import { DC_ONTOLOGY as ONTOLOGY, type Principal, type Proposal } from '@cfd/shared'
import { z } from 'zod'
import { APPROVAL_TTL_MS } from '../agent/approvals.js'
import { isSearchFailure, SEARCH_MODEL_BYTES, searchCorpus } from '../corpus/retrieve.js'
import { ontologyHandle } from '../ontology/handle.js'
import { EngineError } from '../ontology/engine.js'
import { isQueryFailure, QUERY_MODEL_BYTES, runOntologyQuery } from '../ontology/query.js'
import { ONTOLOGY_ACT_TOOL, proposalByToolUseId, proposalPreviewText } from '../ontology/preview.js'
import { errorMessage, fail, okResult, type ToolDef } from './context.js'

const OBJECT_TYPE_NAMES = ONTOLOGY.objectTypeNames() as [string, ...string[]]
const ACTION_NAMES = ONTOLOGY.actionTypeNames() as [string, ...string[]]
// The traverse vocabulary is the link SIDE accessor namespace, not the link-type namespace —
// N1 built linkSideNames() for exactly this enum; never re-derive it from LINK_TYPES.
const SIDE_NAMES = ONTOLOGY.linkSideNames() as [string, ...string[]]
if (OBJECT_TYPE_NAMES.length === 0) throw new Error('ontology.ts: the registry declares no object types; N1 must land first')
if (SIDE_NAMES.length === 0) throw new Error('ontology.ts: the registry declares no link types; N1 must land first')
if (ACTION_NAMES.length === 0) throw new Error('ontology.ts: the registry declares no action types; N1/N4 must land first')

const SIDE_ENUM = (SIDE_NAMES.length <= 96 ? SIDE_NAMES : SIDE_NAMES.slice(0, 96)) as [string, ...string[]]
const SIDE_DESC =
  'A link accessor on objectType: follow it one hop from every matched object and return the far side too. One hop only; null for no hop.' +
  (SIDE_NAMES.length > 96 ? ' (truncated to 96)' : '')

/** "<apiName> (<paramNames>)" one clause per action, joined by ' · ' — the enum's own contract,
 *  grown from the registry so the two can never drift. Capped at 1,200 chars on a clause boundary. */
export const ACTION_HINT: string = (() => {
  const full = ONTOLOGY.actionTypes.map((a) => `${a.apiName} (${a.parameters.map((p) => p.apiName).join(', ')})`).join(' · ')
  if (full.length <= 1200) return full
  const cut = full.lastIndexOf(' · ', 1200)
  return (cut > 0 ? full.slice(0, cut) : full.slice(0, 1200)) + ' …'
})()

/** The where filter, exported so the REST route parses the same JSON with the same schema. */
export const whereSchema = z.object({
  property: z.string().describe('A property api name of objectType.'),
  op: z.enum(['eq', 'ne', 'lt', 'lte', 'gt', 'gte', 'contains', 'startsWith', 'isNull', 'isNotNull']),
  value: z.union([z.string(), z.number(), z.boolean(), z.null()]).describe('Compared against the property; null for isNull/isNotNull.'),
})

const QuerySchema = z.object({
  objectType: z.enum(OBJECT_TYPE_NAMES).describe('The object type to read. Only these exist.'),
  id: z.string().nullable().describe('Primary key of one object; null to list.'),
  where: z.array(whereSchema).max(6).nullable().describe('AND-ed filters; null for no filter.'),
  orderBy: z.string().nullable().describe('Property api name to sort by; null for the primary key.'),
  descending: z.boolean().nullable().describe('Sort direction; null means ascending.'),
  limit: z.number().int().min(1).max(200).nullable().describe('Rows to return; null means 25, hard cap 200.'),
  cursor: z.string().nullable().describe("Opaque cursor from a previous result's nextCursor; null for the first page."),
  traverse: z.enum(SIDE_ENUM).nullable().describe(SIDE_DESC),
  properties: z.array(z.string()).max(30).nullable().describe('Property api names to return; null returns the primary key, the title and up to 12 more.'),
  mode: z.enum(['objects', 'search']).nullable().describe('"objects" (or null) reads typed rows only; "search" also returns document passages with their clause locators.'),
  text: z.string().nullable().describe('The question in the user\'s own words, when mode is "search"; null otherwise.'),
})

const ActSchema = z.object({
  action: z.enum(ACTION_NAMES).describe(ACTION_HINT),
  parameters: z.record(z.string(), z.unknown()).describe("The action's parameters. Read the action list above for the names."),
})

const ApplySchema = z.object({
  proposalId: z.string().describe('The proposalId returned by an ontology_act call in this session.'),
})

// ---------------------------------------------------------------------------
// The approved-proposal gate (C8): proposalId -> the ms at which an EXECUTED
// ontology_act produced it. run is reached only through the loop's execute,
// which runs only on decision === 'approved', so an entry here means a human
// saw this edit set and said yes — or set autoApprove:'all', the same human
// deciding once for the session. The REST routes do not consult this map.
// ---------------------------------------------------------------------------
const approvedProposals = new Map<string, number>()
const APPROVED_MAX = 32

function recordApproved(proposalId: string, now: number): void {
  approvedProposals.delete(proposalId)
  approvedProposals.set(proposalId, now)
  while (approvedProposals.size > APPROVED_MAX) {
    const first = approvedProposals.keys().next().value
    if (first === undefined) break
    approvedProposals.delete(first)
  }
}

function isApproved(proposalId: string, now: number): boolean {
  const at = approvedProposals.get(proposalId)
  if (at === undefined) return false
  if (now - at > APPROVAL_TTL_MS) {
    approvedProposals.delete(proposalId)
    return false
  }
  return true
}

function engineErrorCode(err: unknown): string {
  return err instanceof EngineError ? err.code : 'TOOL_ERROR'
}

export const ontologyQuery: ToolDef<typeof QuerySchema> = {
  name: 'ontology_query',
  description:
    'Read objects and their links from the ontology. Pick an objectType from the enum, filter on its properties, optionally walk one link. Read-only: it can never change anything. Use ontology_act to propose a change. Set mode to "search" and put the question in text to get document passages with their clause locators beside the typed rows.',
  schema: QuerySchema,
  async run(input, ctx) {
    if (input.mode === 'search') {
      if (input.text == null || input.text.trim() === '')
        return fail('SEARCH_WITHOUT_TEXT', 'mode is "search" but text is null; put the question in text, or set mode to "objects".')
      if (input.id != null)
        return fail('ID_IN_SEARCH', 'id and mode "search" cannot be used together; search resolves the object from text. Set id to null, or set mode to "objects".')
      if (input.cursor != null)
        return fail('CURSOR_IN_SEARCH', 'cursor and mode "search" cannot be used together; a search returns one page of at most 6 passages. Set cursor to null, or set mode to "objects".')
      const h = await ontologyHandle({ config: ctx.config, runs: ctx.runs, hub: undefined })
      const r = await searchCorpus(
        h,
        { objectType: input.objectType, text: input.text, traverse: input.traverse, where: input.where, orderBy: input.orderBy, descending: input.descending, limit: input.limit, properties: input.properties },
        { trimTo: SEARCH_MODEL_BYTES },
      )
      if (isSearchFailure(r)) return fail(r.code, r.message)
      return okResult(r)
    }
    if (input.text != null)
      return fail('TEXT_WITHOUT_SEARCH', `text is set but mode is "${input.mode}"; set mode to "search" to use it, or set text to null.`)
    const h = await ontologyHandle({ config: ctx.config, runs: ctx.runs, hub: undefined })
    const r = await runOntologyQuery(h, input, { trimTo: QUERY_MODEL_BYTES })
    if (isQueryFailure(r)) return fail(r.code, r.message)
    return okResult(r)
  },
}

export const ontologyAct: ToolDef<typeof ActSchema> = {
  name: ONTOLOGY_ACT_TOOL,
  description:
    'Propose a change to the ontology. You never write: this stages an edit set and returns it for the operator to approve or reject. Pick an action from the enum and give its parameters exactly as its description says. If the result says state=rejected, read blocking and fix the parameters; do not retry unchanged. If it says state=rendered, the operator has approved it: call ontology_apply with the proposalId.',
  schema: ActSchema,
  async run(input, ctx) {
    const h = await ontologyHandle({ config: ctx.config, runs: ctx.runs, hub: undefined })
    const principal: Principal = { kind: 'agent', id: 'assistant', sessionId: ctx.sessionId }
    // What the operator saw is what applies: the preview bound a proposal to this toolUseId
    // before the card was drawn. A hit returns that proposal unchanged; a miss (policy auto,
    // no card was ever drawn) proposes afresh under the same id.
    const previewed = proposalByToolUseId(h.engine, ctx.toolUseId)
    let proposal: Proposal
    if (previewed) proposal = previewed
    else {
      try {
        proposal = await h.engine.propose(input.action, input.parameters, principal, ctx.toolUseId)
      } catch (err) {
        return fail(engineErrorCode(err), errorMessage(err))
      }
    }
    recordApproved(proposal.proposalId, Date.now())
    return okResult({
      kind: 'ontologyProposal',
      proposalId: proposal.proposalId,
      action: proposal.action,
      state: proposal.state,
      summary: proposalPreviewText(proposal, ONTOLOGY),
      objects: proposal.edits?.objects.length ?? 0,
      links: proposal.edits?.links.length ?? 0,
      blocking: proposal.blocking.map((c) => ({ id: c.id, message: c.message })),
      warnings: proposal.warnings.map((c) => ({ id: c.id, message: c.message })),
      applied: false,
      next: 'call ontology_apply with this proposalId',
    })
  },
}

export const ontologyApply: ToolDef<typeof ApplySchema> = {
  name: 'ontology_apply',
  description:
    'Apply a proposal the operator already approved. It refuses any proposalId that did not come back from an ontology_act call in this session, and refuses a proposal that was already applied. There is nothing else you can do with it: you cannot approve your own proposal.',
  schema: ApplySchema,
  async run(input, ctx) {
    // The gate, not a second card, is what makes policy auto safe here.
    if (!isApproved(input.proposalId, Date.now()))
      return fail('NOT_APPROVED', `proposal ${input.proposalId} did not come from an approved ontology_act in this session; call ontology_act first`)
    const h = await ontologyHandle({ config: ctx.config, runs: ctx.runs, hub: undefined })
    const principal: Principal = { kind: 'user', id: 'local', sessionId: ctx.sessionId }
    try {
      try {
        h.engine.authorise(input.proposalId, principal)
      } catch (err) {
        // authorise refuses every state but 'rendered'; when it refuses, apply still runs and
        // names the precise reason (ALREADY_APPLIED / EXPIRED / NOT_AUTHORISED), writing nothing.
        if (!(err instanceof EngineError && err.code === 'NOT_RENDERED')) throw err
      }
      const entry = await h.engine.apply(input.proposalId, principal)
      return okResult({
        kind: 'ontologyApplied',
        editId: entry.editId,
        proposalId: entry.proposalId,
        action: entry.action,
        appliedAt: entry.appliedAt,
        objects: entry.edits.objects.length,
        links: entry.edits.links.length,
      })
    } catch (err) {
      return fail(engineErrorCode(err), errorMessage(err))
    }
  },
}
