// gui/server/src/corpus/prompt.ts — one ObjectTypeDef becomes the three things
// one extraction pass needs: the inline propose_candidates tool (a zod schema
// through the same sanitiser the agent tools use), the user turn that shows the
// model one passage, and the one corrective retry turn. Nothing here talks to a
// provider or writes a row; extract.ts drives both. Every field of the reply is
// nullable, never optional, and no key of the reply is a bare union — a weaker
// model answers union envelopes with an empty object (seen live 2026-09-11).
import type { BetaTool } from '@anthropic-ai/sdk/resources/beta/messages/messages'
import type { LinkTypeDef, ObjectTypeDef, OntologyRegistry, PropertyDef } from '@cfd/shared'
import { z } from 'zod'
import { sanitizeSchema } from '../tools/index.js'
import type { StagedChunk, StagedDocument } from './stage.js'

export const EXTRACTION_TOOL_NAME = 'propose_candidates'
/** Hard cap on one quote; longer means the model copied the passage instead of citing it. */
export const MAX_QUOTE_CHARS = 400
/** At most three links per candidate. */
export const MAX_LINKS_PER_CANDIDATE = 3

/** The system turn, identical for every pass; English only, the Tier-A corpus is English. */
export const EXTRACTION_SYSTEM = `You extract candidate rows of ONE typed object from ONE passage of an engineering document.
You never write to a database. Everything you return is a proposal a human will review and may reject.

Rules:
1. Every value you return must be STATED IN THE PASSAGE. If the passage does not state it, return null.
   Do not infer, do not complete from your own knowledge, do not summarise.
2. Every non-null value carries a \`quote\`: the shortest run of characters from the passage, copied
   CHARACTER FOR CHARACTER, that states it. A quote that is not in the passage is discarded, so copy,
   never paraphrase.
3. Never convert a unit, never round, never normalise, never reformat a number. Copy it as the passage
   writes it. Every value is a string.
4. Anything the passage states that no property of this type can hold goes in \`unmapped\`, with a verbatim
   quote and one line saying what it is. Reporting it is always better than losing it.
5. Answer with the tool call \`propose_candidates\` and nothing else. Always fill BOTH keys: \`candidates\`
   and \`unmapped\`. An empty array is a valid answer. An empty object is not an answer.`

/** The reply the model is asked for, as extract.ts consumes it after zod. */
export interface ReplyValue { value: string | null; quote: string | null }
export interface ReplyLink { linkApiName: string; targetPrimaryKey: string; quote: string | null }
export interface ReplyCandidate { properties: Record<string, ReplyValue>; links?: ReplyLink[] }
export interface ReplyData { candidates: ReplyCandidate[]; unmapped: { quote: string; why: string }[] }

export interface ExtractionShape {
  def: ObjectTypeDef
  /** The properties that reach the schema, in declaration order. */
  properties: PropertyDef[]
  /** "<apiName> (<why>)" for every property left out. */
  skipped: string[]
  /** Link types whose from-side is this type. May be empty. */
  links: LinkTypeDef[]
}

/** The extractable slice of one definition: every property except the derived,
 *  json and attachmentRef ones — each exclusion reported in `skipped`, never
 *  silently dropped — plus the link types that may leave this type. Throws
 *  nothing; the caller resolves `def` through the registry first. */
export function extractionShape(ontology: OntologyRegistry, def: ObjectTypeDef): ExtractionShape {
  const properties: PropertyDef[] = []
  const skipped: string[] = []
  for (const p of def.properties) {
    if (p.derived !== undefined) skipped.push(`${p.apiName} (derived; computed on read, never written)`)
    else if (p.baseType === 'json') skipped.push(`${p.apiName} (json; a reviewer cannot check it against a span)`)
    else if (p.baseType === 'attachmentRef') skipped.push(`${p.apiName} (attachmentRef; a passage cannot state an attachment id)`)
    else properties.push(p)
  }
  return { def, properties, skipped, links: ontology.linksFrom(def.apiName) }
}

const valueOf = (p: PropertyDef) =>
  z.object({
    value: (p.valueType === 'enum' && p.enumValues !== undefined && p.enumValues.length > 0
      ? z.enum(p.enumValues as [string, ...string[]])
      : z.string()).nullable(),
    quote: z.string().nullable(),
  })

/** The zod validator for one reply. Every field nullable, never optional; the
 *  only union in the emitted JSON is the two-member nullable-enum leaf. */
export function buildExtractionZod(shape: ExtractionShape): z.ZodType<ReplyData> {
  const candidate = z.object({
    properties: z.object(Object.fromEntries(shape.properties.map((p) => [p.apiName, valueOf(p)]))),
    ...(shape.links.length
      ? { links: z.array(z.object({
            linkApiName: z.enum(shape.links.map((l) => l.apiName) as [string, ...string[]]),
            targetPrimaryKey: z.string(),
            quote: z.string().nullable(),
          })).max(MAX_LINKS_PER_CANDIDATE) }
      : {}),
  })
  const reply = z.object({
    candidates: z.array(candidate),
    unmapped: z.array(z.object({ quote: z.string(), why: z.string() })),
  })
  return reply as unknown as z.ZodType<ReplyData>
}

/** The single inline tool sent in LlmStreamParams.tools — already through the shared sanitiser. */
export function buildExtractionTool(shape: ExtractionShape): BetaTool {
  return {
    name: EXTRACTION_TOOL_NAME,
    description: 'Propose candidate rows of one typed object from one passage, each value quoted from the passage.',
    input_schema: sanitizeSchema(z.toJSONSchema(buildExtractionZod(shape), { io: 'input' })) as BetaTool['input_schema'],
  }
}

/** The user turn for one (type, chunk) pass. */
export function buildExtractionPrompt(shape: ExtractionShape, chunk: StagedChunk, doc: StagedDocument): string {
  const def = shape.def
  const propLines = shape.properties
    .map((p) => `- ${p.apiName} (${p.baseType}${p.valueType === 'enum' ? `, one of: ${(p.enumValues ?? []).join(' | ')}` : ''}): ${p.description}`)
    .join('\n')
  const skipped = shape.skipped.length > 0 ? `Not asked for here: ${shape.skipped.join(', ')}.` : ''
  const links = shape.links.length
    ? `Links you may propose from a ${def.apiName}, at most ${MAX_LINKS_PER_CANDIDATE} per candidate, each naming its target by that target's primary key:
${shape.links.map((l) => `- ${l.apiName} -> ${l.to.objectType}: ${l.displayName}`).join('\n')}`
    : ''
  return `Object type: ${def.apiName} - ${def.displayName}
${def.description}
Primary key: ${def.primaryKey}. A candidate whose ${def.primaryKey} is null is still recorded, and a
human is shown that it has no key.

Properties - fill every one, null where the passage does not state it:
${propLines}
${skipped}
${links}

Passage - document ${doc.documentId}, locator ${chunk.locator}, heading "${chunk.heading}":
<<<
${chunk.text}
>>>`
}

/** The single corrective turn sent after an empty-arguments or no-tool-call reply. */
export function buildRetryPrompt(shape: ExtractionShape): string {
  return `Your last reply carried no arguments. Call ${EXTRACTION_TOOL_NAME} again for object type
${shape.def.apiName} over the same passage, and fill both keys: "candidates" (an array, possibly empty) and
"unmapped" (an array, possibly empty). Do not return an empty object. Do not answer in prose.`
}
