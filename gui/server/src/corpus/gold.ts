// The C4 gold set: O2's seeded rows for SPEC-LIT sections 52-55, each carrying a verbatim
// quotation of its passage, plus the scorer that turns a gate report into precision and
// recall per property. Test data: the running server never imports this file.
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { z } from 'zod'
import type { GateCandidateObject, GateReport } from './gate.js'

export interface GoldObject { objectType: string; primaryKey: string; locator: string; evidence: string; properties: Record<string, unknown> }
export interface GoldLink { linkType: string; fromType: string; fromId: string; toType: string; toId: string; locator: string }
export interface GoldSet {
  goldSetId: string; ontologyVersion: string; documentId: string; sourcePath: string; seedSource: string
  objects: GoldObject[]; links: GoldLink[]
}
export type GoldErrorCode =
  | 'GOLD_BAD_JSON' | 'GOLD_SCHEMA' | 'GOLD_FOREIGN_DOCUMENT' | 'GOLD_BAD_LOCATOR'
  | 'GOLD_LOCATOR_MISMATCH' | 'GOLD_NO_CHUNK' | 'GOLD_EVIDENCE_ABSENT'
  | 'GOLD_EVIDENCE_NOT_UNIQUE' | 'GOLD_TOO_SMALL'

export class GoldError extends Error {
  readonly detail: Record<string, string | number>
  constructor(readonly code: GoldErrorCode, message: string, detail: Record<string, string | number> = {}) {
    super(message); this.detail = detail; this.name = 'GoldError'
  }
}

/** tp/fp/fn plus the ratios derived from them; null where a ratio would be 0/0. */
export interface Tally { tp: number; fp: number; fn: number; precision: number | null; recall: number | null; f1: number | null }
/** One property cell of the table; `key` is `<ObjectType>.<property>`, what the table sorts on. */
export interface PropertyScore extends Tally { key: string; objectType: string; property: string }
/** One link type cell; `key` is `link:<linkType>`. */
export interface LinkScore extends Tally { key: string; linkType: string }
export interface GoldScore {
  properties: PropertyScore[]; links: LinkScore[]; micro: Tally
  objectsGold: number; objectsProposed: number; objectsMatched: number
  /** Refusals per object type, straight from the gate - never part of the score. */
  refusedByType: Record<string, number>
}

const LOCATOR = /^5[2-5]\.\d{1,2}$/
const str = z.string()
const goldSetSchema = z.object({
  goldSetId: str, ontologyVersion: str, documentId: str, sourcePath: str, seedSource: str,
  objects: z.array(z.object({
    objectType: str, primaryKey: str, locator: str, evidence: str,
    properties: z.record(str, z.unknown()),
  })),
  links: z.array(z.object({
    linkType: str, fromType: str, fromId: str, toType: str, toId: str, locator: str,
  })),
})

const countOccurrences = (text: string, needle: string): number => {
  let count = 0
  for (let at = text.indexOf(needle); at !== -1; at = text.indexOf(needle, at + 1)) count += 1
  return count
}

/** The [charStart, charEnd) of one entry's evidence inside its passage, or null when it is absent. */
export function resolveEvidence(entry: GoldObject, text: string): { charStart: number; charEnd: number } | null {
  const at = text.indexOf(entry.evidence)
  if (at === -1 || text.indexOf(entry.evidence, at + 1) !== -1) return null
  return { charStart: at, charEnd: at + entry.evidence.length }
}

const bad = (code: GoldErrorCode, message: string, detail: Record<string, string | number> = {}): GoldError =>
  new GoldError(code, message, detail)

function checkEntry(goldSetId: string, e: GoldObject, chunkText: ((locator: string) => string | null) | null): void {
  const where = `${e.objectType}[${e.primaryKey}]`
  if (!LOCATOR.test(e.locator)) throw bad('GOLD_BAD_LOCATOR', `${goldSetId} ${e.locator} ${where}: the locator is not a SPEC-LIT subsection of sections 52 to 55`)
  const spec = e.properties.specSection
  if (typeof spec === 'string' && !('S' + e.locator).startsWith(spec)) throw bad('GOLD_LOCATOR_MISMATCH', `${goldSetId} ${e.locator} ${where}: the passage of locator "${e.locator}" cannot be the passage of specSection "${spec}"`)
  if (chunkText === null) return
  const passage = chunkText(e.locator)
  if (passage === null) throw bad('GOLD_NO_CHUNK', `${goldSetId} ${e.locator} ${where}: no passage with that locator`)
  const times = countOccurrences(passage, e.evidence)
  if (times === 0) throw bad('GOLD_EVIDENCE_ABSENT', `${goldSetId} ${e.locator} ${where}: evidence not found in the passage: "${e.evidence}"`)
  if (times > 1) throw bad('GOLD_EVIDENCE_NOT_UNIQUE', `${goldSetId} ${e.locator} ${where}: evidence occurs ${times} times in the passage: "${e.evidence}"`)
}

/** Check an already-parsed gold value: schema, document, locator, size and, when chunkText is
 *  given, every entry's evidence. Exported so the tests can push a literal through the same
 *  checker; `loadGoldSet` is this plus the disk read. */
export function checkGoldSet(raw: unknown, name: string, chunkText: ((locator: string) => string | null) | null): GoldSet {
  const parsed = goldSetSchema.safeParse(raw)
  if (!parsed.success) {
    const issue = parsed.error.issues[0]
    throw bad('GOLD_SCHEMA', `${name}: the gold file does not match its schema: ${issue === undefined ? '(no detail)' : issue.path.join('.') + ' ' + issue.message}`)
  }
  const set: GoldSet = parsed.data
  if (set.documentId !== 'doc:spec-lit') throw bad('GOLD_FOREIGN_DOCUMENT', `${set.goldSetId}: documentId "${set.documentId}" is not doc:spec-lit; only our own SPEC-LIT may be quoted`, { documentId: set.documentId })
  if (set.objects.length < 12) throw bad('GOLD_TOO_SMALL', `${set.goldSetId}: ${set.objects.length} object rows is under the floor of 12`, { objects: set.objects.length })
  for (const entry of set.objects) checkEntry(set.goldSetId, entry, chunkText)
  for (const link of set.links) {
    if (!LOCATOR.test(link.locator)) throw bad('GOLD_BAD_LOCATOR', `${set.goldSetId} ${link.locator} link ${link.linkType} ${link.fromId}->${link.toId}: the locator is not a SPEC-LIT subsection of sections 52 to 55`)
  }
  return set
}

/** Read, zod-parse and check one gold file. `chunkText(locator)` supplies the passage every
 *  evidence must be found in; null skips the evidence checks but not the rest. GoldError only. */
export function loadGoldSet(name: 'spec-lit-52-55', chunkText: ((locator: string) => string | null) | null): GoldSet {
  const file = path.join(path.dirname(fileURLToPath(import.meta.url)), 'gold', name + '.json')
  let raw: unknown
  try { raw = JSON.parse(readFileSync(file, 'utf8')) } catch {
    throw bad('GOLD_BAD_JSON', `${name}: the gold file is missing or is not valid JSON`)
  }
  return checkGoldSet(raw, name, chunkText)
}

/** A gold value as the string an extractor would stage: a string as is, anything else JSON.stringify'd. */
export function goldText(v: unknown): string {
  return typeof v === 'string' ? v : JSON.stringify(v)
}

const NUL = String.fromCharCode(0)
const objectKey = (objectType: string, primaryKey: string): string => objectType + NUL + primaryKey
const linkKey = (linkType: string, fromId: string, toId: string): string => linkType + NUL + fromId + NUL + toId
const byText = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0)
const byKey = (a: { key: string }, b: { key: string }): number => byText(a.key, b.key)

const ratios = (t: { tp: number; fp: number; fn: number }): Tally => {
  const precision = t.tp + t.fp === 0 ? null : t.tp / (t.tp + t.fp)
  const recall = t.tp + t.fn === 0 ? null : t.tp / (t.tp + t.fn)
  return { tp: t.tp, fp: t.fp, fn: t.fn, precision, recall,
           f1: precision === null || recall === null ? null : precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall) }
}

export function scoreAgainstGold(gold: GoldSet, report: GateReport): GoldScore {
  const blank = (): Tally => ({ tp: 0, fp: 0, fn: 0, precision: null, recall: null, f1: null })
  const cells = new Map<string, PropertyScore>()
  const cell = (objectType: string, property: string): PropertyScore => {
    const key = objectType + '.' + property
    const found = cells.get(key) ?? { ...blank(), key, objectType, property }
    cells.set(key, found)
    return found
  }
  // First candidate of a repeated identity wins; a later duplicate is scored only in the
  // second pass below, where each non-null property it carries is an fp.
  const winners = new Map<string, GateCandidateObject>()
  for (const pass of report.passedObjects) {
    const id = objectKey(pass.row.objectType, pass.row.primaryKey ?? '')
    if (!winners.has(id)) winners.set(id, pass.row)
  }
  for (const g of gold.objects) {
    const winner = winners.get(objectKey(g.objectType, g.primaryKey)) ?? null
    for (const [prop, goldValue] of Object.entries(g.properties)) {
      const staged = winner === null ? null : winner.props[prop]?.value ?? null
      const score = cell(g.objectType, prop)
      if (staged === null) score.fn += 1
      else if (goldText(goldValue).trim() === staged.trim()) score.tp += 1
      else { score.fp += 1; score.fn += 1 }
    }
  }
  for (const pass of report.passedObjects) {
    const row = pass.row
    const id = objectKey(row.objectType, row.primaryKey ?? '')
    const goldRow = gold.objects.find((g) => objectKey(g.objectType, g.primaryKey) === id) ?? null
    const winnerCounted = goldRow !== null && winners.get(id) === row
    for (const [prop, staged] of Object.entries(row.props)) {
      if (staged.value === null || (winnerCounted && goldRow.properties[prop] !== undefined)) continue
      cell(row.objectType, prop).fp += 1
    }
  }
  const linkScores = new Map<string, LinkScore>()
  const linkCell = (linkType: string): LinkScore => {
    const key = 'link:' + linkType
    const found = linkScores.get(key) ?? { ...blank(), key, linkType }
    linkScores.set(key, found)
    return found
  }
  const goldLinkIds = new Set(gold.links.map((l) => linkKey(l.linkType, l.fromId, l.toId)))
  const proposedLinkIds = new Set(report.passedLinks.map((l) => linkKey(l.linkType, l.fromPrimaryKey, l.toPrimaryKey)))
  for (const l of gold.links) linkCell(l.linkType)[proposedLinkIds.has(linkKey(l.linkType, l.fromId, l.toId)) ? 'tp' : 'fn'] += 1
  for (const l of report.passedLinks) if (!goldLinkIds.has(linkKey(l.linkType, l.fromPrimaryKey, l.toPrimaryKey))) linkCell(l.linkType).fp += 1
  const properties = [...cells.values()].sort(byKey)
  const links = [...linkScores.values()].sort(byKey)
  const micro: Tally = blank()
  for (const score of [...properties, ...links]) { Object.assign(score, ratios(score)); micro.tp += score.tp; micro.fp += score.fp; micro.fn += score.fn }
  Object.assign(micro, ratios(micro))
  return {
    properties, links, micro,
    objectsGold: gold.objects.length, objectsProposed: report.passedObjects.length,
    objectsMatched: gold.objects.filter((g) => winners.has(objectKey(g.objectType, g.primaryKey))).length,
    refusedByType: { ...report.counts.refusedByType },
  }
}

const number3 = (v: number | null): string => (v === null ? 'n/a' : v.toFixed(3))
const scoreRow = (label: string, t: Tally): string => `${label} | ${t.tp} | ${t.fp} | ${t.fn} | ${number3(t.precision)} | ${number3(t.recall)} | ${number3(t.f1)}`

/** The score as the table: one row per property, one per link type, micro, then refusals. */
export function formatScoreTable(score: GoldScore): string {
  const refused = Object.entries(score.refusedByType).sort((a, b) => byText(a[0], b[0])).map(([type, count]) => `${type}=${count}`).join(', ')
  const rows = [...score.properties, ...score.links].map((row) => scoreRow(row.key, row))
  rows.push(scoreRow('micro', score.micro), refused === '' ? 'refused | none' : 'refused | ' + refused)
  return ['property | tp | fp | fn | precision | recall | f1', ...rows].join('\n')
}
