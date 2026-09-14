// gui/server/src/ontology/folds/sessions.ts — one Session row per file, one ToolCall row per
// toolCalls[] entry, read straight off the JSON (D7): the session store loads all 80 records
// and repairs the history it loaded, and an importer that repairs is an importer that invents.
// messages[] and ui[] are never imported (D4, R8) - no base64 enters the mirror.
import fs from 'node:fs/promises'
import path from 'node:path'
import type { SessionRecord } from '../../agent/session.js'
import { emptyFoldReport, isoOrNull, linkIfPresent, skip, upsert, type FoldContext, type FoldReport, type MirrorRow } from './base.js'

const APPROVE = ['none', 'reads', 'all']
const EFFORT = ['low', 'medium', 'high', 'xhigh', 'max']
const LOCALES = ['ko', 'en']
const POLICIES = ['auto', 'ask', 'never']
const STATUSES = ['pending', 'awaiting_approval', 'running', 'ok', 'error', 'denied', 'cancelled']

/** The four flattened settings properties are non-nullable enums: an absent settings block or a
 *  value outside an enum is refused as a counted skip naming the key - never defaulted (C8). */
function firstBadSetting(rec: Record<string, unknown>): string | null {
  const s = rec.settings !== null && typeof rec.settings === 'object' ? (rec.settings as Record<string, unknown>) : null
  if (s === null) return 'settings'
  if (!APPROVE.includes(s.autoApprove as string)) return 'settings.autoApprove'
  if (!EFFORT.includes(s.effort as string)) return 'settings.effort'
  if (!LOCALES.includes(s.locale as string)) return 'settings.locale'
  if (typeof s.notifyOnRunEnd !== 'boolean') return 'settings.notifyOnRunEnd'
  if (typeof rec.model !== 'string' || rec.model === '') return 'model'
  if (typeof rec.createdAt !== 'string' || !Number.isFinite(Date.parse(rec.createdAt))) return 'createdAt'
  if (typeof rec.updatedAt !== 'string' || !Number.isFinite(Date.parse(rec.updatedAt))) return 'updatedAt'
  return null
}

export async function foldSessions(ctx: FoldContext): Promise<FoldReport[]> {
  const rep = emptyFoldReport(ctx.type.session)
  const callRep = emptyFoldReport(ctx.type.toolCall)
  const t0 = Date.now()
  const reports = [rep, callRep]
  if (!ctx.writer.hasObjectType(ctx.type.session)) {
    skip(rep, ctx.type.session, 'not declared in the ontology registry; the fold writes no row')
    return reports
  }
  const dir = path.join(ctx.guiDir, 'sessions')
  let names: string[]
  try { names = (await fs.readdir(dir, { withFileTypes: true })).filter((e) => e.isFile() && e.name.endsWith('.json')).map((e) => e.name) } catch { names = [] }
  let nSummaryFallback = 0
  for (const name of names) {
    const abs = path.join(dir, name)
    let text: string
    try { text = await fs.readFile(abs, 'utf8') } catch { continue }
    rep.filesRead++
    const rec = JSON.parse(text) as unknown as SessionRecord & Record<string, unknown>
    const bad = firstBadSetting(rec as Record<string, unknown>)
    if (bad !== null) {
      skip(rep, 'session settings', name + ': ' + bad + ' is missing or outside the declared values; the session row is refused rather than defaulted')
      continue
    }
    const s = rec.settings as Record<string, unknown>
    const rawTitle = typeof rec.title === 'string' && rec.title !== '' ? rec.title : rec.id
    if (rawTitle === rec.id) skip(rep, 'session title fallback', name + ': the title is empty, so the session id stands in; a null title is not declared on Session', 1)
    const sessionId = rec.id
    const session: MirrorRow = {
      objectType: ctx.type.session,
      primaryKey: sessionId,
      properties: {
        sessionId,
        title: rawTitle,
        model: rec.model,
        createdAt: isoOrNull(rec.createdAt) as string,
        updatedAt: isoOrNull(rec.updatedAt) as string,
        autoApprove: String(s.autoApprove),
        effort: String(s.effort),
        locale: String(s.locale),
        notifyOnRunEnd: Boolean(s.notifyOnRunEnd),
        nMessages: Array.isArray(rec.messages) ? rec.messages.length : 0,
        nToolCalls: Array.isArray(rec.toolCalls) ? rec.toolCalls.length : 0,
        casePath: typeof rec.casePath === 'string' && rec.casePath !== '' ? rec.casePath : null,
        allowedTools: Array.isArray(rec.allowedTools) ? rec.allowedTools : [],
      },
      sourcePath: 'gui/sessions/' + name,
      importedAt: ctx.now(),
    }
    await upsert(ctx, rep, session)
    const calls = Array.isArray(rec.toolCalls) ? (rec.toolCalls as Array<Record<string, unknown>>) : []
    for (const tc of calls) {
      if (!POLICIES.includes(tc.policy as string) || !STATUSES.includes(tc.status as string) || typeof tc.toolUseId !== 'string' || typeof tc.name !== 'string') {
        skip(callRep, 'toolCall', 'gui/sessions/' + name + ': a tool call carries a policy, status, toolUseId or name the ontology does not declare; the row is refused')
        continue
      }
      const toolUseId = String(tc.toolUseId)
      // The same toolUseId appears in more than one session file (12 of them, 102 repeated rows
      // on this tree). The primary key holds one row, so the first read wins and the repeat is a
      // counted skip; without this the fold rewrites a row inside one import and a re-import is
      // an update, not unchanged (C6).
      if (ctx.seen.get(ctx.type.toolCall)?.has(toolUseId) === true) {
        skip(callRep, 'repeated toolUseId', 'the same toolUseId appears in a second session file; the first row read wins and the repeat is refused, so one primary key stays one row')
        continue
      }
      const summary = typeof tc.summary === 'string' && tc.summary !== '' ? tc.summary : String(tc.name)
      if (summary === tc.name && typeof tc.summary !== 'string') nSummaryFallback++
      const toolCall: MirrorRow = {
        objectType: ctx.type.toolCall,
        primaryKey: toolUseId,
        properties: {
          toolUseId,
          sessionId,
          name: String(tc.name),
          summary,
          policy: String(tc.policy),
          status: String(tc.status),
          runId: typeof tc.runId === 'string' && tc.runId !== '' ? tc.runId : null,
          startedAt: isoOrNull(tc.startedAt),
          endedAt: isoOrNull(tc.endedAt),
          error: tc.error === null || tc.error === undefined ? null : String(tc.error),
          resultPreview: tc.resultPreview === null || tc.resultPreview === undefined ? null : String(tc.resultPreview),
          input: tc.input ?? null,
        },
        sourcePath: 'gui/sessions/' + name,
        importedAt: ctx.now(),
      }
      await upsert(ctx, callRep, toolCall)
      await linkIfPresent(ctx, callRep, { linkType: ctx.link.belongsTo, fromType: ctx.type.toolCall, fromId: toolCall.primaryKey, toType: ctx.type.session, toId: sessionId, props: null, sourcePath: toolCall.sourcePath, importedAt: toolCall.importedAt })
      if (typeof tc.runId === 'string' && tc.runId !== '')
        await linkIfPresent(ctx, callRep, { linkType: ctx.link.startedRun, fromType: ctx.type.toolCall, fromId: toolCall.primaryKey, toType: ctx.type.run, toId: tc.runId, props: null, sourcePath: toolCall.sourcePath, importedAt: toolCall.importedAt })
    }
    const runIds = Array.isArray(rec.runs) ? (rec.runs as string[]) : []
    for (const runId of runIds)
      await linkIfPresent(ctx, rep, { linkType: ctx.link.touched, fromType: ctx.type.session, fromId: sessionId, toType: ctx.type.run, toId: runId, props: null, sourcePath: session.sourcePath, importedAt: session.importedAt })
    skip(rep, 'messages[]', 'the SDK history is an opaque provider blob and 96% of the session bytes; the mirror keeps the count, not the payload', Array.isArray(rec.messages) ? rec.messages.length : 0)
    skip(rep, 'ui[]', 'the render projection carries base64 image blocks; nothing base64 enters the mirror', Array.isArray(rec.ui) ? rec.ui.length : 0)
  }
  if (nSummaryFallback > 0)
    rep.notes.push(String(nSummaryFallback) + ' tool-call rows had no summary, so the tool name stands in as the title')
  rep.seconds = (Date.now() - t0) / 1000
  callRep.seconds = rep.seconds
  return reports
}
