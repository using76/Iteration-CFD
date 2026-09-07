// Session persistence: <sessionsDir>/<id>.json, written atomically after
// every message. `messages` is the SDK-shaped, append-only history; `ui`
// is the projection the browser renders.
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import type { BetaMessageParam } from '@anthropic-ai/sdk/resources/beta/messages/messages'
import { DEFAULT_SESSION_SETTINGS, type PendingApproval, type SessionSettings, type SessionState, type SessionSummary, type ToolCallRecord, type UiMessage } from '@cfd/shared'
import { projectUser } from './ui-projection.js'

export interface SessionRecord {
  id: string
  title: string
  createdAt: string
  updatedAt: string
  model: string
  settings: SessionSettings
  messages: BetaMessageParam[]
  ui: UiMessage[]
  toolCalls: ToolCallRecord[]
  runs: string[]
  allowedTools: string[]
}

export const TITLE_MAX = 60
const UNTITLED = { ko: '새 대화', en: 'New chat' }

let counter = 0

export function newId(prefix: string): string {
  counter = (counter + 1) % 1e6
  return `${prefix}_${Date.now().toString(36)}${counter.toString(36).padStart(4, '0')}`
}

export function titleFromText(text: string, locale: SessionSettings['locale'] = 'ko'): string {
  const line = text.replace(/\s+/g, ' ').trim()
  if (!line) return UNTITLED[locale]
  return line.length > TITLE_MAX ? `${line.slice(0, TITLE_MAX - 1)}…` : line
}

export function newSessionRecord(model: string, settings: Partial<SessionSettings> = {}): SessionRecord {
  const now = new Date().toISOString()
  const merged = { ...DEFAULT_SESSION_SETTINGS, ...settings }
  return { id: newId('s'), title: UNTITLED[merged.locale], createdAt: now, updatedAt: now, model, settings: merged, messages: [], ui: [], toolCalls: [], runs: [], allowedTools: [] }
}

export function summaryOf(rec: SessionRecord): SessionSummary {
  return { id: rec.id, title: rec.title, createdAt: rec.createdAt, updatedAt: rec.updatedAt, messageCount: rec.ui.length }
}

export function stateOf(rec: SessionRecord, extra: { pendingApprovals: PendingApproval[]; turnActive: boolean; customTools: Array<{ name: string; description: string }> }): SessionState {
  return { id: rec.id, title: rec.title, createdAt: rec.createdAt, updatedAt: rec.updatedAt, settings: rec.settings, messages: rec.ui, pendingApprovals: extra.pendingApprovals, runs: rec.runs, turnActive: extra.turnActive, customTools: extra.customTools }
}

function isRecord(v: unknown): v is SessionRecord {
  if (typeof v !== 'object' || v === null) return false
  const r = v as Record<string, unknown>
  return typeof r.id === 'string' && Array.isArray(r.messages) && Array.isArray(r.ui)
}

/**
 * The assistant message carrying tool_use blocks is saved before any tool has
 * run - it has to be, because a tool round can take two minutes of run_wait or
 * ten of waiting for approval. If the process does not survive that round, the
 * file on disk holds a tool_use with no tool_result, and the API rejects every
 * later request in that session with a 400. Give each unanswered tool_use an
 * is_error result on load; the model reads it the way it reads any failure.
 */
export function repairDanglingToolUses(messages: BetaMessageParam[], reason = 'interrupted: the server stopped before this tool finished'): number {
  let repaired = 0
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i]
    if (m.role !== 'assistant' || typeof m.content === 'string') continue
    const ids = m.content.filter((b) => b.type === 'tool_use').map((b) => b.id)
    if (!ids.length) continue
    const next = messages[i + 1]
    const answered = new Set<string>()
    const nextBlocks = next && next.role === 'user' && typeof next.content !== 'string' ? next.content : null
    if (nextBlocks) for (const b of nextBlocks) if (b.type === 'tool_result') answered.add(b.tool_use_id)
    const missing = ids.filter((id) => !answered.has(id))
    if (!missing.length) continue
    const blocks = missing.map((id) => ({ type: 'tool_result' as const, tool_use_id: id, content: JSON.stringify({ error: { code: 'INTERRUPTED', message: reason } }), is_error: true }))
    // tool_result blocks lead the user message the API replies to.
    if (nextBlocks) nextBlocks.unshift(...blocks)
    else messages.splice(i + 1, 0, { role: 'user', content: blocks })
    repaired += missing.length
  }
  return repaired
}

export interface SessionStore {
  list(): SessionRecord[]
  get(id: string): SessionRecord | undefined
  create(settings?: Partial<SessionSettings>): SessionRecord
  save(rec: SessionRecord): Promise<void>
  delete(id: string): Promise<boolean>
}

export function createSessionStore(dir: string, model: string): SessionStore {
  const records = new Map<string, SessionRecord>()
  fs.mkdirSync(dir, { recursive: true })
  for (const name of fs.readdirSync(dir)) {
    if (!name.endsWith('.json')) continue
    try {
      const parsed: unknown = JSON.parse(fs.readFileSync(path.join(dir, name), 'utf8'))
      if (isRecord(parsed)) {
        const partial = parsed as Partial<SessionRecord> & Pick<SessionRecord, 'id' | 'messages' | 'ui'>
        const rec: SessionRecord = {
          id: partial.id,
          title: partial.title ?? UNTITLED.ko,
          createdAt: partial.createdAt ?? new Date().toISOString(),
          updatedAt: partial.updatedAt ?? new Date().toISOString(),
          model: partial.model ?? model,
          settings: { ...DEFAULT_SESSION_SETTINGS, ...(partial.settings ?? {}) },
          messages: partial.messages,
          ui: partial.ui,
          toolCalls: partial.toolCalls ?? [],
          runs: partial.runs ?? [],
          allowedTools: partial.allowedTools ?? [],
        }
        repairDanglingToolUses(rec.messages)
        records.set(rec.id, rec)
      }
    } catch {
      // a corrupt session file is skipped, never deleted
    }
  }
  const pending = new Map<string, Promise<void>>()

  async function writeAtomic(rec: SessionRecord): Promise<void> {
    const file = path.join(dir, `${rec.id}.json`)
    const tmp = `${file}.${process.pid}.${Date.now().toString(36)}.tmp`
    await fsp.writeFile(tmp, JSON.stringify(rec), 'utf8')
    await fsp.rename(tmp, file)
  }

  return {
    list: () => [...records.values()].sort((a, b) => (a.updatedAt < b.updatedAt ? 1 : -1)),
    get: (id) => records.get(id),
    create(settings) {
      const rec = newSessionRecord(model, settings)
      records.set(rec.id, rec)
      return rec
    },
    save(rec) {
      rec.updatedAt = new Date().toISOString()
      records.set(rec.id, rec)
      // Serialise writes per session so a slow disk cannot reorder snapshots.
      const prev = pending.get(rec.id) ?? Promise.resolve()
      const next = prev.catch(() => {}).then(() => writeAtomic(rec))
      pending.set(rec.id, next)
      return next
    },
    async delete(id) {
      if (!records.delete(id)) return false
      await (pending.get(id) ?? Promise.resolve()).catch(() => {})
      await fsp.rm(path.join(dir, `${id}.json`), { force: true })
      return true
    },
  }
}

// ---------------------------------------------------------------------------
// History helpers (SDK messages + UI projection stay in step)
// ---------------------------------------------------------------------------

export interface UserTurnOptions {
  synthetic: boolean
  notices?: string[]
  /** Set the title from this text when the session is still untitled. */
  entitle?: boolean
}

/** Append a user-role message to both histories; returns the UI message (null for a pure tool_result message). */
export function appendUserTurn(rec: SessionRecord, message: BetaMessageParam, opts: UserTurnOptions): UiMessage | null {
  rec.messages.push(message)
  const ui = projectUser(newId('m'), message, { createdAt: Date.now(), synthetic: opts.synthetic, notices: opts.notices })
  if (ui) rec.ui.push(ui)
  if (opts.entitle && rec.ui.filter((m) => m.role === 'user').length === 1) {
    const text = ui?.blocks.find((b) => b.kind === 'text')
    if (text && text.kind === 'text') rec.title = titleFromText(text.text, rec.settings.locale)
  }
  return ui
}
