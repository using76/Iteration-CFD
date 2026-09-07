// The pure reducer: (SessionData, ServerMsg) -> { state, effects }. Every
// side effect (viewer commands, subscriptions, file reloads) is returned as
// an Effect for the WebSocket client to execute, which keeps this file
// testable with recorded frames.
import { toolPolicy, type LogLine, type MetricRecord, type PendingApproval, type ResidualRecord, type ServerMsg, type SessionState, type ToolCallRecord, type UiBlock, type UiMessage } from '@cfd/shared'
import { EMPTY_TURN, LOG_RING, METRIC_RING, OUTPUT_RING, RESIDUAL_RING, emptyRunData, problemKey, type Effect, type OutputEntry, type RunData, type SessionData } from './types'

export interface ApplyResult {
  state: SessionData
  effects: Effect[]
}

const NO_EFFECTS: Effect[] = []

function pushRing<T>(arr: T[], items: T[], cap: number): T[] {
  if (!items.length) return arr
  const next = arr.length + items.length <= cap ? arr.concat(items) : arr.concat(items).slice(-cap)
  return next
}

function placeholderMessage(id: string, now: number): UiMessage {
  return { id, role: 'assistant', blocks: [], createdAt: now, stopReason: null, model: null, suggestions: [], synthetic: false }
}

function withSession(state: SessionData, session: SessionState): SessionData {
  return { ...state, session }
}

function updateMessages(state: SessionData, fn: (messages: UiMessage[]) => UiMessage[]): SessionData {
  if (!state.session) return state
  const messages = fn(state.session.messages)
  if (messages === state.session.messages) return state
  return withSession(state, { ...state.session, messages })
}

/** Returns the message with `id`, creating an assistant placeholder at the end when absent. */
function ensureMessage(messages: UiMessage[], id: string, now: number): { messages: UiMessage[]; index: number } {
  const index = messages.findIndex((m) => m.id === id)
  if (index >= 0) return { messages, index }
  return { messages: [...messages, placeholderMessage(id, now)], index: messages.length }
}

function setBlock(message: UiMessage, blockIndex: number, block: UiBlock): UiMessage {
  const blocks = message.blocks.slice()
  while (blocks.length < blockIndex) blocks.push({ kind: 'text', text: '' })
  blocks[blockIndex] = block
  return { ...message, blocks }
}

function updateMessageAt(state: SessionData, messageId: string, now: number, fn: (m: UiMessage) => UiMessage): SessionData {
  return updateMessages(state, (messages) => {
    const { messages: list, index } = ensureMessage(messages, messageId, now)
    const updated = fn(list[index])
    if (updated === list[index] && list === messages) return messages
    const out = list.slice()
    out[index] = updated
    return out
  })
}

function updateToolCall(state: SessionData, toolUseId: string, fn: (call: ToolCallRecord) => ToolCallRecord): SessionData {
  return updateMessages(state, (messages) => {
    let out: UiMessage[] | null = null
    messages.forEach((m, mi) => {
      m.blocks.forEach((b, bi) => {
        if (b.kind !== 'tool' || b.call.toolUseId !== toolUseId) return
        const call = fn(b.call)
        if (call === b.call) return
        if (!out) out = messages.slice()
        out[mi] = setBlock(out[mi], bi, { kind: 'tool', call })
      })
    })
    return out ?? messages
  })
}

function runData(state: SessionData, runId: string): RunData {
  return state.runData[runId] ?? emptyRunData()
}

function withRunData(state: SessionData, runId: string, data: RunData): SessionData {
  return { ...state, runData: { ...state.runData, [runId]: data } }
}

/** Append log lines with seq above the last seen one (replays overlap the live stream). */
function pushLogs(state: SessionData, runId: string, lines: LogLine[]): SessionData {
  const data = runData(state, runId)
  let last = data.lastLogSeq
  const fresh: LogLine[] = []
  for (const l of lines) {
    if (l.seq <= last) continue
    fresh.push(l)
    last = l.seq
  }
  if (!fresh.length) return state
  return withRunData(state, runId, { ...data, logs: pushRing(data.logs, fresh, LOG_RING), lastLogSeq: last })
}

function pushResiduals(state: SessionData, runId: string, recs: ResidualRecord[]): SessionData {
  const data = runData(state, runId)
  let last = data.lastResidualSeq
  const fresh: ResidualRecord[] = []
  for (const r of recs) {
    if (r.seq <= last) continue
    fresh.push(r)
    last = r.seq
  }
  if (!fresh.length) return state
  return withRunData(state, runId, { ...data, residuals: pushRing(data.residuals, fresh, RESIDUAL_RING), lastResidualSeq: last })
}

function pushMetrics(state: SessionData, runId: string, recs: MetricRecord[]): SessionData {
  const data = runData(state, runId)
  let last = data.lastMetricSeq
  const fresh: MetricRecord[] = []
  for (const r of recs) {
    if (r.seq <= last) continue
    fresh.push(r)
    last = r.seq
  }
  if (!fresh.length) return state
  return withRunData(state, runId, { ...data, metrics: pushRing(data.metrics, fresh, METRIC_RING), lastMetricSeq: last })
}

function addOutput(state: SessionData, entry: OutputEntry): SessionData {
  return { ...state, outputs: pushRing(state.outputs, [entry], OUTPUT_RING) }
}

function isCurrent(state: SessionData, sessionId: string): boolean {
  return state.currentSessionId === null || state.currentSessionId === sessionId
}

function resolveApprovals(approvals: PendingApproval[], toolUseIds: string[]): PendingApproval[] {
  const done = new Set(toolUseIds)
  const out: PendingApproval[] = []
  for (const a of approvals) {
    const remaining = a.toolUseIds.filter((id) => !done.has(id))
    if (!remaining.length) continue
    if (remaining.length === a.toolUseIds.length) out.push(a)
    else out.push({ ...a, toolUseIds: remaining, calls: a.calls.filter((c) => !done.has(c.toolUseId)) })
  }
  return out
}

export function applyEvent(state: SessionData, msg: ServerMsg, now: number = Date.now()): ApplyResult {
  switch (msg.t) {
    case 'hello': {
      const runs: Record<string, typeof msg.runs[number]> = { ...state.runs }
      for (const r of msg.runs) runs[r.id] = r
      return { state: { ...state, hello: msg.hello, gpu: msg.hello.gpu, sessions: msg.sessions, runs }, effects: [{ type: 'hello' }] }
    }
    case 'pong':
      return { state, effects: NO_EFFECTS }
    case 'error': {
      const next = addOutput({ ...state, serverErrors: pushRing(state.serverErrors, [msg.message], 50) }, { level: 'error', text: msg.message, ts: now, origin: 'server' })
      return { state: next, effects: msg.fatal ? [{ type: 'fatal', message: msg.message }] : NO_EFFECTS }
    }

    case 'session.state': {
      const s = msg.session
      const known = state.sessions.some((x) => x.id === s.id)
      const sessions = known
        ? state.sessions.map((x) => (x.id === s.id ? { ...x, title: s.title, updatedAt: s.updatedAt, messageCount: s.messages.length } : x))
        : [{ id: s.id, title: s.title, createdAt: s.createdAt, updatedAt: s.updatedAt, messageCount: s.messages.length }, ...state.sessions]
      const switching = state.currentSessionId !== s.id
      const turn = switching ? { ...EMPTY_TURN, active: s.turnActive } : { ...state.turn, active: s.turnActive }
      return { state: { ...state, session: s, currentSessionId: s.id, sessions, turn, toolInputJson: switching ? {} : state.toolInputJson }, effects: NO_EFFECTS }
    }
    case 'session.list':
      return { state: { ...state, sessions: msg.sessions }, effects: NO_EFFECTS }
    case 'session.deleted': {
      const sessions = state.sessions.filter((x) => x.id !== msg.sessionId)
      if (state.currentSessionId !== msg.sessionId) return { state: { ...state, sessions }, effects: NO_EFFECTS }
      return { state: { ...state, sessions, session: null, currentSessionId: null, turn: EMPTY_TURN }, effects: [{ type: 'session.deleted', sessionId: msg.sessionId }] }
    }

    case 'turn.start': {
      if (!isCurrent(state, msg.sessionId)) return { state, effects: NO_EFFECTS }
      const next = updateMessageAt(state, msg.messageId, now, (m) => m)
      const session = next.session ? { ...next.session, turnActive: true } : next.session
      return { state: { ...next, session, turn: { active: true, turnId: msg.turnId, messageId: msg.messageId, error: null, refusal: null, warnings: [] } }, effects: NO_EFFECTS }
    }
    case 'turn.done': {
      if (!isCurrent(state, msg.sessionId)) return { state, effects: NO_EFFECTS }
      let next = state.turn.messageId && msg.model ? updateMessageAt(state, state.turn.messageId, now, (m) => (m.model === msg.model ? m : { ...m, model: msg.model })) : state
      if (next.session) next = withSession(next, { ...next.session, turnActive: false })
      return { state: { ...next, turn: { ...next.turn, active: false } }, effects: NO_EFFECTS }
    }
    case 'turn.error': {
      if (!isCurrent(state, msg.sessionId)) return { state, effects: NO_EFFECTS }
      const session = state.session ? { ...state.session, turnActive: false } : null
      return { state: { ...state, session, turn: { ...state.turn, active: false, error: { message: msg.message, retryable: msg.retryable } } }, effects: NO_EFFECTS }
    }
    case 'turn.refusal': {
      if (!isCurrent(state, msg.sessionId)) return { state, effects: NO_EFFECTS }
      const session = state.session ? { ...state.session, turnActive: false } : null
      return { state: { ...state, session, turn: { ...state.turn, active: false, refusal: { category: msg.category, explanation: msg.explanation } } }, effects: NO_EFFECTS }
    }
    case 'turn.warning': {
      if (!isCurrent(state, msg.sessionId)) return { state, effects: NO_EFFECTS }
      return { state: { ...state, turn: { ...state.turn, warnings: [...state.turn.warnings, msg.message] } }, effects: NO_EFFECTS }
    }

    case 'msg.user': {
      if (!isCurrent(state, msg.sessionId)) return { state, effects: NO_EFFECTS }
      const next = updateMessages(state, (messages) => (messages.some((m) => m.id === msg.message.id) ? messages.map((m) => (m.id === msg.message.id ? msg.message : m)) : [...messages, msg.message]))
      return { state: next, effects: NO_EFFECTS }
    }
    case 'msg.block_start': {
      if (!isCurrent(state, msg.sessionId)) return { state, effects: NO_EFFECTS }
      const next = updateMessageAt(state, msg.messageId, now, (m) => {
        const existing = m.blocks[msg.blockIndex]
        if (existing && existing.kind === msg.kind) return m
        return setBlock(m, msg.blockIndex, { kind: msg.kind, text: '' })
      })
      return { state: next, effects: NO_EFFECTS }
    }
    case 'msg.delta': {
      if (!isCurrent(state, msg.sessionId)) return { state, effects: NO_EFFECTS }
      const next = updateMessageAt(state, msg.messageId, now, (m) => {
        const existing = m.blocks[msg.blockIndex]
        if (existing && (existing.kind === 'text' || existing.kind === 'thinking')) return setBlock(m, msg.blockIndex, { kind: existing.kind, text: existing.text + msg.delta })
        return setBlock(m, msg.blockIndex, { kind: 'text', text: msg.delta })
      })
      return { state: next, effects: NO_EFFECTS }
    }
    case 'msg.done': {
      if (!isCurrent(state, msg.sessionId)) return { state, effects: NO_EFFECTS }
      const toolIds = msg.message.blocks.filter((b): b is Extract<UiBlock, { kind: 'tool' }> => b.kind === 'tool').map((b) => b.call.toolUseId)
      let toolInputJson = state.toolInputJson
      if (toolIds.some((id) => id in toolInputJson)) {
        toolInputJson = { ...toolInputJson }
        for (const id of toolIds) delete toolInputJson[id]
      }
      const next = updateMessages(state, (messages) => (messages.some((m) => m.id === msg.message.id) ? messages.map((m) => (m.id === msg.message.id ? msg.message : m)) : [...messages, msg.message]))
      return { state: { ...next, toolInputJson }, effects: NO_EFFECTS }
    }

    case 'tool.start': {
      if (!isCurrent(state, msg.sessionId)) return { state, effects: NO_EFFECTS }
      const call: ToolCallRecord = {
        toolUseId: msg.toolUseId,
        name: msg.name,
        input: null,
        policy: toolPolicy(msg.name),
        status: 'pending',
        summary: '',
        resultPreview: null,
        error: null,
        runId: null,
        startedAt: now,
        endedAt: null,
      }
      const next = updateMessageAt(state, msg.messageId, now, (m) => {
        const existing = m.blocks[msg.blockIndex]
        if (existing && existing.kind === 'tool' && existing.call.toolUseId === msg.toolUseId) return m
        return setBlock(m, msg.blockIndex, { kind: 'tool', call })
      })
      return { state: { ...next, toolInputJson: { ...next.toolInputJson, [msg.toolUseId]: '' } }, effects: NO_EFFECTS }
    }
    case 'tool.input_delta': {
      if (!isCurrent(state, msg.sessionId)) return { state, effects: NO_EFFECTS }
      const prev = state.toolInputJson[msg.toolUseId] ?? ''
      return { state: { ...state, toolInputJson: { ...state.toolInputJson, [msg.toolUseId]: prev + msg.partialJson } }, effects: NO_EFFECTS }
    }
    case 'tool.update': {
      if (!isCurrent(state, msg.sessionId)) return { state, effects: NO_EFFECTS }
      let found = false
      let next = updateToolCall(state, msg.call.toolUseId, (call) => {
        found = true
        return call === msg.call ? call : msg.call
      })
      if (!found && state.turn.messageId) {
        next = updateMessageAt(next, state.turn.messageId, now, (m) => ({ ...m, blocks: [...m.blocks, { kind: 'tool', call: msg.call }] }))
      }
      return { state: next, effects: NO_EFFECTS }
    }
    case 'tool.approval_request': {
      if (!isCurrent(state, msg.sessionId) || !state.session) return { state, effects: NO_EFFECTS }
      const others = state.session.pendingApprovals.filter((a) => a.turnId !== msg.approval.turnId || a.requestedAt !== msg.approval.requestedAt)
      let next = withSession(state, { ...state.session, pendingApprovals: [...others, msg.approval] })
      for (const id of msg.approval.toolUseIds) next = updateToolCall(next, id, (call) => (call.status === 'awaiting_approval' ? call : { ...call, status: 'awaiting_approval' }))
      return { state: next, effects: NO_EFFECTS }
    }
    case 'tool.approval_resolved': {
      if (!isCurrent(state, msg.sessionId) || !state.session) return { state, effects: NO_EFFECTS }
      let next = withSession(state, { ...state.session, pendingApprovals: resolveApprovals(state.session.pendingApprovals, msg.toolUseIds) })
      for (const id of msg.toolUseIds) {
        next = updateToolCall(next, id, (call) => {
          if (call.status !== 'awaiting_approval' && call.status !== 'pending') return call
          if (msg.decision === 'approved') return { ...call, status: 'running' }
          return { ...call, status: 'denied', error: msg.decision === 'expired' ? 'approval expired' : call.error }
        })
      }
      return { state: next, effects: NO_EFFECTS }
    }

    case 'run.started': {
      const next: SessionData = { ...state, runs: { ...state.runs, [msg.run.id]: msg.run } }
      if (!next.runData[msg.run.id]) next.runData = { ...next.runData, [msg.run.id]: emptyRunData() }
      return { state: next, effects: [{ type: 'run.subscribe', runId: msg.run.id }] }
    }
    case 'run.updated':
      return { state: { ...state, runs: { ...state.runs, [msg.run.id]: msg.run } }, effects: NO_EFFECTS }
    case 'run.log':
      return { state: pushLogs(state, msg.runId, msg.lines), effects: NO_EFFECTS }
    case 'run.residual':
      return { state: pushResiduals(state, msg.runId, [msg.rec]), effects: NO_EFFECTS }
    case 'run.metric':
      return { state: pushMetrics(state, msg.runId, [msg.rec]), effects: NO_EFFECTS }
    case 'run.written': {
      const run = state.runs[msg.runId]
      if (!run || run.written.includes(msg.dir)) return { state, effects: NO_EFFECTS }
      return { state: { ...state, runs: { ...state.runs, [msg.runId]: { ...run, written: [...run.written, msg.dir] } } }, effects: NO_EFFECTS }
    }
    case 'run.exit': {
      const r = msg.run
      const text = `${r.label ?? r.binary} [${r.id}] ${r.status}${r.exitCode !== null ? ` (exit ${r.exitCode})` : ''}${r.error ? `: ${r.error}` : ''}`
      const level = r.status === 'done' ? 'info' : r.status === 'killed' ? 'warning' : 'error'
      const next = addOutput({ ...state, runs: { ...state.runs, [r.id]: r } }, { level, text, ts: now, origin: 'client' })
      return { state: next, effects: NO_EFFECTS }
    }

    case 'viewer.command':
      return { state, effects: [{ type: 'viewer.command', requestId: msg.requestId, cmd: msg.cmd }] }
    case 'viewer.open':
      return { state, effects: [{ type: 'viewer.open', path: msg.path, runId: msg.runId }] }
    case 'residuals.open':
      return { state, effects: [{ type: 'residuals.open', runId: msg.runId }] }
    case 'dataset.progress':
      return { state: { ...state, datasetProgress: { ...state.datasetProgress, [msg.progress.datasetId]: msg.progress } }, effects: NO_EFFECTS }

    case 'fs.changed':
      return { state, effects: [{ type: 'fs.changed', paths: msg.paths }] }
    case 'problems': {
      const key = problemKey(msg.source, msg.runId, msg.path)
      const problems = { ...state.problems }
      if (msg.items.length) problems[key] = msg.items
      else delete problems[key]
      return { state: { ...state, problems }, effects: NO_EFFECTS }
    }
    case 'gpu':
      return { state: { ...state, gpu: msg.gpu }, effects: NO_EFFECTS }
    case 'output':
      return { state: addOutput(state, { level: msg.level, text: msg.text, ts: msg.ts, origin: 'server' }), effects: NO_EFFECTS }
  }
}

/**
 * Apply a batch of frames in order, collecting their effects. Consecutive
 * run.residual / run.metric / run.log frames for one run (a replay burst)
 * are coalesced into a single ring append so a 50k replay costs one copy.
 */
export function applyEvents(state: SessionData, msgs: ServerMsg[], now: number = Date.now()): ApplyResult {
  let s = state
  const effects: Effect[] = []
  let i = 0
  while (i < msgs.length) {
    const m = msgs[i]
    if (m.t === 'run.residual' || m.t === 'run.metric' || m.t === 'run.log') {
      let j = i + 1
      while (j < msgs.length && msgs[j].t === m.t && (msgs[j] as { runId: string }).runId === m.runId) j++
      if (j - i > 1) {
        const slice = msgs.slice(i, j)
        if (m.t === 'run.residual') s = pushResiduals(s, m.runId, slice.map((x) => (x as Extract<ServerMsg, { t: 'run.residual' }>).rec))
        else if (m.t === 'run.metric') s = pushMetrics(s, m.runId, slice.map((x) => (x as Extract<ServerMsg, { t: 'run.metric' }>).rec))
        else s = pushLogs(s, m.runId, slice.flatMap((x) => (x as Extract<ServerMsg, { t: 'run.log' }>).lines))
        i = j
        continue
      }
    }
    const r = applyEvent(s, m, now)
    s = r.state
    if (r.effects.length) effects.push(...r.effects)
    i++
  }
  return { state: s, effects }
}
