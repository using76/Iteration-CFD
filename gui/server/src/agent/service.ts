// The agent service: sessions on disk, one active turn per session, the
// approval brokers, the quick actions and the run-end notifications, all
// driven by the client frames the hub hands over.
import fsp from 'node:fs/promises'
import type { BetaMessageParam, BetaTextBlockParam } from '@anthropic-ai/sdk/resources/beta/messages/messages'
import type { ClientMsg, ClientMsgOf, ServerMsg, SessionState, SessionSummary, UserContext } from '@cfd/shared'
import type { ServerConfig } from '../config.js'
import type { DatasetService } from '../datasets/types.js'
import type { RunManager } from '../runs/types.js'
import { loadCustomTools } from '../tools/custom.js'
import type { Hub, ClientConn } from '../ws/types.js'
import { resolveInWorkspace } from '../workspace/paths.js'
import { createAnthropicClient } from './anthropic.js'
import { createApprovalManager, type ApprovalManager } from './approvals.js'
import type { LlmClient } from './llm.js'
import { runTurn, type TurnOutcome } from './loop.js'
import { createMockLlm } from './mockLlm.js'
import { loadPolicyOverrides, type PolicyOverrides } from './policy.js'
import { runNoticeText } from './prompt.js'
import { buildQuickMessage } from './quick.js'
import { appendUserTurn, createSessionStore, newId, stateOf, summaryOf, type SessionRecord, type SessionStore } from './session.js'
import type { AgentService } from './types.js'

export interface AgentServiceDeps {
  config: ServerConfig
  hub: Hub
  runs: RunManager
  datasets: DatasetService
  /** Injected by tests; defaults to the Anthropic client or the mock per config.llm. */
  llm?: LlmClient
  store?: SessionStore
  overrides?: PolicyOverrides
  retryDelayMs?: number
}

export const ATTACHMENT_CAP = 16 * 1024
const MAX_ATTACHMENTS = 8

interface ActiveTurn {
  turnId: string
  controller: AbortController
  done: Promise<TurnOutcome>
}

interface SessionRuntime {
  approvals: ApprovalManager
  active: ActiveTurn | null
  /** Set synchronously while a turn is being assembled, before `active` exists. */
  starting: boolean
  context: UserContext | null
}

const AGENT_FRAMES = new Set<ClientMsg['t']>(['session.open', 'session.new', 'session.list', 'session.delete', 'session.rename', 'user.message', 'turn.cancel', 'tool.approve', 'tool.deny', 'quick', 'settings.set'])

export function createAgentService(deps: AgentServiceDeps): AgentService {
  const { config, hub, runs, datasets } = deps
  const store = deps.store ?? createSessionStore(config.sessionsDir, config.model)
  const llm = deps.llm ?? (config.llm === 'mock' ? createMockLlm({ model: config.model }) : createAnthropicClient(config))
  const overrides = deps.overrides ?? loadPolicyOverrides(config.configDir)
  const runtimes = new Map<string, SessionRuntime>()
  let customTools: Array<{ name: string; description: string }> = []

  const refreshCustomTools = async () => {
    customTools = (await loadCustomTools(config.configDir)).map((t) => ({ name: t.name, description: t.description }))
  }
  void refreshCustomTools()

  function runtime(sessionId: string): SessionRuntime {
    let rt = runtimes.get(sessionId)
    if (!rt) {
      rt = {
        approvals: createApprovalManager((toolUseIds, decision) => hub.sendToSession(sessionId, { t: 'tool.approval_resolved', sessionId, toolUseIds, decision })),
        active: null,
        starting: false,
        context: null,
      }
      runtimes.set(sessionId, rt)
    }
    return rt
  }

  function state(rec: SessionRecord): SessionState {
    const rt = runtime(rec.id)
    return stateOf(rec, { pendingApprovals: rt.approvals.pending(), turnActive: rt.active !== null, customTools })
  }

  const sendState = (rec: SessionRecord) => hub.sendToSession(rec.id, { t: 'session.state', session: state(rec) })
  const broadcastList = () => hub.broadcast({ t: 'session.list', sessions: listSessions() })

  function listSessions(): SessionSummary[] {
    return store.list().map(summaryOf)
  }

  function createSession(): SessionState {
    const rec = store.create()
    void store.save(rec)
    return state(rec)
  }

  function startTurn(rec: SessionRecord): boolean {
    const rt = runtime(rec.id)
    if (rt.active) return false
    const turnId = newId('t')
    const controller = new AbortController()
    const done = runTurn(rec, turnId, controller.signal, {
      config,
      hub,
      runs,
      datasets,
      llm,
      approvals: rt.approvals,
      overrides,
      store,
      customTools: () => customTools.map((t) => t.name),
      userContext: () => rt.context,
      emit: (msg: ServerMsg) => hub.sendToSession(rec.id, msg),
      retryDelayMs: deps.retryDelayMs,
    })
      .catch((err: unknown): TurnOutcome => {
        hub.sendToSession(rec.id, { t: 'turn.error', sessionId: rec.id, turnId, message: `internal error: ${(err as Error).message}`, retryable: false })
        return { status: 'error', rounds: 0, model: null }
      })
      .then(async (outcome) => {
        if (rt.active?.turnId === turnId) rt.active = null
        await refreshCustomTools()
        if (outcome.status === 'cancelled') sendState(rec)
        broadcastList()
        return outcome
      })
    rt.active = { turnId, controller, done }
    return true
  }

  function cancelTurn(rec: SessionRecord): void {
    const rt = runtime(rec.id)
    // Abort first: cancelAll only reaches waiters that already exist, and the
    // loop checks the signal before it creates any more.
    rt.active?.controller.abort()
    rt.approvals.cancelAll('cancelled by user')
  }

  async function attachmentBlocks(context: UserContext): Promise<{ blocks: BetaTextBlockParam[]; notices: string[] }> {
    const blocks: BetaTextBlockParam[] = []
    const notices: string[] = []
    for (const rel of context.attachments.slice(0, MAX_ATTACHMENTS)) {
      try {
        const r = resolveInWorkspace(config.workspaceRoot, rel, { mustExist: true })
        const st = await fsp.stat(r.abs)
        if (st.isDirectory()) {
          const names = (await fsp.readdir(r.abs)).sort().slice(0, 200)
          blocks.push({ type: 'text', text: `<directory path="${r.rel}">\n${names.join('\n')}\n</directory>` })
          notices.push(`@${r.rel} (${names.length} entries)`)
          continue
        }
        const buf = await fsp.readFile(r.abs)
        const truncated = buf.length > ATTACHMENT_CAP
        const text = buf.subarray(0, ATTACHMENT_CAP).toString('utf8')
        blocks.push({ type: 'text', text: `<file path="${r.rel}"${truncated ? ` truncated="true" bytes="${buf.length}"` : ''}>\n${text}\n</file>` })
        notices.push(`@${r.rel} (${buf.length} bytes${truncated ? ', truncated to 16 KB' : ''})`)
      } catch (err) {
        blocks.push({ type: 'text', text: `<file path="${rel}" error="${(err as Error).message}"/>` })
        notices.push(`@${rel}: ${(err as Error).message}`)
      }
    }
    if (context.selection) blocks.push({ type: 'text', text: `<selection${context.activeFile ? ` file="${context.activeFile}"` : ''}>\n${context.selection.slice(0, ATTACHMENT_CAP)}\n</selection>` })
    return { blocks, notices }
  }

  async function userMessage(client: ClientConn, msg: ClientMsgOf<'user.message'>): Promise<void> {
    const rec = store.get(msg.sessionId)
    if (!rec) return client.send({ t: 'error', message: `no session ${msg.sessionId}`, fatal: false })
    const rt = runtime(rec.id)
    if (rt.active || rt.starting) return client.send({ t: 'error', message: 'a turn is already active in this session; cancel it first', fatal: false })
    // Reserved before the first await: reading attachments takes long enough
    // for a second frame to pass an `active`-only guard, and the second turn
    // would then be appended to the history and never answered.
    rt.starting = true
    try {
      rt.context = msg.context
      const text = msg.text.trim()
      const { blocks, notices } = await attachmentBlocks(msg.context)
      if (!text && !blocks.length) return client.send({ t: 'error', message: 'empty message', fatal: false })
      const content: BetaTextBlockParam[] = [...(text ? [{ type: 'text', text } as BetaTextBlockParam] : []), ...blocks]
      const message: BetaMessageParam = { role: 'user', content: text && !blocks.length ? text : content }
      const ui = appendUserTurn(rec, message, { synthetic: false, notices, entitle: true })
      if (ui) {
        if (blocks.length && text) ui.blocks = [{ kind: 'text', text }, ...ui.blocks.filter((b) => b.kind === 'notice')]
        hub.sendToSession(rec.id, { t: 'msg.user', sessionId: rec.id, message: ui })
      }
      await store.save(rec)
      if (!startTurn(rec)) client.send({ t: 'error', message: 'a turn is already active in this session; cancel it first', fatal: false })
    } finally {
      rt.starting = false
    }
  }

  async function quick(client: ClientConn, msg: ClientMsgOf<'quick'>): Promise<void> {
    const rec = store.get(msg.sessionId)
    if (!rec) return client.send({ t: 'error', message: `no session ${msg.sessionId}`, fatal: false })
    const rt = runtime(rec.id)
    if (rt.active || rt.starting) return client.send({ t: 'error', message: 'a turn is already active in this session; cancel it first', fatal: false })
    rt.starting = true
    try {
      const built = buildQuickMessage({ action: msg.action, casePath: msg.casePath, runId: msg.runId, activeFile: rt.context?.activeFile ?? null, locale: rec.settings.locale, runs })
      const ui = appendUserTurn(rec, { role: 'user', content: built.text }, { synthetic: true, entitle: true })
      if (ui) hub.sendToSession(rec.id, { t: 'msg.user', sessionId: rec.id, message: ui })
      await store.save(rec)
      if (!startTurn(rec)) client.send({ t: 'error', message: 'a turn is already active in this session; cancel it first', fatal: false })
    } finally {
      rt.starting = false
    }
  }

  async function handleClientMessage(client: ClientConn, msg: ClientMsg): Promise<boolean> {
    if (!AGENT_FRAMES.has(msg.t)) return false
    switch (msg.t) {
      case 'session.open': {
        let rec = msg.sessionId ? store.get(msg.sessionId) : undefined
        if (!rec) {
          rec = store.create()
          await store.save(rec)
          broadcastList()
        }
        client.sessionId = rec.id
        client.send({ t: 'session.state', session: state(rec) })
        return true
      }
      case 'session.new': {
        const rec = store.create()
        await store.save(rec)
        client.sessionId = rec.id
        client.send({ t: 'session.state', session: state(rec) })
        broadcastList()
        return true
      }
      case 'session.list':
        client.send({ t: 'session.list', sessions: listSessions() })
        return true
      case 'session.delete': {
        await deleteSession(msg.sessionId)
        hub.broadcast({ t: 'session.deleted', sessionId: msg.sessionId })
        broadcastList()
        return true
      }
      case 'session.rename': {
        const rec = store.get(msg.sessionId)
        if (rec) {
          rec.title = msg.title.trim().slice(0, 120) || rec.title
          await store.save(rec)
          broadcastList()
        }
        return true
      }
      case 'settings.set': {
        const rec = store.get(msg.sessionId)
        if (!rec) return true
        rec.settings = { ...rec.settings, ...msg.patch }
        await store.save(rec)
        sendState(rec)
        return true
      }
      case 'user.message':
        await userMessage(client, msg)
        return true
      case 'quick':
        await quick(client, msg)
        return true
      case 'turn.cancel': {
        const rec = store.get(msg.sessionId)
        if (rec) cancelTurn(rec)
        return true
      }
      case 'tool.approve': {
        const rec = store.get(msg.sessionId)
        if (!rec) return true
        const rt = runtime(rec.id)
        if (msg.remember === 'session') {
          const names = rt.approvals
            .pending()
            .flatMap((a) => a.calls)
            .filter((c) => msg.toolUseIds.includes(c.toolUseId))
            .map((c) => c.name)
          for (const name of names) if (!rec.allowedTools.includes(name)) rec.allowedTools.push(name)
          if (names.length) await store.save(rec)
        }
        rt.approvals.resolve(msg.toolUseIds, 'approved')
        return true
      }
      case 'tool.deny': {
        const rec = store.get(msg.sessionId)
        if (rec) runtime(rec.id).approvals.resolve(msg.toolUseIds, 'denied', msg.reason)
        return true
      }
      default:
        return false
    }
  }

  async function deleteSession(sessionId: string): Promise<boolean> {
    const rec = store.get(sessionId)
    if (!rec) return false
    cancelTurn(rec)
    const rt = runtimes.get(sessionId)
    if (rt?.active) await rt.active.done.catch(() => {})
    runtimes.delete(sessionId)
    return store.delete(sessionId)
  }

  function notifyRunEnded(runId: string): void {
    const run = runs.get(runId)
    if (!run) return
    const rec = store.list().find((r) => r.runs.includes(runId))
    if (!rec || !rec.settings.notifyOnRunEnd) return
    const rt = runtime(rec.id)
    if (rt.active) return
    const ui = appendUserTurn(rec, { role: 'user', content: runNoticeText(run, rec.settings.locale) }, { synthetic: true })
    if (ui) hub.sendToSession(rec.id, { t: 'msg.user', sessionId: rec.id, message: ui })
    void store.save(rec).then(() => startTurn(rec))
  }

  return {
    handleClientMessage,
    listSessions,
    getSessionState: (id) => {
      const rec = store.get(id)
      return rec ? state(rec) : null
    },
    createSession,
    deleteSession: (id) => {
      const exists = store.get(id) !== undefined
      if (exists) void deleteSession(id)
      return exists
    },
    notifyRunEnded,
    async shutdown() {
      const waits: Promise<unknown>[] = []
      for (const [id, rt] of runtimes) {
        const rec = store.get(id)
        if (rec) cancelTurn(rec)
        if (rt.active) waits.push(rt.active.done.catch(() => {}))
      }
      await Promise.all(waits)
    },
  }
}
