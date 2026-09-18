// The agent service: sessions on disk, one active turn per session, the
// approval brokers, the quick actions and the run-end notifications, all
// driven by the client frames the hub hands over.
import fsp from 'node:fs/promises'
import type { BetaContentBlockParam, BetaMessageParam, BetaTextBlockParam } from '@anthropic-ai/sdk/resources/beta/messages/messages'
import type { ChatRequest, ChatResponse, ClientMsg, ClientMsgOf, CustomToolSummary, PendingApproval, ServerMsg, SessionSettings, SessionState, SessionSummary, UiBlock, UiMessage, UserContext } from '@cfd/shared'
import { CHAT_TIMEOUT_DEFAULT_MS } from '@cfd/shared'
import { attachmentObjectBlocks, dehydrateAttachmentImages, MAX_ATTACHMENT_IDS, visionMode } from '../attachments/blocks.js'
import { looksBinary } from '../attachments/sniff.js'
import { createAttachmentStore } from '../attachments/store.js'
import type { ServerConfig } from '../config.js'
import type { DatasetService } from '../datasets/types.js'
import type { RunManager } from '../runs/types.js'
import { loadCustomTools } from '../tools/custom.js'
import { mergeTools } from '../tools/defaults.js'
import { ontologyHandle, ontologyPreviewFor } from '../ontology/handle.js'
import type { Hub, ClientConn } from '../ws/types.js'
import { resolveInWorkspace } from '../workspace/paths.js'
import { createAnthropicClient } from './anthropic.js'
import { createApprovalManager, type ApprovalManager } from './approvals.js'
import { emptyUsage, type LlmClient } from './llm.js'
import type { LlmSettingsHandle } from './llmSettings.js'
import { runTurn, type TurnOutcome } from './loop.js'
import { createMockLlm } from './mockLlm.js'
import { loadPolicyOverrides, type PolicyOverrides } from './policy.js'
import { runNoticeText, runNoticeUserText } from './prompt.js'
import { buildQuickMessage } from './quick.js'
import { createZaiClient } from './zai.js'
import { appendUserTurn, createSessionStore, newId, stateOf, summaryOf, type SessionRecord, type SessionStore } from './session.js'
import { ChatError, type AgentService } from './types.js'

export interface AgentServiceDeps {
  config: ServerConfig
  hub: Hub
  runs: RunManager
  datasets: DatasetService
  /** Injected by tests; defaults to the runtime settings handle's client, else one built from config. */
  llm?: LlmClient
  /** The runtime provider/key settings (main wires one); the client is re-read per turn, so a key entered mid-session counts. */
  llmSettings?: LlmSettingsHandle
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
  // The client a turn uses is picked when the turn starts: the runtime handle
  // reflects a key or provider the user entered mid-session. Tests that inject
  // `llm` keep their client, and the config-built one stays as the last resort.
  const staticLlm: LlmClient | null = deps.llm ?? (config.llm === 'mock' ? createMockLlm({ model: config.model }) : config.llm === 'zai' ? createZaiClient(config) : createAnthropicClient(config))
  const llmForTurn = (): LlmClient => deps.llm ?? deps.llmSettings?.client() ?? staticLlm!
  const overrides = deps.overrides ?? loadPolicyOverrides(config.configDir)
  const runtimes = new Map<string, SessionRuntime>()
  let customTools: CustomToolSummary[] = []

  const refreshCustomTools = async () => {
    // The shipped defaults too, so the prompt names tools the user never registered.
    customTools = mergeTools(await loadCustomTools(config.configDir)).map((t) => ({ name: t.name, description: t.description, inputSchema: t.inputSchema }))
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

  function startTurn(rec: SessionRecord): string | null {
    const rt = runtime(rec.id)
    if (rt.active) return null
    const turnId = newId('t')
    const controller = new AbortController()
    const done = runTurn(rec, turnId, controller.signal, {
      config,
      hub,
      runs,
      datasets,
      llm: llmForTurn(),
      approvals: rt.approvals,
      overrides,
      store,
      customTools: () => customTools.map((t) => t.name),
      userContext: () => rt.context,
      emit: (msg: ServerMsg) => hub.sendToSession(rec.id, msg),
      ontologyPreview: ontologyPreviewFor({ config, runs, hub }, rec.id),
      retryDelayMs: deps.retryDelayMs,
    })
      .catch((err: unknown): TurnOutcome => {
        hub.sendToSession(rec.id, { t: 'turn.error', sessionId: rec.id, turnId, message: `internal error: ${(err as Error).message}`, retryable: false })
        return { status: 'error', rounds: 0, model: null, usage: emptyUsage() }
      })
      .then(async (outcome) => {
        if (rt.active?.turnId === turnId) rt.active = null
        // The image goes home the moment the turn ends; only its <attachment/>
        // reference stays in the history and in the session file (D12).
        if (dehydrateAttachmentImages(rec)) await store.save(rec)
        await refreshCustomTools()
        if (outcome.status === 'cancelled') sendState(rec)
        broadcastList()
        return outcome
      })
    rt.active = { turnId, controller, done }
    return turnId
  }

  function cancelTurn(rec: SessionRecord): void {
    const rt = runtime(rec.id)
    // Abort first: cancelAll only reaches waiters that already exist, and the
    // loop checks the signal before it creates any more.
    rt.active?.controller.abort()
    rt.approvals.cancelAll('cancelled by user')
  }

  async function attachmentBlocks(context: UserContext): Promise<{ blocks: BetaContentBlockParam[]; notices: string[]; warnings: string[] }> {
    const blocks: BetaContentBlockParam[] = []
    const notices: string[] = []
    const warnings: string[] = []
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
        // A NUL byte in the first 8 KB means binary: name the file, never utf8-decode
        // it into silent mojibake (D14, the 16 KB truncation defect the row names).
        if (looksBinary(buf.subarray(0, 8 * 1024))) {
          blocks.push({ type: 'text', text: `<file path="${r.rel}" bytes="${buf.length}" binary="true" note="binary file; post it to /api/attachments to show it to the model"/>` })
          notices.push(`@${r.rel} (${buf.length} bytes, binary — not shown as text)`)
          continue
        }
        const text = buf.subarray(0, ATTACHMENT_CAP).toString('utf8')
        blocks.push({ type: 'text', text: `<file path="${r.rel}"${truncated ? ` truncated="true" bytes="${buf.length}"` : ''}>\n${text}\n</file>` })
        notices.push(`@${r.rel} (${buf.length} bytes${truncated ? ', truncated to 16 KB' : ''})`)
      } catch (err) {
        blocks.push({ type: 'text', text: `<file path="${rel}" error="${(err as Error).message}"/>` })
        notices.push(`@${rel}: ${(err as Error).message}`)
      }
    }
    if (context.attachmentIds.length) {
      try {
        // N5's memoised handle: the same instance the REST routes hold, so there is no second database.
        const handle = await ontologyHandle({ config, runs, hub: undefined })
        const attached = await attachmentObjectBlocks(context.attachmentIds, { mode: visionMode(config), store: createAttachmentStore(config), mirror: handle.store })
        blocks.push(...attached.blocks)
        notices.push(...attached.notices)
        warnings.push(...attached.warnings)
      } catch (err) {
        // An unreachable mirror or store must never fail the turn: name each id,
        // exactly the way an unknown one is named (C13).
        for (const id of context.attachmentIds.slice(0, MAX_ATTACHMENT_IDS)) {
          blocks.push({ type: 'text', text: `<attachment id="${id}" error="${(err as Error).message}"/>` })
          notices.push(`attachment ${id}: ${(err as Error).message}`)
        }
      }
    }
    if (context.selection) blocks.push({ type: 'text', text: `<selection${context.activeFile ? ` file="${context.activeFile}"` : ''}>\n${context.selection.slice(0, ATTACHMENT_CAP)}\n</selection>` })
    return { blocks, notices, warnings }
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
      // A crashed turn must not carry its base64 into this turn's history or file (D12).
      dehydrateAttachmentImages(rec)
      // The session list shows the case a conversation was about: the open case
      // the window reports as its active file, kept with the record.
      if (msg.context.activeFile) rec.casePath = msg.context.activeFile
      const text = msg.text.trim()
      const { blocks, notices, warnings } = await attachmentBlocks(msg.context)
      if (!text && !blocks.length) return client.send({ t: 'error', message: 'empty message', fatal: false })
      const content: BetaContentBlockParam[] = [...(text ? [{ type: 'text', text } as BetaContentBlockParam] : []), ...blocks]
      const message: BetaMessageParam = { role: 'user', content: text && !blocks.length ? text : content }
      const ui = appendUserTurn(rec, message, { synthetic: false, notices, warnings, entitle: true })
      if (ui) {
        // The bubble never carries the image bytes: text, then notices and warnings only.
        if (blocks.length) ui.blocks = [...(text ? [{ kind: 'text', text } as UiBlock] : []), ...ui.blocks.filter((b) => b.kind === 'notice')]
        hub.sendToSession(rec.id, { t: 'msg.user', sessionId: rec.id, message: ui })
      }
      await store.save(rec)
      if (!startTurn(rec)) client.send({ t: 'error', message: 'a turn is already active in this session; cancel it first', fatal: false })
    } finally {
      rt.starting = false
    }
  }

  /** N7: the REST entry a program drives. Everything up to the userMessage call is
   *  synchronous, so two concurrent POSTs cannot both pass the guard. */
  async function chat(req: ChatRequest): Promise<ChatResponse> {
    const patch: Partial<SessionSettings> = {}
    if (req.autoApprove) patch.autoApprove = req.autoApprove
    if (req.locale) patch.locale = req.locale
    const fresh = req.sessionId === null
    const rec = req.sessionId === null ? store.create(patch) : store.get(req.sessionId)
    if (!rec) throw new ChatError(404, `no session ${req.sessionId}`)
    const rt = runtime(rec.id)
    const busy = rt.active !== null || rt.starting
    if (busy) throw new ChatError(409, 'a turn is already active in this session; cancel it first')
    if (!fresh && Object.keys(patch).length) rec.settings = { ...rec.settings, ...patch }
    const uiBefore = rec.ui.length
    const runsBefore = rec.runs.length
    const context: UserContext = {
      activeFile: req.activeFile, activeRun: null, activeStep: null, activeTab: null,
      attachments: req.attachments, attachmentIds: req.attachmentIds, selection: null,
    }
    const errors: string[] = []
    const conn: ClientConn = {
      id: 'http', sessionId: rec.id, runs: new Set(), viewerState: null, uiState: null,
      send: (m) => { if (m.t === 'error') errors.push(m.message) },
      close: () => {},
    }
    await userMessage(conn, { t: 'user.message', sessionId: rec.id, text: req.text, context })
    if (errors.length) {
      // A session created here must not survive a refusal, or it is an orphan.
      if (fresh) await store.delete(rec.id)
      throw new ChatError(errors[0] === 'empty message' ? 400 : 409, errors[0])
    }
    // userMessage set rt.active before its first await; the busy boolean above kept
    // the checker from narrowing the property to null in the meantime.
    const active = rt.active
    if (!active) throw new ChatError(500, 'the turn did not start')
    if (fresh) broadcastList()
    const timeoutMs = req.timeoutMs ?? CHAT_TIMEOUT_DEFAULT_MS
    let timedOut = false
    let approvalsAtTimeout: PendingApproval[] = []
    const outcome = await Promise.race([
      active.done,
      new Promise<null>((resolve) => {
        const t = setTimeout(() => { timedOut = true; approvalsAtTimeout = rt.approvals.pending(); resolve(null) }, timeoutMs)
        void active.done.finally(() => clearTimeout(t))
      }),
    ])
    return {
      sessionId: rec.id,
      turnId: active.turnId,
      status: timedOut || !outcome ? 'timeout' : outcome.status,
      rounds: outcome?.rounds ?? 0,
      model: outcome?.model ?? null,
      usage: outcome?.usage ?? emptyUsage(),
      messages: rec.ui.slice(uiBefore),
      pendingApprovals: timedOut ? approvalsAtTimeout : rt.approvals.pending(),
      runs: rec.runs.slice(runsBefore),
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
      if (msg.casePath) rec.casePath = msg.casePath
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
    // The model still answers a user turn - the API needs one - but the screen
    // shows a system notice on an assistant message: the operator never sees
    // words attributed to them that they did not write.
    rec.messages.push({ role: 'user', content: runNoticeUserText(run, rec.settings.locale) })
    const level = run.status === 'failed' || run.status === 'diverged' ? 'error' : run.status === 'killed' ? 'warning' : 'info'
    const ui: UiMessage = {
      id: newId('m'),
      role: 'assistant',
      blocks: [{ kind: 'notice', level, text: runNoticeText(run, rec.settings.locale) }],
      createdAt: Date.now(),
      stopReason: null,
      model: null,
      suggestions: [],
      synthetic: true,
    }
    rec.ui.push(ui)
    hub.sendToSession(rec.id, { t: 'msg.done', sessionId: rec.id, message: ui })
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
    chat,
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
