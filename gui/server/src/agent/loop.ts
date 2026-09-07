// One turn of the agent: stream a response, persist it, run the tools it
// asked for (auto ones immediately, `ask` ones after one batched approval),
// return every tool_result in ONE user message, and repeat until the model
// stops. Every stop_reason, cancellation and error path leaves the history
// consistent (each tool_use has exactly one tool_result) before turn.done.
import type { BetaContentBlockParam, BetaMessage, BetaToolResultBlockParam, BetaToolUseBlock, BetaUsage } from '@anthropic-ai/sdk/resources/beta/messages/messages'
import { summarizeToolCall, toolLabel, type ServerMsg, type ToolCallRecord, type UiMessage, type UserContext } from '@cfd/shared'
import type { ServerConfig } from '../config.js'
import type { DatasetService } from '../datasets/types.js'
import type { RunManager } from '../runs/types.js'
import { isToolResult, previewCaseEdit, type CaseEditInput } from '../tools/case.js'
import { fail, type ToolContext, type ToolResult } from '../tools/context.js'
import { getTool, runTool, toolDefinitions, toolResultBlock } from '../tools/index.js'
import { meshArgs } from '../tools/mesh.js'
import type { Hub } from '../ws/types.js'
import { describeError, isAbortError, isRetryableError, isSystemRoleRejection } from './anthropic.js'
import type { ApprovalManager } from './approvals.js'
import { emptyUsage, type LlmClient } from './llm.js'
import { classifyTool, type PolicyOverrides } from './policy.js'
import { BUDGET_EXHAUSTED_TEXT, buildVolatileContext, foldContextIntoUser, systemParam, volatileSystemMessage } from './prompt.js'
import { appendUserTurn, newId, type SessionRecord, type SessionStore } from './session.js'
import { createStreamProjector, projectAssistant, type AssistantExtras, type UiStopReason } from './ui-projection.js'

export const MAX_TOOL_ROUNDS = 40
export const MAX_TOKENS = 64_000
export const RETRY_DELAY_MS = 2000

export interface TurnDeps {
  config: ServerConfig
  hub: Hub
  runs: RunManager
  datasets: DatasetService
  llm: LlmClient
  approvals: ApprovalManager
  overrides: PolicyOverrides
  store: SessionStore
  customTools(): string[]
  userContext(): UserContext | null
  emit(msg: ServerMsg): void
  retryDelayMs?: number
  now?: () => Date
}

export type TurnStatus = 'done' | 'error' | 'refusal' | 'cancelled'

export interface TurnOutcome {
  status: TurnStatus
  rounds: number
  model: string | null
}

type Usage = ReturnType<typeof emptyUsage>

function addUsage(total: Usage, u: BetaUsage | undefined): void {
  if (!u) return
  total.inputTokens += u.input_tokens ?? 0
  total.outputTokens += u.output_tokens ?? 0
  total.cacheReadTokens += u.cache_read_input_tokens ?? 0
  total.cacheWriteTokens += u.cache_creation_input_tokens ?? 0
}

function sleep(ms: number, signal: AbortSignal): Promise<boolean> {
  return new Promise((resolve) => {
    if (signal.aborted) return resolve(false)
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', onAbort)
      resolve(true)
    }, ms)
    const onAbort = () => {
      clearTimeout(timer)
      resolve(false)
    }
    signal.addEventListener('abort', onAbort, { once: true })
  })
}

function isToolUse(b: BetaContentBlockParam | BetaMessage['content'][number]): b is BetaToolUseBlock {
  return b.type === 'tool_use'
}

function isToolJsonParseError(err: unknown): boolean {
  return err instanceof Error && /Unable to parse tool parameter JSON/.test(err.message)
}

function newCall(tu: BetaToolUseBlock, locale: 'ko' | 'en'): ToolCallRecord {
  return { toolUseId: tu.id, name: tu.name, input: tu.input, policy: 'auto', status: 'pending', summary: toolLabel(tu.name, locale), resultPreview: null, error: null, runId: null, startedAt: null, endedAt: null }
}

function errorResultBlock(toolUseId: string, code: string, message: string): BetaToolResultBlockParam {
  return { type: 'tool_result', tool_use_id: toolUseId, content: JSON.stringify({ error: { code, message } }), is_error: true }
}

function formatArgs(args: Array<{ flag: string; value: unknown }>): string {
  return args.map((a) => (a.value === true || a.value === null ? a.flag : `${a.flag} ${String(a.value)}`)).join(' ')
}

/** The preview text on the approval card. */
export async function approvalPreview(name: string, input: unknown, workspaceRoot: string): Promise<string | null> {
  const i = (input ?? {}) as Record<string, unknown>
  switch (name) {
    case 'case_edit': {
      const p = await previewCaseEdit(workspaceRoot, input as CaseEditInput)
      if (isToolResult(p)) return p.error?.message ?? null
      return p.errors.length ? p.errors.join('\n') : p.diff || '(no change)'
    }
    case 'case_create':
      return `${String(i.path)} ← ${String(i.template)}${Array.isArray(i.overrides) && i.overrides.length ? `\n${(i.overrides as Array<{ pointer: string; valueJson: string }>).map((o) => `${o.pointer} = ${o.valueJson}`).join('\n')}` : ''}`
    case 'run_start':
      return [String(i.binary), i.casePath ? String(i.casePath) : '', ...(Array.isArray(i.positionals) ? (i.positionals as string[]) : []), formatArgs((i.args as Array<{ flag: string; value: unknown }>) ?? [])].filter(Boolean).join(' ')
    case 'mesh_generate': {
      const m = meshArgs(input as Parameters<typeof meshArgs>[0])
      return ['ofgpu-generate-mesh', ...m.positionals, formatArgs(m.args)].filter(Boolean).join(' ')
    }
    case 'file_write': {
      const content = typeof i.content === 'string' ? i.content : ''
      const head = content.split('\n').slice(0, 20).join('\n')
      return `${String(i.path)} (${Buffer.byteLength(content, 'utf8')} bytes${i.createOnly ? ', create only' : ''})\n${head}${content.split('\n').length > 20 ? '\n…' : ''}`
    }
    case 'run_stop':
      return `stop ${String(i.runId)}`
    case 'custom_tool_run':
      return `${String(i.name)} ${String(i.inputJson ?? '')}`
    case 'custom_tool_create': {
      const impl = i.impl as { kind?: string; argv?: string[]; source?: string } | undefined
      return `${String(i.name)}: ${impl?.kind ?? '?'} ${impl?.kind === 'command' ? (impl.argv ?? []).join(' ') : (impl?.source ?? '').split('\n').slice(0, 12).join('\n')}`
    }
    case 'shell_exec':
      return Array.isArray(i.argv) ? (i.argv as string[]).join(' ') : null
    default:
      return null
  }
}

// ---------------------------------------------------------------------------

export async function runTurn(rec: SessionRecord, turnId: string, signal: AbortSignal, deps: TurnDeps): Promise<TurnOutcome> {
  const sessionId = rec.id
  const { emit } = deps
  const locale = rec.settings.locale
  const persist = () => deps.store.save(rec)
  const usage = emptyUsage()
  const suggestions: string[] = []
  let model: string | null = null
  let rounds = 0
  let foldContext = false
  let retried = false
  let budgetNoticeSent = false
  let lastAssistantUi: UiMessage | null = null
  const firstMessageId = newId('m')
  emit({ t: 'turn.start', sessionId, turnId, messageId: firstMessageId })

  const emitCall = (call: ToolCallRecord) => emit({ t: 'tool.update', sessionId, call: { ...call } })

  function appendAssistant(content: ReadonlyArray<BetaContentBlockParam | BetaMessage['content'][number]>, stopReason: UiStopReason, messageId: string, calls: Map<string, ToolCallRecord>, extras?: AssistantExtras): UiMessage {
    rec.messages.push({ role: 'assistant', content: content as BetaContentBlockParam[] })
    for (const c of calls.values()) rec.toolCalls.push(c)
    const ui = projectAssistant(messageId, content, calls, { createdAt: Date.now(), stopReason, model, suggestions: [], extras })
    rec.ui.push(ui)
    lastAssistantUi = ui
    return ui
  }

  async function finish(status: TurnStatus): Promise<TurnOutcome> {
    if (lastAssistantUi && suggestions.length) {
      lastAssistantUi.suggestions = [...suggestions]
      emit({ t: 'msg.done', sessionId, message: lastAssistantUi })
    }
    await persist()
    emit({ t: 'turn.done', sessionId, turnId, usage, model })
    return { status, rounds, model }
  }

  /**
   * Keep the complete blocks of a turn that stopped early and give any dangling
   * tool_use an is_error result. `content` is the projector's reconstruction
   * when the stream never produced a final message, and final.content itself
   * when it did - the latter is authoritative and carries blocks the projector
   * has no case for.
   */
  async function repairPartial(content: ReadonlyArray<BetaContentBlockParam | BetaMessage['content'][number]>, messageId: string, stopReason: UiStopReason, code: string, message: string, warning: string | null): Promise<void> {
    const toolUses = content.filter(isToolUse)
    if (stopReason === 'cancelled' && !toolUses.length) return
    if (!content.length) return
    const calls = new Map<string, ToolCallRecord>()
    for (const tu of toolUses) {
      const call = newCall(tu, locale)
      call.status = stopReason === 'cancelled' ? 'cancelled' : 'error'
      call.error = message
      calls.set(tu.id, call)
    }
    const ui = appendAssistant(content, stopReason, messageId, calls)
    if (toolUses.length) rec.messages.push({ role: 'user', content: toolUses.map((tu) => errorResultBlock(tu.id, code, message)) })
    emit({ t: 'msg.done', sessionId, message: ui })
    for (const c of calls.values()) emitCall(c)
    if (warning) emit({ t: 'turn.warning', sessionId, turnId, message: warning })
  }

  for (;;) {
    if (rounds >= MAX_TOOL_ROUNDS && !budgetNoticeSent) {
      budgetNoticeSent = true
      const ui = appendUserTurn(rec, { role: 'user', content: BUDGET_EXHAUSTED_TEXT }, { synthetic: true })
      if (ui) emit({ t: 'msg.user', sessionId, message: ui })
      await persist()
    }
    const messageId = rounds === 0 ? firstMessageId : newId('m')
    const volatile = buildVolatileContext({
      workspaceRoot: deps.config.workspaceRoot,
      mode: deps.config.demo ? 'demo' : 'real',
      gpu: deps.runs.gpu(),
      runs: deps.runs.list(),
      context: deps.userContext(),
      customTools: deps.customTools(),
      locale,
      now: deps.now ? deps.now() : new Date(),
    })
    const messages = foldContext ? foldContextIntoUser(rec.messages, volatile) : [...rec.messages, volatileSystemMessage(volatile)]
    const projector = createStreamProjector(emit, sessionId, messageId)
    let final: BetaMessage
    try {
      const stream = deps.llm.stream({ system: systemParam(), messages, tools: toolDefinitions(), maxTokens: MAX_TOKENS, effort: rec.settings.effort, signal })
      for await (const ev of stream.events) projector.onEvent(ev)
      final = await stream.finalMessage()
    } catch (err) {
      if (signal.aborted || isAbortError(err)) {
        await repairPartial(projector.completeContent(), messageId, 'cancelled', 'CANCELLED', 'cancelled by user', null)
        return finish('cancelled')
      }
      if (isToolJsonParseError(err)) {
        rounds++
        model = projector.partial().model ?? model
        await repairPartial(projector.completeContent(), messageId, 'max_tokens', 'TRUNCATED', 'not executed: the response was truncated', 'The response was cut off before a tool call was complete; that call was not executed.')
        return finish('done')
      }
      if (isSystemRoleRejection(err) && !foldContext) {
        foldContext = true
        continue
      }
      if (isRetryableError(err) && !retried) {
        retried = true
        emit({ t: 'turn.warning', sessionId, turnId, message: `${describeError(err)} — retrying in ${Math.round((deps.retryDelayMs ?? RETRY_DELAY_MS) / 1000)} s` })
        if (await sleep(deps.retryDelayMs ?? RETRY_DELAY_MS, signal)) continue
        return finish('cancelled')
      }
      await persist()
      emit({ t: 'turn.error', sessionId, turnId, message: describeError(err), retryable: isRetryableError(err) })
      return { status: 'error', rounds, model }
    }

    rounds++
    model = final.model
    addUsage(usage, final.usage)

    if (final.stop_reason === 'refusal') {
      // A refusal can arrive mid-stream with a finished tool_use already in the
      // content. Persisting that leaves a tool_use no tool_result will ever
      // answer, and the session is a 400 from then on. Keep the text the user
      // can see; the call itself is not part of a refused turn.
      const kept = final.content.filter((b) => !isToolUse(b))
      if (kept.length) {
        const ui = appendAssistant(kept, 'refusal', messageId, new Map())
        emit({ t: 'msg.done', sessionId, message: ui })
      }
      emit({ t: 'turn.refusal', sessionId, turnId, category: final.stop_details?.category ?? null, explanation: final.stop_details?.explanation ?? null })
      return finish('refusal')
    }

    if (final.stop_reason === 'max_tokens' || final.stop_reason === 'model_context_window_exceeded') {
      await repairPartial(final.content, messageId, 'max_tokens', 'TRUNCATED', 'not executed: the response was truncated (max_tokens)', 'The response hit the output token limit; tool calls in it were not executed.')
      return finish('done')
    }

    const toolUses = final.content.filter(isToolUse)
    const calls = new Map<string, ToolCallRecord>()
    for (const tu of toolUses) calls.set(tu.id, newCall(tu, locale))
    const stopReason: UiStopReason = final.stop_reason === 'tool_use' ? 'tool_use' : 'end_turn'
    const ui = appendAssistant(final.content, stopReason, messageId, calls)
    await persist()
    emit({ t: 'msg.done', sessionId, message: ui })

    if (final.stop_reason !== 'tool_use' || !toolUses.length) return finish('done')

    if (budgetNoticeSent) {
      for (const c of calls.values()) {
        c.status = 'denied'
        c.error = 'tool budget exhausted'
        emitCall(c)
      }
      rec.messages.push({ role: 'user', content: toolUses.map((tu) => errorResultBlock(tu.id, 'BUDGET', 'tool budget exhausted; answer without tools')) })
      emit({ t: 'turn.warning', sessionId, turnId, message: `Tool budget (${MAX_TOOL_ROUNDS} rounds) exhausted.` })
      return finish('done')
    }

    const results = await executeRound(toolUses, calls)
    const blocks: BetaToolResultBlockParam[] = []
    const extras: AssistantExtras = { diffs: [], images: [] }
    for (const tu of toolUses) {
      const result = results.get(tu.id) ?? fail('INTERNAL', 'no result')
      const parts = toolResultBlock(tu.id, result)
      blocks.push(parts.block)
      if (result.diff) extras.diffs!.push({ ...result.diff, toolUseId: tu.id })
      for (const img of result.images ?? []) extras.images!.push({ base64: img.base64, mime: img.mime, alt: `${tu.name} screenshot` })
      if (result.suggestions) suggestions.splice(0, suggestions.length, ...result.suggestions)
      if (result.runId && !rec.runs.includes(result.runId)) rec.runs.push(result.runId)
    }
    rec.messages.push({ role: 'user', content: blocks })
    if (extras.diffs!.length || extras.images!.length) {
      const refreshed = projectAssistant(ui.id, final.content, calls, { createdAt: ui.createdAt, stopReason, model, suggestions: ui.suggestions, extras })
      ui.blocks = refreshed.blocks
    }
    await persist()
    emit({ t: 'msg.done', sessionId, message: ui })
    if (signal.aborted) return finish('cancelled')
  }

  // -------------------------------------------------------------------------

  async function executeRound(toolUses: BetaToolUseBlock[], calls: Map<string, ToolCallRecord>): Promise<Map<string, ToolResult>> {
    const results = new Map<string, ToolResult>()
    const pending: Promise<void>[] = []
    const asks: Array<{ tu: BetaToolUseBlock; call: ToolCallRecord; input: unknown }> = []

    const settle = (tu: BetaToolUseBlock, call: ToolCallRecord, input: unknown, result: ToolResult) => {
      const r = signal.aborted && !result.ok ? fail('CANCELLED', 'cancelled by user') : result
      results.set(tu.id, r)
      const code = r.error?.code
      call.status = r.ok ? 'ok' : code === 'DENIED' ? 'denied' : code === 'CANCELLED' ? 'cancelled' : 'error'
      call.summary = summarizeToolCall(tu.name, input, r.data, r.ok, locale)
      call.resultPreview = toolResultBlock(tu.id, r).preview
      call.error = r.ok ? null : (r.error?.message ?? 'failed')
      call.runId = r.runId ?? null
      call.endedAt = Date.now()
      emitCall(call)
    }

    const execute = async (tu: BetaToolUseBlock, call: ToolCallRecord, input: unknown) => {
      call.status = 'running'
      call.startedAt = Date.now()
      emitCall(call)
      const ctx: ToolContext = { config: deps.config, hub: deps.hub, runs: deps.runs, datasets: deps.datasets, sessionId, signal, workspaceRoot: deps.config.workspaceRoot, settings: rec.settings, toolUseId: tu.id }
      settle(tu, call, input, await runTool(tu.name, input, ctx))
    }

    for (const tu of toolUses) {
      const call = calls.get(tu.id)!
      const tool = getTool(tu.name)
      if (!tool) {
        settle(tu, call, tu.input, fail('UNKNOWN_TOOL', `no tool named ${tu.name}`))
        continue
      }
      const parsed = tool.schema.safeParse(tu.input)
      if (!parsed.success) {
        settle(tu, call, tu.input, fail('INVALID_INPUT', `invalid input for ${tu.name}: ${parsed.error.issues.map((i) => `${i.path.join('.') || '(root)'}: ${i.message}`).join('; ')}`))
        continue
      }
      const policy = classifyTool(tu.name, parsed.data, { settings: rec.settings, allowedTools: rec.allowedTools, overrides: deps.overrides })
      call.policy = policy
      if (policy === 'never') {
        settle(tu, call, parsed.data, fail('DENIED', `${tu.name} is disabled by policy (config/policy.json can set it to "ask"); do not retry it`))
        continue
      }
      if (policy === 'auto') {
        pending.push(execute(tu, call, parsed.data))
        continue
      }
      asks.push({ tu, call, input: parsed.data })
    }

    if (asks.length && signal.aborted) {
      // Nothing to approve on a cancelled turn: asking would put a card on
      // screen after the cancel and hold the session for the approval TTL.
      for (const a of asks) settle(a.tu, a.call, a.input, fail('CANCELLED', 'cancelled by user'))
      asks.length = 0
    }

    if (asks.length) {
      const previews = await Promise.all(asks.map((a) => approvalPreview(a.tu.name, a.input, deps.config.workspaceRoot).catch(() => null)))
      const req = deps.approvals.request(
        turnId,
        asks.map((a, i) => ({ toolUseId: a.tu.id, name: a.tu.name, input: a.input, summary: toolLabel(a.tu.name, locale), preview: previews[i] })),
        undefined,
        signal,
      )
      for (const a of asks) {
        a.call.status = 'awaiting_approval'
        emitCall(a.call)
      }
      emit({ t: 'tool.approval_request', sessionId, approval: req.approval })
      for (const a of asks) {
        pending.push(
          (async () => {
            const outcome = await req.outcomes.get(a.tu.id)!
            if (outcome.decision === 'approved' && !signal.aborted) await execute(a.tu, a.call, a.input)
            else settle(a.tu, a.call, a.input, fail('DENIED', outcome.decision === 'expired' ? 'approval timed out; the user did not answer' : `denied by user${outcome.reason ? `: ${outcome.reason}` : ''}`))
          })(),
        )
      }
    }
    await Promise.all(pending)
    return results
  }
}
