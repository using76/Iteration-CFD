// From SDK content to what the browser renders: UiMessage blocks for
// persisted messages, and the streaming events (msg.block_start/delta,
// tool.start/input_delta) while a response is in flight. The projector also
// keeps a partial copy of the message so a cancelled or truncated stream can
// be repaired without the SDK's finalMessage().
import type { BetaContentBlock, BetaContentBlockParam, BetaMessageParam, BetaRawMessageStreamEvent, BetaRefusalStopDetails, BetaStopReason } from '@anthropic-ai/sdk/resources/beta/messages/messages'
import type { ServerMsg, ToolCallRecord, UiBlock, UiMessage } from '@cfd/shared'

export type UiStopReason = UiMessage['stopReason']

export interface AssistantExtras {
  diffs?: Array<{ path: string; before: string; after: string; applied: boolean; toolUseId: string }>
  images?: Array<{ base64: string; mime: string; alt: string }>
}

export function projectAssistant(
  id: string,
  content: ReadonlyArray<BetaContentBlock | BetaContentBlockParam>,
  calls: ReadonlyMap<string, ToolCallRecord>,
  meta: { createdAt: number; stopReason: UiStopReason; model: string | null; suggestions?: string[]; extras?: AssistantExtras },
): UiMessage {
  const blocks: UiBlock[] = []
  for (const block of content) {
    switch (block.type) {
      case 'text':
        blocks.push({ kind: 'text', text: block.text })
        break
      case 'thinking':
        blocks.push({ kind: 'thinking', text: block.thinking })
        break
      case 'redacted_thinking':
        blocks.push({ kind: 'thinking', text: '' })
        break
      case 'tool_use': {
        const call = calls.get(block.id) ?? { toolUseId: block.id, name: block.name, input: block.input, policy: 'auto', status: 'pending', summary: block.name, resultPreview: null, error: null, runId: null, startedAt: null, endedAt: null }
        blocks.push({ kind: 'tool', call })
        break
      }
      default:
        blocks.push({ kind: 'notice', level: 'info', text: `[${block.type}]` })
    }
  }
  for (const d of meta.extras?.diffs ?? []) blocks.push({ kind: 'diff', path: d.path, before: d.before, after: d.after, applied: d.applied, toolUseId: d.toolUseId })
  for (const img of meta.extras?.images ?? []) blocks.push({ kind: 'image', mime: img.mime, base64: img.base64, alt: img.alt })
  return { id, role: 'assistant', blocks, createdAt: meta.createdAt, stopReason: meta.stopReason, model: meta.model, suggestions: meta.suggestions ?? [], synthetic: false }
}

/** A user message; null when it only carries tool results (those are shown on the tool cards). */
export function projectUser(id: string, message: BetaMessageParam, meta: { createdAt: number; synthetic: boolean; notices?: string[] }): UiMessage | null {
  const blocks: UiBlock[] = []
  if (typeof message.content === 'string') blocks.push({ kind: 'text', text: message.content })
  else {
    for (const block of message.content) {
      if (block.type === 'text') blocks.push({ kind: 'text', text: block.text })
      else if (block.type === 'image' && block.source.type === 'base64') blocks.push({ kind: 'image', mime: block.source.media_type, base64: block.source.data, alt: 'attachment' })
    }
  }
  for (const n of meta.notices ?? []) blocks.push({ kind: 'notice', level: 'info', text: n })
  if (!blocks.length) return null
  return { id, role: 'user', blocks, createdAt: meta.createdAt, stopReason: null, model: null, suggestions: [], synthetic: meta.synthetic }
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

export type PartialBlock =
  | { type: 'text'; text: string; stopped: boolean }
  | { type: 'thinking'; thinking: string; signature: string; stopped: boolean }
  | { type: 'redacted_thinking'; data: string; stopped: boolean }
  | { type: 'tool_use'; id: string; name: string; json: string; stopped: boolean }

export interface PartialMessage {
  model: string | null
  blocks: PartialBlock[]
  stopReason: BetaStopReason | null
  stopDetails: BetaRefusalStopDetails | null
}

export interface StreamProjector {
  onEvent(ev: BetaRawMessageStreamEvent): void
  partial(): PartialMessage
  /** Complete blocks of the partial message as SDK content params (tool_use inputs parsed; unparsable ones dropped). */
  completeContent(): BetaContentBlockParam[]
}

export function createStreamProjector(emit: (msg: ServerMsg) => void, sessionId: string, messageId: string): StreamProjector {
  const state: PartialMessage = { model: null, blocks: [], stopReason: null, stopDetails: null }
  return {
    onEvent(ev) {
      switch (ev.type) {
        case 'message_start':
          state.model = ev.message.model
          break
        case 'content_block_start': {
          const cb = ev.content_block
          const index = ev.index
          if (cb.type === 'text') {
            state.blocks[index] = { type: 'text', text: cb.text ?? '', stopped: false }
            emit({ t: 'msg.block_start', sessionId, messageId, blockIndex: index, kind: 'text' })
            if (cb.text) emit({ t: 'msg.delta', sessionId, messageId, blockIndex: index, delta: cb.text })
          } else if (cb.type === 'thinking') {
            state.blocks[index] = { type: 'thinking', thinking: cb.thinking ?? '', signature: cb.signature ?? '', stopped: false }
            emit({ t: 'msg.block_start', sessionId, messageId, blockIndex: index, kind: 'thinking' })
          } else if (cb.type === 'redacted_thinking') {
            state.blocks[index] = { type: 'redacted_thinking', data: cb.data, stopped: false }
            emit({ t: 'msg.block_start', sessionId, messageId, blockIndex: index, kind: 'thinking' })
          } else if (cb.type === 'tool_use') {
            state.blocks[index] = { type: 'tool_use', id: cb.id, name: cb.name, json: '', stopped: false }
            emit({ t: 'tool.start', sessionId, messageId, blockIndex: index, toolUseId: cb.id, name: cb.name })
          }
          break
        }
        case 'content_block_delta': {
          const block = state.blocks[ev.index]
          const d = ev.delta
          if (d.type === 'text_delta' && block?.type === 'text') {
            block.text += d.text
            emit({ t: 'msg.delta', sessionId, messageId, blockIndex: ev.index, delta: d.text })
          } else if (d.type === 'thinking_delta' && block?.type === 'thinking') {
            block.thinking += d.thinking
            emit({ t: 'msg.delta', sessionId, messageId, blockIndex: ev.index, delta: d.thinking })
          } else if (d.type === 'signature_delta' && block?.type === 'thinking') {
            block.signature += d.signature
          } else if (d.type === 'input_json_delta' && block?.type === 'tool_use') {
            block.json += d.partial_json
            emit({ t: 'tool.input_delta', sessionId, toolUseId: block.id, partialJson: d.partial_json })
          }
          break
        }
        case 'content_block_stop': {
          const block = state.blocks[ev.index]
          if (block) block.stopped = true
          break
        }
        case 'message_delta':
          state.stopReason = ev.delta.stop_reason
          state.stopDetails = ev.delta.stop_details ?? null
          break
        case 'message_stop':
          break
      }
    },
    partial: () => state,
    completeContent() {
      const out: BetaContentBlockParam[] = []
      for (const b of state.blocks) {
        if (!b || !b.stopped) continue
        if (b.type === 'text') out.push({ type: 'text', text: b.text })
        else if (b.type === 'thinking') out.push({ type: 'thinking', thinking: b.thinking, signature: b.signature })
        else if (b.type === 'redacted_thinking') out.push({ type: 'redacted_thinking', data: b.data })
        else {
          try {
            out.push({ type: 'tool_use', id: b.id, name: b.name, input: b.json.trim() ? JSON.parse(b.json) : {} })
          } catch {
            // an unfinished tool call is dropped
          }
        }
      }
      return out
    },
  }
}
