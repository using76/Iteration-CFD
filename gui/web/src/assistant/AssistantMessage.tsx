// Renders one assistant message: blocks in order, with consecutive tool
// calls collapsed into a bordered group and run cards after run-starting
// calls.
import { memo, useMemo } from 'react'
import type { ToolCallRecord, UiMessage } from '@cfd/shared'
import { useT } from '../app/hooks'
import { Icon } from '../components/common/Icon'
import { ImageBlock, NoticeCard, SuggestionChips, ThinkingBlock } from './Cards'
import { DiffCard } from './DiffCard'
import { Markdown } from './Markdown'
import { RunProgressCard } from './RunProgressCard'
import { ToolGroup } from './ToolStepCard'

type Segment = { kind: 'tools'; calls: ToolCallRecord[]; key: string } | { kind: 'run'; runId: string; key: string } | { kind: 'block'; index: number; key: string }

const RUN_TOOLS = new Set(['run_start', 'mesh_generate'])

function segments(m: UiMessage): Segment[] {
  const out: Segment[] = []
  let tools: ToolCallRecord[] = []
  const flush = () => {
    if (tools.length) out.push({ kind: 'tools', calls: tools, key: `tools:${tools[0].toolUseId}` })
    tools = []
  }
  m.blocks.forEach((b, i) => {
    if (b.kind === 'tool') {
      tools.push(b.call)
      if (b.call.runId && RUN_TOOLS.has(b.call.name)) {
        flush()
        out.push({ kind: 'run', runId: b.call.runId, key: `run:${b.call.runId}` })
      }
      return
    }
    if (b.kind === 'text' && !b.text) return
    flush()
    out.push({ kind: 'block', index: i, key: `b:${i}` })
  })
  flush()
  return out
}

export const AssistantMessage = memo(function AssistantMessage({ message, streaming, showSuggestions }: { message: UiMessage; streaming: boolean; showSuggestions: boolean }) {
  const t = useT()
  const segs = useMemo(() => segments(message), [message])
  const lastIndex = message.blocks.length - 1
  if (!segs.length && !streaming && !message.stopReason && !message.suggestions.length) return null
  return (
    <div className="assistant-msg" data-testid="assistant-message" data-message-id={message.id}>
      <span className="assistant-avatar">
        <Icon name="logo" size={16} />
      </span>
      <div className="assistant-content">
        {segs.map((seg) => {
          if (seg.kind === 'tools') return <ToolGroup key={seg.key} calls={seg.calls} />
          if (seg.kind === 'run') return <RunProgressCard key={seg.key} runId={seg.runId} />
          const b = message.blocks[seg.index]
          const isLast = seg.index === lastIndex
          switch (b.kind) {
            case 'text':
              return (
                <div key={seg.key} className="assistant-text" data-testid="assistant-text">
                  <Markdown text={b.text} />
                </div>
              )
            case 'thinking':
              return <ThinkingBlock key={seg.key} text={b.text} streaming={streaming && isLast} />
            case 'diff':
              return <DiffCard key={seg.key} block={b} id={`diff:${b.toolUseId ?? `${message.id}:${seg.index}`}`} />
            case 'image':
              return <ImageBlock key={seg.key} mime={b.mime} base64={b.base64} alt={b.alt} />
            case 'notice':
              return <NoticeCard key={seg.key} level={b.level} text={b.text} />
            default:
              return null
          }
        })}
        {streaming && !segs.length ? (
          <div className="turn-status" style={{ paddingLeft: 0 }}>
            <span className="spinner" /> {t('assistant.thinking')}
          </div>
        ) : null}
        {message.stopReason === 'max_tokens' ? <NoticeCard level="warning" text={t('assistant.maxTokens')} /> : null}
        {message.stopReason === 'cancelled' ? <NoticeCard level="info" text={t('assistant.cancelled')} /> : null}
        {showSuggestions ? <SuggestionChips items={message.suggestions} /> : null}
      </div>
    </div>
  )
})
