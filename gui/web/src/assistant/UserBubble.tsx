import { memo } from 'react'
import type { UiMessage } from '@cfd/shared'

function textOf(m: UiMessage): string {
  return m.blocks
    .map((b) => (b.kind === 'text' ? b.text : b.kind === 'notice' ? b.text : ''))
    .filter(Boolean)
    .join('\n')
}

export const UserBubble = memo(function UserBubble({ message }: { message: UiMessage }) {
  const text = textOf(message)
  if (message.synthetic) return <div className="system-line" data-testid="system-line">{text}</div>
  return (
    <div className="user-bubble" data-testid="user-message">
      {text}
    </div>
  )
})
