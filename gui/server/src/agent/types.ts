// Agent service interface: owns sessions, the Claude (or mock) streaming
// loop, tool execution and approvals. Implemented in agent/service.ts.
import type { ChatRequest, ChatResponse, ClientMsg, SessionState, SessionSummary } from '@cfd/shared'
import type { ClientConn } from '../ws/types.js'

/** A refusal a route can map: errorToResponse() turns any thrown object with a numeric
 *  `status` into that HTTP status, so this needs nothing from http/. */
export class ChatError extends Error {
  constructor(readonly status: number, message: string) {
    super(message)
    this.name = 'ChatError'
  }
}

export interface AgentService {
  /** Handle a session/turn/tool/quick/settings client message. Returns false when the message is not the agent's. */
  handleClientMessage(client: ClientConn, msg: ClientMsg): Promise<boolean>
  listSessions(): SessionSummary[]
  getSessionState(sessionId: string): SessionState | null
  createSession(): SessionState
  /** Start or continue a session over REST: append the user turn, run it, and answer with
   *  the UI messages that turn appended. Rejects with ChatError(404|409|400|503). */
  chat(req: ChatRequest): Promise<ChatResponse>
  deleteSession(sessionId: string): boolean
  /** Called by the run manager when a run ends; the service may start a notification turn. */
  notifyRunEnded(runId: string): void
  shutdown(): Promise<void>
}
