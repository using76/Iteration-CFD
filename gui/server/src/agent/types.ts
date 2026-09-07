// Agent service interface: owns sessions, the Claude (or mock) streaming
// loop, tool execution and approvals. Implemented in agent/service.ts.
import type { ClientMsg, SessionState, SessionSummary } from '@cfd/shared'
import type { ClientConn } from '../ws/types.js'

export interface AgentService {
  /** Handle a session/turn/tool/quick/settings client message. Returns false when the message is not the agent's. */
  handleClientMessage(client: ClientConn, msg: ClientMsg): Promise<boolean>
  listSessions(): SessionSummary[]
  getSessionState(sessionId: string): SessionState | null
  createSession(): SessionState
  deleteSession(sessionId: string): boolean
  /** Called by the run manager when a run ends; the service may start a notification turn. */
  notifyRunEnded(runId: string): void
  shutdown(): Promise<void>
}
