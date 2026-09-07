// The WebSocket hub interface the agent loop, tools and run manager talk to.
// Implemented in ws/hub.ts.
import type { ClientMsg, ServerMsg, ViewerCommand, ViewerResult, ViewerState } from '@cfd/shared'

export interface ClientConn {
  id: string
  /** Session this client has open, if any. */
  sessionId: string | null
  /** Run ids the client subscribed to. */
  runs: Set<string>
  /** Latest viewer state the client reported (null if it has no viewer mounted). */
  viewerState: ViewerState | null
  send(msg: ServerMsg): void
  close(code?: number, reason?: string): void
}

export type ClientMsgHandler = (client: ClientConn, msg: ClientMsg) => void | Promise<void>

export interface Hub {
  broadcast(msg: ServerMsg): void
  /** Send to every client that has this session open. */
  sendToSession(sessionId: string, msg: ServerMsg): void
  /** Send to every client subscribed to this run (plus session clients if given). */
  sendToRun(runId: string, msg: ServerMsg): void
  clients(): ClientConn[]
  /** True when at least one client reported a mounted viewer. */
  hasViewerClient(): boolean
  /**
   * Forward a viewer command to the most recently active viewer client and
   * await its `viewer.result`. Rejects with a ViewerResult-shaped error
   * ({ok:false, error:{code:'NO_VIEWER'|'TIMEOUT'}}) rather than throwing.
   */
  requestViewer(cmd: ViewerCommand, opts?: { timeoutMs?: number; sessionId?: string | null }): Promise<ViewerResult>
  onClientMessage(handler: ClientMsgHandler): () => void
  onClientOpen(handler: (client: ClientConn) => void): () => void
  onClientClose(handler: (client: ClientConn) => void): () => void
}
