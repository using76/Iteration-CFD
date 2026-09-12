// The WebSocket hub interface the agent loop, tools and run manager talk to.
// Implemented in ws/hub.ts.
import type { ClientMsg, ServerMsg, UiCommand, UiState, ViewerCommand, ViewerResult, ViewerState } from '@cfd/shared'

export interface ClientConn {
  id: string
  /** Session this client has open, if any. */
  sessionId: string | null
  /** Run ids the client subscribed to. */
  runs: Set<string>
  /** Latest viewer state the client reported (null if it has no viewer mounted). */
  viewerState: ViewerState | null
  /** Latest screen projection the client reported via ui.state, if any. */
  uiState?: UiState | null
  send(msg: ServerMsg): void
  close(code?: number, reason?: string): void
}

export type ClientMsgHandler = (client: ClientConn, msg: ClientMsg) => void | Promise<void>

/** What requestUi resolves with: the error shape mirrors ViewerResult's. */
export interface UiRequestResult {
  ok: boolean
  state: UiState | null
  error: { code: string; message: string } | null
}

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
   * Forward a viewer command to the session's most recently active viewer
   * client (any viewer client when no session is named) and await its
   * `viewer.result`. Rejects with a ViewerResult-shaped error
   * ({ok:false, error:{code:'NO_VIEWER'|'TIMEOUT'}}) rather than throwing;
   * a session whose windows are all gone gets NO_VIEWER, never a stranger's.
   */
  requestViewer(cmd: ViewerCommand, opts?: { timeoutMs?: number; sessionId?: string | null }): Promise<ViewerResult>
  /**
   * Latest UiState reported by the client that has the session open (the most
   * recently active one when several do); null when none has reported one.
   */
  getUiState(sessionId?: string | null): UiState | null
  /**
   * Forward a UI command to the client that has the session open (the most
   * recently active one when several do; any client when no session is named)
   * and await its `ui.result` for that requestId. Never throws: a timeout, a
   * vanished client or no client of the session resolves with
   * { ok:false, error:{ code:'TIMEOUT'|'NO_UI' } } - another session's window
   * is never adopted.
   */
  requestUi(cmd: UiCommand, opts?: { timeoutMs?: number; sessionId?: string | null }): Promise<UiRequestResult>
  onClientMessage(handler: ClientMsgHandler): () => void
  onClientOpen(handler: (client: ClientConn) => void): () => void
  onClientClose(handler: (client: ClientConn) => void): () => void
}
