// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The census's allow-list of named gui_control refusals (docs/17 STU-19). A gui_control
// answer is either applied or refused; a refusal counts as answered only when its code is
// one of these names, each a precondition the screen states and the operator can meet
// (open a case first, split the view first, name a tool that exists). "does not implement
// it" is never on the list: that is the default branch of the window's apply, not a refusal.
// No imports, so the Playwright walk (e2e/studio-control.e2e.ts) loads it as is.

/** Every refusal code a gui_control answer may carry and still count as a named refusal. */
export const UI_REFUSAL_ALLOW_LIST: readonly string[] = [
  // the window's own (uiBridge.ts apply): "CODE (type): why"
  'BUSY',
  'CASE_INVALID',
  'DIRTY',
  'INVALID',
  'LOCATE_FAILED',
  'MISS',
  'NO_CASE',
  'NO_DATASET',
  'NO_EDITOR',
  'NO_FIELD',
  'NO_GEOMETRY',
  'NO_IMAGE',
  'NO_LINE',
  'NO_MESH',
  'NO_METRIC',
  'NO_RUN',
  'NO_STEP',
  'NO_SUCH_LAYER',
  'NO_SURFACE_METRICS',
  'NO_TAB',
  'NO_VIEWER',
  'UNSUPPORTED',
  // the server-side delegates' (guiRun / guiCase / guiMesh / guiBoundary / guiCenterline / custom_tool_run)
  'NO_SUCH_RUN',
  'NO_SUCH_TOOL',
  'NO_RUN_BINARY',
  'NO_MESH_DRAFT',
  'NO_RESULT',
  'NOT_A_CASE',
  'RUN_SETTING_EMPTY',
]

/** The text of the window's default branch; an answer that carries it is never a named refusal. */
export const UI_UNIMPLEMENTED_TEXT = 'does not implement it'

/** The name of a refusal. A window refusal reaches the server as UI_ERROR with the message
 *  "CODE (type): why", so its name is that prefix; any other code is its own name. */
export function refusalCodeOf(code: string, message: string): string {
  if (code !== 'UI_ERROR') return code
  const m = /^([A-Z][A-Z0-9_]*) \(/.exec(message)
  return m ? m[1] : code
}

/** True when a refusal is a named one from the allow-list. */
export function isNamedRefusal(code: string, message: string): boolean {
  return !message.includes(UI_UNIMPLEMENTED_TEXT) && UI_REFUSAL_ALLOW_LIST.includes(refusalCodeOf(code, message))
}
