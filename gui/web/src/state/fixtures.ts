// Recorded ServerMsg frames (shape-checked against ServerMsgSchema in the
// tests) covering every variant of the protocol. Shared by the reducer and
// frame-validation tests.
import type { RunInfo, ServerMsg, SessionState, ToolCallRecord, UiMessage } from '@cfd/shared'

export const NOW = 1_760_000_000_000

export const RUN_1: RunInfo = {
  id: 'r_1',
  binary: 'ofgpu-k-epsilon',
  argv: ['ofgpu-k-epsilon', 'cases/plume.jsonc', '-iters', '400', '-check', '25'],
  cwd: '',
  casePath: 'cases/plume.jsonc',
  outputRoot: 'cases/plume_jsonc',
  status: 'running',
  pid: 4242,
  startedAt: '2026-09-07T10:00:00.000Z',
  endedAt: null,
  exitCode: null,
  signal: null,
  iter: 0,
  targetIter: 400,
  time: null,
  endTime: null,
  lastResidual: null,
  written: [],
  error: null,
  converged: false,
  device: 'Demo GPU',
  logLines: 0,
  mode: 'demo',
  label: 'k-epsilon plume',
}

export const SESSION_1: SessionState = {
  id: 's_1',
  title: 'Channel mesh',
  createdAt: '2026-09-07T09:00:00.000Z',
  updatedAt: '2026-09-07T09:30:00.000Z',
  settings: { autoApprove: 'reads', effort: 'high', notifyOnRunEnd: true, locale: 'ko' },
  messages: [],
  pendingApprovals: [],
  runs: [],
  turnActive: false,
  customTools: [],
}

export const USER_MSG: UiMessage = {
  id: 'm_u1',
  role: 'user',
  blocks: [{ kind: 'text', text: 'Generate a mesh for the channel case.' }],
  createdAt: NOW,
  stopReason: null,
  model: null,
  suggestions: [],
  synthetic: false,
}

export const TOOL_CALL_1: ToolCallRecord = {
  toolUseId: 'tu_1',
  name: 'mesh_generate',
  input: { kind: 'channel', outputDir: 'cases/channel', cells: [200, 120, 1] },
  policy: 'ask',
  status: 'ok',
  summary: 'Generated structured mesh (24,000 cells)',
  resultPreview: '{"cells":24000}',
  error: null,
  runId: 'r_0',
  startedAt: NOW,
  endedAt: NOW + 1500,
}

export const DONE_MSG: UiMessage = {
  id: 'm_a1',
  role: 'assistant',
  blocks: [
    { kind: 'thinking', text: 'The user wants a mesh.' },
    { kind: 'text', text: "I'll generate a structured mesh for the channel case." },
    { kind: 'tool', call: TOOL_CALL_1 },
  ],
  createdAt: NOW,
  stopReason: 'tool_use',
  model: 'claude-opus-5',
  suggestions: ['Run the k-epsilon solver', 'Show the mesh in 3D'],
  synthetic: false,
}

export const FRAMES: Record<ServerMsg['t'], ServerMsg> = {
  hello: {
    t: 'hello',
    hello: {
      version: '0.1.0',
      mode: 'demo',
      llm: 'mock',
      model: 'claude-opus-5',
      gpu: { state: 'demo', name: 'Demo GPU', memUsedMB: 4096, memTotalMB: 16303, source: 'demo' },
      workspaceRoot: '/home/user/Iteration-CFD',
      availableBinaries: ['ofgpu-k-epsilon', 'ofgpu-generate-mesh'],
      platform: 'linux',
    },
    sessions: [{ id: 's_1', title: 'Channel mesh', createdAt: SESSION_1.createdAt, updatedAt: SESSION_1.updatedAt, messageCount: 4 }],
    runs: [RUN_1],
  },
  pong: { t: 'pong', ts: NOW },
  error: { t: 'error', message: 'no such run: r_9', fatal: false },
  'session.state': { t: 'session.state', session: SESSION_1 },
  'session.list': { t: 'session.list', sessions: [{ id: 's_2', title: 'Second', createdAt: SESSION_1.createdAt, updatedAt: SESSION_1.updatedAt, messageCount: 0 }] },
  'session.deleted': { t: 'session.deleted', sessionId: 's_1' },
  'turn.start': { t: 'turn.start', sessionId: 's_1', turnId: 't_1', messageId: 'm_a1' },
  'turn.done': { t: 'turn.done', sessionId: 's_1', turnId: 't_1', usage: { inputTokens: 1200, outputTokens: 300, cacheReadTokens: 1000, cacheWriteTokens: 0 }, model: 'claude-opus-5' },
  'turn.error': { t: 'turn.error', sessionId: 's_1', turnId: 't_1', message: 'overloaded_error', retryable: true },
  'turn.refusal': { t: 'turn.refusal', sessionId: 's_1', turnId: 't_1', category: 'harmful', explanation: null },
  'turn.warning': { t: 'turn.warning', sessionId: 's_1', turnId: 't_1', message: 'max_tokens reached' },
  'msg.user': { t: 'msg.user', sessionId: 's_1', message: USER_MSG },
  'msg.block_start': { t: 'msg.block_start', sessionId: 's_1', messageId: 'm_a1', blockIndex: 1, kind: 'text' },
  'msg.delta': { t: 'msg.delta', sessionId: 's_1', messageId: 'm_a1', blockIndex: 1, delta: "I'll generate" },
  'msg.done': { t: 'msg.done', sessionId: 's_1', message: DONE_MSG },
  'tool.start': { t: 'tool.start', sessionId: 's_1', messageId: 'm_a1', blockIndex: 2, toolUseId: 'tu_1', name: 'mesh_generate' },
  'tool.input_delta': { t: 'tool.input_delta', sessionId: 's_1', toolUseId: 'tu_1', partialJson: '{"kind":"chan' },
  'tool.update': { t: 'tool.update', sessionId: 's_1', call: { ...TOOL_CALL_1, status: 'running', summary: '', resultPreview: null, endedAt: null } },
  'tool.approval_request': {
    t: 'tool.approval_request',
    sessionId: 's_1',
    approval: {
      turnId: 't_1',
      toolUseIds: ['tu_1'],
      calls: [{ toolUseId: 'tu_1', name: 'mesh_generate', input: { kind: 'channel' }, summary: 'Generate channel mesh', preview: 'ofgpu-generate-mesh channel cases/channel 200 120 1' }],
      requestedAt: NOW,
      expiresAt: NOW + 600_000,
    },
  },
  'tool.approval_resolved': { t: 'tool.approval_resolved', sessionId: 's_1', toolUseIds: ['tu_1'], decision: 'approved' },
  'run.started': { t: 'run.started', run: RUN_1 },
  'run.updated': { t: 'run.updated', run: { ...RUN_1, iter: 200 } },
  'run.log': {
    t: 'run.log',
    runId: 'r_1',
    lines: [
      { seq: 1, stream: 'system', text: '$ ofgpu-k-epsilon cases/plume.jsonc -iters 400', ts: NOW },
      { seq: 2, stream: 'stdout', text: 'ofgpu k-epsilon | Demo GPU sm_120 | 16303 MiB | precision double', ts: NOW + 1 },
      { seq: 3, stream: 'stdout', text: '     25  epsilon res 3.612e-04 (14)  k res 2.081e-04 (9)  max dk/k 1.734e-02', ts: NOW + 2 },
    ],
  },
  'run.residual': { t: 'run.residual', runId: 'r_1', rec: { seq: 1, iter: 25, time: null, wall: null, fields: { epsilon: 3.612e-4, k: 2.081e-4, dk_k: 1.734e-2 }, solverIters: { epsilon: 14, k: 9 }, raw: '     25  epsilon res ...' } },
  'run.metric': { t: 'run.metric', runId: 'r_1', rec: { seq: 1, iter: 25, time: null, metrics: { Tmin: 293.1, Tmax: 310.2 }, raw: '...' } },
  'run.written': { t: 'run.written', runId: 'r_1', dir: 'cases/plume_jsonc/400' },
  'run.exit': { t: 'run.exit', run: { ...RUN_1, status: 'done', iter: 400, endedAt: '2026-09-07T10:01:00.000Z', exitCode: 0, converged: true, written: ['cases/plume_jsonc/400'] } },
  'viewer.command': { t: 'viewer.command', requestId: 'vq_1', cmd: { type: 'load', path: 'cases/plume_jsonc', timeIndex: 'last', field: 'U' } },
  'viewer.open': { t: 'viewer.open', path: 'cases/plume_jsonc/400', runId: 'r_1' },
  'residuals.open': { t: 'residuals.open', runId: 'r_1' },
  'dataset.progress': { t: 'dataset.progress', progress: { datasetId: 'ds_1', stage: 'fields', pct: 60, message: 'reading U' } },
  'fs.changed': { t: 'fs.changed', paths: ['cases/plume.jsonc'] },
  problems: {
    t: 'problems',
    source: 'solver',
    path: 'cases/plume.jsonc',
    runId: 'r_1',
    items: [{ id: 'r_1:7', severity: 'error', message: 'error: kOmegaSST is not supported by ofgpu; available: kEpsilon', source: 'solver', path: 'cases/plume.jsonc', line: null, col: null, runId: 'r_1', logSeq: 7, hint: 'Use one of the listed values in the case file.' }],
  },
  gpu: { t: 'gpu', gpu: { state: 'busy', name: 'Demo GPU', memUsedMB: 9000, memTotalMB: 16303, source: 'demo' } },
  output: { t: 'output', level: 'info', text: 'dataset cached', ts: NOW },
}

export const ALL_FRAME_TYPES = Object.keys(FRAMES) as Array<ServerMsg['t']>
