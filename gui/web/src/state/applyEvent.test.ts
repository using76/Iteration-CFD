import { describe, expect, it } from 'vitest'
import { ServerMsgSchema, type ServerMsg } from '@cfd/shared'
import { applyEvent, applyEvents } from './applyEvent'
import { ALL_FRAME_TYPES, DONE_MSG, FRAMES, NOW, RUN_1, SESSION_1, USER_MSG } from './fixtures'
import { LOG_RING, RESIDUAL_RING, initialSessionData, type SessionData } from './types'

function opened(): SessionData {
  return applyEvents(initialSessionData(), [FRAMES.hello, FRAMES['session.state']], NOW).state
}

describe('applyEvent fixtures', () => {
  it('covers every ServerMsg variant with a schema-valid frame', () => {
    const variants = ServerMsgSchema.options.map((o) => o.shape.t.value)
    expect([...ALL_FRAME_TYPES].sort()).toEqual([...variants].sort())
    for (const t of ALL_FRAME_TYPES) {
      const r = ServerMsgSchema.safeParse(FRAMES[t])
      expect(r.success, `${t}: ${r.success ? '' : r.error.message}`).toBe(true)
    }
  })

  it('never mutates its input and never throws for any frame', () => {
    const base = opened()
    const snapshot = JSON.stringify(base)
    for (const t of ALL_FRAME_TYPES) {
      expect(() => applyEvent(base, FRAMES[t], NOW)).not.toThrow()
    }
    expect(JSON.stringify(base)).toBe(snapshot)
  })
})

describe('session switching and the turn message', () => {
  it('resets the turn when the server confirms a different session', () => {
    let s = opened()
    s = applyEvent(s, { t: 'turn.error', sessionId: 's_1', turnId: 't_1', message: 'boom', retryable: false }, NOW).state
    expect(s.turn.error?.message).toBe('boom')
    // actions.openSession sets currentSessionId optimistically before the
    // server answers; the reducer must not read that as "same session".
    s = { ...s, currentSessionId: 's_2' }
    const other = { ...SESSION_1, id: 's_2', title: 'Second', messages: [] }
    s = applyEvent(s, { t: 'session.state', session: other }, NOW).state
    expect(s.session?.id).toBe('s_2')
    expect(s.turn.error).toBeNull()
    expect(s.toolInputJson).toEqual({})
  })

  it('keeps the turn when the same session is refreshed', () => {
    let s = opened()
    s = applyEvent(s, { t: 'turn.warning', sessionId: 's_1', turnId: 't_1', message: 'retrying' }, NOW).state
    s = applyEvent(s, FRAMES['session.state'], NOW).state
    expect(s.turn.warnings).toEqual(['retrying'])
  })

  it('follows the turn onto the message of each round', () => {
    let s = opened()
    s = applyEvent(s, { t: 'turn.start', sessionId: 's_1', turnId: 't_1', messageId: 'm_r1' }, NOW).state
    expect(s.turn.messageId).toBe('m_r1')
    // The server allocates a new messageId per round; the first block of the
    // second round is what says so.
    s = applyEvent(s, { t: 'msg.block_start', sessionId: 's_1', messageId: 'm_r2', blockIndex: 0, kind: 'text' }, NOW).state
    expect(s.turn.messageId).toBe('m_r2')
    s = applyEvent(s, { t: 'tool.start', sessionId: 's_1', messageId: 'm_r3', blockIndex: 0, toolUseId: 'toolu_9', name: 'gpu_info' }, NOW).state
    expect(s.turn.messageId).toBe('m_r3')
  })

  it('does not move the turn message when no turn is running', () => {
    let s = opened()
    s = applyEvent(s, { t: 'msg.block_start', sessionId: 's_1', messageId: 'm_z', blockIndex: 0, kind: 'text' }, NOW).state
    expect(s.turn.active).toBe(false)
    expect(s.turn.messageId).toBeNull()
  })
})

describe('hello / sessions / gpu', () => {
  it('hello stores the server facts, runs and emits the hello effect', () => {
    const r = applyEvent(initialSessionData(), FRAMES.hello, NOW)
    expect(r.state.hello?.mode).toBe('demo')
    expect(r.state.gpu?.state).toBe('demo')
    expect(r.state.sessions.map((s) => s.id)).toEqual(['s_1'])
    expect(r.state.runs.r_1?.binary).toBe('ofgpu-k-epsilon')
    expect(r.effects).toEqual([{ type: 'hello' }])
  })

  it('session.state becomes the current session and refreshes its summary', () => {
    const s = opened()
    expect(s.currentSessionId).toBe('s_1')
    expect(s.session?.title).toBe('Channel mesh')
    expect(s.sessions).toHaveLength(1)
    const other: ServerMsg = { t: 'session.state', session: { ...SESSION_1, id: 's_9', title: 'New' } }
    const r = applyEvent(s, other, NOW)
    expect(r.state.currentSessionId).toBe('s_9')
    expect(r.state.sessions.map((x) => x.id)).toEqual(['s_9', 's_1'])
  })

  it('session.deleted of the current session clears it and asks for another', () => {
    const r = applyEvent(opened(), FRAMES['session.deleted'], NOW)
    expect(r.state.session).toBeNull()
    expect(r.state.currentSessionId).toBeNull()
    expect(r.state.sessions).toHaveLength(0)
    expect(r.effects).toEqual([{ type: 'session.deleted', sessionId: 's_1' }])
  })

  it('session.list replaces the list; gpu replaces the gpu state; pong is a no-op', () => {
    const s = opened()
    expect(applyEvent(s, FRAMES['session.list'], NOW).state.sessions[0].id).toBe('s_2')
    expect(applyEvent(s, FRAMES.gpu, NOW).state.gpu?.state).toBe('busy')
    expect(applyEvent(s, FRAMES.pong, NOW).state).toBe(s)
  })

  it('error frames land in the output pane; fatal ones emit an effect', () => {
    const r = applyEvent(opened(), FRAMES.error, NOW)
    expect(r.state.outputs.at(-1)?.text).toBe('no such run: r_9')
    expect(r.effects).toEqual([])
    const fatal = applyEvent(opened(), { t: 'error', message: 'boom', fatal: true }, NOW)
    expect(fatal.effects).toEqual([{ type: 'fatal', message: 'boom' }])
  })
})

describe('streaming a turn', () => {
  it('turn.start creates the assistant placeholder and marks the turn active', () => {
    const r = applyEvent(opened(), FRAMES['turn.start'], NOW)
    expect(r.state.turn).toMatchObject({ active: true, turnId: 't_1', messageId: 'm_a1' })
    expect(r.state.session?.turnActive).toBe(true)
    expect(r.state.session?.messages.at(-1)).toMatchObject({ id: 'm_a1', role: 'assistant', blocks: [] })
  })

  it('block_start + delta append text by blockIndex, padding holes', () => {
    const s = applyEvents(opened(), [FRAMES['turn.start'], FRAMES['msg.block_start'], FRAMES['msg.delta'], { ...FRAMES['msg.delta'], delta: ' a mesh' } as ServerMsg], NOW).state
    const m = s.session!.messages.at(-1)!
    expect(m.blocks).toHaveLength(2)
    expect(m.blocks[0]).toEqual({ kind: 'text', text: '' })
    expect(m.blocks[1]).toEqual({ kind: 'text', text: "I'll generate a mesh" })
  })

  it('thinking blocks stream separately from text', () => {
    const s = applyEvents(
      opened(),
      [FRAMES['turn.start'], { t: 'msg.block_start', sessionId: 's_1', messageId: 'm_a1', blockIndex: 0, kind: 'thinking' }, { t: 'msg.delta', sessionId: 's_1', messageId: 'm_a1', blockIndex: 0, delta: 'hmm' }],
      NOW,
    ).state
    expect(s.session!.messages.at(-1)!.blocks[0]).toEqual({ kind: 'thinking', text: 'hmm' })
  })

  it('tool.start inserts a pending tool block and tool.input_delta accumulates JSON', () => {
    const s = applyEvents(opened(), [FRAMES['turn.start'], FRAMES['tool.start'], FRAMES['tool.input_delta'], { ...FRAMES['tool.input_delta'], partialJson: 'nel"}' } as ServerMsg], NOW).state
    const block = s.session!.messages.at(-1)!.blocks[2]
    expect(block.kind).toBe('tool')
    if (block.kind === 'tool') {
      expect(block.call).toMatchObject({ toolUseId: 'tu_1', name: 'mesh_generate', status: 'pending', policy: 'ask' })
    }
    expect(s.toolInputJson.tu_1).toBe('{"kind":"channel"}')
  })

  it('tool.update replaces the call in place wherever it lives', () => {
    const s = applyEvents(opened(), [FRAMES['turn.start'], FRAMES['tool.start'], FRAMES['tool.update']], NOW).state
    const block = s.session!.messages.at(-1)!.blocks[2]
    expect(block.kind === 'tool' && block.call.status).toBe('running')
    expect(s.session!.messages.at(-1)!.blocks).toHaveLength(3)
  })

  it('tool.update for an unknown call appends it to the streaming message', () => {
    const s = applyEvents(opened(), [FRAMES['turn.start'], FRAMES['tool.update']], NOW).state
    expect(s.session!.messages.at(-1)!.blocks).toHaveLength(1)
  })

  it('msg.done replaces the streamed message and clears its tool input buffers', () => {
    const s = applyEvents(opened(), [FRAMES['turn.start'], FRAMES['msg.block_start'], FRAMES['msg.delta'], FRAMES['tool.start'], FRAMES['tool.input_delta'], FRAMES['msg.done']], NOW).state
    expect(s.session!.messages).toHaveLength(1)
    expect(s.session!.messages[0]).toEqual(DONE_MSG)
    expect(s.toolInputJson).toEqual({})
  })

  it('msg.user appends once (idempotent by id)', () => {
    const s = applyEvents(opened(), [FRAMES['msg.user'], FRAMES['msg.user']], NOW).state
    expect(s.session!.messages).toEqual([USER_MSG])
  })

  it('turn.done / error / refusal / warning end the turn with the right marker', () => {
    const started = applyEvents(opened(), [FRAMES['turn.start'], FRAMES['msg.done']], NOW).state
    const done = applyEvent(started, FRAMES['turn.done'], NOW).state
    expect(done.turn.active).toBe(false)
    expect(done.session?.turnActive).toBe(false)
    expect(done.session?.messages[0].model).toBe('claude-opus-5')
    const err = applyEvent(started, FRAMES['turn.error'], NOW).state
    expect(err.turn).toMatchObject({ active: false, error: { message: 'overloaded_error', retryable: true } })
    const refusal = applyEvent(started, FRAMES['turn.refusal'], NOW).state
    expect(refusal.turn.refusal).toEqual({ category: 'harmful', explanation: null })
    const warn = applyEvent(started, FRAMES['turn.warning'], NOW).state
    expect(warn.turn.warnings).toEqual(['max_tokens reached'])
  })

  it('ignores frames for other sessions', () => {
    const s = opened()
    const foreign: ServerMsg = { ...FRAMES['turn.start'], sessionId: 's_other' } as ServerMsg
    expect(applyEvent(s, foreign, NOW).state).toBe(s)
  })
})

describe('approvals', () => {
  it('approval_request stores the approval and flags the tool blocks', () => {
    const s = applyEvents(opened(), [FRAMES['turn.start'], FRAMES['tool.start'], FRAMES['tool.approval_request']], NOW).state
    expect(s.session!.pendingApprovals).toHaveLength(1)
    const block = s.session!.messages.at(-1)!.blocks[2]
    expect(block.kind === 'tool' && block.call.status).toBe('awaiting_approval')
    const again = applyEvent(s, FRAMES['tool.approval_request'], NOW).state
    expect(again.session!.pendingApprovals).toHaveLength(1)
  })

  it('approval_resolved removes resolved ids and updates the block status', () => {
    const s = applyEvents(opened(), [FRAMES['turn.start'], FRAMES['tool.start'], FRAMES['tool.approval_request']], NOW).state
    const ok = applyEvent(s, FRAMES['tool.approval_resolved'], NOW).state
    expect(ok.session!.pendingApprovals).toHaveLength(0)
    expect(ok.session!.messages.at(-1)!.blocks[2]).toMatchObject({ call: { status: 'running' } })
    const denied = applyEvent(s, { ...FRAMES['tool.approval_resolved'], decision: 'denied' } as ServerMsg, NOW).state
    expect(denied.session!.messages.at(-1)!.blocks[2]).toMatchObject({ call: { status: 'denied' } })
    const expired = applyEvent(s, { ...FRAMES['tool.approval_resolved'], decision: 'expired' } as ServerMsg, NOW).state
    expect(expired.session!.messages.at(-1)!.blocks[2]).toMatchObject({ call: { status: 'denied', error: 'approval expired' } })
  })

  it('partially resolved approvals keep their remaining calls', () => {
    const multi: ServerMsg = { t: 'tool.approval_request', sessionId: 's_1', approval: { ...FRAMES['tool.approval_request'].t === 'tool.approval_request' ? FRAMES['tool.approval_request'].approval : (undefined as never), toolUseIds: ['tu_1', 'tu_2'], calls: [{ toolUseId: 'tu_1', name: 'a', input: null, summary: 'a', preview: null }, { toolUseId: 'tu_2', name: 'b', input: null, summary: 'b', preview: null }] } }
    const s = applyEvents(opened(), [multi, FRAMES['tool.approval_resolved']], NOW).state
    expect(s.session!.pendingApprovals[0].toolUseIds).toEqual(['tu_2'])
    expect(s.session!.pendingApprovals[0].calls.map((c) => c.toolUseId)).toEqual(['tu_2'])
  })
})

describe('runs', () => {
  it('run.started registers the run and asks for a subscription', () => {
    const r = applyEvent(initialSessionData(), FRAMES['run.started'], NOW)
    expect(r.state.runs.r_1).toEqual(RUN_1)
    expect(r.state.runData.r_1).toBeDefined()
    expect(r.effects).toEqual([{ type: 'run.subscribe', runId: 'r_1' }])
  })

  it('run.updated / run.exit replace the run; exit writes an output note', () => {
    const s = applyEvents(initialSessionData(), [FRAMES['run.started'], FRAMES['run.updated']], NOW).state
    expect(s.runs.r_1.iter).toBe(200)
    const done = applyEvent(s, FRAMES['run.exit'], NOW).state
    expect(done.runs.r_1.status).toBe('done')
    expect(done.outputs.at(-1)?.text).toContain('done')
  })

  it('run.log dedupes by seq and keeps a bounded ring', () => {
    let s = applyEvents(initialSessionData(), [FRAMES['run.started'], FRAMES['run.log'], FRAMES['run.log']], NOW).state
    expect(s.runData.r_1.logs).toHaveLength(3)
    expect(s.runData.r_1.lastLogSeq).toBe(3)
    const bulk: ServerMsg = { t: 'run.log', runId: 'r_1', lines: Array.from({ length: LOG_RING + 10 }, (_, i) => ({ seq: 4 + i, stream: 'stdout' as const, text: `line ${i}`, ts: NOW })) }
    s = applyEvent(s, bulk, NOW).state
    expect(s.runData.r_1.logs).toHaveLength(LOG_RING)
    expect(s.runData.r_1.logs[0].seq).toBe(14)
    expect(s.runData.r_1.lastLogSeq).toBe(3 + LOG_RING + 10)
  })

  it('run.residual / run.metric dedupe by seq and bound the ring', () => {
    let s = applyEvents(initialSessionData(), [FRAMES['run.residual'], FRAMES['run.residual'], FRAMES['run.metric'], FRAMES['run.metric']], NOW).state
    expect(s.runData.r_1.residuals).toHaveLength(1)
    expect(s.runData.r_1.metrics).toHaveLength(1)
    const rec = FRAMES['run.residual'].t === 'run.residual' ? FRAMES['run.residual'].rec : (undefined as never)
    for (let i = 2; i <= 6; i++) s = applyEvent(s, { t: 'run.residual', runId: 'r_1', rec: { ...rec, seq: i, iter: i * 25 } }, NOW).state
    expect(s.runData.r_1.residuals.map((r) => r.seq)).toEqual([1, 2, 3, 4, 5, 6])
    // a replay burst (one flush) is coalesced into a single ring append
    const burst: ServerMsg[] = Array.from({ length: RESIDUAL_RING + 10 }, (_, k) => ({ t: 'run.residual', runId: 'r_1', rec: { ...rec, seq: k + 1, iter: (k + 1) * 25 } }))
    const t0 = performance.now()
    s = applyEvents(s, burst, NOW).state
    expect(performance.now() - t0).toBeLessThan(1500)
    expect(s.runData.r_1.residuals).toHaveLength(RESIDUAL_RING)
    expect(s.runData.r_1.residuals.at(-1)?.seq).toBe(RESIDUAL_RING + 10)
    expect(s.runData.r_1.residuals[0].seq).toBe(11)
    expect(s.runData.r_1.lastResidualSeq).toBe(RESIDUAL_RING + 10)
    // metrics coalesce the same way and stay deduped
    const mrec = FRAMES['run.metric'].t === 'run.metric' ? FRAMES['run.metric'].rec : (undefined as never)
    s = applyEvents(s, [1, 1, 2, 3].map((seq) => ({ t: 'run.metric', runId: 'r_1', rec: { ...mrec, seq } }) as ServerMsg), NOW).state
    expect(s.runData.r_1.metrics.map((m) => m.seq)).toEqual([1, 2, 3])
  })

  it('run.written appends unique directories to the run', () => {
    const s = applyEvents(initialSessionData(), [FRAMES['run.started'], FRAMES['run.written'], FRAMES['run.written']], NOW).state
    expect(s.runs.r_1.written).toEqual(['cases/plume_jsonc/400'])
  })
})

describe('delegated and misc frames', () => {
  it('viewer.command / viewer.open / residuals.open / fs.changed are pure effects', () => {
    const s = opened()
    expect(applyEvent(s, FRAMES['viewer.command'], NOW)).toEqual({ state: s, effects: [{ type: 'viewer.command', requestId: 'vq_1', cmd: { type: 'load', path: 'cases/plume_jsonc', timeIndex: 'last', field: 'U' } }] })
    expect(applyEvent(s, FRAMES['viewer.open'], NOW).effects).toEqual([{ type: 'viewer.open', path: 'cases/plume_jsonc/400', runId: 'r_1' }])
    expect(applyEvent(s, FRAMES['residuals.open'], NOW).effects).toEqual([{ type: 'residuals.open', runId: 'r_1' }])
    expect(applyEvent(s, FRAMES['fs.changed'], NOW).effects).toEqual([{ type: 'fs.changed', paths: ['cases/plume.jsonc'] }])
  })

  it('problems replace their key and empty lists clear it', () => {
    const s = applyEvent(opened(), FRAMES.problems, NOW).state
    expect(Object.keys(s.problems)).toEqual(['solver:r_1'])
    const cleared = applyEvent(s, { ...FRAMES.problems, items: [] } as ServerMsg, NOW).state
    expect(cleared.problems).toEqual({})
  })

  it('dataset.progress and output are recorded', () => {
    const s = applyEvents(opened(), [FRAMES['dataset.progress'], FRAMES.output], NOW).state
    expect(s.datasetProgress.ds_1.pct).toBe(60)
    expect(s.outputs.at(-1)).toMatchObject({ level: 'info', text: 'dataset cached', origin: 'server' })
  })
})
