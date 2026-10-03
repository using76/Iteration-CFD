// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// A1-A9, R1-R2 and F1-F4 of GUI-5: cad_template_propose refuses before any card, file or python by
// the five CAD-AUTHOR ids, every ADM rule comes back by its own id with the retry class the server
// counts by its own table (3 execution + 5 geometry caps), the candidate file is written once wx and
// never overwritten, a crashing admit.py is TOOL_FAILED and uncounted, and only a person freezes an
// admitted candidate - exactly one ALWAYS_ASK card, the lock byte-identical on a denial, one new
// entry and one freeze row on an approval, FREEZE-IMMUTABLE by its own id.
import crypto from 'node:crypto'
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'
import type { BetaMessage, BetaRawMessageStreamEvent } from '@anthropic-ai/sdk/resources/beta/messages/messages'
import { runTurn } from '../agent/loop.js'
import { makeMessage, mockEvents, type MockPlan } from '../agent/mockLlm.js'
import { ALWAYS_ASK, classifyTool } from '../agent/policy.js'
import { appendUserTurn } from '../agent/session.js'
import { fakeDatasets, fakeHub, fakeRuns, makeWorkspace, REPO_ROOT, type TempWorkspace } from '../agent/test-fakes.js'
import { makeDeps, toolResultsOf, until } from '../agent/test-util.js'
import { ADM_RETRY_CLASS, ADM_RULES, SOURCE_MAX, admissionRecord, attemptsLog, candidateFile, writeCandidate } from './cadTemplate.js'
import type { ToolContext } from './context.js'
import { runTool, toolDefinitions } from './index.js'
import { pythonCommand } from './pytool.js'
import { spawnCapture } from './shell.js'

vi.setConfig({ testTimeout: 300_000, hookTimeout: 300_000 })

const BRIEF = 'a straight circular pipe, axisymmetric on +x, metres, for the authoring graft'
/** The candidate source a stub check reads its verdict from: STUB_RULE = "<x>" then a distinct k. */
const S = (x: string, k: number): string => `STUB_RULE = "${x}"${'\n'}# ${k}${'\n'}`
const proposeInput = (candidate_id: string, source: string, brief = BRIEF): Record<string, unknown> => ({ candidate_id, brief, source })
const sha256Hex = (data: string | Buffer): string => crypto.createHash('sha256').update(data).digest('hex')

/**
 * The stub admit.py the stub workspace runs instead of the real judge: every invocation notes its
 * verb in stub_calls.txt (the nothing-ran oracle), `check` refuses by the STUB_RULE line or admits,
 * `freeze` always refuses FREEZE-IMMUTABLE. Test fixture code - python, imports os/sys/json/hashlib only.
 */
const STUB_ADMIT = `import os
import sys
import json
import hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(os.path.dirname(HERE))

RULES = ("ADM-AST-LOCK", "ADM-AST-IMPORT", "ADM-AST-NAME", "ADM-AST-FALLBACK", "ADM-CONTRACT",
         "ADM-BUILD", "ADM-DETERM", "ADM-STAGE", "ADM-TAGS", "ADM-INSENSITIVE")
CLASS = {"ADM-AST-LOCK": "execution", "ADM-AST-IMPORT": "execution", "ADM-AST-NAME": "execution",
         "ADM-AST-FALLBACK": "execution", "ADM-CONTRACT": "execution", "ADM-BUILD": "execution",
         "ADM-DETERM": "execution", "ADM-STAGE": "geometry", "ADM-TAGS": "geometry",
         "ADM-INSENSITIVE": "geometry"}
KEYS = ("schema", "source", "source_sha256", "template_id", "status", "rule", "detail",
        "retry_class", "point", "traceback", "hint", "drivers_mode", "counts", "sensitivity",
        "env", "rules")


def note(verb):
    with open(os.path.join(WS, "stub_calls.txt"), "a", encoding="utf-8") as f:
        f.write(verb + chr(10))


def stub_record(src_path, rule):
    with open(src_path, "rb") as f:
        b = f.read()
    rec = {}
    for k in KEYS:
        rec[k] = None
    rec["schema"] = "cad-admission/1"
    rec["source"] = os.path.relpath(os.path.abspath(src_path), WS).replace(os.sep, "/")
    rec["source_sha256"] = hashlib.sha256(b).hexdigest()
    rec["counts"] = {"sweep": 0, "sobol": 0, "corners": 0, "accepted": 0, "pipelines": 0}
    rec["sensitivity"] = {}
    rec["env"] = {}
    rec["rules"] = []
    if rule == "ADMIT":
        rec["template_id"] = "stub/1"
        rec["status"] = "admitted"
        rec["drivers_mode"] = "declared"
        rec["counts"] = {"sweep": 1, "sobol": 0, "corners": 0, "accepted": 1, "pipelines": 0}
    else:
        rec["status"] = "refused"
        rec["rule"] = rule
        rec["retry_class"] = CLASS[rule]
        rec["detail"] = "stub detail " + rule
        rec["point"] = {"D": 0.05}
        rec["traceback"] = "Traceback (most recent call last):" + chr(10) + "ValueError: stub " + rule
        rec["hint"] = "stub hint " + rule
    return rec


def main():
    verb = sys.argv[1]
    note(verb)
    if verb == "check":
        src, out = sys.argv[2], sys.argv[3]
        with open(src, "r", encoding="utf-8") as f:
            text = f.read()
        x = None
        for line in text.splitlines():
            if line.startswith("STUB_RULE = "):
                x = line[len("STUB_RULE = "):].strip().strip('"')
                break
        if x == "CRASH":
            sys.stderr.write("stub crash" + chr(10))
            return 3
        rec = stub_record(src, x)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(rec, f)
        print(json.dumps({"status": rec["status"], "rule": rec["rule"], "detail": rec["detail"]}))
        return 0 if rec["status"] == "admitted" else 1
    if verb == "freeze":
        sys.stderr.write("FREEZE-IMMUTABLE: the stub lock refuses stub/1" + chr(10))
        return 1
    sys.stderr.write("usage" + chr(10))
    return 2


if __name__ == "__main__":
    sys.exit(main())
`

/** sha-hex of every file of one flat directory - the nothing-written oracle. */
function dirState(dir: string): Record<string, string> {
  const out: Record<string, string> = {}
  try {
    for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
      if (e.isFile()) out[e.name] = fs.readFileSync(path.join(dir, e.name)).toString('hex')
    }
  } catch {
    // the directory is not there yet
  }
  return out
}

/** The planLlm helper of cadEdit.test.ts, copied: a scripted LLM the agent loop talks to. */
function planLlm(kind: 'zai' | 'anthropic', model: string, plans: MockPlan[]): import('../agent/llm.js').LlmClient {
  let n = 0
  return {
    kind,
    model,
    stream(params) {
      const p = plans[Math.min(n++, plans.length - 1)]
      let final: BetaMessage | null = null
      const gen = mockEvents(p, model, { signal: params.signal, delayMs: 0, inputTokens: 10 }, (m) => (final = m))
      return {
        events: gen as AsyncIterable<BetaRawMessageStreamEvent>,
        finalMessage: async () => final ?? makeMessage(model, [], 'end_turn', null, 10),
      }
    },
  }
}
const say = (text: string): MockPlan => ({ blocks: [{ type: 'text', text }], stopReason: 'end_turn' })
const freezeCall = (): MockPlan => ({
  blocks: [
    { type: 'text', text: 'The candidate is admitted; I ask the person to freeze it.' },
    { type: 'tool_use', name: 'cad_template_freeze', input: { candidate_id: 'pipe_straight' } },
  ],
  stopReason: 'tool_use',
})

describe('cad_template_propose / cad_template_freeze on the stub admit.py (docs/16 §G, §I GUI-5)', () => {
  let ws: TempWorkspace
  let calls = 0
  beforeAll(async () => {
    ws = await makeWorkspace()
    await fsp.mkdir(path.join(ws.root, 'tools', 'cad', 'templates', 'nozzle_contraction'), { recursive: true })
    await fsp.writeFile(path.join(ws.root, 'tools', 'cad', 'admit.py'), STUB_ADMIT)
  })
  afterAll(async () => ws.cleanup())

  const candidatesAbs = () => path.join(ws.root, 'tools', 'cad', 'templates', 'candidates')
  const authoringAbs = () => path.join(ws.root, 'cad', 'authoring')
  const stubCalls = (): string[] => {
    try {
      return fs.readFileSync(path.join(ws.root, 'stub_calls.txt'), 'utf8').split(/\r?\n/).filter((l) => l.trim().length > 0)
    } catch {
      return []
    }
  }
  const logRows = (id: string): any[] => {
    try {
      return fs
        .readFileSync(path.join(ws.root, attemptsLog(id)), 'utf8')
        .split(/\r?\n/)
        .filter((l) => l.trim().length > 0)
        .map((l) => JSON.parse(l))
    } catch {
      return []
    }
  }
  function ctx(): ToolContext {
    calls += 1
    return {
      config: ws.config,
      hub: fakeHub(),
      runs: fakeRuns(),
      datasets: fakeDatasets(),
      sessionId: 's1',
      signal: new AbortController().signal,
      workspaceRoot: ws.root,
      settings: { autoApprove: 'all', effort: 'high', notifyOnRunEnd: true, locale: 'en' },
      toolUseId: `toolu_g5s${calls}`,
      llm: { provider: 'mock', model: 'test' },
    }
  }

  /** One refused proposal: the rule id, no candidates/authoring byte, and admit.py not invoked. */
  async function refusedProposal(input: Record<string, unknown>, id: string, message?: (m: string) => void): Promise<any> {
    const candBefore = dirState(candidatesAbs())
    const authBefore = dirState(authoringAbs())
    const callsBefore = stubCalls()
    const r = await runTool('cad_template_propose', input, ctx())
    expect(r.ok, JSON.stringify(r.data)).toBe(false)
    expect(r.error?.code, JSON.stringify(r.data)).toBe(id)
    expect(dirState(candidatesAbs())).toEqual(candBefore)
    expect(dirState(authoringAbs())).toEqual(authBefore)
    expect(stubCalls()).toEqual(callsBefore)
    if (message) message(r.error!.message)
    return r
  }

  it('A1: freeze is ALWAYS_ASK that nothing relaxes, propose is a plain ask; both schemas plain objects', () => {
    expect(ALWAYS_ASK.has('cad_template_freeze')).toBe(true)
    expect(ALWAYS_ASK.has('cad_template_propose')).toBe(false)
    const settings = { autoApprove: 'all' as const, effort: 'high' as const, notifyOnRunEnd: true, locale: 'en' as const }
    expect(classifyTool('cad_template_freeze', {}, { settings, allowedTools: ['cad_template_freeze'], overrides: {} })).toBe('ask')
    expect(classifyTool('cad_template_freeze', {}, { settings, allowedTools: [], overrides: { cad_template_freeze: 'auto' } })).toBe('ask')
    expect(classifyTool('cad_template_freeze', {}, { settings, allowedTools: [], overrides: { cad_template_freeze: 'never' } })).toBe('never')
    expect(classifyTool('cad_template_propose', {}, { settings, allowedTools: [], overrides: {} })).toBe('auto')
    for (const name of ['cad_template_propose', 'cad_template_freeze']) {
      const s = toolDefinitions().find((t) => t.name === name)!.input_schema as Record<string, unknown>
      expect(s.type, name).toBe('object')
      expect(s.oneOf, name).toBeUndefined()
      expect(s.anyOf, name).toBeUndefined()
    }
  })

  it('A2 CAD-AUTHOR-ID: bad slugs, the reserved candidates, and a hand-reviewed template dir refuse before anything', async () => {
    await refusedProposal(proposeInput('Bad-Id', S('ADMIT', 0)), 'CAD-AUTHOR-ID')
    await refusedProposal(proposeInput('../x', S('ADMIT', 0)), 'CAD-AUTHOR-ID')
    await refusedProposal(proposeInput('ab', S('ADMIT', 0)), 'CAD-AUTHOR-ID')
    await refusedProposal(proposeInput('candidates', S('ADMIT', 0)), 'CAD-AUTHOR-ID')
    await refusedProposal(proposeInput('nozzle_contraction', S('ADMIT', 0)), 'CAD-AUTHOR-ID', (m) => expect(m).toContain('nozzle_contraction'))
  })

  it('A3 CAD-AUTHOR-TYPE: empty or oversized brief and source, and a non-string source, refuse', async () => {
    await refusedProposal({ candidate_id: 'typ1', brief: '', source: S('ADMIT', 0) }, 'CAD-AUTHOR-TYPE')
    await refusedProposal({ candidate_id: 'typ1', brief: '   ', source: S('ADMIT', 0) }, 'CAD-AUTHOR-TYPE')
    await refusedProposal({ candidate_id: 'typ1', brief: 'b', source: '' }, 'CAD-AUTHOR-TYPE')
    await refusedProposal({ candidate_id: 'typ1', brief: 'b', source: 'x'.repeat(SOURCE_MAX + 1) }, 'CAD-AUTHOR-TYPE')
    await refusedProposal({ candidate_id: 'typ1', brief: 'b', source: 42 }, 'CAD-AUTHOR-TYPE')
  })

  it.each(ADM_RULES.map((rule, index) => ({ rule, index })))(
    'A4: $rule refuses by its own id with its retry class, record detail, point, hint and counts',
    async ({ rule, index }) => {
      const cid = `rule_${index}`
      const r = await runTool('cad_template_propose', proposeInput(cid, S(rule, 0)), ctx())
      expect(r.ok, JSON.stringify(r.data)).toBe(false)
      expect(r.error?.code, JSON.stringify(r.data)).toBe(rule)
      const d = r.data as any
      const cls = (ADM_RETRY_CLASS as Record<string, string>)[rule]
      expect(d.retry_class).toBe(cls)
      expect(d.detail).toBe(`stub detail ${rule}`)
      expect(d.hint).toBe(`stub hint ${rule}`)
      expect(d.point).toEqual({ D: 0.05 })
      expect(String(d.traceback_tail)).toMatch(new RegExp(`ValueError: stub ${rule}$`))
      expect(d.refused_execution).toBe(cls === 'execution' ? 1 : 0)
      expect(d.refused_geometry).toBe(cls === 'geometry' ? 1 : 0)
      expect(d.may_retry).toBe(true)
      expect(d.n).toBe(1)
      expect(fs.readFileSync(path.join(ws.root, candidateFile(cid, 1)))).toEqual(Buffer.from(S(rule, 0), 'utf8'))
      expect(fs.existsSync(path.join(ws.root, admissionRecord(cid, 1)))).toBe(true)
      const rows = logRows(cid).filter((row) => row.kind === 'attempt')
      expect(rows).toHaveLength(1)
      expect(rows[0].status).toBe('refused')
      expect(rows[0].rule).toBe(rule)
    },
  )

  it('A6: the caps - 4th execution and 6th geometry refusal are the last; 3 + 5 still admit a 9th', async () => {
    for (let k = 0; k < 4; k++) {
      const r = await runTool('cad_template_propose', proposeInput('capx', S('ADM-BUILD', k)), ctx())
      expect(r.error?.code, `capx ${k}`).toBe('ADM-BUILD')
      expect((r.data as any).may_retry, `capx ${k}`).toBe(k < 3)
    }
    const callsBefore = stubCalls().length
    const cap = await runTool('cad_template_propose', proposeInput('capx', S('ADM-BUILD', 4)), ctx())
    expect(cap.error?.code).toBe('CAD-AUTHOR-CAP')
    expect(stubCalls()).toHaveLength(callsBefore)
    for (let k = 0; k < 6; k++) {
      const r = await runTool('cad_template_propose', proposeInput('capg', S('ADM-TAGS', k)), ctx())
      expect(r.error?.code, `capg ${k}`).toBe('ADM-TAGS')
    }
    const capg = await runTool('cad_template_propose', proposeInput('capg', S('ADM-TAGS', 6)), ctx())
    expect(capg.error?.code).toBe('CAD-AUTHOR-CAP')
    for (let k = 0; k < 3; k++) {
      const r = await runTool('cad_template_propose', proposeInput('capmix', S('ADM-CONTRACT', k)), ctx())
      expect(r.error?.code, `capmix contract ${k}`).toBe('ADM-CONTRACT')
    }
    for (let k = 0; k < 5; k++) {
      const r = await runTool('cad_template_propose', proposeInput('capmix', S('ADM-STAGE', k)), ctx())
      expect(r.error?.code, `capmix stage ${k}`).toBe('ADM-STAGE')
    }
    const ninth = await runTool('cad_template_propose', proposeInput('capmix', S('ADMIT', 9)), ctx())
    expect(ninth.ok, JSON.stringify(ninth.data)).toBe(true)
    const d = ninth.data as any
    expect(d.status).toBe('admitted')
    expect(d.n).toBe(9)
    expect(d.template_id).toBe('stub/1')
    expect(d.refused_execution).toBe(3)
    expect(d.refused_geometry).toBe(5)
    const tenth = await refusedProposal(proposeInput('capmix', S('ADMIT', 10)), 'CAD-AUTHOR-DONE')
    expect(tenth.error!.message).toContain('cad_template_freeze')
  })

  it('A7 CAD-AUTHOR-SAME: the very same bytes refuse after the first judgement', async () => {
    const first = await runTool('cad_template_propose', proposeInput('same1', S('ADM-TAGS', 0)), ctx())
    expect(first.error?.code).toBe('ADM-TAGS')
    await refusedProposal(proposeInput('same1', S('ADM-TAGS', 0)), 'CAD-AUTHOR-SAME')
  })

  it('A8: writeCandidate never overwrites - planted c_x.1.py and c_x.2.py give n 3, then n 4', async () => {
    const dir = await fsp.mkdtemp(path.join(os.tmpdir(), 'cand-'))
    try {
      const p1 = path.join(dir, 'c_x.1.py')
      const p2 = path.join(dir, 'c_x.2.py')
      await fsp.writeFile(p1, 'one')
      await fsp.writeFile(p2, 'two')
      const first = await writeCandidate(dir, 'c_x', 'three')
      expect(first.n).toBe(3)
      expect(first.file).toBe(path.join(dir, 'c_x.3.py'))
      expect(fs.readFileSync(p1, 'utf8')).toBe('one')
      expect(fs.readFileSync(p2, 'utf8')).toBe('two')
      const second = await writeCandidate(dir, 'c_x', 'four')
      expect(second.n).toBe(4)
      expect(fs.readFileSync(path.join(dir, 'c_x.4.py'), 'utf8')).toBe('four')
    } finally {
      await fsp.rm(dir, { recursive: true, force: true })
    }
  })

  it('A9: a crashing admit.py is TOOL_FAILED, logged failed, counted in neither class', async () => {
    const crash = await runTool('cad_template_propose', proposeInput('crash1', S('CRASH', 0)), ctx())
    expect(crash.ok).toBe(false)
    expect(crash.error?.code).toBe('TOOL_FAILED')
    const rows = logRows('crash1').filter((row) => row.kind === 'attempt')
    expect(rows).toHaveLength(1)
    expect(rows[0].status).toBe('failed')
    expect(rows[0].refused_execution).toBe(0)
    expect(rows[0].refused_geometry).toBe(0)
    // The brief says S('ADM-NAME', 1), but "ADM-NAME" is no ADM rule id (admit.py's is ADM-AST-NAME);
    // the crash row counts in neither class, so the next attempt runs and refuses.
    const next = await runTool('cad_template_propose', proposeInput('crash1', S('ADM-AST-NAME', 1)), ctx())
    expect(next.error?.code).toBe('ADM-AST-NAME')
    expect((next.data as any).n).toBe(2)
    const rows2 = logRows('crash1').filter((row) => row.kind === 'attempt')
    expect(rows2).toHaveLength(2)
  })

  it('F4: an admit.py freeze refusal comes back by its own id (FREEZE-IMMUTABLE)', async () => {
    // The brief names this candidate "fz", but CANDIDATE_ID_RE needs 3..48 chars (A2 refuses "ab");
    // the slug here is the same intent at the shortest legal spelling.
    const admitted = await runTool('cad_template_propose', proposeInput('fz1', S('ADMIT', 0)), ctx())
    expect(admitted.ok, JSON.stringify(admitted.data)).toBe(true)
    const callsBefore = stubCalls()
    const frozen = await runTool('cad_template_freeze', { candidate_id: 'fz1' }, ctx())
    expect(frozen.ok).toBe(false)
    expect(frozen.error?.code).toBe('FREEZE-IMMUTABLE')
    expect(stubCalls().length).toBe(callsBefore.length + 1)
    expect(stubCalls()[stubCalls().length - 1]).toBe('freeze')
  })
})

describe('cad_template_propose / cad_template_freeze on the real admit.py', () => {
  let ws: TempWorkspace
  let calls = 0
  beforeAll(async () => {
    ws = await makeWorkspace()
    const cad = path.join(ws.root, 'tools', 'cad')
    await fsp.cp(path.join(REPO_ROOT, 'tools', 'cad'), cad, {
      recursive: true,
      filter: (src) => !src.split(path.sep).includes('__pycache__') && path.basename(src) !== 'studies.jsonl',
    })
    const geom = path.join(ws.root, 'tools', 'geom')
    await fsp.mkdir(geom, { recursive: true })
    await fsp.copyFile(path.join(REPO_ROOT, 'tools', 'geom', 'stl_repair.py'), path.join(geom, 'stl_repair.py'))
  })
  afterAll(async () => ws.cleanup())

  const candidatesAbs = () => path.join(ws.root, 'tools', 'cad', 'templates', 'candidates')
  const authoringAbs = () => path.join(ws.root, 'cad', 'authoring')
  const lockAbs = () => path.join(ws.root, 'tools', 'cad', 'templates.lock')
  const logRows = (id: string): any[] => {
    try {
      return fs
        .readFileSync(path.join(ws.root, attemptsLog(id)), 'utf8')
        .split(/\r?\n/)
        .filter((l) => l.trim().length > 0)
        .map((l) => JSON.parse(l))
    } catch {
      return []
    }
  }
  const fixtureSource = (name: string): string => fs.readFileSync(path.join(REPO_ROOT, 'tools', 'cad', 'fixtures', 'admit', name), 'utf8')
  function ctx(): ToolContext {
    calls += 1
    return {
      config: ws.config,
      hub: fakeHub(),
      runs: fakeRuns(),
      datasets: fakeDatasets(),
      sessionId: 's1',
      signal: new AbortController().signal,
      workspaceRoot: ws.root,
      settings: { autoApprove: 'all', effort: 'high', notifyOnRunEnd: true, locale: 'en' },
      toolUseId: `toolu_g5r${calls}`,
      llm: { provider: 'mock', model: 'test' },
    }
  }

  it('A5: the server tables equal admit.py RETRY_CLASS and RULES, verbatim', async () => {
    const r = await spawnCapture(
      [pythonCommand(ws.config), '-c', 'import json, admit; print(json.dumps(admit.RETRY_CLASS)); print(json.dumps(admit.RULES))'],
      { cwd: path.join(ws.root, 'tools', 'cad'), timeoutMs: 120_000, env: { PYTHONIOENCODING: 'utf-8' } },
    )
    expect(r.exitCode).toBe(0)
    const [cls, rules] = r.stdout.trim().split(/\r?\n/).map((l) => JSON.parse(l))
    expect(cls).toEqual(ADM_RETRY_CLASS)
    expect(rules).toEqual(ADM_RULES)
  })

  it('R1: bad_import_json is refused ADM-AST-IMPORT, execution class, with its line detail', async () => {
    const candBefore = dirState(candidatesAbs())
    const authBefore = dirState(authoringAbs())
    const r = await runTool('cad_template_propose', proposeInput('real_bad', fixtureSource('bad_import_json.py')), ctx())
    expect(r.ok, JSON.stringify(r.data)).toBe(false)
    expect(r.error?.code).toBe('ADM-AST-IMPORT')
    const d = r.data as any
    expect(d.retry_class).toBe('execution')
    expect(d.detail).toBe('line 22: import json is not on the whitelist')
    // The real judge really did run: exactly one candidate file and one admission record landed.
    expect(Object.keys(dirState(candidatesAbs())).length - Object.keys(candBefore).length).toBe(1)
    expect(Object.keys(dirState(authoringAbs())).length - Object.keys(authBefore).length).toBe(2)
  })

  it('R2: good_pipe is admitted with its measured counts in about a minute', async () => {
    const r = await runTool('cad_template_propose', proposeInput('pipe_straight', fixtureSource('good_pipe.py')), ctx())
    expect(r.ok, JSON.stringify(r.data)).toBe(true)
    const d = r.data as any
    expect(d.kind).toBe('cadTemplateAdmission')
    expect(d.candidate_id).toBe('pipe_straight')
    expect(d.n).toBe(1)
    expect(d.status).toBe('admitted')
    expect(d.template_id).toBe('pipe_straight/1')
    expect(d.drivers_mode).toBe('declared')
    expect(d.counts).toEqual({ sweep: 73, sobol: 64, corners: 8, accepted: 73, pipelines: 8 })
    expect(d.refused_execution).toBe(0)
    expect(d.refused_geometry).toBe(0)
    expect(d.next).toContain('cad_template_freeze')
    expect(fs.readFileSync(path.join(ws.root, candidateFile('pipe_straight', 1)), 'utf8')).toBe(fixtureSource('good_pipe.py'))
    const rows = logRows('pipe_straight')
    expect(rows).toHaveLength(1)
    expect(rows[0].status).toBe('admitted')
    expect(rows[0].template_id).toBe('pipe_straight/1')
  })

  it('F1: freeze refuses before the card with the lock byte-identical', async () => {
    const before = fs.readFileSync(lockAbs())
    const r1 = await runTool('cad_template_freeze', { candidate_id: 'nolog_x' }, ctx())
    expect(r1.ok).toBe(false)
    expect(r1.error?.code).toBe('CAD-AUTHOR-ID')
    const r2 = await runTool('cad_template_freeze', { candidate_id: 'real_bad' }, ctx())
    expect(r2.ok).toBe(false)
    expect(r2.error?.code).toBe('FREEZE-UNADMITTED')
    const recAbs = path.join(ws.root, 'cad', 'authoring', 'pipe_straight.1.admission.json')
    const recOriginal = fs.readFileSync(recAbs)
    try {
      fs.writeFileSync(recAbs, Buffer.concat([recOriginal, Buffer.from(' ')]))
      const r3 = await runTool('cad_template_freeze', { candidate_id: 'pipe_straight' }, ctx())
      expect(r3.ok).toBe(false)
      expect(r3.error?.code).toBe('FREEZE-UNADMITTED')
    } finally {
      fs.writeFileSync(recAbs, recOriginal)
    }
    const lockOriginal = fs.readFileSync(lockAbs())
    try {
      const lock = JSON.parse(lockOriginal.toString('utf8'))
      lock.templates.push({
        template_id: 'pipe_straight/1',
        source: 'x',
        source_sha256: '0'.repeat(64),
        declaration_sha256: null,
        cadquery: 'x',
        occt: 'x',
        frozen_by: 'x',
        admission_sha256: 'x',
      })
      fs.writeFileSync(lockAbs(), JSON.stringify(lock))
      const r4 = await runTool('cad_template_freeze', { candidate_id: 'pipe_straight' }, ctx())
      expect(r4.ok).toBe(false)
      expect(r4.error?.code).toBe('FREEZE-IMMUTABLE')
    } finally {
      fs.writeFileSync(lockAbs(), lockOriginal)
    }
    expect(fs.readFileSync(lockAbs()).equals(before)).toBe(true)
    expect(logRows('pipe_straight').some((row) => row.kind === 'freeze')).toBe(false)
  })

  it('F2: through runTurn a freeze draws exactly one card; denying leaves the lock byte-identical', async () => {
    const before = fs.readFileSync(lockAbs())
    const deps = makeDeps(ws, { llm: planLlm('zai', 'claude-opus-5', [freezeCall(), say('The freeze was not approved, so nothing changed.')]) })
    const rec = deps.store.create({ locale: 'en', autoApprove: 'all' })
    appendUserTurn(rec, { role: 'user', content: 'Freeze the admitted pipe_straight candidate.' }, { synthetic: false, entitle: true })
    const turn = runTurn(rec, 'f2', new AbortController().signal, deps)
    await until(() => deps.hub.of('tool.approval_request').length === 1, 60_000)
    const req = (deps.hub.of('tool.approval_request')[0] as { approval: { toolUseIds: string[]; calls: Array<{ name: string; preview: string }> } }).approval
    expect(req.calls).toHaveLength(1)
    expect(req.calls[0].name).toBe('cad_template_freeze')
    expect(req.calls[0].preview).toContain('pipe_straight/1')
    expect(req.calls[0].preview).toContain('frozen_by:')
    deps.approvals.resolve(req.toolUseIds, 'denied')
    await turn
    const results = rec.messages.flatMap((m) => toolResultsOf(m))
    expect(results).toHaveLength(1)
    const denied = JSON.parse(results[0].content) as { error: { code: string } }
    expect(denied.error.code).toBe('DENIED')
    expect(results[0].is_error).toBe(true)
    expect(fs.readFileSync(lockAbs()).equals(before)).toBe(true)
    expect(logRows('pipe_straight').some((row) => row.kind === 'freeze')).toBe(false)
  })

  it('F3: approving writes exactly one lock entry and the freeze row', async () => {
    const beforeEntries = (JSON.parse(fs.readFileSync(lockAbs(), 'utf8')) as { templates: any[] }).templates
    expect(beforeEntries).toHaveLength(1)
    const deps = makeDeps(ws, { llm: planLlm('zai', 'claude-opus-5', [freezeCall(), say('The template is frozen.')]) })
    const rec = deps.store.create({ locale: 'en', autoApprove: 'all' })
    appendUserTurn(rec, { role: 'user', content: 'Freeze the admitted pipe_straight candidate.' }, { synthetic: false, entitle: true })
    const turn = runTurn(rec, 'f3', new AbortController().signal, deps)
    await until(() => deps.hub.of('tool.approval_request').length === 1, 60_000)
    const req = (deps.hub.of('tool.approval_request')[0] as { approval: { toolUseIds: string[] } }).approval
    deps.approvals.resolve(req.toolUseIds, 'approved')
    await turn
    const results = rec.messages.flatMap((m) => toolResultsOf(m))
    expect(results).toHaveLength(1)
    expect(results[0].is_error).toBe(false)
    const d = JSON.parse(results[0].content) as any
    expect(d.kind).toBe('cadTemplateFreeze')
    expect(d.candidate_id).toBe('pipe_straight')
    expect(d.n).toBe(1)
    expect(d.template_id).toBe('pipe_straight/1')
    expect(d.entry.template_id).toBe('pipe_straight/1')
    expect(d.entry.source).toBe('tools/cad/templates/candidates/pipe_straight.1.py')
    expect(d.entry.source_sha256).toBe(sha256Hex(fixtureSource('good_pipe.py')))
    expect(d.entry.declaration_sha256).toBeNull()
    expect(String(d.entry.frozen_by).length).toBeGreaterThan(0)
    const after = JSON.parse(fs.readFileSync(lockAbs(), 'utf8')) as { templates: any[] }
    expect(after.templates).toHaveLength(2)
    expect(after.templates.find((e) => e.template_id === 'nozzle_contraction/1')).toEqual(beforeEntries[0])
    expect(after.templates.find((e) => e.template_id === 'pipe_straight/1')).toBeTruthy()
    const rows = logRows('pipe_straight')
    expect(rows[rows.length - 1].kind).toBe('freeze')
    expect(rows[rows.length - 1].template_id).toBe('pipe_straight/1')
    expect(rows[rows.length - 1].frozen_by).toBe(d.entry.frozen_by)
  })
})
