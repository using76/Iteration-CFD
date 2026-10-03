// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// P1-P2 and C1-C3 of GUI-3: file_write, case_edit, shell_exec, custom_tool_create and
// custom_tool_run refuse every protected requirements/gate/split/template path and everything
// the CAD loop owns (WL-PROTECTED) and any CadQuery/OCP/gmsh code (CUSTOM-CAD), one test per
// path or pattern, while ordinary paths still pass every refuser and really write.
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { fakeDatasets, fakeHub, fakeRuns, makeWorkspace, type TempWorkspace } from '../agent/test-fakes.js'
import type { ToolContext } from './context.js'
import { runTool } from './index.js'
import { refuseCaseEdit, refuseCustomCreate, refuseCustomRun, refuseFileWrite, refuseShellExec, protectedMention, protectedPath } from './writeGuards.js'

let ws: TempWorkspace
beforeAll(async () => {
  ws = await makeWorkspace()
})
afterAll(async () => {
  await ws.cleanup()
})

function ctx(): ToolContext {
  return {
    config: ws.config,
    hub: fakeHub(),
    runs: fakeRuns(),
    datasets: fakeDatasets(),
    sessionId: 's1',
    signal: new AbortController().signal,
    workspaceRoot: ws.root,
    settings: { autoApprove: 'all', effort: 'high', notifyOnRunEnd: true, locale: 'en' },
    toolUseId: 'toolu_g3p',
  }
}

const ruleIdOf = (r: { ok: boolean; data: unknown }): string => (r.data as { ruleId?: string }).ruleId ?? ''

/** sha-like state of one target file (null when absent) - the nothing-changed oracle. */
const fileState = (p: string): string | null => {
  const abs = path.isAbsolute(p) ? p : path.join(ws.root, p.replace(/\\/g, '/'))
  try {
    return fs.readFileSync(abs).toString('hex')
  } catch {
    return null
  }
}

const createInput = (p: string, argv?: string[]): Record<string, unknown> => ({
  name: 'x_probe_tool',
  description: 'probe tool',
  inputSchemaJson: '{"type":"object"}',
  impl: { kind: 'command', argv: argv ?? ['python', '-c', `open('${p}','w')`], cwd: null },
})

// The last case holds the <WS_ROOT> placeholder, substituted with the workspace root at run time.
const PROTECTED_CASES = [
  'cad/v1/requirements/requirements.json',
  'cad/v1/requirements/requirements.lock',
  'tools/cad/gates.json',
  'tools/cad/gates.lock',
  'campaigns/det_a/split.lock',
  'tools/cad/templates.lock',
  'tools/cad/templates/nozzle_contraction/template.py',
  'tools/cad/templates/candidate_x/template.json',
  'cad/proposals/toolu_1/proposal.json',
  'cad/v1/study/params/stable.json',
  'cad/studies.jsonl',
  'cad/v1/edits/v1.cad9.json',
  'cad/v1/cad_edits.jsonl',
  'tools\\cad\\gates.lock',
  'TOOLS/CAD/GATES.LOCK',
  './tools/x/../cad/split.lock',
  '<WS_ROOT>/tools/autonomy/gates.lock',
]

describe('protected write paths (docs/16 §E.8, §I GUI-3)', () => {
  it.each(PROTECTED_CASES)('P1: %s is refused WL-PROTECTED by all five write tools', async (raw) => {
    const p = raw.replace('<WS_ROOT>', ws.root)
    const before = fileState(p)
    const inputs: Array<[string, Record<string, unknown>]> = [
      ['file_write', { path: p, content: 'x', createOnly: false }],
      ['case_edit', { path: p, edits: [{ pointer: '/a', valueJson: '1' }], dryRun: false }],
      ['shell_exec', { argv: ['python', '-c', `open('${p}','w').write('x')`], cwd: null, timeoutSec: null }],
      ['custom_tool_create', createInput(p)],
      ['custom_tool_run', { name: 'anything', inputJson: JSON.stringify({ path: p }) }],
    ]
    for (const [tool, input] of inputs) {
      const r = await runTool(tool, input, ctx())
      expect(r.ok, `${tool} on ${p}: ${JSON.stringify(r.data)}`).toBe(false)
      expect(ruleIdOf(r), `${tool} on ${p}`).toBe('WL-PROTECTED')
      expect(r.error?.message.startsWith('WL-PROTECTED: '), `${tool} on ${p}`).toBe(true)
    }
    expect(fileState(p)).toBe(before)
  })

  it('P2: ordinary paths pass every refuser (null) and file_write really writes them', async () => {
    for (const p of ['notes/a.md', 'cad/v1/start.json', 'tools/cad/notes.md', 'requirements.txt', 'configs/box.json']) {
      expect(protectedPath(p), p).toBeNull()
      expect(protectedMention(`open('${p}','w')`), p).toBeNull()
      expect(refuseFileWrite({ path: p, content: 'x', createOnly: false }), p).toBeNull()
      expect(refuseCaseEdit({ path: p, edits: [{ pointer: '/a', valueJson: '1' }], dryRun: false }), p).toBeNull()
      expect(refuseShellExec({ argv: ['python', '-c', `open('${p}','w')`], cwd: null, timeoutSec: null }), p).toBeNull()
      expect(refuseCustomCreate(createInput(p)), p).toBeNull()
      expect(refuseCustomRun({ name: 'anything', inputJson: JSON.stringify({ path: p }) }), p).toBeNull()
      const r = await runTool('file_write', { path: p, content: 'x', createOnly: false }, ctx())
      expect(r.ok, p).toBe(true)
      expect(fs.readFileSync(path.join(ws.root, p.replace(/\\/g, '/')), 'utf8'), p).toBe('x')
    }
  })

  // Built at run time by split('/').join('\\') - a heredoc would eat the backslashes.
  const P3_CASES = ['tools/cad/templates/nozzle_contraction/template.py', 'cad/v1/study/params/stable.json', 'cad/v1/edits/v1.cad1.json']

  it.each(P3_CASES)('P3: Windows-spelled %s inside a token is refused WL-PROTECTED', async (p) => {
    const bp = p.split('/').join('\\')
    expect(protectedPath(bp), bp).not.toBeNull()
    const before = fileState(p)
    const inputs: Array<[string, Record<string, unknown>]> = [
      ['file_write', { path: bp, content: 'x', createOnly: false }],
      ['shell_exec', { argv: ['python', '-c', `open('${bp}','w').write('x')`], cwd: null, timeoutSec: null }],
      ['custom_tool_create', createInput(bp)],
      ['custom_tool_run', { name: 'anything', inputJson: JSON.stringify({ path: bp }) }],
    ]
    for (const [tool, input] of inputs) {
      const r = await runTool(tool, input, ctx())
      expect(r.ok, `${tool} on ${bp}: ${JSON.stringify(r.data)}`).toBe(false)
      expect(ruleIdOf(r), `${tool} on ${bp}`).toBe('WL-PROTECTED')
      expect(r.error?.message.startsWith('WL-PROTECTED: '), `${tool} on ${bp}`).toBe(true)
    }
    expect(fileState(p)).toBe(before)
  })

  const C1_CASES: Array<Record<string, unknown>> = [
    { kind: 'command', argv: ['python', '-c', 'import cadquery as cq'], cwd: null },
    { kind: 'command', argv: ['python', '-c', 'from OCP.BRepPrimAPI import x'], cwd: null },
    { kind: 'command', argv: ['python', '-m', 'gmsh', 'a.geo'], cwd: null },
    { kind: 'command', argv: ['gmsh', 'a.geo'], cwd: null },
    { kind: 'command', argv: ['C:/bin/gmsh.exe', 'a.geo'], cwd: null },
    { kind: 'command', argv: ['python', '-c', "__import__('OCP')"], cwd: null },
    { kind: 'js', source: 'const body = 1 // cadquery would build it here' },
  ]

  it.each(C1_CASES)('C1: custom_tool_create refuses CadQuery/OCP/gmsh code (CUSTOM-CAD): %j', async (impl) => {
    const r = await runTool('custom_tool_create', { name: 'x_cad_probe', description: 'probe', inputSchemaJson: '{"type":"object"}', impl }, ctx())
    expect(r.ok, JSON.stringify(impl)).toBe(false)
    expect(r.error?.code).toBe('EDIT_REFUSED')
    expect(ruleIdOf(r), JSON.stringify(impl)).toBe('CUSTOM-CAD')
    expect(r.error?.message.startsWith('CUSTOM-CAD: ')).toBe(true)
  })

  it('C2: custom_tool_create of a plain json-reading tool is ok and registered', async () => {
    const r = await runTool('custom_tool_create', { name: 'x_json_tool', description: 'json probe', inputSchemaJson: '{"type":"object"}', impl: { kind: 'command', argv: ['python', '-c', 'import json'], cwd: null } }, ctx())
    expect(r.ok).toBe(true)
    expect((r.data as { tools?: string[] }).tools).toContain('x_json_tool')
  })

  it('C3: a hand-written custom-tools.json is re-checked at run time (WL-PROTECTED, CUSTOM-CAD) and spawns nothing', async () => {
    await fsp.mkdir(ws.config.configDir, { recursive: true })
    const spec = (name: string, argv: string[]): Record<string, unknown> => ({
      name,
      description: 'hand written',
      inputSchema: { type: 'object' },
      impl: { kind: 'command', argv, cwd: null },
      createdAt: '2026-10-04T00:00:00.000Z',
    })
    await fsp.writeFile(
      path.join(ws.config.configDir, 'custom-tools.json'),
      JSON.stringify({ tools: [spec('t_lock', ['python', '-c', "open('tools/cad/gates.lock','w')"]), spec('t_cad', ['python', '-c', 'import gmsh'])] }),
      'utf8',
    )
    const lockAbs = path.join(ws.root, 'tools', 'cad', 'gates.lock')
    expect(fs.existsSync(lockAbs)).toBe(false)
    const r1 = await runTool('custom_tool_run', { name: 't_lock', inputJson: '{}' }, ctx())
    expect(r1.ok).toBe(false)
    expect(ruleIdOf(r1)).toBe('WL-PROTECTED')
    const r2 = await runTool('custom_tool_run', { name: 't_cad', inputJson: '{}' }, ctx())
    expect(r2.ok).toBe(false)
    expect(ruleIdOf(r2)).toBe('CUSTOM-CAD')
    expect(fs.existsSync(lockAbs)).toBe(false)
  })
})
