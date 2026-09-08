// The shipped default tools: tools.defaults.json loads with resolved <repo> /
// <OFGPU_BIN_DIR> paths, registers itself under the user's custom-tools.json
// without custom_tool_create, loses to a user tool of the same name, and its
// optional argv inputs splice (or vanish) instead of leaving empty tokens.
import fsp from 'node:fs/promises'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { fakeDatasets, fakeHub, fakeRuns, GUI_DIR, makeWorkspace, type TempWorkspace } from '../agent/test-fakes.js'
import { findBinary } from '../runs/dispatch.js'
import type { ToolContext } from './context.js'
import { loadCustomTools, saveCustomTools, type CustomToolSpec } from './custom.js'
import { defaultToolCandidates, loadDefaultTools, mergeTools, resolveDefaultArgv, setDefaultTools } from './defaults.js'
import { runTool } from './index.js'

let ws: TempWorkspace
let hub: ReturnType<typeof fakeHub>
let runs: ReturnType<typeof fakeRuns>
beforeAll(async () => {
  ws = await makeWorkspace()
  hub = fakeHub()
  runs = fakeRuns()
})
afterAll(async () => {
  setDefaultTools([])
  await ws.cleanup()
})

function ctx(): ToolContext {
  return { config: ws.config, hub, runs, datasets: fakeDatasets(), sessionId: 's1', signal: new AbortController().signal, workspaceRoot: ws.root, settings: { autoApprove: 'reads', effort: 'high', notifyOnRunEnd: true, locale: 'en' }, toolUseId: 'toolu_1' }
}

function echoDefault(name: string, required: string[]): CustomToolSpec {
  return {
    name,
    description: 'test default',
    inputSchema: { type: 'object', properties: { config: { type: 'string' }, extra: { type: 'string' } }, required },
    impl: { kind: 'command', argv: [process.execPath, '-e', 'console.log("ran", process.argv.slice(1).join("|"))', '{{input.config}}', '{{input.extra}}'], cwd: null },
    createdAt: new Date().toISOString(),
  }
}

describe('default tools', () => {
  it('loads the two shipped mesh tools with resolved paths', async () => {
    const specs = await loadDefaultTools(ws.config)
    expect(specs.map((t) => t.name)).toEqual(['mesh_from_step', 'mesh_to_fluent'])
    expect(defaultToolCandidates(ws.config)[0]).toBe(path.join(GUI_DIR, 'server', 'tools.defaults.json'))

    const fromStep = specs[0]
    expect(fromStep.impl.kind).toBe('command')
    expect(fromStep.inputSchema.required).toEqual(['config'])
    const stepArgv = fromStep.impl.kind === 'command' ? fromStep.impl.argv : []
    expect(stepArgv[0]).toBe('python')
    expect(stepArgv[1]).toBe(path.resolve(GUI_DIR, '..', 'tools', 'mesh', 'step_mesh.py'))
    expect(stepArgv).toContain('{{input.config}}')
    expect(stepArgv).toContain('{{input.extra}}')

    const toFluent = specs[1]
    expect(toFluent.inputSchema.required).toEqual(['msh', 'case', 'fluent'])
    const fluentArgv = toFluent.impl.kind === 'command' ? toFluent.impl.argv : []
    const exe = process.platform === 'win32' ? 'ofgpu-convert-mesh.exe' : 'ofgpu-convert-mesh'
    expect(fluentArgv).toEqual([path.join(ws.root, 'rust', 'target', 'release', exe), '{{input.msh}}', '{{input.case}}', '-fluent', '{{input.fluent}}', '{{input.types}}'])
  })

  it('resolves <repo> and <OFGPU_BIN_DIR> the way the solver binaries are found', async () => {
    expect(resolveDefaultArgv(['<repo>/tools/mesh/step_mesh.py'], ws.config)).toEqual([path.resolve(GUI_DIR, '..', 'tools', 'mesh', 'step_mesh.py')])
    // Nothing is built in the temp workspace: the first search directory still supplies a path.
    const fallback = resolveDefaultArgv(['<OFGPU_BIN_DIR>/ofgpu-convert-mesh'], ws.config)[0]
    expect(fallback).toBe(path.join(ws.root, 'rust', 'target', 'release', process.platform === 'win32' ? 'ofgpu-convert-mesh.exe' : 'ofgpu-convert-mesh'))
    // An OFGPU_BIN_DIR that really holds the binary wins, like runs/dispatch.ts.
    const binDir = path.join(ws.tmp, 'bin')
    await fsp.mkdir(binDir, { recursive: true })
    await fsp.writeFile(path.join(binDir, process.platform === 'win32' ? 'ofgpu-convert-mesh.exe' : 'ofgpu-convert-mesh'), '')
    const withBin = { ...ws.config, binDir }
    expect(resolveDefaultArgv(['<OFGPU_BIN_DIR>/ofgpu-convert-mesh'], withBin)[0]).toBe(findBinary(withBin, 'ofgpu-convert-mesh'))
    expect(resolveDefaultArgv(['<OFGPU_BIN_DIR>/ofgpu-convert-mesh'], withBin)[0]).toBe(path.join(binDir, process.platform === 'win32' ? 'ofgpu-convert-mesh.exe' : 'ofgpu-convert-mesh'))
  })

  it('registers the defaults without custom_tool_create and runs them by name', async () => {
    setDefaultTools(await loadDefaultTools(ws.config))
    expect((await loadCustomTools(ws.config.configDir))).toEqual([]) // nothing in the user's file
    const merged = mergeTools(await loadCustomTools(ws.config.configDir))
    expect(merged.map((t) => t.name)).toEqual(['mesh_from_step', 'mesh_to_fluent'])

    const unknown = await runTool('custom_tool_run', { name: 'no_such_tool', inputJson: '{}' }, ctx())
    expect(unknown.error?.code).toBe('NO_SUCH_TOOL')
    expect(unknown.error?.message).toContain('mesh_from_step, mesh_to_fluent')

    // A default really dispatches: argv placeholders substitute, the optional
    // tail splices on whitespace, an absent one leaves no empty token.
    setDefaultTools([echoDefault('echo_argv', ['config'])])
    const withExtra = await runTool('custom_tool_run', { name: 'echo_argv', inputJson: '{"config":"cases/step.json","extra":"--from-checkpoint z=3.05"}' }, ctx())
    expect(withExtra.ok).toBe(true)
    expect((withExtra.data as { argv: string[] }).argv).toEqual([process.execPath, '-e', expect.any(String), 'cases/step.json', '--from-checkpoint', 'z=3.05'])
    const withoutExtra = await runTool('custom_tool_run', { name: 'echo_argv', inputJson: '{"config":"my dir/step.json"}' }, ctx())
    expect((withoutExtra.data as { argv: string[] }).argv).toEqual([process.execPath, '-e', expect.any(String), 'my dir/step.json'])
    const ran = (withoutExtra.data as { stdout: string }).stdout
    expect(ran).toContain('my dir/step.json')
    expect(ran).not.toMatch(/\|\|/) // no empty element between the pipes
  })

  it('a user tool of the same name replaces the default', async () => {
    setDefaultTools(await loadDefaultTools(ws.config))
    const user: CustomToolSpec = {
      name: 'mesh_from_step',
      description: 'user wins',
      inputSchema: { type: 'object', properties: { config: { type: 'string' } }, required: ['config'] },
      impl: { kind: 'command', argv: [process.execPath, '-e', 'console.log("user-wins")', '{{input.config}}'], cwd: null },
      createdAt: new Date().toISOString(),
    }
    await saveCustomTools(ws.config.configDir, [user])
    const merged = mergeTools(await loadCustomTools(ws.config.configDir))
    expect(merged.filter((t) => t.name === 'mesh_from_step')).toEqual([user])
    expect(merged.map((t) => t.name)).toEqual(['mesh_from_step', 'mesh_to_fluent'])

    const run = await runTool('custom_tool_run', { name: 'mesh_from_step', inputJson: '{"config":"cases/step.json"}' }, ctx())
    expect(run.ok).toBe(true)
    expect((run.data as { stdout: string }).stdout.trim()).toBe('user-wins')
  })
})
