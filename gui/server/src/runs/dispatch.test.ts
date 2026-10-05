// The shell line a pipeline is started with. A .cmd runs through cmd.exe, whose
// separators are live outside quotes, so every argument is quoted - the one
// place a run request's free string (a --tag value) could otherwise become a
// second command.
import { describe, expect, it } from 'vitest'
import { BINARIES } from '@cfd/shared'
import { STATIC_SYSTEM } from '../prompts/system.js'
import { availableBinaries, buildArgv, pipelineCommandLine, pipelineSpawn, quoteForCmd } from './dispatch.js'

describe('pipeline command line', () => {
  it('quotes every argument, doubling embedded quotes, so cmd separators stay literal', () => {
    expect(quoteForCmd('plain')).toBe('"plain"')
    expect(quoteForCmd('cases/a b.json')).toBe('"cases/a b.json"')
    expect(quoteForCmd('say "hi"')).toBe('"say ""hi"""')
    expect(quoteForCmd('x&whoami')).toBe('"x&whoami"')
    const argv = buildArgv({ casePath: null, positionals: ['cases/foo.json'], args: [{ flag: '--tag', value: 'x&whoami' }, { flag: '--dry-run', value: true }] })
    expect(argv).toEqual(['cases/foo.json', '--tag', 'x&whoami', '--dry-run'])
    const line = pipelineCommandLine('C:\\ws\\tools\\mesh\\run_step_mesh.cmd', argv)
    expect(line).toBe('"C:\\ws\\tools\\mesh\\run_step_mesh.cmd" "cases/foo.json" "--tag" "x&whoami" "--dry-run"')
    // nothing is left outside a quote pair
    expect(line.replace(/"[^"]*"/g, '').trim()).toBe('')
  })
})

describe('pipeline spawn', () => {
  it('py_pipeline_spawn: a .py runs exec-style with nothing quoted, a .cmd keeps the quoted cmd.exe line, anything else is refused', () => {
    expect(pipelineSpawn({ name: 'geom-tool', source: 'tools/geom/geom_tool.py' }, 'C:\\ws\\tools\\geom\\geom_tool.py', ['info', 'a b.step', '--json', 'x&y.json'], 'python', 'cmd.exe')).toEqual({
      command: 'python',
      args: ['C:\\ws\\tools\\geom\\geom_tool.py', 'info', 'a b.step', '--json', 'x&y.json'],
      verbatim: false,
    })
    expect(pipelineSpawn({ name: 'mesh-step', source: 'tools/mesh/run_step_mesh.cmd' }, 'C:\\ws\\tools\\mesh\\run_step_mesh.cmd', ['cases/foo.json', '--tag', 'x&whoami', '--dry-run'], 'python', 'cmd.exe')).toEqual({
      command: 'cmd.exe',
      args: ['/d', '/s', '/c', '""C:\\ws\\tools\\mesh\\run_step_mesh.cmd" "cases/foo.json" "--tag" "x&whoami" "--dry-run""'],
      verbatim: true,
    })
    expect(() => pipelineSpawn({ name: 'x', source: 'tools/x.sh' }, 'C:\\ws\\tools\\x.sh', [], 'python', 'cmd.exe')).toThrow(/\.cmd or a \.py/)
  })
})

// ofgpu-regions and ofgpu-sample are Cargo targets now, so nothing in the
// registry is pending any more: every declared ofgpu-* binary is offered in
// demo mode, and in real mode only the ones actually built on this machine.
describe('arrived binaries', () => {
  it('availableBinaries offers ofgpu-regions and ofgpu-sample once they are Cargo targets', () => {
    expect(availableBinaries({ binDir: null, workspaceRoot: '/nowhere', demo: true }, ['ofgpu-cht', 'ofgpu-regions', 'ofgpu-sample'])).toEqual(['ofgpu-cht', 'ofgpu-regions', 'ofgpu-sample'])
    expect(availableBinaries({ binDir: null, workspaceRoot: '/nowhere', demo: false }, ['ofgpu-regions'])).toEqual([])
    expect(STATIC_SYSTEM).toContain('ofgpu-regions')
    expect(STATIC_SYSTEM).toContain('ofgpu-sample')
    expect(BINARIES.some((b) => b.pending === true)).toBe(false)
  })
})
