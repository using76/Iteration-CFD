// The shell line a pipeline is started with. A .cmd runs through cmd.exe, whose
// separators are live outside quotes, so every argument is quoted - the one
// place a run request's free string (a --tag value) could otherwise become a
// second command.
import { spawn } from 'node:child_process'
import { describe, expect, it } from 'vitest'
import { BINARIES, checkArgValue, getBinary } from '@cfd/shared'
import { STATIC_SYSTEM } from '../prompts/system.js'
import { availableBinaries, buildArgv, pipelineCommandLine, pipelineSpawn, quoteForCmd, type DispatchOptions } from './dispatch.js'
import { createRunManager } from './manager.js'
import { makeTempWorkspace } from './test-helpers.js'

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

// PAR-17: the studio launches a parallel run through the ranks / transport
// run option (-np, -comm). The transport is threads only: PAR-06 (MPI) is
// deferred, so the registry offers no mpi and a request for it is refused
// before anything is spawned.
describe('ofgpu-par dispatch', () => {
  it('builds `ofgpu-par case -np 2 -comm threads` and refuses -comm mpi', () => {
    const par = getBinary('ofgpu-par')!
    const req = { casePath: 'cases/plume.jsonc', positionals: [], args: [{ flag: '-np', value: '2' }, { flag: '-comm', value: 'threads' }] }
    for (const a of req.args) expect(checkArgValue(par.flags.find((f) => f.name === a.flag)!, a.value)).toBeNull()
    expect(buildArgv(req)).toEqual(['cases/plume.jsonc', '-np', '2', '-comm', 'threads'])
    expect(checkArgValue(par.flags.find((f) => f.name === '-comm')!, 'mpi')).toMatch(/must be one of threads/)
    expect(par.runOptions?.[0].flags).toEqual(['-np', '-comm'])
    expect(availableBinaries({ binDir: null, workspaceRoot: '/nowhere', demo: true }, ['ofgpu-par'])).toEqual(['ofgpu-par'])
    expect(availableBinaries({ binDir: null, workspaceRoot: '/nowhere', demo: false }, ['ofgpu-par'])).toEqual([])
  })

  it('the run manager hands the dispatcher that argv, and refuses -comm mpi before anything is spawned', async () => {
    const ws = await makeTempWorkspace()
    try {
      const seen: DispatchOptions[] = []
      const runs = await createRunManager({
        config: ws.config,
        dispatch: (opts) => {
          seen.push(opts)
          const child = spawn(process.execPath, ['-e', ''], { stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true })
          return { child, argv: [opts.binary, ...opts.argv], mode: 'real' as const }
        },
      })
      try {
        const base = { binary: 'ofgpu-par', casePath: 'cases/plume.jsonc', positionals: [], label: null }
        await expect(runs.start({ ...base, args: [{ flag: '-np', value: '2' }, { flag: '-comm', value: 'mpi' }] })).rejects.toThrow(/-comm must be one of threads/)
        const run = await runs.start({ ...base, args: [{ flag: '-np', value: '2' }, { flag: '-comm', value: 'threads' }] })
        for (let i = 0; i < 100 && seen.length === 0; i++) await new Promise((r) => setTimeout(r, 20))
        expect(seen).toHaveLength(1)
        expect(seen[0].binary).toBe('ofgpu-par')
        expect(seen[0].argv).toEqual(['cases/plume.jsonc', '-np', '2', '-comm', 'threads'])
        expect(run.binary).toBe('ofgpu-par')
      } finally {
        await runs.shutdown()
      }
    } finally {
      await ws.cleanup()
    }
  })
})
