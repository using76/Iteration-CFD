// The shell line a pipeline is started with. A .cmd runs through cmd.exe, whose
// separators are live outside quotes, so every argument is quoted - the one
// place a run request's free string (a --tag value) could otherwise become a
// second command.
import { spawn } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'
import { BINARIES, checkArgValue, getBinary } from '@cfd/shared'
import { STATIC_SYSTEM } from '../prompts/system.js'
import { availableBinaries, binarySearchDirs, buildArgv, dispatch, findBinary, pipelineCommandLine, pipelineSpawn, quoteForCmd, runHeaderLine, type DispatchOptions } from './dispatch.js'
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

// F32-19: the studio runs a case at f32. precision 'single' looks only in
// rust/target/single/release (OFGPU_BIN_DIR and target/release hold the f64
// build), the run records its precision, and the log's first line is the run
// header naming the build. Single was a preview until F32-18; F32-19b
// cleared the flag.
describe('single precision (F32-19)', () => {
  it('binarySearchDirs and findBinary: single looks only in rust/target/single/release', async () => {
    const ws = await makeTempWorkspace({ plume: false })
    try {
      const single = path.join(ws.root, 'rust', 'target', 'single', 'release')
      expect(binarySearchDirs({ binDir: '/b', workspaceRoot: ws.root }, 'single')).toEqual([single])
      const f64 = ['/b', path.join(ws.root, 'rust', 'target', 'release'), path.join(ws.root, 'rust', 'target', 'debug')]
      expect(binarySearchDirs({ binDir: '/b', workspaceRoot: ws.root })).toEqual(f64)
      expect(binarySearchDirs({ binDir: '/b', workspaceRoot: ws.root }, 'double')).toEqual(f64)
      const exe = process.platform === 'win32' ? 'ofgpu-k-epsilon.exe' : 'ofgpu-k-epsilon'
      fs.mkdirSync(single, { recursive: true })
      fs.writeFileSync(path.join(single, exe), '')
      expect(findBinary(ws.config, 'ofgpu-k-epsilon', 'single')).toBe(path.join(single, exe))
      expect(findBinary(ws.config, 'ofgpu-k-epsilon', 'double')).toBeNull()
      expect(findBinary(ws.config, 'ofgpu-k-epsilon')).toBeNull()
    } finally {
      await ws.cleanup()
    }
  })

  it('precision "single" spawns the single path; a missing single build is refused naming the build command', async () => {
    const ws = await makeTempWorkspace({ plume: false })
    try {
      const config = { ...ws.config, demo: false }
      const base = { config, binary: 'ofgpu-k-epsilon', argv: ['-e', 'process.exit(0)'], cwd: ws.root }
      expect(() => dispatch({ ...base, precision: 'single' })).toThrow(/no single-precision build.*target[\\/]single[\\/]release.*--features single --target-dir target\/single/)
      // A real executable at the single path: node itself, linked (or copied) in under the solver's name.
      const single = path.join(ws.root, 'rust', 'target', 'single', 'release')
      fs.mkdirSync(single, { recursive: true })
      const exe = path.join(single, process.platform === 'win32' ? 'ofgpu-k-epsilon.exe' : 'ofgpu-k-epsilon')
      try {
        fs.linkSync(process.execPath, exe)
      } catch {
        fs.copyFileSync(process.execPath, exe)
      }
      const spawned = dispatch({ ...base, precision: 'single' })
      const code = await new Promise<number | null>((resolve) => spawned.child.on('close', (c) => resolve(c)))
      expect(spawned.mode).toBe('real')
      expect(spawned.argv).toEqual([exe, '-e', 'process.exit(0)'])
      expect(code).toBe(0)
      // The f64 build is still looked for where it always was, and is not there.
      expect(() => dispatch(base)).toThrow(/is not built on this machine/)
      expect(runHeaderLine('ofgpu-k-epsilon', 'single', exe)).toBe(`run header: ofgpu-k-epsilon precision single (f32 build, ${exe})`)
      expect(runHeaderLine('ofgpu-k-epsilon', 'single')).toBe('run header: ofgpu-k-epsilon precision single (f32 build, demo)')
      expect(runHeaderLine('ofgpu-k-epsilon', 'double', '/x/ofgpu-k-epsilon')).toBe('run header: ofgpu-k-epsilon precision double (f64 build, /x/ofgpu-k-epsilon)')
      expect(runHeaderLine('ofgpu-k-epsilon', 'double')).toBe('run header: ofgpu-k-epsilon precision double (f64 build, demo)')
    } finally {
      await ws.cleanup()
    }
  })

  it('the run manager hands precision to the dispatcher, records it and writes the run header first; a pipeline refuses single', async () => {
    const ws = await makeTempWorkspace()
    try {
      const seen: DispatchOptions[] = []
      const runs = await createRunManager({
        config: ws.config,
        dispatch: (opts) => {
          seen.push(opts)
          const child = spawn(process.execPath, ['-e', ''], { stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true })
          return { child, argv: [opts.binary, ...opts.argv], mode: 'demo' as const }
        },
      })
      try {
        const base = { binary: 'ofgpu-k-epsilon', casePath: 'cases/plume.jsonc', positionals: [], args: [], label: null }
        await expect(runs.start({ binary: 'mesh-step', casePath: null, positionals: ['cases/x.json'], args: [], label: null, precision: 'single' })).rejects.toThrow(/mesh-step is a pipeline script.*no single-precision build/)
        const single = await runs.start({ ...base, precision: 'single' })
        expect(single.precision).toBe('single')
        for (let i = 0; i < 100 && seen.length === 0; i++) await new Promise((r) => setTimeout(r, 20))
        expect(seen).toHaveLength(1)
        expect(seen[0].precision).toBe('single')
        await runs.wait(single.id, { maxMs: 5_000 })
        const first = runs.log(single.id, 1, 1).lines[0]
        expect(first.stream).toBe('system')
        expect(first.text).toBe('run header: ofgpu-k-epsilon precision single (f32 build, demo)')
        const dbl = await runs.start(base)
        expect(dbl.precision).toBe('double')
        for (let i = 0; i < 100 && seen.length < 2; i++) await new Promise((r) => setTimeout(r, 20))
        expect(seen[1].precision).toBe('double')
        await runs.wait(dbl.id, { maxMs: 5_000 })
        expect(runs.log(dbl.id, 1, 1).lines[0].text).toBe('run header: ofgpu-k-epsilon precision double (f64 build, demo)')
      } finally {
        await runs.shutdown()
      }
    } finally {
      await ws.cleanup()
    }
  })
})
