// The shell line a pipeline is started with. A .cmd runs through cmd.exe, whose
// separators are live outside quotes, so every argument is quoted - the one
// place a run request's free string (a --tag value) could otherwise become a
// second command.
import { describe, expect, it } from 'vitest'
import { buildArgv, pipelineCommandLine, quoteForCmd } from './dispatch.js'

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
