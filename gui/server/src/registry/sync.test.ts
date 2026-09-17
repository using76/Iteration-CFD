// Keeps shared/registry.ts in step with the Rust sources: Cargo [[bin]]
// names/paths, the flag set of every `fn usage()` binary, the `const USAGE`
// text of the others, and the model names.
import fs from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'
import { BINARIES, BINARY_NAMES, getBinary, MODELS, PIPELINES } from '@cfd/shared'
import { REPO_ROOT } from '../runs/test-helpers.js'

const RUST = path.join(REPO_ROOT, 'rust')
const read = (p: string) => fs.readFileSync(p, 'utf8')

function cargoBins(): Array<{ name: string; path: string }> {
  const text = read(path.join(RUST, 'Cargo.toml'))
  const out: Array<{ name: string; path: string }> = []
  for (const m of text.matchAll(/\[\[bin\]\]\s*\nname = "([^"]+)"\s*\npath = "([^"]+)"/g)) out.push({ name: m[1], path: m[2] })
  return out
}

/** The concatenated string literal of `fn usage()` with Rust's `\`-newline continuations and `{}` expanded. */
function usageText(source: string): string {
  const m = source.match(/fn usage\(\)\s*\{\s*eprintln!\(\s*"((?:[^"\\]|\\[\s\S])*)"/)
  if (!m) throw new Error('no fn usage() literal')
  return m[1]
    .replace(/\\\r?\n\s*/g, '')
    .replace(/\\n/g, '\n')
    .replace(/\{\}/g, '\n  -permissive     downgrade unsupported-setting errors to warnings')
}

function constUsage(source: string): string {
  const m = source.match(/const USAGE: &str = "((?:[^"\\]|\\[\s\S])*)"/)
  if (!m) throw new Error('no const USAGE')
  return m[1].replace(/\\\r?\n\s*/g, '')
}

const flagTokens = (text: string) => new Set([...text.matchAll(/(?<=[\s[])(-[A-Za-z][A-Za-z0-9]*)(?=[\s\]])/g)].map((m) => m[1]))

describe('registry <-> rust sources', () => {
  it('lists exactly the Cargo [[bin]] targets with their source paths', () => {
    const cargo = cargoBins().sort((a, b) => a.name.localeCompare(b.name))
    const reg = BINARIES.filter((b) => !b.pending).map((b) => ({ name: b.name, path: b.source })).sort((a, b) => a.name.localeCompare(b.name))
    expect(reg).toEqual(cargo)
    for (const b of BINARIES.filter((b) => !b.pending)) expect(fs.existsSync(path.join(RUST, b.source)), b.source).toBe(true)
    // A pending entry is a promise the tree keeps honest: as long as the flag
    // is set, the Cargo target must NOT exist yet.
    for (const b of BINARIES.filter((b) => b.pending)) expect(fs.existsSync(path.join(RUST, b.source)), `${b.name} is pending but ${b.source} exists: drop pending and pin it`).toBe(false)
  })

  it('has the same flag set as every fn usage()', () => {
    for (const b of BINARIES.filter((x) => !x.pending && x.usageKind === 'usageFn')) {
      const usage = usageText(read(path.join(RUST, b.source)))
      const inUsage = [...flagTokens(usage)].sort()
      const inSpec = b.flags.map((f) => f.name).sort()
      expect(inSpec, `${b.name}: registry flags vs usage()`).toEqual(inUsage)
    }
  })

  it('mentions every registry flag in the const USAGE binaries', () => {
    for (const b of BINARIES.filter((x) => !x.pending && x.usageKind === 'constUsage')) {
      const usage = constUsage(read(path.join(RUST, b.source)))
      for (const f of b.flags) expect(usage, `${b.name} ${f.name}`).toContain(f.name)
      const tokens = [...flagTokens(usage)].filter((t) => t.length > 2)
      for (const t of tokens) expect(b.flags.map((f) => f.name), `${b.name}: ${t} documented but not in the registry`).toContain(t)
    }
  })

  it('names every model somewhere in rust/src', () => {
    const files: string[] = []
    const walk = (dir: string) => {
      for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
        const p = path.join(dir, e.name)
        if (e.isDirectory()) walk(p)
        else if (e.name.endsWith('.rs')) files.push(p)
      }
    }
    walk(path.join(RUST, 'src'))
    const corpus = files.map(read).join('\n')
    for (const m of MODELS) expect(corpus.includes(m.name), `model ${m.name}`).toBe(true)
    for (const m of MODELS) for (const d of m.drivers) expect(BINARIES.map((b) => b.name), `${m.name} driver ${d}`).toContain(d)
  })

  it('keeps the mesh-step pipeline out of the Cargo list and in step with step_mesh.py\'s own argparse', () => {
    const step = PIPELINES.find((p) => p.name === 'mesh-step')!
    expect(step.pipeline).toBe(true)
    expect(BINARY_NAMES).not.toContain('mesh-step')
    expect(getBinary('mesh-step')).toBe(step)
    expect(fs.existsSync(path.join(REPO_ROOT, step.source)), step.source).toBe(true)
    // the script the .cmd wraps declares the flags; the registry must offer exactly those
    const py = read(path.join(REPO_ROOT, 'tools', 'mesh', 'step_mesh.py'))
    const declared = [...py.matchAll(/ap\.add_argument\('(--[a-z-]+)'/g)].map((m) => m[1]).sort()
    expect(step.flags.map((f) => f.name).sort()).toEqual(declared)
    expect(py).toMatch(/ap\.add_argument\('config'/)
    expect(step.positionals.map((p) => p.name)).toEqual(['config'])
    // a flag argparse gives a metavar takes a value; the rest are bare. Read that from the
    // script rather than naming the flags here, so a new one cannot be typed wrongly in silence.
    const takesValue = new Set(
      [...py.matchAll(/ap\.add_argument\('(--[a-z-]+)'[^)]*metavar=/g)].map((m) => m[1]),
    )
    expect(takesValue.size, 'argparse declares at least one value-taking flag').toBeGreaterThan(0)
    for (const f of step.flags) expect(f.type, f.name).toBe(takesValue.has(f.name) ? 'string' : 'flag')
  })

  it('geometry_pipelines_shape: geom-tool and regions-from-msh are .py pipelines beside the binaries', () => {
    expect(PIPELINES.map((p) => p.name)).toEqual(['mesh-step', 'geom-tool', 'regions-from-msh'])
    for (const [name, positionals] of [['geom-tool', ['command', 'file']], ['regions-from-msh', ['msh', 'outDir']]] as const) {
      const pipeline = PIPELINES.find((p) => p.name === name)!
      expect(pipeline.pipeline).toBe(true)
      expect(BINARY_NAMES).not.toContain(name)
      expect(getBinary(name)).toBe(pipeline)
      expect(pipeline.source.endsWith('.py'), pipeline.source).toBe(true)
      expect(pipeline.kind).toBe('mesh')
      expect(pipeline.gpu).toBe(false)
      expect(pipeline.positionals.map((p) => p.name)).toEqual([...positionals])
    }
    const geom = PIPELINES.find((p) => p.name === 'geom-tool')!
    expect(geom.flags.map((f) => f.name).sort()).toEqual(['--json', '--ops', '--out', '--scale', '--stl-size'])
    const regions = PIPELINES.find((p) => p.name === 'regions-from-msh')!
    expect(regions.flags.map((f) => f.name).sort()).toEqual(['--fluid', '--material', '--overwrite', '--tolerance', '--units'])
    expect(regions.flags.some((f) => f.name === '--fluent')).toBe(false)
    expect(regions.flags.find((f) => f.name === '--material')?.repeatable).toBe(true)
    expect(regions.flags.find((f) => f.name === '--overwrite')?.type).toBe('flag')
    // The scripts land with the feat/automesher branch (M1-M4) and are not
    // asserted to exist here; when one is present, every registry flag must be
    // declared by its argparse. The script may declare more (--fluent, which
    // M4 refuses by name), so the check is a subset, not an equality.
    for (const pipeline of [geom, regions]) {
      const abs = path.join(REPO_ROOT, pipeline.source)
      if (!fs.existsSync(abs)) continue
      const py = fs.readFileSync(abs, 'utf8')
      const declared = [...py.matchAll(/add_argument\('(--[a-z-]+)'/g)].map((m) => m[1])
      for (const f of pipeline.flags) expect(declared, `${pipeline.name} ${f.name}`).toContain(f.name)
    }
  })

  it('agrees with driver_for on which driver builds the k-epsilon family', () => {
    const common = read(path.join(RUST, 'src', 'bin', 'common', 'mod.rs'))
    expect(common).toMatch(/RasModel::KEpsilon \| RasModel::RealizableKE \| RasModel::RNGkEpsilon => "ofgpu-k-epsilon"/)
    const ke = BINARIES.find((b) => b.name === 'ofgpu-k-epsilon')!
    expect(ke.builds).toEqual(expect.arrayContaining(['kEpsilon', 'realizableKE', 'RNGkEpsilon']))
  })
})
