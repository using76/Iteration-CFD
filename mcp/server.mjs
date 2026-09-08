#!/usr/bin/env node
// An MCP server that hands the ofgpu solvers to any MCP client.
//
// Stdio JSON-RPC, no dependencies. The Model Context Protocol is a small
// protocol and a 2 MB SDK to speak it would be the largest thing in this
// repository by file count, so it is spoken directly here: `initialize`,
// `tools/list`, `tools/call`, and the notifications a client sends that need
// no answer. Anything else gets a proper JSON-RPC error rather than silence.
//
// What it exposes is the binaries, not a reimplementation of them: probe,
// mesh generation, the seven solver drivers, the validation suite, and two
// read-only helpers for finding cases and reading back what a run wrote. The
// long ones stream nothing -- MCP tool calls are request/response -- so they
// return the tail of the log and the exit code, which is what a model needs to
// decide what to do next.
//
// Safety, deliberately: no shell. Every child process is spawned with an
// argument array, the binary is looked up by name in one directory, and every
// path a caller gives is resolved and checked to be inside the workspace root.
// A tool call cannot run something that is not an ofgpu binary and cannot
// touch a file outside the workspace.
import { spawn } from 'node:child_process'
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs'
import path from 'node:path'
import process from 'node:process'
import readline from 'node:readline'

const PROTOCOL_VERSION = '2025-06-18'
const SERVER = { name: 'ofgpu', version: '0.1.0' }

// ---------------------------------------------------------------- config ---

/** Where the ofgpu-*.exe live. Set OFGPU_BIN_DIR, or let it find a build tree. */
function binDir() {
  if (process.env.OFGPU_BIN_DIR) return path.resolve(process.env.OFGPU_BIN_DIR)
  const here = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'))
  for (const rel of ['../rust/target/release', '../bin', '../../bin']) {
    const dir = path.resolve(here, rel)
    if (existsSync(dir)) return dir
  }
  return path.resolve(here, '../rust/target/release')
}

/** The one directory tool calls may read and write inside. */
function workspace() {
  if (process.env.OFGPU_WORKSPACE) return path.resolve(process.env.OFGPU_WORKSPACE)
  return process.cwd()
}

const BIN_DIR = binDir()
const WORKSPACE = workspace()
const EXE = process.platform === 'win32' ? '.exe' : ''

/** Seconds a tool call may run before it is killed and reports what it had. */
const DEFAULT_TIMEOUT_S = Number(process.env.OFGPU_MCP_TIMEOUT_S || 3600)

/** How much of a solver's log to hand back; the interesting part is the end. */
const LOG_TAIL_LINES = 60

const SOLVERS = {
  'k-epsilon': 'ofgpu-k-epsilon',
  'k-omega': 'ofgpu-k-omega',
  sa: 'ofgpu-sa',
  lowmach: 'ofgpu-lowmach',
  cht: 'ofgpu-cht',
  buoyant: 'ofgpu-buoyant',
  plume: 'ofgpu-plume',
  vof: 'ofgpu-vof',
  datacentre: 'ofgpu-datacentre',
}

const MESH_PRESETS = ['channel', 'cavity', 'step', 'big', 'plume', 'room', 'damBreak']

// ------------------------------------------------------------- utilities ---

class ToolError extends Error {}

/** Resolve a caller-supplied path inside the workspace, or refuse it. */
function insideWorkspace(rel, what = 'path') {
  if (typeof rel !== 'string' || rel.length === 0) throw new ToolError(`${what} is required`)
  const abs = path.resolve(WORKSPACE, rel)
  const root = WORKSPACE.endsWith(path.sep) ? WORKSPACE : WORKSPACE + path.sep
  if (abs !== WORKSPACE && !abs.startsWith(root)) {
    throw new ToolError(`${what} must stay inside the workspace (${WORKSPACE}): got ${rel}`)
  }
  return abs
}

function binary(name) {
  const abs = path.join(BIN_DIR, name + EXE)
  if (!existsSync(abs)) {
    throw new ToolError(
      `${name}${EXE} is not in ${BIN_DIR}. Build it with \`cargo build --release\` in rust/, or set OFGPU_BIN_DIR to the installed bin directory.`,
    )
  }
  return abs
}

function tail(text, lines = LOG_TAIL_LINES) {
  const all = text.split(/\r?\n/)
  if (all.length <= lines) return text.trimEnd()
  return ['… ' + (all.length - lines) + ' earlier line(s) omitted', ...all.slice(-lines)].join('\n').trimEnd()
}

/**
 * Run one ofgpu binary. No shell: argv is an array and the executable is one
 * this file resolved by name, so nothing a caller writes can become a command.
 */
function runBinary(name, args, timeoutS = DEFAULT_TIMEOUT_S) {
  const exe = binary(name)
  return new Promise((resolve) => {
    const child = spawn(exe, args, { cwd: WORKSPACE, windowsHide: true })
    let out = ''
    let err = ''
    let killed = false
    const cap = (s) => (s.length > 4_000_000 ? s.slice(-2_000_000) : s)
    child.stdout.on('data', (d) => (out = cap(out + d)))
    child.stderr.on('data', (d) => (err = cap(err + d)))
    const timer = setTimeout(() => {
      killed = true
      child.kill('SIGKILL')
    }, timeoutS * 1000)
    child.on('error', (e) => {
      clearTimeout(timer)
      resolve({ code: null, out, err: String(e.message), killed })
    })
    child.on('close', (code) => {
      clearTimeout(timer)
      resolve({ code, out, err, killed })
    })
  })
}

/** One text block, plus isError when the process did not succeed. */
function reportOf(label, r) {
  const parts = [`$ ${label}`]
  if (r.killed) parts.push(`(killed after ${DEFAULT_TIMEOUT_S}s: raise OFGPU_MCP_TIMEOUT_S if the case needs longer)`)
  parts.push(`exit code: ${r.code === null ? 'n/a' : r.code}`)
  const body = tail(r.out)
  if (body) parts.push('', body)
  const e = tail(r.err, 20)
  if (e) parts.push('', 'stderr:', e)
  return { text: parts.join('\n'), isError: r.killed || (r.code !== 0 && r.code !== null) }
}

// ----------------------------------------------------------------- tools ---

const TOOLS = [
  {
    name: 'ofgpu_probe',
    title: 'Check the GPU',
    description:
      'Report the GPU the solvers would run on and verify that the device result is bitwise identical to the host result. Run this first: everything else needs a CUDA 13 capable NVIDIA GPU.',
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
    readOnly: true,
    async run() {
      return reportOf('ofgpu-probe', await runBinary('ofgpu-probe', [], 120))
    },
  },
  {
    name: 'ofgpu_list_cases',
    title: 'List the cases in the workspace',
    description:
      'List the runnable cases under the workspace: the *.jsonc case definitions and any OpenFOAM case directories (a directory holding constant/ and system/). Use it before meshing or solving to find out what is there.',
    inputSchema: {
      type: 'object',
      properties: {
        dir: { type: 'string', description: 'Workspace-relative directory to look in. Default: cases' },
      },
      additionalProperties: false,
    },
    readOnly: true,
    async run(args) {
      const rel = args.dir ?? 'cases'
      const abs = insideWorkspace(rel, 'dir')
      if (!existsSync(abs)) throw new ToolError(`${rel} does not exist`)
      const lines = []
      for (const entry of readdirSync(abs, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
        const child = path.join(abs, entry.name)
        if (entry.isFile() && entry.name.endsWith('.jsonc')) {
          lines.push(`${path.posix.join(rel.replace(/\\/g, '/'), entry.name)}  (jsonc case definition)`)
        } else if (entry.isDirectory() && existsSync(path.join(child, 'system'))) {
          const times = readdirSync(child).filter((n) => /^\d+(\.\d+)?$/.test(n) && n !== '0')
          const mesh = existsSync(path.join(child, 'constant', 'polyMesh')) ? 'meshed' : 'no polyMesh'
          const solved = times.length ? `results at ${times.sort().join(', ')}` : 'no results'
          lines.push(`${path.posix.join(rel.replace(/\\/g, '/'), entry.name)}/  (case directory, ${mesh}, ${solved})`)
        }
      }
      return { text: lines.length ? lines.join('\n') : `${rel} holds no cases`, isError: false }
    },
  },
  {
    name: 'ofgpu_generate_mesh',
    title: 'Generate a mesh',
    description:
      'Write a complete, ready-to-run case: constant/polyMesh, the transport and turbulence dictionaries, system/, and a 0/ directory. With `stl` the geometry is cut out of the block (cut-cell), which is how external-aerodynamics cases are built. Cut-cell meshing is CPU-bound and a 128-cube can take 10-20 minutes.',
    inputSchema: {
      type: 'object',
      properties: {
        preset: { type: 'string', enum: MESH_PRESETS, description: 'Base geometry. `big` is a unit cube tunnel and takes a single cell count.' },
        outputDir: { type: 'string', description: 'Workspace-relative directory to write the case into. Must not already exist.' },
        cells: {
          type: 'array',
          items: { type: 'integer', minimum: 1 },
          minItems: 1,
          maxItems: 3,
          description: 'Cell counts: [n] for the `big` preset (n^3), otherwise [nx, ny, nz].',
        },
        stl: { type: 'string', description: 'Workspace-relative STL to carve out of the block. Must be a closed manifold (SPEC-LIT §23.2).' },
        stlPatchName: { type: 'string', description: 'Wall patch name for the STL surface. Default: the STL file stem.' },
        cutCell: { type: 'boolean', description: 'Cut cells at the STL surface rather than stair-stepping. Default true when stl is given.' },
        wallModel: { type: 'string', enum: ['standard', 'spalding', 'rough', 'lowRe'], description: 'Wall treatment preset (SPEC-LIT §29.1).' },
        cyclic: { type: 'string', enum: ['x', 'y', 'z'], description: 'Make this axis periodic.' },
      },
      required: ['preset', 'outputDir'],
      additionalProperties: false,
    },
    async run(args) {
      const out = insideWorkspace(args.outputDir, 'outputDir')
      if (existsSync(out)) throw new ToolError(`${args.outputDir} already exists; the mesher will not overwrite a case`)
      const argv = [args.preset, path.relative(WORKSPACE, out).replace(/\\/g, '/')]
      if (args.cells) argv.push(...args.cells.map(String))
      if (args.stl) {
        const stl = insideWorkspace(args.stl, 'stl')
        if (!existsSync(stl)) throw new ToolError(`${args.stl} does not exist`)
        const name = args.stlPatchName ?? path.basename(stl).replace(/\.stl$/i, '')
        argv.push('-stl', `${name}=${path.relative(WORKSPACE, stl).replace(/\\/g, '/')}`)
        if (args.cutCell !== false) argv.push('-cutcell')
      } else if (args.cutCell) {
        throw new ToolError('cutCell needs an stl to cut against')
      }
      if (args.wallModel) argv.push('-wallModel', args.wallModel)
      if (args.cyclic) argv.push('-cyclic', args.cyclic)
      return reportOf(`ofgpu-generate-mesh ${argv.join(' ')}`, await runBinary('ofgpu-generate-mesh', argv))
    },
  },
  {
    name: 'ofgpu_solve',
    title: 'Run a solver',
    description:
      'Run one of the GPU-resident solvers on a case directory or a *.jsonc case, and return the tail of its log: residuals at each check interval, the wall time, and where the results were written. The whole time-integration loop stays on the device, so a 2-million-cell steady RANS case is seconds to a minute, not hours. `k-epsilon`, `k-omega` and `sa` solve the turbulence equations ONLY, on a velocity field they never touch: no U and no p are written and what comes back is the initial field. A case that needs a velocity field must use `lowmach`, `buoyant`, `plume` or `vof`.',
    inputSchema: {
      type: 'object',
      properties: {
        solver: { type: 'string', enum: Object.keys(SOLVERS), description: 'Which driver to run. `k-epsilon`, `k-omega` and `sa` solve turbulence only on a frozen velocity field; for a velocity field use `lowmach`, `buoyant`, `plume` or `vof`.' },
        casePath: { type: 'string', description: 'Workspace-relative case directory or *.jsonc file.' },
        iterations: { type: 'integer', minimum: 1, description: 'Iterations (or time steps) to run.' },
        checkEvery: { type: 'integer', minimum: 1, description: 'How often to print residuals. Default: iterations / 10.' },
        output: { type: 'string', enum: ['foam', 'vtu', 'nvdb', 'vdb', 'usda'], description: 'Result format. Default foam (OpenFOAM ASCII).' },
      },
      required: ['solver', 'casePath', 'iterations'],
      additionalProperties: false,
    },
    async run(args) {
      const bin = SOLVERS[args.solver]
      if (!bin) throw new ToolError(`unknown solver ${args.solver}; one of ${Object.keys(SOLVERS).join(', ')}`)
      const abs = insideWorkspace(args.casePath, 'casePath')
      if (!existsSync(abs)) throw new ToolError(`${args.casePath} does not exist`)
      const rel = path.relative(WORKSPACE, abs).replace(/\\/g, '/')
      const check = args.checkEvery ?? Math.max(1, Math.floor(args.iterations / 10))
      const argv = [rel, '-iters', String(args.iterations), '-check', String(check)]
      if (args.output) argv.push('-output', args.output)
      return reportOf(`${bin} ${argv.join(' ')}`, await runBinary(bin, argv))
    },
  },
  {
    name: 'ofgpu_validate',
    title: 'Run the validation suite',
    description:
      'Run ofgpu-validate: manufactured solutions, analytic solutions and published benchmarks. It never compares against another CFD code (SPEC-LIT §10, §22). Use it to check a build, not to check a case.',
    inputSchema: {
      type: 'object',
      properties: { filter: { type: 'string', description: 'Only run checks whose name contains this.' } },
      additionalProperties: false,
    },
    readOnly: true,
    async run(args) {
      const argv = args.filter ? [args.filter] : []
      return reportOf(`ofgpu-validate ${argv.join(' ')}`, await runBinary('ofgpu-validate', argv))
    },
  },
  {
    name: 'ofgpu_read_case_file',
    title: 'Read a case file',
    description:
      'Read one text file from the workspace: a case definition, a dictionary under constant/ or system/, or a field file under a time directory. Field files in a solved case are large, so this returns at most 400 lines.',
    inputSchema: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'Workspace-relative file path.' },
        maxLines: { type: 'integer', minimum: 1, maximum: 2000, description: 'Lines to return. Default 400.' },
      },
      required: ['path'],
      additionalProperties: false,
    },
    readOnly: true,
    async run(args) {
      const abs = insideWorkspace(args.path, 'path')
      if (!existsSync(abs) || !statSync(abs).isFile()) throw new ToolError(`${args.path} is not a file`)
      const max = args.maxLines ?? 400
      const lines = readFileSync(abs, 'utf8').split(/\r?\n/)
      const head = lines.slice(0, max).join('\n')
      const note = lines.length > max ? `\n\n… ${lines.length - max} more line(s); ask for a larger maxLines to see them.` : ''
      return { text: head + note, isError: false }
    },
  },
]

const BY_NAME = new Map(TOOLS.map((t) => [t.name, t]))

// -------------------------------------------------------------- protocol ---

function send(msg) {
  process.stdout.write(JSON.stringify(msg) + '\n')
}

function result(id, value) {
  send({ jsonrpc: '2.0', id, result: value })
}

function failure(id, code, message) {
  send({ jsonrpc: '2.0', id, error: { code, message } })
}

async function handle(msg) {
  const { id, method, params } = msg
  // A notification has no id and takes no reply, whatever it asks for.
  const isNotification = id === undefined || id === null

  switch (method) {
    case 'initialize':
      return result(id, {
        protocolVersion: PROTOCOL_VERSION,
        capabilities: { tools: { listChanged: false } },
        serverInfo: SERVER,
        instructions:
          `The ofgpu solvers, running on this machine's GPU. Workspace: ${WORKSPACE}. Binaries: ${BIN_DIR}.\n` +
          'Call ofgpu_probe first to confirm the GPU. Cut-cell mesh generation is CPU-bound and slow (10-20 min for a 128-cube); ' +
          'solving is GPU-resident and fast. Every path is workspace-relative and paths outside the workspace are refused.',
      })

    case 'tools/list':
      return result(id, {
        tools: TOOLS.map((t) => ({
          name: t.name,
          title: t.title,
          description: t.description,
          inputSchema: t.inputSchema,
          annotations: { readOnlyHint: Boolean(t.readOnly), destructiveHint: false, openWorldHint: false },
        })),
      })

    case 'tools/call': {
      const tool = BY_NAME.get(params?.name)
      if (!tool) return failure(id, -32602, `no such tool: ${params?.name}`)
      try {
        const r = await tool.run(params.arguments ?? {})
        return result(id, { content: [{ type: 'text', text: r.text }], isError: r.isError })
      } catch (e) {
        // A tool that refuses reports through the result, not the transport:
        // the model is supposed to read the reason and try something else.
        const why = e instanceof ToolError ? e.message : `unexpected failure: ${e?.message ?? e}`
        return result(id, { content: [{ type: 'text', text: why }], isError: true })
      }
    }

    case 'ping':
      return result(id, {})

    default:
      if (isNotification) return
      return failure(id, -32601, `method not found: ${method}`)
  }
}

const rl = readline.createInterface({ input: process.stdin })
rl.on('line', (line) => {
  const text = line.trim()
  if (!text) return
  let msg
  try {
    msg = JSON.parse(text)
  } catch {
    return send({ jsonrpc: '2.0', id: null, error: { code: -32700, message: 'parse error' } })
  }
  // One in-flight call at a time would serialise a probe behind a 20-minute
  // mesh; let the client decide what to run concurrently.
  void handle(msg).catch((e) => {
    if (msg.id !== undefined && msg.id !== null) failure(msg.id, -32603, String(e?.message ?? e))
  })
})
rl.on('close', () => process.exit(0))

export { TOOLS, insideWorkspace, tail, WORKSPACE, BIN_DIR }
