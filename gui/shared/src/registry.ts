// The catalogue of everything the existing program exposes: its binaries and
// their command lines, the turbulence models and which driver builds each, the
// wall treatments, algorithms, mesh presets, patch kinds and output formats.
// Facts come from rust/Cargo.toml, rust/src/bin/*.rs (usage() / const USAGE),
// rust/src/bin/common/mod.rs (driver_for, json case handling) and
// docs/schema/case-1.json. A vitest sync test on the server keeps this in step
// with the Rust sources.

export type ArgType = 'int' | 'float' | 'string' | 'path' | 'enum' | 'flag' | 'list' | 'time'

export interface FlagSpec {
  /** As typed on the command line, e.g. "-iters". */
  name: string
  type: ArgType
  values?: string[]
  default?: string | number | boolean | null
  repeatable?: boolean
  description: string
  /** Another flag this one only makes sense with (e.g. "-Ks" needs "-wallModel rough"). */
  requires?: string
}

export interface PositionalSpec {
  name: string
  type: ArgType
  values?: string[]
  optional?: boolean
  description: string
}

export type CaseFormat = 'foamDir' | 'jsonc'
export type BinaryKind = 'mesh' | 'solver' | 'analysis' | 'bench' | 'diagnostic'
export type OutputFormat = 'foam' | 'vtu' | 'nvdb' | 'vdb' | 'usda'
export type UsageKind = 'usageFn' | 'constUsage' | 'inline' | 'none'

export interface BinarySpec {
  /** Cargo [[bin]] name. */
  name: string
  /** Source path under rust/, from Cargo.toml. */
  source: string
  purpose: string
  kind: BinaryKind
  positionals: PositionalSpec[]
  flags: FlagSpec[]
  /** Which case formats the driver reads. Empty for binaries that take no case. */
  accepts: CaseFormat[]
  /** Turbulence models this driver can construct. */
  builds: string[]
  residualStyle: 'kEpsilon' | 'kOmega' | 'sa' | 'plume' | 'buoyant' | 'lowmach' | 'vof' | 'datacentre' | 'generic' | 'none'
  writes: { formats: OutputFormat[]; restart: boolean; csv: boolean }
  longRunning: boolean
  /** Whether the run needs the GPU (serialised on the single-GPU queue). */
  gpu: boolean
  usageKind: UsageKind
  /** One-line summary shown in the CFD Tools panel and the system prompt. */
  summary: string
}

const OUTPUT_LIST: FlagSpec = {
  name: '-output',
  type: 'list',
  values: ['foam', 'vtu', 'nvdb', 'vdb', 'usda'],
  default: 'foam',
  description: 'Comma list of result formats. Must not be combined with an `output` block in a JSONC case.',
}
const PERMISSIVE: FlagSpec = {
  name: '-permissive',
  type: 'flag',
  description: 'Downgrade unsupported settings from an error to a warning that says what was substituted (SPEC-LIT §13.4).',
}
const ITERS: FlagSpec = { name: '-iters', type: 'int', description: 'Outer iterations (default: from the case endTime).' }
const FIXED_ITERS: FlagSpec = { name: '-fixedIters', type: 'int', description: 'Run the linear solver for exactly N sweeps with zero host transfers.' }
const CHECK: FlagSpec = { name: '-check', type: 'int', default: 25, description: 'Test convergence and print residuals every N iterations.' }
const WRITE: FlagSpec = { name: '-write', type: 'string', description: 'Time directory name to write into (default: from the case).' }
const NO_WRITE: FlagSpec = { name: '-noWrite', type: 'flag', description: 'Do not write fields (timing runs).' }
const GRAPH: FlagSpec = { name: '-graph', type: 'flag', description: 'Capture the time loop as a CUDA graph.' }
const END_TIME: FlagSpec = { name: '-endTime', type: 'time', description: 'Transient mode: physical end time in seconds.' }
const DELTA_T: FlagSpec = { name: '-deltaT', type: 'time', description: 'Time step in seconds.' }
const WRITE_INTERVAL: FlagSpec = { name: '-writeInterval', type: 'time', description: 'Seconds of physical time between writes.' }
const OUTER_ITERS: FlagSpec = { name: '-outerIters', type: 'int', description: 'PIMPLE outer correctors per time step.' }
const NO_POTENTIAL: FlagSpec = { name: '-noPotential', type: 'flag', description: 'Skip the potential-flow initial flux.' }
const INLET_PATCH: FlagSpec = { name: '-inletPatch', type: 'string', description: 'Name of the inlet patch for the potential-flow start.' }
const OUTLET_PATCH: FlagSpec = { name: '-outletPatch', type: 'string', description: 'Name of the outlet patch for the potential-flow start.' }
const RESTART_WRITE: FlagSpec = { name: '-restartWrite', type: 'int', description: 'Write a restart checkpoint every N steps.' }
const RESTART_FROM: FlagSpec = { name: '-restartFrom', type: 'path', description: 'Resume from a checkpoint file.' }

const STEADY_TURB_FLAGS = [ITERS, FIXED_ITERS, WRITE, NO_WRITE, CHECK, PERMISSIVE, OUTPUT_LIST]

export const MESH_KINDS = ['channel', 'cavity', 'step', 'big', 'plume', 'room', 'damBreak'] as const
export type MeshKind = (typeof MESH_KINDS)[number]

export interface MeshPreset {
  kind: MeshKind
  title: string
  defaultCells: [number, number, number]
  description: string
  /** Which solver runs the generated case. */
  solvers: string[]
}

export const MESH_PRESETS: MeshPreset[] = [
  { kind: 'channel', title: 'Plane channel', defaultCells: [200, 120, 1], description: '2-D plane channel graded to both walls (expansion 20), 1/7-power frozen velocity.', solvers: ['ofgpu-k-epsilon', 'ofgpu-k-omega', 'ofgpu-sa'] },
  { kind: 'cavity', title: 'Lid-driven cavity', defaultCells: [128, 128, 1], description: '2-D square cavity, four walls, stream-function recirculation.', solvers: ['ofgpu-k-epsilon', 'ofgpu-k-omega', 'ofgpu-sa'] },
  { kind: 'step', title: 'Backward-facing step box', defaultCells: [300, 100, 1], description: '2-D downstream box of a backward-facing step.', solvers: ['ofgpu-k-epsilon', 'ofgpu-k-omega', 'ofgpu-sa'] },
  { kind: 'big', title: 'Uniform benchmark cube', defaultCells: [160, 160, 160], description: 'Uniform n^3 box for benchmarks (second form: `big <dir> [n]`).', solvers: ['ofgpu-k-epsilon', 'ofgpu-bench'] },
  { kind: 'plume', title: 'Buoyant plume', defaultCells: [98, 42, 20], description: '3-D room with a hot floor inlet window and an x-max outlet.', solvers: ['ofgpu-plume', 'ofgpu-buoyant'] },
  { kind: 'room', title: 'Ventilated room', defaultCells: [60, 40, 30], description: '3-D room with a supply inlet and an outlet (data-centre style).', solvers: ['ofgpu-buoyant'] },
  { kind: 'damBreak', title: 'Dam break (VOF)', defaultCells: [150, 90, 1], description: '2-D collapsing water column (Martin & Moyce 1952).', solvers: ['ofgpu-vof'] },
]

export const BINARIES: BinarySpec[] = [
  {
    name: 'ofgpu-generate-mesh',
    source: 'src/bin/generate_mesh.rs',
    purpose: 'Write a complete, ready-to-run OpenFOAM-format test case: structured block mesh, initial fields and system dictionaries.',
    summary: 'Generate a structured mesh + case directory from a preset.',
    kind: 'mesh',
    positionals: [
      { name: 'kind', type: 'enum', values: [...MESH_KINDS], description: 'Case preset.' },
      { name: 'outputDir', type: 'path', description: 'Directory to create.' },
      { name: 'nx', type: 'int', optional: true, description: 'Cells in x (for `big` a single n means n^3).' },
      { name: 'ny', type: 'int', optional: true, description: 'Cells in y.' },
      { name: 'nz', type: 'int', optional: true, description: 'Cells in z.' },
    ],
    flags: [
      { name: '-stl', type: 'string', repeatable: true, description: '`[name=]path` — carve the block against a closed STL surface (castellated).' },
      { name: '-cutcell', type: 'flag', description: 'Embedded-boundary cut cells instead of castellation (needs -stl).' },
      { name: '-s', type: 'int', default: 16, description: 'Cut-cell supersample lattice size.', requires: '-cutcell' },
      { name: '-thetaMin', type: 'float', default: 0.2, description: 'Small-cell merge threshold.', requires: '-cutcell' },
      { name: '-wallModel', type: 'enum', values: ['standard', 'spalding', 'rough', 'lowRe'], description: 'Wall-treatment preset written into 0/ (SPEC-LIT §29.1).' },
      { name: '-Ks', type: 'float', description: 'Sand-grain roughness height in m.', requires: '-wallModel' },
      { name: '-Cs', type: 'float', default: 0.5, description: 'Roughness constant.', requires: '-wallModel' },
      { name: '-cyclic', type: 'enum', values: ['x', 'y', 'z'], repeatable: true, description: 'Make the two faces of this axis a cyclic pair.' },
      PERMISSIVE,
    ],
    accepts: [],
    builds: [],
    residualStyle: 'none',
    writes: { formats: ['foam'], restart: false, csv: false },
    longRunning: false,
    gpu: false,
    usageKind: 'usageFn',
  },
  {
    name: 'ofgpu-k-epsilon',
    source: 'src/bin/k_epsilon.rs',
    purpose: 'Solve the k and epsilon transport equations on a frozen velocity field (standard, realizable or RNG k-epsilon, chosen by the case).',
    summary: 'k-epsilon family on a frozen U (JSONC or OpenFOAM case).',
    kind: 'solver',
    positionals: [{ name: 'case', type: 'path', description: 'OpenFOAM case directory or a .jsonc case file.' }],
    flags: STEADY_TURB_FLAGS,
    accepts: ['foamDir', 'jsonc'],
    builds: ['kEpsilon', 'realizableKE', 'RNGkEpsilon', 'laminar'],
    residualStyle: 'kEpsilon',
    writes: { formats: ['foam', 'vtu', 'nvdb', 'vdb', 'usda'], restart: false, csv: false },
    longRunning: true,
    gpu: true,
    usageKind: 'usageFn',
  },
  {
    name: 'ofgpu-k-omega',
    source: 'src/bin/k_omega.rs',
    purpose: 'Solve k and omega (Wilcox k-omega or Menter SST) on a frozen velocity field.',
    summary: 'k-omega / k-omega SST on a frozen U (OpenFOAM case directory).',
    kind: 'solver',
    positionals: [{ name: 'caseDir', type: 'path', description: 'OpenFOAM case directory.' }],
    flags: STEADY_TURB_FLAGS,
    accepts: ['foamDir'],
    builds: ['kOmega', 'kOmegaSST'],
    residualStyle: 'kOmega',
    writes: { formats: ['foam', 'vtu', 'nvdb', 'vdb', 'usda'], restart: false, csv: false },
    longRunning: true,
    gpu: true,
    usageKind: 'usageFn',
  },
  {
    name: 'ofgpu-sa',
    source: 'src/bin/sa.rs',
    purpose: 'Solve the Spalart-Allmaras nuTilda equation (and its DES/DDES/IDDES hybrids) on a frozen velocity field.',
    summary: 'Spalart-Allmaras (+ SA-background DES) on a frozen U.',
    kind: 'solver',
    positionals: [{ name: 'caseDir', type: 'path', description: 'OpenFOAM case directory.' }],
    flags: STEADY_TURB_FLAGS,
    accepts: ['foamDir'],
    builds: ['SpalartAllmaras', 'SpalartAllmarasDES', 'SpalartAllmarasDDES', 'SpalartAllmarasIDDES'],
    residualStyle: 'sa',
    writes: { formats: ['foam', 'vtu', 'nvdb', 'vdb', 'usda'], restart: false, csv: false },
    longRunning: true,
    gpu: true,
    usageKind: 'usageFn',
  },
  {
    name: 'ofgpu-plume',
    source: 'src/bin/plume.rs',
    purpose: 'Buoyant plume solver: momentum, pressure, temperature and k-epsilon with buoyancy production, steady or transient.',
    summary: 'Buoyant plume (U, p, T, k, epsilon) — OpenFOAM case directory.',
    kind: 'solver',
    positionals: [{ name: 'caseDir', type: 'path', description: 'OpenFOAM case directory.' }],
    flags: [ITERS, FIXED_ITERS, CHECK, WRITE, NO_WRITE, GRAPH, END_TIME, DELTA_T, WRITE_INTERVAL, OUTER_ITERS, NO_POTENTIAL, INLET_PATCH, OUTLET_PATCH, PERMISSIVE],
    accepts: ['foamDir'],
    builds: ['kEpsilon'],
    residualStyle: 'plume',
    writes: { formats: ['foam'], restart: false, csv: false },
    longRunning: true,
    gpu: true,
    usageKind: 'usageFn',
  },
  {
    name: 'ofgpu-buoyant',
    source: 'src/bin/buoyant.rs',
    purpose: 'General buoyant SIMPLE/PIMPLE solver with the coupled turbulence registry (SST, LM transition, LES, DES), pressure backends and restart.',
    summary: 'Coupled buoyant solver (SIMPLE/PIMPLE, any registry model, restart).',
    kind: 'solver',
    positionals: [{ name: 'caseDir', type: 'path', description: 'OpenFOAM case directory.' }],
    flags: [
      ITERS, FIXED_ITERS, CHECK, WRITE, NO_WRITE, GRAPH, END_TIME, DELTA_T, WRITE_INTERVAL, OUTER_ITERS,
      { name: '-nCorrectors', type: 'int', description: 'PISO/PIMPLE pressure correctors per outer iteration.' },
      { name: '-backend', type: 'enum', values: ['auto', 'pbicgstab', 'fft', 'amgx'], default: 'auto', description: 'Pressure linear backend.' },
      { name: '-probe', type: 'flag', description: 'Print per-layer probes.' },
      NO_POTENTIAL, INLET_PATCH, OUTLET_PATCH, OUTPUT_LIST, RESTART_WRITE, RESTART_FROM, PERMISSIVE,
    ],
    accepts: ['foamDir'],
    builds: ['kEpsilon', 'realizableKE', 'RNGkEpsilon', 'kOmega', 'kOmegaSST', 'kOmegaSSTLM', 'LaunderSharmaKE', 'Smagorinsky', 'WALE', 'kEqn', 'SSTDES', 'SSTDDES', 'SSTIDDES', 'laminar'],
    residualStyle: 'buoyant',
    writes: { formats: ['foam', 'vtu', 'nvdb', 'vdb', 'usda'], restart: true, csv: false },
    longRunning: true,
    gpu: true,
    usageKind: 'usageFn',
  },
  {
    name: 'ofgpu-vof',
    source: 'src/bin/vof.rs',
    purpose: 'Two-phase VOF solver (interface compression, Zalesak FCT, CSF surface tension, contact angles) with adaptive time stepping.',
    summary: 'Two-phase VOF (dam break etc.), adaptive dt.',
    kind: 'solver',
    positionals: [{ name: 'caseDir', type: 'path', description: 'OpenFOAM case directory.' }],
    flags: [
      END_TIME, DELTA_T,
      { name: '-maxCo', type: 'float', description: 'Courant ceiling for the adaptive step.' },
      { name: '-maxDeltaT', type: 'time', description: 'Upper bound on the adaptive time step.' },
      WRITE_INTERVAL, NO_WRITE,
      { name: '-surge', type: 'flag', description: 'Report the Martin & Moyce surge-front position.' },
      { name: '-a', type: 'float', description: 'Initial column width (m).' },
      OUTPUT_LIST, RESTART_WRITE, RESTART_FROM,
      { name: '-reportEvery', type: 'int', description: 'Print a step line every N steps.' },
      PERMISSIVE,
    ],
    accepts: ['foamDir'],
    builds: ['laminar'],
    residualStyle: 'vof',
    writes: { formats: ['foam', 'vtu', 'nvdb', 'vdb', 'usda'], restart: true, csv: false },
    longRunning: true,
    gpu: true,
    usageKind: 'usageFn',
  },
  {
    name: 'ofgpu-lowmach',
    source: 'src/bin/lowmach.rs',
    purpose: 'Low-Mach variable-density solver (SPEC-LIT §25/§26): sealed or open enclosures, heaters, species, wall heat transfer.',
    summary: 'Low-Mach variable-density solver (JSONC case).',
    kind: 'solver',
    positionals: [{ name: 'case', type: 'path', description: 'A .jsonc case file (or OpenFOAM directory).' }],
    flags: [
      ITERS, CHECK, END_TIME, DELTA_T,
      { name: '-sealed', type: 'flag', description: 'Closed enclosure: thermodynamic pressure p0 evolves.' },
      { name: '-p0', type: 'float', description: 'Initial thermodynamic pressure in Pa.' },
      { name: '-heaterPower', type: 'float', description: 'Heater power in W.' },
      OUTPUT_LIST, WRITE_INTERVAL, RESTART_WRITE, RESTART_FROM, PERMISSIVE,
    ],
    accepts: ['jsonc', 'foamDir'],
    builds: ['kEpsilon', 'realizableKE', 'RNGkEpsilon', 'kOmega', 'kOmegaSST', 'kOmegaSSTLM', 'LaunderSharmaKE', 'Smagorinsky', 'WALE', 'kEqn', 'SSTDES', 'SSTDDES', 'SSTIDDES', 'laminar'],
    residualStyle: 'lowmach',
    writes: { formats: ['foam', 'vtu', 'nvdb', 'vdb', 'usda'], restart: true, csv: false },
    longRunning: true,
    gpu: true,
    usageKind: 'usageFn',
  },
  {
    name: 'ofgpu-cht',
    source: 'src/bin/cht.rs',
    purpose: 'Multi-region conduction with conjugate interfaces, and conjugate natural convection in a closed cavity (SPEC-LIT §46/§47, §59/§60).',
    summary: 'Conjugate heat transfer (JSONC case), optional CSV.',
    kind: 'solver',
    positionals: [{ name: 'case', type: 'path', description: 'A .jsonc CHT case file.' }],
    flags: [{ name: '-csv', type: 'path', description: 'Write interface/region results to this CSV.' }],
    accepts: ['jsonc'],
    builds: ['laminar'],
    residualStyle: 'generic',
    writes: { formats: [], restart: false, csv: true },
    longRunning: true,
    gpu: true,
    usageKind: 'constUsage',
  },
  {
    name: 'ofgpu-datacentre',
    source: 'src/bin/datacentre.rs',
    purpose: 'Data-centre airflow: fans with performance curves, porous jumps, humid air, RCI/RTI/SHI/RHI metrics.',
    summary: 'Data-centre airflow with fans and rack metrics (JSONC case).',
    kind: 'solver',
    positionals: [{ name: 'case', type: 'path', description: 'A .jsonc data-centre case file.' }],
    flags: [{ name: '-csv', type: 'path', description: 'Write rack/fan metrics to this CSV.' }, PERMISSIVE],
    accepts: ['jsonc'],
    builds: ['kEpsilon'],
    residualStyle: 'datacentre',
    writes: { formats: ['foam'], restart: false, csv: true },
    longRunning: true,
    gpu: true,
    usageKind: 'constUsage',
  },
  {
    name: 'ofgpu-decompose',
    source: 'src/bin/decompose.rs',
    purpose: 'Partition a mesh (Hilbert, linear, round-robin) and verify partition-invariant reductions and distributed solves on one GPU.',
    summary: 'Mesh decomposition diagnostics.',
    kind: 'analysis',
    positionals: [{ name: 'case', type: 'path', description: 'OpenFOAM case directory or .jsonc case.' }],
    flags: [
      { name: '-parts', type: 'list', default: '2,3,4', description: 'Part counts to try.' },
      { name: '-method', type: 'enum', values: ['hilbert', 'linear', 'roundrobin', 'all'], default: 'hilbert', description: 'Partitioner.' },
      { name: '-sweeps', type: 'int', default: 8, description: 'Jacobi sweeps per run.' },
      { name: '-alpha', type: 'float', default: 0.5, description: 'Under-relaxation factor.' },
      { name: '-quiet', type: 'flag', description: 'Print only the verdict.' },
    ],
    accepts: ['foamDir', 'jsonc'],
    builds: [],
    residualStyle: 'none',
    writes: { formats: [], restart: false, csv: false },
    longRunning: false,
    gpu: true,
    usageKind: 'constUsage',
  },
  {
    name: 'ofgpu-validate',
    source: 'src/bin/validate.rs',
    purpose: 'Run every validation gate (MMS, analytic solutions, published benchmarks) and print the pass/miss/open registry.',
    summary: 'Full validation suite (900+ gates, long).',
    kind: 'diagnostic',
    positionals: [],
    flags: [],
    accepts: [],
    builds: [],
    residualStyle: 'none',
    writes: { formats: [], restart: false, csv: false },
    longRunning: true,
    gpu: true,
    usageKind: 'none',
  },
  {
    name: 'ofgpu-bench',
    source: 'src/bin/bench.rs',
    purpose: 'Time the turbulence closure on a uniform box of the given size.',
    summary: 'Solver throughput benchmark on an nx ny nz box.',
    kind: 'bench',
    positionals: [
      { name: 'nx', type: 'int', description: 'Cells in x.' },
      { name: 'ny', type: 'int', description: 'Cells in y.' },
      { name: 'nz', type: 'int', description: 'Cells in z.' },
    ],
    flags: [ITERS, FIXED_ITERS, { name: '-model', type: 'string', description: 'Turbulence model name to benchmark.' }],
    accepts: [],
    builds: [],
    residualStyle: 'none',
    writes: { formats: [], restart: false, csv: false },
    longRunning: true,
    gpu: true,
    usageKind: 'none',
  },
  {
    name: 'ofgpu-graph-bench',
    source: 'src/bin/graph_bench.rs',
    purpose: 'Compare eager kernel launches against a captured CUDA graph.',
    summary: 'CUDA-graph launch benchmark.',
    kind: 'bench',
    positionals: [],
    flags: [],
    accepts: [],
    builds: [],
    residualStyle: 'none',
    writes: { formats: [], restart: false, csv: false },
    longRunning: true,
    gpu: true,
    usageKind: 'inline',
  },
  {
    name: 'ofgpu-dispatch-bench',
    source: 'src/bin/dispatch_bench.rs',
    purpose: 'Measure kernel dispatch overhead.',
    summary: 'Kernel dispatch benchmark.',
    kind: 'bench',
    positionals: [],
    flags: [],
    accepts: [],
    builds: [],
    residualStyle: 'none',
    writes: { formats: [], restart: false, csv: false },
    longRunning: false,
    gpu: true,
    usageKind: 'none',
  },
  {
    name: 'ofgpu-probe',
    source: 'src/bin/probe.rs',
    purpose: 'Toolchain vertical slice: load a kernel, run it, compare bit-for-bit. Prints the device name.',
    summary: 'GPU self-test (device name, precision).',
    kind: 'diagnostic',
    positionals: [],
    flags: [],
    accepts: [],
    builds: [],
    residualStyle: 'none',
    writes: { formats: [], restart: false, csv: false },
    longRunning: false,
    gpu: true,
    usageKind: 'none',
  },
]

export const BINARY_NAMES = BINARIES.map((b) => b.name)

export function getBinary(name: string): BinarySpec | undefined {
  return BINARIES.find((b) => b.name === name)
}

// ---------------------------------------------------------------------------
// Models and the pick-lists
// ---------------------------------------------------------------------------

export type ModelFamily = 'RAS' | 'LES' | 'DES' | 'transition' | 'laminar'

export interface ModelSpec {
  name: string
  family: ModelFamily
  title: string
  equations: string[]
  /** Drivers that build this model, in preference order. */
  drivers: string[]
  /** JSONC `turbulence.kind` this model needs. */
  kind: 'RAS' | 'LES'
  /** Extra requirements the validator checks. */
  requires: { wallTreatment?: 'lowRe'; initial?: string[] }
  specRef: string
  notes: string
}

export const MODELS: ModelSpec[] = [
  { name: 'kEpsilon', family: 'RAS', title: 'Standard k-epsilon', equations: ['k', 'epsilon'], drivers: ['ofgpu-k-epsilon', 'ofgpu-plume', 'ofgpu-buoyant', 'ofgpu-lowmach', 'ofgpu-datacentre'], kind: 'RAS', requires: { initial: ['k', 'epsilon'] }, specRef: '§6.1', notes: 'Launder & Spalding 1974 coefficients.' },
  { name: 'realizableKE', family: 'RAS', title: 'Realizable k-epsilon', equations: ['k', 'epsilon'], drivers: ['ofgpu-k-epsilon', 'ofgpu-buoyant', 'ofgpu-lowmach'], kind: 'RAS', requires: { initial: ['k', 'epsilon'] }, specRef: '§40', notes: 'C_mu is a field; refuses standard-model coefficients like C1.' },
  { name: 'RNGkEpsilon', family: 'RAS', title: 'RNG k-epsilon', equations: ['k', 'epsilon'], drivers: ['ofgpu-k-epsilon', 'ofgpu-buoyant', 'ofgpu-lowmach'], kind: 'RAS', requires: { initial: ['k', 'epsilon'] }, specRef: '§41', notes: 'Diffusivity alpha (nu + nu_t).' },
  { name: 'LaunderSharmaKE', family: 'RAS', title: 'Launder-Sharma low-Re k-epsilon', equations: ['k', 'epsilon'], drivers: ['ofgpu-buoyant', 'ofgpu-lowmach'], kind: 'RAS', requires: { wallTreatment: 'lowRe', initial: ['k', 'epsilon'] }, specRef: '§33', notes: 'Needs wallTreatment lowRe and a wall-resolving mesh.' },
  { name: 'kOmega', family: 'RAS', title: 'Wilcox k-omega', equations: ['k', 'omega'], drivers: ['ofgpu-k-omega', 'ofgpu-buoyant', 'ofgpu-lowmach'], kind: 'RAS', requires: { initial: ['k', 'omega'] }, specRef: '§6.2', notes: '' },
  { name: 'kOmegaSST', family: 'RAS', title: 'Menter k-omega SST', equations: ['k', 'omega'], drivers: ['ofgpu-k-omega', 'ofgpu-buoyant', 'ofgpu-lowmach'], kind: 'RAS', requires: { initial: ['k', 'omega'] }, specRef: '§6.3', notes: 'Blending functions F1/F2.' },
  { name: 'kOmegaSSTLM', family: 'transition', title: 'k-omega SST-LM (Langtry-Menter transition)', equations: ['k', 'omega', 'ReThetat', 'gammaInt'], drivers: ['ofgpu-buoyant', 'ofgpu-lowmach'], kind: 'RAS', requires: { initial: ['k', 'omega'] }, specRef: '§88/§89', notes: 'Not reachable from ofgpu-k-omega (would run fully turbulent).' },
  { name: 'SpalartAllmaras', family: 'RAS', title: 'Spalart-Allmaras', equations: ['nuTilda'], drivers: ['ofgpu-sa'], kind: 'RAS', requires: { initial: ['nuTilda'] }, specRef: '§56', notes: 'Refused in buoyant solvers (no k equation for G_b).' },
  { name: 'SpalartAllmarasDES', family: 'DES', title: 'SA-DES97', equations: ['nuTilda'], drivers: ['ofgpu-sa'], kind: 'RAS', requires: { initial: ['nuTilda'] }, specRef: '§57', notes: '' },
  { name: 'SpalartAllmarasDDES', family: 'DES', title: 'SA-DDES', equations: ['nuTilda'], drivers: ['ofgpu-sa'], kind: 'RAS', requires: { initial: ['nuTilda'] }, specRef: '§57', notes: '' },
  { name: 'SpalartAllmarasIDDES', family: 'DES', title: 'SA-IDDES', equations: ['nuTilda'], drivers: ['ofgpu-sa'], kind: 'RAS', requires: { initial: ['nuTilda'] }, specRef: '§57', notes: '' },
  { name: 'SSTDES', family: 'DES', title: 'SST-DES', equations: ['k', 'omega'], drivers: ['ofgpu-buoyant', 'ofgpu-lowmach'], kind: 'RAS', requires: { initial: ['k', 'omega'] }, specRef: '§57', notes: '' },
  { name: 'SSTDDES', family: 'DES', title: 'SST-DDES', equations: ['k', 'omega'], drivers: ['ofgpu-buoyant', 'ofgpu-lowmach'], kind: 'RAS', requires: { initial: ['k', 'omega'] }, specRef: '§57', notes: '' },
  { name: 'SSTIDDES', family: 'DES', title: 'SST-IDDES', equations: ['k', 'omega'], drivers: ['ofgpu-buoyant', 'ofgpu-lowmach'], kind: 'RAS', requires: { initial: ['k', 'omega'] }, specRef: '§57', notes: '' },
  { name: 'Smagorinsky', family: 'LES', title: 'Smagorinsky LES', equations: [], drivers: ['ofgpu-buoyant', 'ofgpu-lowmach'], kind: 'LES', requires: {}, specRef: '§30', notes: 'Werner-Wengle wall model by default.' },
  { name: 'WALE', family: 'LES', title: 'WALE LES', equations: [], drivers: ['ofgpu-buoyant', 'ofgpu-lowmach'], kind: 'LES', requires: {}, specRef: '§30', notes: '' },
  { name: 'kEqn', family: 'LES', title: 'Deardorff one-equation LES', equations: ['k'], drivers: ['ofgpu-buoyant', 'ofgpu-lowmach'], kind: 'LES', requires: { initial: ['k'] }, specRef: '§30', notes: '' },
  { name: 'laminar', family: 'laminar', title: 'Laminar (no closure)', equations: [], drivers: ['ofgpu-k-epsilon', 'ofgpu-buoyant', 'ofgpu-lowmach', 'ofgpu-vof', 'ofgpu-cht'], kind: 'RAS', requires: {}, specRef: '§5', notes: 'nu_t frozen at zero.' },
]

export function getModel(name: string): ModelSpec | undefined {
  return MODELS.find((m) => m.name === name)
}

/** Static pick-lists. The server overwrites the enum-backed ones from docs/schema/case-1.json at startup. */
export const PICK_LISTS = {
  turbulenceKinds: ['RAS', 'LES'],
  wallTreatments: ['standard', 'spalding', 'rough', 'lowRe'],
  algorithms: ['SIMPLE', 'PISO', 'PIMPLE'],
  meshKinds: ['cartesian'],
  patchKinds: ['wall', 'inlet', 'open', 'empty', 'symmetry'],
  buoyancy: ['densityRatio', 'boussinesq'],
  ddt: ['steadyState', 'Euler', 'backward', 'CrankNicolson', 'localEuler'],
  exactFormats: ['foam', 'vtu'],
  visualisationFormats: ['vdb', 'nvdb'],
  sceneFormats: ['usda'],
  linearSolvers: ['PBiCGStab', 'PCG'],
  preconditioners: ['diagonal', 'DIC', 'DILU', 'Jacobi', 'none'],
  divSchemes: ['Gauss upwind', 'Gauss linear', 'Gauss linearUpwind grad(U)', 'Gauss cubic', 'Gauss QUICK', 'Gauss Gamma', 'Gauss limitedLinear 1', 'Gauss vanLeer', 'Gauss Minmod', 'bounded Gauss upwind'],
}

/** Where a JSONC case's output goes: `<stem>_jsonc/` next to the file (bin/common/mod.rs). */
export function jsonCaseOutputDir(casePath: string): string {
  const slash = Math.max(casePath.lastIndexOf('/'), casePath.lastIndexOf('\\'))
  const dir = slash >= 0 ? casePath.slice(0, slash + 1) : ''
  const file = slash >= 0 ? casePath.slice(slash + 1) : casePath
  const dot = file.lastIndexOf('.')
  const stem = dot > 0 ? file.slice(0, dot) : file
  return `${dir}${stem}_jsonc`
}

export function isJsonCase(path: string): boolean {
  return /\.jsonc?$/i.test(path)
}

/** Drivers that can run a given model on a given case format, in preference order. */
export function driversFor(model: string | null, format: CaseFormat): string[] {
  const out: string[] = []
  for (const b of BINARIES) {
    if (b.kind !== 'solver') continue
    if (!b.accepts.includes(format)) continue
    if (model && !b.builds.includes(model)) continue
    out.push(b.name)
  }
  return out
}

/** A number, or a string that is entirely a number. Anything else is NaN. */
function numericValue(value: unknown): number {
  if (typeof value === 'number') return value
  if (typeof value === 'string' && value.trim() !== '') return Number(value.trim())
  return Number.NaN
}

/** Validate one command-line value against a flag or positional spec. Returns an error message or null. */
export function checkArgValue(spec: FlagSpec | PositionalSpec, value: unknown): string | null {
  switch (spec.type) {
    case 'flag':
      return value === true || value === null || value === undefined ? null : `${spec.name} takes no value`
    // Number(null) is 0, Number(true) is 1 and Number('') is 0, so the old
    // Number(value) test let null, true and the empty string through and handed
    // the driver the literal strings "null", "true" and "".
    case 'int':
      return Number.isInteger(numericValue(value)) ? null : `${spec.name} expects an integer, got ${JSON.stringify(value)}`
    case 'float':
    case 'time':
      return Number.isFinite(numericValue(value)) ? null : `${spec.name} expects a number, got ${JSON.stringify(value)}`
    case 'enum':
      return spec.values && spec.values.includes(String(value)) ? null : `${spec.name} must be one of ${spec.values?.join(', ')}`
    case 'list':
      if (typeof value !== 'string') return `${spec.name} expects a comma list`
      if (spec.values) {
        const bad = value.split(',').map((s) => s.trim()).filter((s) => !spec.values!.includes(s))
        if (bad.length) return `${spec.name}: unknown ${bad.join(', ')} (available: ${spec.values.join(', ')})`
      }
      return null
    case 'string':
    case 'path':
      return typeof value === 'string' && value.length > 0 ? null : `${spec.name} expects a string`
  }
}
