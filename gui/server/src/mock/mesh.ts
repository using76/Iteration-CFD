// Mock `ofgpu-generate-mesh`: the blockgen presets (rust/src/blockgen.rs
// case_block_spec) as CartesianSpecs, written as a complete OpenFOAM case
// through the shared format writers, with the same patch names and 0/ files
// the real generator produces.
import fsp from 'node:fs/promises'
import path from 'node:path'
import { MESH_PRESETS, type MeshKind } from '@cfd/shared'
import { buildCartesianGrid, cartesianToPolyMesh, type BoxFace, type CartesianGrid, type CartesianSpec } from '../formats/cartesian.js'
import { writeFoamField, type FoamPatchOut, type FoamWriteOptions } from '../formats/foam.js'
import { writePolyMesh } from '../formats/polymesh.js'
import { analyticFields, uniformGrid } from './fields.js'
import { g } from './format.js'

export const DAM_BREAK_A = 0.05
export const PLUME_T_INLET = 1173.15
export const ROOM_T_INLET = 573.15
const FACES: BoxFace[] = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']

export interface PatchDesc {
  name: string
  /** wall | patch | empty | cyclic | symmetry */
  type: string
}

export interface Preset {
  kind: MeshKind
  spec: CartesianSpec
  patches: PatchDesc[]
  uRef: number
  nu: number
  thermal: boolean
  vof: boolean
  tInlet: number
}

export function defaultCells(kind: MeshKind): [number, number, number] {
  const p = MESH_PRESETS.find((m) => m.kind === kind)
  return p ? [...p.defaultCells] : [40, 20, 20]
}

export function presetSpec(kind: MeshKind, cells: [number, number, number], opts: { cyclic?: string[] } = {}): Preset {
  const [nx, ny, nz] = cells
  let bounds: CartesianSpec['bounds']
  let names: string[]
  let types: string[]
  let grading: CartesianSpec['grading'] = null
  const regions: CartesianSpec['regions'] = []
  let uRef = 1
  let nu = 1.5e-5
  let thermal = false
  let vof = false
  let tInlet = PLUME_T_INLET
  switch (kind) {
    case 'channel':
      bounds = { min: [0, -1, 0], max: [4, 1, 0.1] }
      grading = { y: { expansion: 20, twoSided: true } }
      names = ['inlet', 'outlet', 'bottomWall', 'topWall', 'back', 'front']
      types = ['patch', 'patch', 'wall', 'wall', 'empty', 'empty']
      break
    case 'cavity':
      bounds = { min: [0, 0, 0], max: [0.1, 0.1, 0.1] }
      names = ['leftWall', 'rightWall', 'fixedWall', 'movingWall', 'back', 'front']
      types = ['wall', 'wall', 'wall', 'wall', 'empty', 'empty']
      nu = 1e-5
      break
    case 'step':
      bounds = { min: [0, 0, 0], max: [30, 2, 1] }
      grading = { x: { expansion: 4, twoSided: false }, y: { expansion: 10, twoSided: true } }
      names = ['inlet', 'outlet', 'lowerWall', 'upperWall', 'back', 'front']
      types = ['patch', 'patch', 'wall', 'wall', 'empty', 'empty']
      break
    case 'big':
      bounds = { min: [0, 0, 0], max: [1, 1, 1] }
      names = ['inlet', 'outlet', 'bottomWall', 'topWall', 'backWall', 'frontWall']
      types = ['patch', 'patch', 'wall', 'wall', 'wall', 'wall']
      break
    case 'damBreak':
      bounds = { min: [0, 0, 0], max: [5 * DAM_BREAK_A, 3 * DAM_BREAK_A, DAM_BREAK_A / 20] }
      names = ['leftWall', 'rightWall', 'lowerWall', 'atmosphere', 'back', 'front']
      types = ['wall', 'wall', 'wall', 'patch', 'empty', 'empty']
      vof = true
      nu = 1e-6
      break
    case 'room':
      bounds = { min: [0, 0, 0], max: [10, 10, 3] }
      names = ['inlet', 'wallXMax', 'wallYMin', 'wallYMax', 'floor', 'ceiling']
      types = ['patch', 'wall', 'wall', 'wall', 'wall', 'wall']
      regions.push({ name: 'door', on: 'xmax', shape: { kind: 'box', min: [9, 4.5, -1], max: [11, 5.5, 2] } })
      uRef = 2
      thermal = true
      tInlet = ROOM_T_INLET
      break
    case 'plume':
      bounds = { min: [-8.32, -2.62, 0], max: [6.32, 3.62, 3] }
      names = ['wallXMin', 'outlet', 'wallYMin', 'wallYMax', 'floor', 'ceiling']
      types = ['wall', 'patch', 'wall', 'wall', 'wall', 'wall']
      regions.push({ name: 'inlet', on: 'zmin', shape: { kind: 'box', min: [-0.6, -0.6, -1], max: [0.6, 0.6, 1] } })
      uRef = 2
      thermal = true
      break
  }
  if (nz > 1) types = types.map((t, i) => (i >= 4 && t === 'empty' ? 'wall' : t))
  const boundaries = Object.fromEntries(FACES.map((f, i) => [f, names[i]])) as Record<BoxFace, string>
  const patchTypes: Record<string, string> = Object.fromEntries(names.map((n, i) => [n, types[i]]))
  const cyclic: CartesianSpec['cyclic'] = []
  for (const axis of opts.cyclic ?? []) {
    const idx = axis === 'x' ? 0 : axis === 'y' ? 1 : 2
    const a = names[2 * idx]
    const b = names[2 * idx + 1]
    patchTypes[a] = 'cyclic'
    patchTypes[b] = 'cyclic'
    cyclic.push({ a, b })
  }
  for (const r of regions) patchTypes[r.name] = 'patch'
  const patches: PatchDesc[] = [...names.map((n) => ({ name: n, type: patchTypes[n] })), ...regions.map((r) => ({ name: r.name, type: 'patch' }))]
  return {
    kind,
    spec: { bounds, cells: [nx, ny, nz], grading, boundaries, regions, cyclic, patchTypes },
    patches,
    uRef,
    nu,
    thermal,
    vof,
    tInlet,
  }
}

export function gridFor(spec: CartesianSpec): CartesianGrid {
  try {
    return buildCartesianGrid(spec)
  } catch {
    return uniformGrid(spec.bounds.min, spec.bounds.max, spec.cells)
  }
}

const DIMENSIONS: Record<string, string> = {
  U: '[0 1 -1 0 0 0 0]',
  p: '[0 2 -2 0 0 0 0]',
  p_rgh: '[1 -1 -2 0 0 0 0]',
  k: '[0 2 -2 0 0 0 0]',
  epsilon: '[0 2 -3 0 0 0 0]',
  omega: '[0 0 -1 0 0 0 0]',
  nut: '[0 2 -1 0 0 0 0]',
  nuTilda: '[0 2 -1 0 0 0 0]',
  T: '[0 0 0 1 0 0 0]',
  'alpha.water': '[0 0 0 0 0 0 0]',
}

export interface BcContext {
  uRef: number
  tInlet: number
  k0: number
  eps0: number
  omega0: number
  nu: number
}

function uniform(v: number | [number, number, number]): string {
  return Array.isArray(v) ? `uniform (${v.map((x) => g(x)).join(' ')})` : `uniform ${g(v)}`
}

/** The boundaryField block the real writers put on each patch, by patch type and field. */
export function foamPatchesFor(field: string, patches: PatchDesc[], bc: BcContext): FoamPatchOut[] {
  return patches.map((p): FoamPatchOut => {
    if (p.type === 'empty') return { name: p.name, type: 'empty' }
    if (p.type === 'cyclic') return { name: p.name, type: 'cyclic' }
    if (p.type === 'symmetry') return { name: p.name, type: 'symmetry' }
    const inlet = p.name === 'inlet' || p.name === 'movingWall'
    const wall = p.type === 'wall'
    switch (field) {
      case 'U':
        if (inlet) return { name: p.name, type: 'fixedValue', entries: { value: uniform(p.name === 'movingWall' ? [bc.uRef, 0, 0] : [0, 0, bc.tInlet !== 293.15 ? bc.uRef : 0]) } }
        if (wall) return { name: p.name, type: 'fixedValue', entries: { value: uniform([0, 0, 0]) } }
        return { name: p.name, type: 'inletOutlet', entries: { inletValue: uniform([0, 0, 0]), value: uniform([0, 0, 0]) } }
      case 'p':
      case 'p_rgh':
        if (wall || inlet) return { name: p.name, type: 'zeroGradient' }
        return { name: p.name, type: 'fixedValue', entries: { value: uniform(0) } }
      case 'k':
        if (wall && !inlet) return { name: p.name, type: 'kqRWallFunction', entries: { value: uniform(bc.k0) } }
        if (inlet) return { name: p.name, type: 'fixedValue', entries: { value: uniform(bc.k0) } }
        return { name: p.name, type: 'zeroGradient' }
      case 'epsilon':
        if (wall && !inlet) return { name: p.name, type: 'epsilonWallFunction', entries: { value: uniform(bc.eps0) } }
        if (inlet) return { name: p.name, type: 'fixedValue', entries: { value: uniform(bc.eps0) } }
        return { name: p.name, type: 'zeroGradient' }
      case 'omega':
        if (wall && !inlet) return { name: p.name, type: 'omegaWallFunction', entries: { value: uniform(bc.omega0) } }
        if (inlet) return { name: p.name, type: 'fixedValue', entries: { value: uniform(bc.omega0) } }
        return { name: p.name, type: 'zeroGradient' }
      case 'nut':
        if (wall && !inlet) return { name: p.name, type: 'nutkWallFunction', entries: { value: uniform(0) } }
        return { name: p.name, type: 'calculated', entries: { value: uniform(0) } }
      case 'nuTilda':
        if (wall && !inlet) return { name: p.name, type: 'fixedValue', entries: { value: uniform(0) } }
        if (inlet) return { name: p.name, type: 'fixedValue', entries: { value: uniform(3 * bc.nu) } }
        return { name: p.name, type: 'zeroGradient' }
      case 'T':
        if (inlet) return { name: p.name, type: 'fixedValue', entries: { value: uniform(bc.tInlet) } }
        if (wall) return { name: p.name, type: 'zeroGradient' }
        return { name: p.name, type: 'inletOutlet', entries: { inletValue: uniform(293.15), value: uniform(293.15) } }
      case 'alpha.water':
        if (wall || inlet) return { name: p.name, type: 'zeroGradient' }
        return { name: p.name, type: 'inletOutlet', entries: { inletValue: uniform(0), value: uniform(0) } }
      default:
        return { name: p.name, type: 'zeroGradient' }
    }
  })
}

export function bcContext(uRef: number, halfHeight: number, nu: number, tInlet: number): BcContext {
  const iTurb = 0.05
  const cmu = 0.09
  const k0 = 1.5 * (iTurb * uRef) ** 2
  const l = 0.07 * halfHeight * 2
  const eps0 = (cmu ** 0.75 * k0 ** 1.5) / l
  const omega0 = Math.sqrt(k0) / (cmu ** 0.25 * l)
  return { uRef, tInlet, k0, eps0, omega0, nu }
}

export interface FieldWrite {
  name: string
  data: Float32Array
  components: 1 | 3
}

/** Write a set of fields into a time directory; a formats failure is reported once and never fatal. */
export async function writeFieldSet(dirAbs: string, time: string, fields: FieldWrite[], patches: PatchDesc[], bc: BcContext, report: (msg: string) => void): Promise<boolean> {
  await fsp.mkdir(dirAbs, { recursive: true })
  let ok = true
  for (const f of fields) {
    const opts: FoamWriteOptions = {
      name: f.name,
      class: f.components === 3 ? 'volVectorField' : 'volScalarField',
      dimensions: DIMENSIONS[f.name] ?? '[0 0 0 0 0 0 0]',
      time,
      data: f.data,
      components: f.components,
      patches: foamPatchesFor(f.name, patches, bc),
    }
    try {
      await writeFoamField(path.join(dirAbs, f.name), opts)
    } catch (err) {
      if (ok) report(`[mock] result writing unavailable: ${(err as Error).message}`)
      ok = false
    }
  }
  return ok
}

function controlDict(endTime: number, deltaT: number, writeInterval: number): string {
  return `FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}

application     ofgpu;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         ${g(endTime)};
deltaT          ${g(deltaT)};
writeControl    timeStep;
writeInterval   ${g(writeInterval)};
purgeWrite      0;
writeFormat     ascii;
writePrecision  12;
timeFormat      general;
timePrecision   6;
`
}

const FV_SCHEMES = `FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}

ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; }
divSchemes
{
    default             none;
    div(phi,U)          Gauss linearUpwind grad(U);
    div(phi,k)          bounded Gauss upwind;
    div(phi,epsilon)    bounded Gauss upwind;
    div(phi,omega)      bounded Gauss upwind;
    div(phi,nuTilda)    bounded Gauss upwind;
    div(phi,T)          bounded Gauss upwind;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
`

const FV_SOLUTION = `FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}

solvers
{
    p       { solver PBiCGStab; preconditioner DIC; tolerance 1e-08; relTol 0.01; maxIter 1000; }
    "(U|k|epsilon|omega|nuTilda|T)" { solver PBiCGStab; preconditioner diagonal; tolerance 1e-08; relTol 0.1; maxIter 200; }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    residualControl { }
}

relaxationFactors
{
    fields    { p 0.3; }
    equations { U 0.7; k 0.7; epsilon 0.7; omega 0.7; nuTilda 0.7; T 0.7; }
}
`

function momentumTransport(model: string): string {
  return `FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      momentumTransport;
}

simulationType  RAS;

RAS
{
    model           ${model};
    turbulence      on;
    printCoeffs     on;
}
`
}

export interface GeneratedMesh {
  cells: number
  faces: number
  points: number
}

export function faceCount(cells: [number, number, number]): { internal: number; boundary: number } {
  const [nx, ny, nz] = cells
  const internal = (nx - 1) * ny * nz + nx * (ny - 1) * nz + nx * ny * (nz - 1)
  const boundary = 2 * (ny * nz + nx * nz + nx * ny)
  return { internal, boundary }
}

/** Write the whole case: constant/polyMesh, 0/, system/, constant dictionaries. */
export async function writeMockCase(dirAbs: string, preset: Preset, out: (line: string) => void, opts: { model?: string; wallModel?: string } = {}): Promise<GeneratedMesh> {
  const { spec } = preset
  const [nx, ny, nz] = spec.cells
  const grid = gridFor(spec)
  const nCells = nx * ny * nz
  out(`[mesh] ${preset.kind}: block ${nx} x ${ny} x ${nz}, bounds x [${g(spec.bounds.min[0])}, ${g(spec.bounds.max[0])}] y [${g(spec.bounds.min[1])}, ${g(spec.bounds.max[1])}] z [${g(spec.bounds.min[2])}, ${g(spec.bounds.max[2])}]`)
  if (spec.grading) {
    for (const [axis, gr] of Object.entries(spec.grading)) if (gr) out(`[mesh] grading ${axis}: expansion ${g(gr.expansion)}${gr.twoSided ? ' (two-sided)' : ''}`)
  }
  for (const r of spec.regions) out(`[mesh] window ${r.name} on ${r.on}: x [${g(r.shape.min[0])}, ${g(r.shape.max[0])}] y [${g(r.shape.min[1])}, ${g(r.shape.max[1])}]`)

  await fsp.mkdir(path.join(dirAbs, 'constant', 'polyMesh'), { recursive: true })
  await fsp.mkdir(path.join(dirAbs, '0'), { recursive: true })
  await fsp.mkdir(path.join(dirAbs, 'system'), { recursive: true })

  let faces = 0
  let points = (nx + 1) * (ny + 1) * (nz + 1)
  try {
    const mesh = cartesianToPolyMesh(grid, spec)
    out(`[mesh] polyMesh: ${mesh.nPoints} points, ${mesh.nFaces} faces (${mesh.nInternalFaces} internal), ${mesh.boundary.length} patches`)
    await writePolyMesh(path.join(dirAbs, 'constant', 'polyMesh'), mesh, { note: `nPoints: ${mesh.nPoints} nCells: ${mesh.nCells} nFaces: ${mesh.nFaces} nInternalFaces: ${mesh.nInternalFaces}` })
    faces = mesh.nFaces
    points = mesh.nPoints
  } catch (err) {
    out(`[mock] result writing unavailable: ${(err as Error).message}`)
    const fc = faceCount(spec.cells)
    faces = fc.internal + fc.boundary
  }

  const halfHeight = 0.5 * (spec.bounds.max[1] - spec.bounds.min[1])
  const bc = bcContext(preset.uRef, halfHeight, preset.nu, preset.thermal ? preset.tInlet : 293.15)
  const f = analyticFields(grid, { uRef: preset.uRef, thermal: preset.thermal, vof: preset.vof, time: 0, nu: preset.nu })
  const zero = new Float32Array(nCells)
  const fill = (v: number) => new Float32Array(nCells).fill(v)
  const initial: FieldWrite[] = preset.vof
    ? [
        { name: 'U', data: new Float32Array(3 * nCells), components: 3 },
        { name: 'p_rgh', data: zero, components: 1 },
        { name: 'alpha.water', data: f.alpha, components: 1 },
      ]
    : [
        { name: 'U', data: preset.kind === 'cavity' || preset.thermal ? new Float32Array(3 * nCells) : f.U, components: 3 },
        { name: 'p', data: zero, components: 1 },
        { name: 'k', data: fill(bc.k0), components: 1 },
        { name: 'epsilon', data: fill(bc.eps0), components: 1 },
        { name: 'omega', data: fill(bc.omega0), components: 1 },
        { name: 'nut', data: zero, components: 1 },
        { name: 'nuTilda', data: fill(3 * preset.nu), components: 1 },
        ...(preset.thermal ? [{ name: 'T', data: fill(293.15), components: 1 as const }] : []),
      ]
  await writeFieldSet(path.join(dirAbs, '0'), '0', initial, preset.patches, bc, out)
  out(`  0/ fields: Uref ${g(bc.uRef)}  k ${g(bc.k0)}  epsilon ${g(bc.eps0)}  omega ${g(bc.omega0)}  nuTilda ${g(3 * preset.nu)}  (nu ${g(preset.nu)})`)

  const transient = preset.vof
  await fsp.writeFile(path.join(dirAbs, 'system', 'controlDict'), transient ? controlDict(0.5, 0.0002, 250) : controlDict(4000, 1, 4000))
  await fsp.writeFile(path.join(dirAbs, 'system', 'fvSchemes'), FV_SCHEMES)
  await fsp.writeFile(path.join(dirAbs, 'system', 'fvSolution'), FV_SOLUTION)
  await fsp.writeFile(path.join(dirAbs, 'constant', 'momentumTransport'), momentumTransport(opts.model ?? (preset.vof ? 'laminar' : 'kEpsilon')))
  await fsp.writeFile(
    path.join(dirAbs, 'constant', 'transportProperties'),
    `FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       dictionary;\n    object      transportProperties;\n}\n\ntransportModel  Newtonian;\nnu              ${g(preset.nu)};\n`,
  )
  if (preset.thermal || preset.vof) {
    const gv = preset.vof ? '(0 -9.81 0)' : '(0 0 -9.81)'
    await fsp.writeFile(path.join(dirAbs, 'constant', 'g'), `FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       uniformDimensionedVectorField;\n    object      g;\n}\n\ndimensions      [0 1 -2 0 0 0 0];\nvalue           ${gv};\n`)
  }
  if (opts.wallModel) out(`  wallTreatment ${opts.wallModel} written into 0/nut (SPEC-LIT §29.1)`)
  return { cells: nCells, faces, points }
}
