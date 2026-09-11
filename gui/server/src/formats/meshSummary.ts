// The mesh statistics the meshers print, kept in one record per mesh.
//
// Three writers feed this: ofgpu-generate-mesh (preset block/carve lines),
// ofgpu-convert-mesh (`[convert] ... cells, points, faces (internal, boundary)`,
// `regions:`, the patch lines) and ofgpu-automesher (the §92.3 quality block
// and its refusal naming cells). The STEP pipeline (tools/mesh/step_mesh.py)
// is not a run-manager binary, but its log and its <name>_summary.json carry
// the gmsh quality lines (minSICN, the thickness gate), so both are parsed
// too. Whatever a mesh run ended with is written beside the mesh as
// <caseDir>/constant/polyMesh/.meshSummary.json; a mesh with no record on
// disk still reports its counts and patches straight off the polyMesh files.
import fs from 'node:fs/promises'
import path from 'node:path'
import { parseBoundaryText, readOwnerHeaderNote, type PolyMeshPatch } from './polymesh.js'

/** parseBoundaryText for a file that may not be there; null instead of a throw. */
export async function parseBoundaryTextSafe(filePath: string): Promise<PolyMeshPatch[] | null> {
  try {
    return parseBoundaryText(await fs.readFile(filePath, 'latin1'))
  } catch {
    return null
  }
}

export interface MeshPatchInfo {
  name: string
  /** The boundary file's own type word, when known. */
  type?: string
  faces?: number
}

/** One subject a gate refusal named: its id, where it sits, what was measured. */
export interface MeshGateSubject {
  kind: 'cell' | 'face'
  id: number
  /** Centroid (cells) or face centre, when the line carried one. */
  xyz?: [number, number, number]
  text: string
}

export interface MeshQuality {
  /** The automesher's §92.3 verdict; null when no gate line was seen. */
  gate: 'passed' | 'FAILED' | null
  failedGates: string[]
  /** G4, internal faces. */
  nonOrthMaxDeg: number | null
  nonOrthMeanDeg: number | null
  nonOrthOverReport: number | null
  /** G5, tau_c = 3 V_c / A_max^(3/2). */
  minThicknessTau: number | null
  minThicknessCell: number | null
  /** G2. */
  maxClosure: number | null
  /** The STEP pipeline's gmsh minSICN block. */
  minSICN: number | null
  minSICNp05: number | null
  sicnBelow01: number | null
  sicnNegative: number | null
  /** The cells (or faces) a refusal or the thickness gate named. */
  subjects: MeshGateSubject[]
}

export interface MeshSummaryRecord {
  /** The binary (or `step_mesh`) that produced the numbers. */
  tool: string
  /** `run`: a mesh run wrote it; `polyMesh`: read back from the files. */
  source: 'run' | 'polyMesh'
  /** Workspace-relative case directory. */
  caseDir: string
  runId: string | null
  writtenAt: string
  stoppedAfter: string | null
  totalSeconds: number | null
  counts: {
    cells: number | null
    points: number | null
    faces: number | null
    internalFaces: number | null
    boundaryFaces: number | null
    regions: number | null
    regionSizes: number[] | null
  }
  patches: MeshPatchInfo[]
  quality: MeshQuality
}

export const MESH_SUMMARY_FILE = '.meshSummary.json'

export class MeshSummaryError extends Error {
  constructor(
    readonly code: 'NOT_FOUND' | 'INVALID',
    message: string,
  ) {
    super(message)
    this.name = 'MeshSummaryError'
  }
}

const NUM = '[\\d,]+'
const num = (s: string): number => Number(s.replace(/,/g, ''))
const flt = (s: string): number => Number(s)
const triple = (s: string): [number, number, number] | undefined => {
  const parts = s.split(/\s*,\s*/).map(Number)
  return parts.length === 3 && parts.every((v) => Number.isFinite(v)) ? (parts as [number, number, number]) : undefined
}

export function emptyQuality(): MeshQuality {
  return {
    gate: null,
    failedGates: [],
    nonOrthMaxDeg: null,
    nonOrthMeanDeg: null,
    nonOrthOverReport: null,
    minThicknessTau: null,
    minThicknessCell: null,
    maxClosure: null,
    minSICN: null,
    minSICNp05: null,
    sicnBelow01: null,
    sicnNegative: null,
    subjects: [],
  }
}

/** What the log lines of any number of meshers said; every field fills in as it is seen. */
export interface MeshLogFacts {
  tools: string[]
  cells: number | null
  points: number | null
  faces: number | null
  internalFaces: number | null
  boundaryFaces: number | null
  regions: number | null
  regionSizes: number[] | null
  patches: MeshPatchInfo[]
  quality: MeshQuality
  /** `<caseDir>/constant/polyMesh` the automesher said it wrote. */
  polyMeshPath: string | null
  /** The `<name>_summary.json` the automesher said it wrote. */
  summaryJsonPath: string | null
  /** The case directory a `[convert] <dir>:` or `... cells -> <dir>` line named. */
  caseDirHint: string | null
}

export function emptyLogFacts(): MeshLogFacts {
  return { tools: [], cells: null, points: null, faces: null, internalFaces: null, boundaryFaces: null, regions: null, regionSizes: null, patches: [], quality: emptyQuality(), polyMeshPath: null, summaryJsonPath: null, caseDirHint: null }
}

const seen = (f: MeshLogFacts, tool: string): void => {
  if (!f.tools.includes(tool)) f.tools.push(tool)
}

/**
 * Parse the lines any of the meshers print. Lines that do not match a known
 * shape are ignored, so it is safe to feed a whole run log. One exception is
 * deliberately generic - the last "N cells" figure wins, as parseMeshCells in
 * tools/mesh.ts has always read it - and only fills `cells` when no line said
 * a case directory or counts outright.
 */
export function parseMeshLog(lines: string[]): MeshLogFacts {
  const f = emptyLogFacts()
  let genericCells: number | null = null
  for (const line of lines) {
    let m: RegExpExecArray | null

    // ofgpu-convert-mesh
    if ((m = /^\[convert\] (.+): ([\d,]+) cells, ([\d,]+) points, ([\d,]+) faces \(([\d,]+) internal, ([\d,]+) boundary\)/.exec(line))) {
      seen(f, 'ofgpu-convert-mesh')
      f.cells = num(m[2])
      f.points = num(m[3])
      f.faces = num(m[4])
      f.internalFaces = num(m[5])
      f.boundaryFaces = num(m[6])
      f.caseDirHint = m[1].replace(/[\\/]constant[\\/]polyMesh$/i, '')
      continue
    }
    if ((m = /^\[convert\] patch '(.+?)' \(([^,]*), [^)]*\): ([\d,]+) face/.exec(line))) {
      seen(f, 'ofgpu-convert-mesh')
      f.patches.push({ name: m[1], type: m[2].trim() || undefined, faces: num(m[3]) })
      continue
    }
    // `regions: 1 (4096 cells)` when every region is kept, and
    // `regions: 9; dropped 8 sealed region(s), ...` when the converter drops
    // the pockets - both say how many regions the mesh had.
    if ((m = /^regions: ([\d,]+)(?: \(([\d,\s.]+?)\s*cells?\))?(?:;.*)?\s*$/.exec(line))) {
      seen(f, 'ofgpu-convert-mesh')
      f.regions = num(m[1])
      // Past twelve regions the list ends in `...`; keep the numbers it did print.
      f.regionSizes = m[2] ? m[2].split(',').map((s) => num(s.trim())).filter((n) => Number.isFinite(n)) : null
      continue
    }

    // ofgpu-generate-mesh (real carve path, mock polyMesh line, final lines)
    if ((m = /^\[carve\] cells: ([\d,]+) block -> ([\d,]+) fluid \/ ([\d,]+) solid/.exec(line))) {
      seen(f, 'ofgpu-generate-mesh')
      f.cells = num(m[2])
      continue
    }
    if ((m = /^\[carve\] faces: ([\d,]+) internal, ([\d,]+) kept on domain patches/.exec(line))) {
      seen(f, 'ofgpu-generate-mesh')
      f.internalFaces = num(m[1])
      f.boundaryFaces = num(m[2])
      continue
    }
    if ((m = /^\[mesh\] polyMesh: ([\d,]+) points, ([\d,]+) faces \(([\d,]+) internal\), ([\d,]+) patches/.exec(line))) {
      seen(f, 'ofgpu-generate-mesh')
      f.points = num(m[1])
      f.faces = num(m[2])
      f.internalFaces = num(m[3])
      f.boundaryFaces = num(m[2]) - num(m[3])
      continue
    }
    if ((m = /^\w+: \d+ x \d+ x \d+ = ([\d,]+) cells -> (.+)$/.exec(line))) {
      seen(f, 'ofgpu-generate-mesh')
      f.cells = num(m[1])
      f.caseDirHint = m[2].trim()
      continue
    }
    if ((m = /^mesh: ([\d,]+) cells(?:, ([\d,]+) faces)?/.exec(line))) {
      seen(f, 'ofgpu-generate-mesh')
      f.cells = num(m[1])
      if (m[2]) f.faces = num(m[2])
      continue
    }

    // ofgpu-automesher: the §92.3 quality block
    if ((m = /^automesher: ([\d,]+) cells, ([\d,]+) internal faces, ([\d,]+) boundary faces, ([\d,]+) points/.exec(line))) {
      seen(f, 'ofgpu-automesher')
      f.cells = num(m[1])
      f.internalFaces = num(m[2])
      f.boundaryFaces = num(m[3])
      f.points = num(m[4])
      f.faces = f.internalFaces + f.boundaryFaces
      continue
    }
    if ((m = /^  volume: min (\S+) \(cell ([\d,]+)\); regions: ([\d,]+)/.exec(line))) {
      seen(f, 'ofgpu-automesher')
      f.regions = num(m[3])
      continue
    }
    if ((m = /^  closure: max (\S+) \(cell ([\d,]+)\)/.exec(line))) {
      seen(f, 'ofgpu-automesher')
      f.quality.maxClosure = flt(m[1])
      continue
    }
    if ((m = /^  non-orthogonality: max ([\d.]+) deg, mean ([\d.]+) deg, ([\d,]+) face\(s\) past the report mark/.exec(line))) {
      seen(f, 'ofgpu-automesher')
      f.quality.nonOrthMaxDeg = flt(m[1])
      f.quality.nonOrthMeanDeg = flt(m[2])
      f.quality.nonOrthOverReport = num(m[3])
      continue
    }
    if ((m = /^  thickness: min tau (\S+) \(cell ([\d,]+)\); conditioning: max cond/.exec(line))) {
      seen(f, 'ofgpu-automesher')
      f.quality.minThicknessTau = flt(m[1])
      f.quality.minThicknessCell = num(m[2])
      continue
    }
    if ((m = /^  duplicate faces: \d+; ldu ordered: (?:yes|no); gate: (passed|FAILED)/.exec(line))) {
      seen(f, 'ofgpu-automesher')
      f.quality.gate = m[1] as MeshQuality['gate']
      continue
    }
    // Gate::name() spells the gate out - `G4 (non-orthogonality)` - so the
    // number is all that is read between "gate" and "failed".
    if ((m = /^automesher: quality gate (G\d)\b[^:]*? failed/.exec(line))) {
      seen(f, 'ofgpu-automesher')
      if (!f.quality.failedGates.includes(m[1])) f.quality.failedGates.push(m[1])
      f.quality.gate = 'FAILED'
      continue
    }
    // `cell 90 at (x, y, z): V = ...`, `face 3 at (x, y, z) why: theta = ...`
    // and G7's bare `face 7: why`.
    if ((m = /^  (cell|face) (\d+)(?: (?:at|between) ([^:]+))?: (.+)$/.exec(line))) {
      seen(f, 'ofgpu-automesher')
      const where = m[3] ?? ''
      const at = /\(([^)]*)\)/.exec(where)
      f.quality.subjects.push({ kind: m[1] as 'cell' | 'face', id: num(m[2]), xyz: at ? triple(at[1]) : undefined, text: where ? `${where}: ${m[4]}` : m[4] })
      continue
    }
    if ((m = /^ofgpu-automesher: wrote (\S+?_summary\.json)$/.exec(line))) {
      seen(f, 'ofgpu-automesher')
      f.summaryJsonPath = m[1]
      continue
    }
    if ((m = /^ofgpu-automesher: wrote (\S+) \(([\d,]+) cells\)$/.exec(line))) {
      seen(f, 'ofgpu-automesher')
      f.polyMeshPath = m[1]
      f.cells = num(m[2])
      f.caseDirHint = m[1].replace(/[\\/]constant[\\/]polyMesh.*$/i, '')
      continue
    }

    // tools/mesh/step_mesh.py: the gmsh quality lines and the thickness gate
    if ((m = /\b(minSICN|gamma)\s+min\s+(-?[\d.eE+]+)\s+p1\s+(-?[\d.]+)\s+p5\s+(-?[\d.]+)\s+p50\s+(-?[\d.]+)\s+<0\.1:\s*(\d+)\s+<0:\s*(\d+)/.exec(line))) {
      seen(f, 'step_mesh')
      if (m[1] === 'minSICN') {
        f.quality.minSICN = flt(m[2])
        f.quality.minSICNp05 = flt(m[4])
        f.quality.sicnBelow01 = num(m[6])
        f.quality.sicnNegative = num(m[7])
      }
      continue
    }
    if ((m = /thickness gate: worst tau (-?[\d.eE+]+) at \(([^)]*)\)/.exec(line))) {
      seen(f, 'step_mesh')
      f.quality.minThicknessTau = flt(m[1])
      f.quality.subjects.push({ kind: 'cell', id: -1, xyz: triple(m[2]), text: `worst tau ${m[1]} at (${m[2]})` })
      continue
    }

    // The generic fallback: the last "N cells" figure in the log.
    for (const c of line.matchAll(/(\d[\d,]*)\s+cells/g)) genericCells = num(c[1])
  }
  if (f.cells === null && genericCells !== null) f.cells = genericCells
  return f
}

function pickQuality(q: Record<string, unknown>, key: string): number | null {
  const v = q[key]
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}

/**
 * The automesher's own `<name>_summary.json` (92.57) - the same numbers the
 * quality block prints, plus the patch list and the stop rule.
 */
export function parseAutomesherSummary(json: unknown): Partial<MeshLogFacts> & { stoppedAfter?: string | null; totalSeconds?: number | null } {
  const f: Partial<MeshLogFacts> & { stoppedAfter?: string | null; totalSeconds?: number | null } = { patches: [], quality: emptyQuality(), stoppedAfter: null, totalSeconds: null }
  if (typeof json !== 'object' || json === null) return f
  const o = json as Record<string, unknown>
  const mesh = (o.mesh ?? {}) as Record<string, unknown>
  const q = (o.quality ?? {}) as Record<string, unknown>
  f.cells = typeof mesh.n_cells === 'number' ? mesh.n_cells : null
  f.points = typeof mesh.n_points === 'number' ? mesh.n_points : null
  f.internalFaces = typeof mesh.n_internal_faces === 'number' ? mesh.n_internal_faces : null
  f.boundaryFaces = typeof mesh.n_boundary_faces === 'number' ? mesh.n_boundary_faces : null
  f.faces = f.internalFaces !== null && f.boundaryFaces !== null ? f.internalFaces + f.boundaryFaces : null
  if (Array.isArray(mesh.patches)) {
    f.patches = mesh.patches
      .filter((p): p is Record<string, unknown> => typeof p === 'object' && p !== null && typeof (p as Record<string, unknown>).name === 'string')
      .map((p) => ({ name: p.name as string, type: typeof p.kind === 'string' ? p.kind : undefined, faces: typeof p.size === 'number' ? p.size : undefined }))
  }
  const quality = emptyQuality()
  quality.nonOrthMaxDeg = pickQuality(q, 'max_non_orth_deg')
  quality.nonOrthMeanDeg = pickQuality(q, 'mean_non_orth_deg')
  quality.nonOrthOverReport = pickQuality(q, 'n_non_orth_over_report')
  quality.minThicknessTau = pickQuality(q, 'min_thickness_ratio')
  quality.minThicknessCell = pickQuality(q, 'min_thickness_cell')
  quality.maxClosure = pickQuality(q, 'max_closure')
  quality.gate = q.passed === true ? 'passed' : q.passed === false ? 'FAILED' : null
  f.quality = quality
  f.regions = typeof q.n_regions === 'number' ? q.n_regions : null
  f.regionSizes = Array.isArray(q.region_sizes) ? q.region_sizes.filter((v): v is number => typeof v === 'number') : null
  f.stoppedAfter = typeof o.stopped_after === 'string' ? o.stopped_after : null
  f.totalSeconds = typeof o.total_seconds === 'number' ? o.total_seconds : null
  return f
}

/**
 * The STEP pipeline's `<name>_summary.json`: gmsh's minSICN/gamma blocks and
 * the thickness gate's named cells (`thickness_gate: [{tet, tau, xyz}, ...]`).
 */
export function parseStepSummary(json: unknown): Partial<MeshLogFacts> {
  const f: Partial<MeshLogFacts> = { quality: emptyQuality(), patches: [] }
  if (typeof json !== 'object' || json === null) return f
  const o = json as Record<string, unknown>
  const quality = emptyQuality()
  const q = (o.quality ?? {}) as Record<string, unknown>
  const after = (o.quality_after_flat_removal ?? q) as Record<string, unknown>
  const sicn = (after.minSICN ?? q.minSICN ?? {}) as Record<string, unknown>
  quality.minSICN = pickQuality(sicn, 'min')
  quality.minSICNp05 = pickQuality(sicn, 'p05')
  quality.sicnBelow01 = pickQuality(sicn, 'below_0.1')
  quality.sicnNegative = pickQuality(sicn, 'negative')
  if (Array.isArray(o.thickness_gate)) {
    for (const g of o.thickness_gate) {
      if (typeof g !== 'object' || g === null) continue
      const gate = g as Record<string, unknown>
      if (typeof gate.tau !== 'number') continue
      if (quality.minThicknessTau === null || gate.tau < quality.minThicknessTau) quality.minThicknessTau = gate.tau
      quality.subjects.push({
        kind: 'cell',
        id: typeof gate.tet === 'number' ? gate.tet : -1,
        xyz: Array.isArray(gate.xyz) && gate.xyz.length === 3 && gate.xyz.every((v) => typeof v === 'number') ? (gate.xyz as [number, number, number]) : undefined,
        text: `tau ${gate.tau}`,
      })
    }
  }
  f.cells = typeof o.tetrahedra === 'number' ? o.tetrahedra : null
  f.quality = quality
  return f
}

/** Build the record that is written beside the mesh. */
export function buildMeshSummary(tool: string, caseDir: string, facts: Partial<MeshLogFacts>, opts: { runId?: string | null; source?: 'run' | 'polyMesh'; stoppedAfter?: string | null; totalSeconds?: number | null } = {}): MeshSummaryRecord {
  const q = facts.quality ?? emptyQuality()
  const counts = {
    cells: facts.cells ?? null,
    points: facts.points ?? null,
    faces: facts.faces ?? null,
    internalFaces: facts.internalFaces ?? null,
    boundaryFaces: facts.boundaryFaces ?? null,
    regions: facts.regions ?? null,
    regionSizes: facts.regionSizes ?? null,
  }
  return {
    tool,
    source: opts.source ?? 'run',
    caseDir,
    runId: opts.runId ?? null,
    writtenAt: new Date().toISOString(),
    stoppedAfter: opts.stoppedAfter ?? null,
    totalSeconds: opts.totalSeconds ?? null,
    counts,
    patches: facts.patches ?? [],
    quality: { ...emptyQuality(), ...q, subjects: q.subjects ?? [] },
  }
}

/** The record a mesh run wrote, or null when the mesh has none. */
export async function readMeshSummaryRecord(caseDirAbs: string): Promise<MeshSummaryRecord | null> {
  let text: string
  try {
    text = await fs.readFile(path.join(caseDirAbs, 'constant', 'polyMesh', MESH_SUMMARY_FILE), 'utf8')
  } catch {
    return null
  }
  try {
    return JSON.parse(text) as MeshSummaryRecord
  } catch {
    return null
  }
}

/** Write the record beside the mesh, creating constant/polyMesh if needed. */
export async function writeMeshSummaryRecord(caseDirAbs: string, record: MeshSummaryRecord): Promise<void> {
  const dir = path.join(caseDirAbs, 'constant', 'polyMesh')
  await fs.mkdir(dir, { recursive: true })
  await fs.writeFile(path.join(dir, MESH_SUMMARY_FILE), JSON.stringify(record, null, 2), 'utf8')
}

/**
 * No run wrote a record: read the counts off the mesh itself - the owner
 * header's note line and the boundary file, both cheap - so an existing mesh
 * still reports cells and patches. Null when there is no polyMesh here.
 */
export async function summaryFromPolyMesh(caseDirAbs: string): Promise<MeshSummaryRecord | null> {
  const dir = path.join(caseDirAbs, 'constant', 'polyMesh')
  let patches
  try {
    patches = parseBoundaryText(await fs.readFile(path.join(dir, 'boundary'), 'latin1'))
  } catch {
    return null
  }
  const note = await readOwnerHeaderNote(dir).catch(() => ({}) as Record<string, number>)
  const nFaces = note.nFaces ?? null
  const nInternal = note.nInternalFaces ?? null
  return buildMeshSummary(
    'polyMesh',
    caseDirAbs,
    {
      cells: note.nCells ?? null,
      points: note.nPoints ?? null,
      faces: nFaces,
      internalFaces: nInternal,
      boundaryFaces: nFaces !== null && nInternal !== null ? nFaces - nInternal : null,
      patches: patches.map((p) => ({ name: p.name, type: p.type, faces: p.nFaces })),
      quality: emptyQuality(),
    },
    { source: 'polyMesh' },
  )
}

/** The route's worker: the record a run wrote, else the mesh itself, else NOT_FOUND. `caseDir` is workspace-relative. */
export async function meshSummaryForCase(rootAbs: string, caseDir: string): Promise<MeshSummaryRecord> {
  const abs = path.join(rootAbs, caseDir)
  const record = await readMeshSummaryRecord(abs)
  if (record) return record
  const fallback = await summaryFromPolyMesh(abs)
  if (fallback) {
    fallback.caseDir = caseDir
    return fallback
  }
  throw new MeshSummaryError('NOT_FOUND', `no mesh summary and no constant/polyMesh in ${caseDir}`)
}
