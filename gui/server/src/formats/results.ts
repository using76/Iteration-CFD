// Result discovery: time directories with OpenFOAM time fallback, exponent
// and named directories, VTK files.
import fs from 'node:fs/promises'
import path from 'node:path'
import { jsonCaseOutputDir } from '@cfd/shared'
import { readFoamFieldHeader } from './foam.js'

export interface TimeDirInfo {
  /** Directory name as on disk ("0", "0.5", "1e-05", "4000", "results"). */
  name: string
  /** Numeric value, or null for a named directory (sorted after numeric ones, in mtime order). */
  value: number | null
  abs: string
  /** vol*Field files present (surface fields like phi excluded). */
  fields: string[]
  /** FoamFile class of each listed field (volScalarField, volVectorField, ...). */
  fieldClasses: Record<string, string>
  mtimeMs: number
}

export type ResultRootKind = 'jsoncCase' | 'foamCase' | 'outputDir' | 'timeDir' | 'vtu' | 'pvd'

export interface ResultRoot {
  kind: ResultRootKind
  /** Absolute directory holding the time directories (or containing the vtk file). */
  rootAbs: string
  /** Absolute path of the case.jsonc that produced this output, if known. */
  caseJsoncAbs: string | null
  /** For 'timeDir': the selected time directory name. */
  timeDir: string | null
  /** VTK files under <root>/VTK (the whole directory listing, sorted by name). */
  vtk: Array<{ abs: string; kind: 'pvd' | 'vtu' | 'vtp' }>
  /** For 'vtu' / 'pvd': the file that was opened explicitly (it alone defines the time series). */
  vtkFile: string | null
  hasPolyMesh: boolean
}

const NOT_TIME_DIRS = new Set(['constant', 'system', 'VTK', 'VDB', 'postProcessing', 'logs', 'processor', 'uniform'])

/** "0" -> 0, "1e-05" -> 1e-5, "4000" -> 4000, "results" -> null. */
export function parseTimeName(name: string): number | null {
  if (!/^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(name)) return null
  const v = Number(name)
  return Number.isFinite(v) ? v : null
}

async function isDir(p: string): Promise<boolean> {
  try {
    return (await fs.stat(p)).isDirectory()
  } catch {
    return false
  }
}

async function isFile(p: string): Promise<boolean> {
  try {
    return (await fs.stat(p)).isFile()
  } catch {
    return false
  }
}

/** vol*Field files directly inside a directory, by FoamFile class; anything else (phi, binary, dictionaries) is skipped. */
export async function listVolFields(dirAbs: string): Promise<Record<string, string>> {
  const out: Record<string, string> = {}
  let entries: import('node:fs').Dirent[]
  try {
    entries = await fs.readdir(dirAbs, { withFileTypes: true })
  } catch {
    return out
  }
  const files = entries.filter((e) => e.isFile() && !e.name.startsWith('.') && !e.name.endsWith('.orig')).map((e) => e.name)
  await Promise.all(
    files.map(async (name) => {
      try {
        const h = await readFoamFieldHeader(path.join(dirAbs, name))
        if (/^vol\w*Field$/.test(h.class)) out[name] = h.class
      } catch {
        // Not a field file.
      }
    }),
  )
  return out
}

export async function listTimeDirs(rootAbs: string): Promise<TimeDirInfo[]> {
  let entries: import('node:fs').Dirent[]
  try {
    entries = await fs.readdir(rootAbs, { withFileTypes: true })
  } catch {
    return []
  }
  const candidates = entries.filter((e) => e.isDirectory() && !e.name.startsWith('.') && !NOT_TIME_DIRS.has(e.name))
  const infos = await Promise.all(
    candidates.map(async (e): Promise<TimeDirInfo | null> => {
      const abs = path.join(rootAbs, e.name)
      const fieldClasses = await listVolFields(abs)
      const fields = Object.keys(fieldClasses).sort()
      if (fields.length === 0) return null
      const st = await fs.stat(abs)
      return { name: e.name, value: parseTimeName(e.name), abs, fields, fieldClasses, mtimeMs: st.mtimeMs }
    }),
  )
  const out = infos.filter((t): t is TimeDirInfo => t !== null)
  out.sort((a, b) => {
    if (a.value !== null && b.value !== null) return a.value - b.value || a.name.localeCompare(b.name)
    if (a.value !== null) return -1
    if (b.value !== null) return 1
    return a.mtimeMs - b.mtimeMs || a.name.localeCompare(b.name)
  })
  return out
}

/** Latest time directory at or before `targetIndex` (into `times`) that carries `field`; 0/ counts. Null when none. */
export function fieldTimeFallback(times: TimeDirInfo[], field: string, targetIndex: number): TimeDirInfo | null {
  for (let i = Math.min(targetIndex, times.length - 1); i >= 0; i--) if (times[i].fields.includes(field)) return times[i]
  return null
}

async function listVtk(rootAbs: string): Promise<ResultRoot['vtk']> {
  const dir = path.join(rootAbs, 'VTK')
  let entries: string[]
  try {
    entries = await fs.readdir(dir)
  } catch {
    return []
  }
  const out: ResultRoot['vtk'] = []
  for (const name of entries.sort()) {
    const ext = path.extname(name).toLowerCase()
    if (ext === '.pvd' || ext === '.vtu' || ext === '.vtp') out.push({ abs: path.join(dir, name), kind: ext.slice(1) as 'pvd' | 'vtu' | 'vtp' })
  }
  return out
}

async function hasPolyMeshAt(rootAbs: string): Promise<boolean> {
  return isFile(path.join(rootAbs, 'constant', 'polyMesh', 'points'))
}

/** `<dir>/<stem>_jsonc` -> `<dir>/<stem>.jsonc` (or .json) when that file exists. */
async function siblingCaseJsonc(outputDirAbs: string): Promise<string | null> {
  const base = path.basename(outputDirAbs)
  if (!base.endsWith('_jsonc')) return null
  const stem = base.slice(0, -'_jsonc'.length)
  for (const ext of ['.jsonc', '.json']) {
    const candidate = path.join(path.dirname(outputDirAbs), stem + ext)
    if (await isFile(candidate)) return candidate
  }
  return null
}

async function hasTimeDirs(rootAbs: string): Promise<boolean> {
  return (await listTimeDirs(rootAbs)).length > 0
}

async function classifyDir(absPath: string): Promise<ResultRoot> {
  const caseJsoncAbs = await siblingCaseJsonc(absPath)
  const [polyMesh, vtk] = await Promise.all([hasPolyMeshAt(absPath), listVtk(absPath)])
  if (caseJsoncAbs) return { kind: 'outputDir', rootAbs: absPath, caseJsoncAbs, timeDir: null, vtk, vtkFile: null, hasPolyMesh: polyMesh }
  if (polyMesh || (await isDir(path.join(absPath, 'system'))) || (await isDir(path.join(absPath, '0')))) {
    return { kind: 'foamCase', rootAbs: absPath, caseJsoncAbs: null, timeDir: null, vtk, vtkFile: null, hasPolyMesh: polyMesh }
  }
  if (await hasTimeDirs(absPath)) return { kind: 'outputDir', rootAbs: absPath, caseJsoncAbs: null, timeDir: null, vtk, vtkFile: null, hasPolyMesh: polyMesh }
  if (Object.keys(await listVolFields(absPath)).length > 0) {
    const parentAbs = path.dirname(absPath)
    const parent = await classifyDir(parentAbs)
    return { ...parent, kind: 'timeDir', rootAbs: parentAbs, timeDir: path.basename(absPath) }
  }
  return { kind: 'outputDir', rootAbs: absPath, caseJsoncAbs: null, timeDir: null, vtk, vtkFile: null, hasPolyMesh: polyMesh }
}

/** Classify what a path points at and where its results live (JSONC -> <stem>_jsonc/, case dir -> itself, time dir -> parent, ...). */
export async function resolveResultRoot(absPath: string): Promise<ResultRoot> {
  const st = await fs.stat(absPath)
  if (st.isDirectory()) return classifyDir(absPath)
  const ext = path.extname(absPath).toLowerCase()
  if (ext === '.jsonc' || ext === '.json') {
    const rootAbs = jsonCaseOutputDir(absPath)
    const [vtk, polyMesh] = await Promise.all([listVtk(rootAbs), hasPolyMeshAt(rootAbs)])
    return { kind: 'jsoncCase', rootAbs, caseJsoncAbs: absPath, timeDir: null, vtk, vtkFile: null, hasPolyMesh: polyMesh }
  }
  if (ext === '.vtu' || ext === '.pvd' || ext === '.vtp') {
    const dir = path.dirname(absPath)
    const rootAbs = path.basename(dir) === 'VTK' ? path.dirname(dir) : dir
    const [caseJsoncAbs, polyMesh, listed] = await Promise.all([siblingCaseJsonc(rootAbs), hasPolyMeshAt(rootAbs), listVtk(rootAbs)])
    const kind = ext === '.pvd' ? 'pvd' : 'vtu'
    const vtk = listed.some((v) => v.abs === absPath) ? listed : [{ abs: absPath, kind: ext.slice(1) as 'pvd' | 'vtu' | 'vtp' }, ...listed]
    return { kind, rootAbs, caseJsoncAbs, timeDir: null, vtk, vtkFile: absPath, hasPolyMesh: polyMesh }
  }
  throw new Error(`not a case, result directory or VTK file: ${absPath}`)
}
