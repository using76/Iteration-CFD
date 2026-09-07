// Result discovery: time directories with OpenFOAM time fallback, exponent
// and named directories, VTK files. STUB.

export interface TimeDirInfo {
  /** Directory name as on disk ("0", "0.5", "1e-05", "4000", "results"). */
  name: string
  /** Numeric value, or null for a named directory (sorted after numeric ones, in mtime order). */
  value: number | null
  abs: string
  /** vol*Field files present (surface fields like phi excluded). */
  fields: string[]
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
  /** VTK files under <root>/VTK or given directly. */
  vtk: Array<{ abs: string; kind: 'pvd' | 'vtu' | 'vtp' }>
  hasPolyMesh: boolean
}

/** "0" -> 0, "1e-05" -> 1e-5, "4000" -> 4000, "results" -> null. */
export function parseTimeName(_name: string): number | null {
  throw new Error('not implemented: parseTimeName')
}

export async function listTimeDirs(_rootAbs: string): Promise<TimeDirInfo[]> {
  throw new Error('not implemented: listTimeDirs')
}

/** Latest time directory at or before `targetIndex` (into `times`) that carries `field`; 0/ counts. Null when none. */
export function fieldTimeFallback(_times: TimeDirInfo[], _field: string, _targetIndex: number): TimeDirInfo | null {
  throw new Error('not implemented: fieldTimeFallback')
}

/** Classify what a path points at and where its results live (JSONC -> <stem>_jsonc/, case dir -> itself, time dir -> parent, ...). */
export async function resolveResultRoot(_absPath: string): Promise<ResultRoot> {
  throw new Error('not implemented: resolveResultRoot')
}
