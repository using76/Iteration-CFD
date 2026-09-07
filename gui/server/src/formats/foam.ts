// OpenFOAM ASCII vol*Field reader/writer, matching rust/src/io/fields.rs.
// STUB: signatures are the contract; the implementation is filled in by the
// formats module owner.

export interface FoamFieldHeader {
  version: string
  format: string
  /** volScalarField | volVectorField | surfaceScalarField | ... */
  class: string
  location: string
  object: string
  dimensions: string | null
}

export interface FoamField {
  name: string
  header: FoamFieldHeader
  components: 1 | 3
  /** Number of tuples in `data`. */
  count: number
  data: Float32Array
  uniform: boolean
  uniformValue: number[] | null
  /** Patch names and types found in boundaryField (type may be null when absent). */
  patches: Array<{ name: string; type: string | null }>
}

export interface FoamPatchOut {
  name: string
  /** e.g. fixedValue, zeroGradient, empty, kqRWallFunction ... */
  type: string
  /** Extra entries written verbatim inside the patch block, e.g. { value: 'uniform (0 0 0)' }. */
  entries?: Record<string, string>
}

export interface FoamWriteOptions {
  name: string
  class: 'volScalarField' | 'volVectorField'
  /** e.g. "[0 1 -1 0 0 0 0]" */
  dimensions: string
  /** Time directory name, written into the header `location`. */
  time: string
  data: Float32Array | Float64Array
  components: 1 | 3
  patches: FoamPatchOut[]
  /** Significant digits (default 12). */
  precision?: number
}

export async function readFoamFieldHeader(_path: string): Promise<FoamFieldHeader> {
  throw new Error('not implemented: readFoamFieldHeader')
}

/**
 * Read a field. `uniform x;` internalFields are expanded to `nCells` tuples
 * when given (otherwise count = 1 and uniform = true). Handles
 * `nonuniform 0()`, newlines between `List<scalar>`, N and `(`, and
 * double-quoted patch names.
 */
export async function readFoamField(_path: string, _opts: { nCells?: number | null } = {}): Promise<FoamField> {
  throw new Error('not implemented: readFoamField')
}

export async function writeFoamField(_path: string, _opts: FoamWriteOptions): Promise<void> {
  throw new Error('not implemented: writeFoamField')
}

/** Port of rust/src/io/case.rs format_time_name: 6 significant digits, exponent form when needed, "0" for zero. */
export function formatTimeName(_v: number): string {
  throw new Error('not implemented: formatTimeName')
}
