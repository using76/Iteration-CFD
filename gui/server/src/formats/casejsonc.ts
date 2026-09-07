// JSONC case reading (comments + trailing commas stripped with jsonc-parser)
// and extraction of the parts the GUI needs. STUB.
import type { CartesianSpec } from './cartesian.js'

export interface JsoncParseError {
  message: string
  offset: number
  length: number
  line: number
  col: number
}

export interface CaseJsoncInfo {
  /** Workspace-relative path. */
  path: string
  raw: string
  /** Parsed value (null when unparseable). */
  json: unknown
  errors: JsoncParseError[]
  name: string | null
  mesh: CartesianSpec | null
  turbulenceKind: string | null
  model: string | null
  gravity: [number, number, number] | null
  run: { endTime: number; deltaT: number } | null
  /** The `output` block verbatim, if any. */
  output: unknown | null
  /** Workspace-relative `<stem>_jsonc` directory. */
  outputDir: string
  /** Fields the case initialises (keys of `initial`). */
  initialFields: string[]
}

export function parseJsonc(_text: string): { json: unknown; errors: JsoncParseError[] } {
  throw new Error('not implemented: parseJsonc')
}

export async function readCaseJsonc(_absPath: string, _relPath: string): Promise<CaseJsoncInfo> {
  throw new Error('not implemented: readCaseJsonc')
}

/** The `mesh` block as a CartesianSpec (null when kind != cartesian or malformed). Patch types come from `patches[]` rules (wall/inlet/open/empty/symmetry) matched against boundary names. */
export function extractCartesianSpec(_json: unknown): CartesianSpec | null {
  throw new Error('not implemented: extractCartesianSpec')
}
