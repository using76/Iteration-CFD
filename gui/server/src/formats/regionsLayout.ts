// The region layout behind GET /api/mesh/regions: docs/10 §C's regions.json
// read back with one answer per region - whether its polyMesh is there and how
// many cells the owner header's note declares (M4 wrote the note). The GUI's
// Regions table and the case editor's region pickers read this response, not
// the manifest itself: the manifest is the writer's contract, this is the
// reader's.
//
// No GPL-licensed source was consulted.
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import { readOwnerHeaderNote } from './polymesh.js'

export interface RegionLayoutRegion {
  name: string
  kind: 'fluid' | 'solid'
  polyMesh: string
  material: string | null
  cells: number | null
  ok: boolean
  error: string | null
}

export interface RegionLayoutInterface {
  regions: [string, string]
  patches: [string, string]
  faces: number
  tolerance: number
}

export interface RegionLayoutResponse {
  dir: string
  version: number
  units: string
  regions: RegionLayoutRegion[]
  interfaces: RegionLayoutInterface[]
  source: Record<string, unknown> | null
}

export class RegionLayoutError extends Error {
  constructor(
    public readonly code: 'NOT_FOUND' | 'BAD_MANIFEST',
    message: string,
  ) {
    super(message)
    this.name = 'RegionLayoutError'
  }
}

interface RawRegion {
  name?: unknown
  kind?: unknown
  polyMesh?: unknown
  material?: unknown
}

/** Read `<rootAbs>/<dirRel>/regions.json` and answer it per region. `dirRel` is workspace-relative. */
export async function readRegionLayout(rootAbs: string, dirRel: string): Promise<RegionLayoutResponse> {
  let text: string
  try {
    text = await fsp.readFile(path.join(rootAbs, dirRel, 'regions.json'), 'utf8')
  } catch {
    throw new RegionLayoutError('NOT_FOUND', `no regions.json in ${dirRel}`)
  }
  let raw: { version?: unknown; units?: unknown; regions?: unknown; interfaces?: unknown; source?: unknown }
  try {
    raw = JSON.parse(text)
  } catch {
    throw new RegionLayoutError('BAD_MANIFEST', `regions.json in ${dirRel} is not JSON`)
  }
  if (!raw || raw.version !== 1 || !Array.isArray(raw.regions)) {
    throw new RegionLayoutError('BAD_MANIFEST', `regions.json in ${dirRel} is not a version-1 region manifest`)
  }
  const regions: RegionLayoutRegion[] = []
  for (const r of raw.regions as RawRegion[]) {
    if (typeof r?.name !== 'string' || !r.name) {
      throw new RegionLayoutError('BAD_MANIFEST', `a region without a name in ${dirRel}/regions.json`)
    }
    const polyMesh = typeof r.polyMesh === 'string' ? r.polyMesh : ''
    let ok = false
    let error: string | null = null
    let cells: number | null = null
    if (path.isAbsolute(polyMesh) || polyMesh.includes('..')) {
      error = 'polyMesh path is not relative to the manifest (R6)'
    } else {
      const abs = path.join(rootAbs, dirRel, polyMesh)
      ok = fs.existsSync(path.join(abs, 'boundary'))
      error = ok ? null : 'polyMesh missing'
      if (ok) cells = (await readOwnerHeaderNote(abs).catch(() => ({} as Record<string, number>))).nCells ?? null
    }
    regions.push({ name: r.name, kind: r.kind === 'fluid' ? 'fluid' : 'solid', polyMesh, material: typeof r.material === 'string' ? r.material : null, cells, ok, error })
  }
  return {
    dir: dirRel.split(path.sep).join('/'),
    version: 1,
    units: typeof raw.units === 'string' ? raw.units : 'm',
    regions,
    interfaces: readInterfaces(raw.interfaces),
    source: raw.source && typeof raw.source === 'object' && !Array.isArray(raw.source) ? (raw.source as Record<string, unknown>) : null,
  }
}

/** The manifest's interface entries that carry two region names and two patch names; the rest are dropped. */
function readInterfaces(raw: unknown): RegionLayoutInterface[] {
  if (!Array.isArray(raw)) return []
  const out: RegionLayoutInterface[] = []
  for (const i of raw) {
    const regions = (i as { regions?: unknown }).regions
    const patches = (i as { patches?: unknown }).patches
    if (!Array.isArray(regions) || regions.length !== 2 || !regions.every((v) => typeof v === 'string')) continue
    if (!Array.isArray(patches) || patches.length !== 2 || !patches.every((v) => typeof v === 'string')) continue
    const faces = (i as { faces?: unknown }).faces
    const tolerance = (i as { tolerance?: unknown }).tolerance
    out.push({
      regions: [regions[0] as string, regions[1] as string],
      patches: [patches[0] as string, patches[1] as string],
      faces: typeof faces === 'number' && Number.isFinite(faces) ? faces : 0,
      tolerance: typeof tolerance === 'number' && Number.isFinite(tolerance) ? tolerance : 0,
    })
  }
  return out
}
