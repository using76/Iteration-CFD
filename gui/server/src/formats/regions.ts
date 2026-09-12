// Multi-region discovery over a result root (docs/10 §C): the case's
// mesh.regions manifest, a regions.json in the root, region result
// directories, and a .cht.jsonc's own regions[] merge into one RegionEntry
// per region name; later sources fill only the keys an earlier one left null.
import fs from 'node:fs/promises'
import path from 'node:path'
import type { RegionEntry } from '@cfd/shared'
import { parseJsonc, type CaseJsoncInfo } from './casejsonc.js'
import { caseRootOfPolyMesh, findPolyMeshDir, type ResultRoot } from './results.js'
import { readOwnerHeaderNote } from './polymesh.js'
import { readVtuInfo } from './vtu.js'

export interface RegionsManifestEntry {
  name: string
  kind?: string
  polyMesh: string
  material?: string
}

export interface RegionsManifest {
  version: number
  regions: RegionsManifestEntry[]
  /** Kept verbatim, untyped. */
  interfaces: unknown[]
}

const statIs = (want: 'isDirectory' | 'isFile') => async (p: string): Promise<boolean> => {
  try {
    return (await fs.stat(p))[want]()
  } catch {
    return false
  }
}
const isDir = statIs('isDirectory')
const isFile = statIs('isFile')

/** Parse a regions.json (paths inside it are relative to its own directory). */
export async function readRegionsManifest(fileAbs: string): Promise<RegionsManifest> {
  const text = await fs.readFile(fileAbs, 'utf8')
  const name = path.basename(fileAbs)
  const { json } = parseJsonc(text)
  if (!json || typeof json !== 'object' || Array.isArray(json)) throw new Error(`${name}: no regions array`)
  const obj = json as Record<string, unknown>
  if (obj.version !== 1) throw new Error(`${name}: version ${JSON.stringify(obj.version)} is not 1`)
  if (!Array.isArray(obj.regions)) throw new Error(`${name}: no regions array`)
  const regions: RegionsManifestEntry[] = []
  for (const r of obj.regions) {
    if (!r || typeof r !== 'object' || Array.isArray(r)) continue
    const e = r as Record<string, unknown>
    if (typeof e.name !== 'string' || typeof e.polyMesh !== 'string') continue
    regions.push({ name: e.name, kind: typeof e.kind === 'string' ? e.kind : undefined, polyMesh: e.polyMesh, material: typeof e.material === 'string' ? e.material : undefined })
  }
  return { version: 1, regions, interfaces: Array.isArray(obj.interfaces) ? obj.interfaces : [] }
}

async function mergeManifest(manifestAbs: string, get: (name: string) => RegionEntry): Promise<void> {
  const m = await readRegionsManifest(manifestAbs)
  const manifestDir = path.dirname(manifestAbs)
  for (const r of m.regions) {
    const e = get(r.name)
    if (e.kind === null && (r.kind === 'fluid' || r.kind === 'solid')) e.kind = r.kind
    if (e.material === null && r.material !== undefined) e.material = r.material
    if (e.polyMeshDir === null) {
      const dir = await findPolyMeshDir(path.resolve(manifestDir, r.polyMesh))
      if (dir) {
        e.polyMeshDir = dir
        e.meshPath = caseRootOfPolyMesh(dir)
      }
    }
  }
}

/** Merge every source into one entry per region, first appearance first, paths ABSOLUTE. */
export async function discoverRegions(root: ResultRoot, caseInfo: CaseJsoncInfo | null): Promise<RegionEntry[]> {
  const entries = new Map<string, RegionEntry>()
  const get = (name: string): RegionEntry => {
    let e = entries.get(name)
    if (!e) {
      e = { name, kind: null, material: null, path: null, meshPath: null, polyMeshDir: null, resultPath: null, source: null, cellCount: null }
      entries.set(name, e)
    }
    return e
  }

  // (a) the case's mesh.regions manifest
  if (caseInfo?.regionsManifest && root.caseJsoncAbs) {
    try {
      await mergeManifest(path.join(path.dirname(root.caseJsoncAbs), caseInfo.regionsManifest), get)
    } catch {
      // a malformed manifest is a silent skip here; the checker owns refusals
    }
  }
  // (b) a manifest in the root
  for (const cand of [path.join(root.rootAbs, 'regions.json'), path.join(root.rootAbs, 'mesh', 'regions.json')]) {
    if (!(await isFile(cand))) continue
    try {
      await mergeManifest(cand, get)
    } catch {
      // silent skip
    }
  }

  // (c) region result directories
  const regionsDir = path.join(root.rootAbs, 'regions')
  let listing: string[] = []
  try {
    listing = await fs.readdir(regionsDir)
  } catch {
    listing = []
  }
  for (const name of listing.sort()) {
    const dirAbs = path.join(regionsDir, name)
    if (!(await isDir(dirAbs))) continue
    const e = get(name)
    if (e.resultPath === null) {
      e.resultPath = dirAbs
      e.source = 'dir'
    }
  }

  // (d) the case's own regions[]: VTK per region, and imported polyMesh meshes
  if (caseInfo) {
    const caseDir = root.caseJsoncAbs ? path.dirname(root.caseJsoncAbs) : null
    for (const r of caseInfo.regions) {
      const e = get(r.name)
      if (e.kind === null) e.kind = r.kind
      if (e.resultPath === null) {
        const vtuAbs = path.join(root.rootAbs, 'VTK', `${r.name}.vtu`)
        if (await isFile(vtuAbs)) {
          e.resultPath = vtuAbs
          e.source = 'vtu'
        }
      }
      if (r.mesh?.kind === 'polyMesh' && caseDir !== null && e.polyMeshDir === null) {
        const dir = await findPolyMeshDir(path.resolve(caseDir, r.mesh.path))
        if (dir) {
          e.polyMeshDir = dir
          e.meshPath = caseRootOfPolyMesh(dir)
        }
      }
    }
  }

  // resolve what `open` opens and the cell count behind it
  for (const e of entries.values()) {
    e.path = e.resultPath ?? e.meshPath ?? null
    if (e.source === null && e.path !== null) e.source = 'polyMesh'
    if (e.source === 'vtu' && e.resultPath !== null) {
      e.cellCount = await readVtuInfo(e.resultPath)
        .then((i) => i.nCells)
        .catch(() => null)
    } else if (e.polyMeshDir !== null) {
      e.cellCount = await readOwnerHeaderNote(e.polyMeshDir)
        .then((n) => (Number.isFinite(n.nCells) ? n.nCells : null))
        .catch(() => null)
    }
  }
  return [...entries.values()]
}
