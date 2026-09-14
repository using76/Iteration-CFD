// gui/server/src/ontology/folds/regions.ts — regions.json becomes one RegionLayout, one Region
// per entry and one Interface per entry, read through readRegionsManifest (C11). The per-side
// facts live on the joins link, whose three properties N1 declares and the Interface type does
// not; the reader drops `units` and `source`, so the same 625-byte file is read twice.
import fs from 'node:fs/promises'
import path from 'node:path'
import { readRegionsManifest } from '../../formats/regions.js'
import { emptyFoldReport, linkIfPresent, skip, upsert, walkFiles, type FoldContext, type FoldReport, type MirrorRow } from './base.js'

interface NarrowedInterface {
  regions: string[]
  patches: [string, string]
  faces: number | null
  tolerance: number | null
}

/** An interface entry is used only when regions and patches are both two-element string arrays
 *  and faces is a number or absent; otherwise it is a counted skip. */
function narrowInterface(e: unknown): NarrowedInterface | null {
  if (e === null || typeof e !== 'object') return null
  const o = e as Record<string, unknown>
  const regions = Array.isArray(o.regions) && o.regions.every((v) => typeof v === 'string') ? (o.regions as string[]) : []
  const patches = Array.isArray(o.patches) && o.patches.every((v) => typeof v === 'string') ? (o.patches as string[]) : []
  if (regions.length !== 2 || patches.length !== 2) return null
  if (o.faces !== undefined && o.faces !== null && typeof o.faces !== 'number') return null
  if (o.tolerance !== undefined && o.tolerance !== null && typeof o.tolerance !== 'number') return null
  return { regions, patches: [patches[0], patches[1]], faces: typeof o.faces === 'number' ? o.faces : null, tolerance: typeof o.tolerance === 'number' ? o.tolerance : null }
}

/** D16: the layout's tolerance is the interfaces' tolerance when they agree, else null. */
function layoutTolerance(infos: NarrowedInterface[]): number | null {
  if (infos.length === 0) return null
  const first = infos[0].tolerance
  for (const i of infos) {
    if (i.tolerance === null) return null
    if (first === null || i.tolerance !== first) return null
  }
  return first
}

export async function foldRegions(ctx: FoldContext): Promise<FoldReport[]> {
  const layoutRep = emptyFoldReport(ctx.type.regionLayout)
  const regionRep = emptyFoldReport(ctx.type.region)
  const ifaceRep = emptyFoldReport(ctx.type.interface)
  const t0 = Date.now()
  const reports = [layoutRep, regionRep, ifaceRep]
  if (!ctx.writer.hasObjectType(ctx.type.regionLayout)) {
    skip(layoutRep, ctx.type.regionLayout, 'not declared in the ontology registry; the fold writes no row')
    return reports
  }
  for await (const f of walkFiles(ctx.workspaceRoot, ctx.workspaceRoot, (name) => name === 'regions.json')) {
    layoutRep.filesRead++
    let manifest
    try {
      manifest = await readRegionsManifest(f.abs)
    } catch (e) {
      const err = new Error('reading ' + f.rel + ': ' + (e instanceof Error ? e.message : String(e)))
      ;(err as Error & { source?: string; reports?: FoldReport[] }).source = f.rel
      ;(err as Error & { source?: string; reports?: FoldReport[] }).reports = reports
      throw err
    }
    const raw = JSON.parse(await fs.readFile(f.abs, 'utf8')) as Record<string, unknown>
    const rawRegions = Array.isArray(raw.regions) ? raw.regions.length : manifest.regions.length
    const interfaces = (manifest.interfaces ?? []).map(narrowInterface)
    const good = interfaces.filter((i): i is NarrowedInterface => i !== null)
    const bad = interfaces.length - good.length
    if (bad > 0) skip(layoutRep, 'malformed interface entries', 'an interface entry whose regions or patches are not two string arrays is refused; the manifest keeps the row', bad)
    if (rawRegions !== manifest.regions.length)
      skip(layoutRep, 'dropped region entries', 'the manifest reader silently drops a region entry missing name or polyMesh', rawRegions - manifest.regions.length)
    if (manifest.regions.length === 0)
      skip(layoutRep, 'empty regions', f.rel + ': the manifest declares no region; the layout row is still written', 0)
    const src = raw.source !== null && typeof raw.source === 'object' ? (raw.source as Record<string, unknown>) : {}
    const str = (v: unknown): string | null => (typeof v === 'string' ? v : null)
    const layoutId = f.rel
    const layout: MirrorRow = {
      objectType: ctx.type.regionLayout,
      primaryKey: layoutId,
      properties: {
        layoutId, name: path.basename(path.dirname(f.rel)),
        version: manifest.version,
        units: str(raw.units),
        tolerance: layoutTolerance(good),
        sourceTool: str(src.tool), sourceVersion: str(src.version),
        sourceGeometry: str(src.geometry), sourceConfig: str(src.config),
        nRegions: manifest.regions.length,
        nInterfaces: good.length,
      },
      sourcePath: f.rel,
      importedAt: ctx.now(),
    }
    await upsert(ctx, layoutRep, layout)
    for (const r of manifest.regions) {
      const regionId = layoutId + '#' + r.name
      const region: MirrorRow = {
        objectType: ctx.type.region,
        primaryKey: regionId,
        properties: {
          regionId, layoutId, name: r.name,
          kind: r.kind === 'fluid' || r.kind === 'solid' ? r.kind : null,
          polyMeshPath: r.polyMesh,
          materialName: typeof r.material === 'string' ? r.material : null,
          cellCount: null,
        },
        sourcePath: f.rel,
        importedAt: ctx.now(),
      }
      await upsert(ctx, regionRep, region)
      await linkIfPresent(ctx, regionRep, { linkType: ctx.link.contains, fromType: ctx.type.regionLayout, fromId: layoutId, toType: ctx.type.region, toId: regionId, props: null, sourcePath: f.rel, importedAt: ctx.now() })
    }
    for (const info of good) {
      const patchA = info.patches[0]
      const patchB = info.patches[1]
      const interfaceId = layoutId + '#' + patchA + '|' + patchB
      const iface: MirrorRow = {
        objectType: ctx.type.interface,
        primaryKey: interfaceId,
        properties: {
          interfaceId, layoutId, name: patchA + ' / ' + patchB,
          tolerance: info.tolerance,
          rc: null, kappaLayers: null,
        },
        sourcePath: f.rel,
        importedAt: ctx.now(),
      }
      await upsert(ctx, ifaceRep, iface)
      await linkIfPresent(ctx, ifaceRep, { linkType: ctx.link.declares, fromType: ctx.type.regionLayout, fromId: layoutId, toType: ctx.type.interface, toId: interfaceId, props: null, sourcePath: f.rel, importedAt: ctx.now() })
      // Two joins rows per interface; the per-side facts are link properties (C11).
      for (const side of [0, 1] as const) {
        const regionId = layoutId + '#' + info.regions[side]
        await linkIfPresent(ctx, ifaceRep, {
          linkType: ctx.link.joins,
          fromType: ctx.type.interface, fromId: interfaceId,
          toType: ctx.type.region, toId: regionId,
          props: { side: side === 0 ? 'a' : 'b', patch: info.patches[side], faces: info.faces },
          sourcePath: f.rel,
          importedAt: ctx.now(),
        })
      }
    }
  }
  layoutRep.notes.push('rc and kappaLayers stay null: no regions.json on disk carries either (facts §4.4)')
  layoutRep.seconds = (Date.now() - t0) / 1000
  return reports
}
