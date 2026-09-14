// Run provenance (docs/12 §C, unit N0): the five edges a new run record grows
// -- gitSha, gitDirty, caseId, meshId, machine -- plus the reader-side null
// fill that makes a 24-key legacy record and a 29-key new one the same shape.
import fsp from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import type { MachineRef, RunInfo } from '@cfd/shared'
import { MESH_SUMMARY_FILE } from '../formats/meshSummary.js'
import { caseRootOfPolyMesh, resolveResultRoot } from '../formats/results.js'
import { toWorkspaceRel } from '../workspace/paths.js'
import { gitHead } from '../workspace/git.js'

/** The five edges a run record grows. Every field is null when it could not be established. */
export interface RunProvenance {
  gitSha: string | null
  gitDirty: boolean | null
  caseId: string | null
  meshId: string | null
  machine: MachineRef | null
}

/** All five null: what a legacy record and a failed probe both look like. */
export const NO_PROVENANCE: RunProvenance = {
  gitSha: null,
  gitDirty: null,
  caseId: null,
  meshId: null,
  machine: null,
}

/**
 * The Case and Mesh primary keys a workspace-relative path resolves to.
 * `probeRel` is `casePath ?? outputRoot` as `validate()` returned them (already
 * confined, already forward-slashed). `isMeshRun` is `spec.kind === 'mesh'`.
 */
export async function caseAndMeshIds(
  workspaceRoot: string,
  probeRel: string | null,
  isMeshRun: boolean,
): Promise<{ caseId: string | null; meshId: string | null }> {
  if (probeRel === null) return { caseId: null, meshId: null }
  try {
    // Throws on anything that is not a case, result directory or VTK file.
    const rr = await resolveResultRoot(path.join(workspaceRoot, probeRel))
    const caseRel = rr.caseJsoncAbs !== null
      ? toWorkspaceRel(workspaceRoot, rr.caseJsoncAbs)
      : rr.kind === 'foamCase'
        ? toWorkspaceRel(workspaceRoot, rr.rootAbs)
        : ''
    const caseId = caseRel === '' ? null : caseRel
    // A mesh run produces a mesh (`produced-by`); it never reads one (D6).
    let meshId: string | null = null
    if (!isMeshRun && rr.polyMeshDirAbs !== null) {
      meshId = await meshKeyOf(workspaceRoot, rr.polyMeshDirAbs)
    }
    return { caseId, meshId }
  } catch {
    return { caseId: null, meshId: null }
  }
}

/**
 * D6's mesh key: `<workspace-relative path of the summary file>#<mesh name>`,
 * where the name is basename(record.caseDir) when the summary carries one and
 * the case root's basename otherwise. No readable summary => null: the mesh is
 * not a Mesh object yet, and any invented string would dangle the usesMesh
 * edge. (readMeshSummaryRecord is not reused here: it hard-codes
 * `<caseDir>/constant/polyMesh`, while findPolyMeshDir accepts two more shapes.)
 */
async function meshKeyOf(workspaceRoot: string, polyMeshDirAbs: string): Promise<string | null> {
  const sumAbs = path.join(polyMeshDirAbs, MESH_SUMMARY_FILE)
  try {
    const rec = JSON.parse(await fsp.readFile(sumAbs, 'utf8')) as { caseDir?: unknown }
    if (rec === null || typeof rec !== 'object') return null
    const meshName = typeof rec.caseDir === 'string' && rec.caseDir !== ''
      ? path.basename(rec.caseDir)
      : path.basename(caseRootOfPolyMesh(polyMeshDirAbs))
    const sumRel = toWorkspaceRel(workspaceRoot, sumAbs)
    return sumRel === '' || meshName === '' ? null : `${sumRel}#${meshName}`
  } catch {
    return null
  }
}

/** N1's machine struct: hostname, the cached GPU name ('' when unknown), and process.platform. */
export function machineOf(gpuName: string | null): MachineRef {
  return { hostname: os.hostname(), gpu: gpuName ?? '', platform: process.platform }
}

/** Everything a new run record records about where and against what it ran. Never throws. */
export async function collectRunProvenance(opts: {
  workspaceRoot: string
  probeRel: string | null
  isMeshRun: boolean
  /** `gpuMonitor.state().name` — passed in so this module never imports the gpu monitor. */
  gpuName: string | null
}): Promise<RunProvenance> {
  try {
    const [head, ids] = await Promise.all([
      gitHead(opts.workspaceRoot),
      caseAndMeshIds(opts.workspaceRoot, opts.probeRel, opts.isMeshRun),
    ])
    return { gitSha: head.sha, gitDirty: head.dirty, caseId: ids.caseId, meshId: ids.meshId, machine: machineOf(opts.gpuName) }
  } catch {
    return NO_PROVENANCE
  }
}

/** A record read from disk, with every missing provenance key made an explicit null. Pure. */
export function normalizeProvenance(run: RunInfo): RunInfo {
  return {
    ...run,
    gitSha: run.gitSha ?? null,
    gitDirty: run.gitDirty ?? null,
    caseId: run.caseId ?? null,
    meshId: run.meshId ?? null,
    machine: run.machine ?? null,
  }
}
