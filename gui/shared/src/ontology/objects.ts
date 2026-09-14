// gui/shared/src/ontology/objects.ts — the twelve object types, one const per
// type. Every property name, enum word and count comes from the tree (protocol.ts,
// casejsonc.ts, meshSummary.ts, regions.ts, session.ts, registry.ts, quality.rs,
// facts-data.md) — never invented.
import type { BaseType, ObjectTypeDef, PropertyDef } from './types.js'

const p = (apiName: string, baseType: BaseType, displayName: string, description: string, nullable = false, extra: Partial<PropertyDef> = {}): PropertyDef =>
  ({ apiName, displayName, baseType, nullable, description, ...extra })
const en = (values: string[]): Partial<PropertyDef> => ({ valueType: 'enum', enumValues: values })
const arr = (items: BaseType): Partial<PropertyDef> => ({ items })
const wp: Partial<PropertyDef> = { valueType: 'workspacePath' }

/** One execution of a binary. cwd, time and endTime are deliberately not declared:
 *  cwd is "" in all 221 records on disk, endTime null in all, time non-null once
 *  (facts-data §1.2) — R:518 stores each fact once, R:513 keeps dead fields out. */
const Run: ObjectTypeDef = {
  apiName: 'Run', displayName: 'Run', pluralName: 'Runs',
  description: 'One execution of a solver, mesh or analysis binary, with its command line, outcome and last residual.',
  icon: 'run',
  source: { projection: 'runs', paths: ['gui/runs/*/run.json'] },
  primaryKey: 'runId', titleKey: 'label',
  properties: [
    p('runId', 'string', 'Run', "Minted id, r_<n> (runs/manager.ts:332)."),
    p('label', 'string', 'Label', 'Short label shown in the run list.', true),
    p('binary', 'string', 'Binary', 'Registry binary or pipeline name.'),
    p('argv', 'array', 'Arguments', 'Command line as launched; argv[0] is the registry name.', false, arr('string')),
    p('casePath', 'string', 'Case path', 'Workspace-relative case file or directory; null for binaries that take no case.', true, wp),
    p('outputRoot', 'string', 'Output root', 'Where results land, workspace-relative.', true, wp),
    p('status', 'string', 'Status', 'queued, running, done, failed, killed or diverged; the 17 running rows on disk are servers that died mid-run and were never rewritten.', false, en(['queued', 'running', 'done', 'failed', 'killed', 'diverged'])),
    p('pid', 'integer', 'Pid', 'OS process id while running.', true),
    p('startedAt', 'timestamp', 'Started at', 'ISO-8601 launch time.'),
    p('endedAt', 'timestamp', 'Ended at', 'ISO-8601 end time, null while still running.', true),
    p('exitCode', 'integer', 'Exit code', 'Process exit code.', true),
    p('signal', 'string', 'Signal', 'Termination signal, when killed.', true),
    p('iter', 'integer', 'Iterations', 'Latest iteration or step seen in the log.'),
    p('targetIter', 'integer', 'Target iterations', 'Planned iterations when known.', true),
    p('lastResidual', 'json', 'Last residual', 'Field name to residual; never queried by field.', true),
    p('written', 'array', 'Written', 'Directories named by written-to lines, in order, workspace-relative.', false, { items: 'string', valueType: 'workspacePath' }),
    p('error', 'string', 'Error', 'First error line captured from the log.', true),
    p('converged', 'boolean', 'Converged', 'Whether the run met its convergence check.'),
    p('device', 'string', 'Device', 'Free-text device banner, not a machine id.', true),
    p('logLines', 'integer', 'Log lines', 'Line count of the run log.'),
    p('mode', 'string', 'Mode', 'real or demo.', false, en(['real', 'demo'])),
    p('startedBy', 'string', 'Started by', 'Principal id that launched the run; null on every imported row.', true),
    p('gitSha', 'string', 'Git sha', 'Full HEAD sha when the run was created; null until N0 stamps it.', true, { valueType: 'gitSha' }),
    p('gitDirty', 'boolean', 'Git dirty', 'True when tracked content differed from HEAD at launch; null until N0.', true),
    p('caseId', 'string', 'Case', 'The Case this run is a run of, workspace-relative; null until N0.', true, wp),
    p('meshId', 'string', 'Mesh', 'Mesh primary key, summary path # name; null until N0.', true),
    p('machine', 'struct', 'Machine', 'The machine the run executed on; hostname is the main field.', true, { fields: { hostname: { baseType: 'string', main: true }, gpu: { baseType: 'string' }, platform: { baseType: 'string' } } }),
  ],
  ontologyVersion: '0.1.0',
}

/** A simulation case file. The path is the primary key; the name is not the path
 *  (cases/plume.jsonc is named plumeB). A cht case is identified by its filename
 *  suffix, not by content. */
const Case: ObjectTypeDef = {
  apiName: 'Case', displayName: 'Case', pluralName: 'Cases',
  description: 'A simulation case: physics, turbulence, regions and the run block, read from one .jsonc or OpenFOAM directory.',
  icon: 'case',
  source: { projection: 'cases', paths: ['cases/**/*.jsonc'] },
  primaryKey: 'caseId', titleKey: 'name',
  properties: [
    p('caseId', 'string', 'Case', 'Workspace-relative path of the case file.', false, wp),
    p('name', 'string', 'Name', 'Case name; not the path (plume.jsonc is named plumeB).'),
    p('format', 'string', 'Format', 'case-1, cht-1, dc or foamDir.', false, en(['case-1', 'cht-1', 'dc', 'foamDir'])),
    p('turbulenceKind', 'string', 'Turbulence kind', 'Turbulence family, when the case closes with one.', true),
    p('turbulenceModel', 'string', 'Turbulence model', 'Model name, when the case closes with one.', true),
    p('mode', 'string', 'Mode', 'run.mode as a string, e.g. stress; null when absent.', true),
    p('regionsManifest', 'string', 'Regions manifest', 'mesh.regions manifest path, relative to the case file directory.', true, wp),
    p('outputDir', 'string', 'Output directory', 'Workspace-relative output directory.', false, wp),
    p('nRegions', 'integer', 'Regions', 'Region count.'),
    p('nInterfaces', 'integer', 'Interfaces', 'Interface count.'),
    p('nPatchRules', 'integer', 'Patch rules', 'Ordered patch rule count.'),
    p('runBlock', 'json', 'Run block', 'The run block verbatim.', true),
    p('schemaUrl', 'string', 'Schema URL', '$schema URL when present (7 of 14 files).', true, { valueType: 'url' }),
  ],
  ontologyVersion: '0.1.0',
}

/** A binary or script pipeline the program can run. Declared, never imported:
 *  the registry constant is the source of truth, kept in step with Cargo by an
 *  existing vitest sync test (registry.ts:1-7). */
const Driver: ObjectTypeDef = {
  apiName: 'Driver', displayName: 'Driver', pluralName: 'Drivers',
  description: 'Declared, never imported: the registry constant is the source of truth for the binaries the program exposes.',
  icon: 'driver',
  source: { projection: 'driverRegistry', paths: ['gui/shared/src/registry.ts'] },
  primaryKey: 'name', titleKey: 'name',
  properties: [
    p('name', 'string', 'Name', 'Cargo [[bin]] name or pipeline name.'),
    p('source', 'string', 'Source', 'Source path under rust/.', false, wp),
    p('purpose', 'string', 'Purpose', 'One-line purpose.'),
    p('kind', 'string', 'Kind', 'mesh, solver, analysis, bench or diagnostic.'),
    p('accepts', 'array', 'Accepts', 'Case formats the driver reads.', false, arr('string')),
    p('builds', 'array', 'Builds', 'Turbulence models this driver can construct.', false, arr('string')),
    p('residualStyle', 'string', 'Residual style', 'Residual style of the log.'),
    p('longRunning', 'boolean', 'Long running', 'Whether the run is long running.'),
    p('gpu', 'boolean', 'GPU', 'Whether the run needs the GPU.'),
    p('pending', 'boolean', 'Pending', 'Declared before its Cargo bin exists.'),
    p('pipeline', 'boolean', 'Pipeline', 'A shell script pipeline, not a Cargo binary.'),
    p('summary', 'string', 'Summary', 'One-line summary shown in the tools panel.'),
  ],
  ontologyVersion: '0.1.0',
}

/** A git commit; the tree's provenance backbone (255 of them). */
const Commit: ObjectTypeDef = {
  apiName: 'Commit', displayName: 'Commit', pluralName: 'Commits',
  description: 'One git commit, with its subject, author and branch.',
  icon: 'commit',
  source: { projection: 'gitLog', paths: ['.git'] },
  primaryKey: 'sha', titleKey: 'subject',
  properties: [
    p('sha', 'string', 'Sha', 'Full 40-hex commit sha; the short form is its own property, never a key.', false, { valueType: 'gitSha' }),
    p('shortSha', 'string', 'Short sha', 'The abbreviated sha.'),
    p('subject', 'string', 'Subject', 'Subject line.'),
    p('author', 'string', 'Author', 'Author name.'),
    p('authoredAt', 'timestamp', 'Authored at', 'Author date.'),
    p('committedAt', 'timestamp', 'Committed at', 'Commit date.'),
    p('nFiles', 'integer', 'Files', 'Changed file count.'),
    p('branch', 'string', 'Branch', 'Branch, null when detached.', true),
  ],
  ontologyVersion: '0.1.0',
}

/** A mesh, keyed by its summary file path and mesh name joined with # — never
 *  caseDir + / + name (CONTRACT §3). No bbox: the bounding box belongs to the
 *  surface, i.e. to Geometry, which is not one of the twelve. */
const Mesh: ObjectTypeDef = {
  apiName: 'Mesh', displayName: 'Mesh', pluralName: 'Meshes',
  description: 'A generated mesh, with its counts, tool and wall seconds, as one mesh summary states them.',
  icon: 'mesh',
  source: { projection: 'meshSummaries', paths: ['**/*_summary.json', '**/constant/polyMesh/.meshSummary.json'] },
  primaryKey: 'meshId', titleKey: 'name',
  properties: [
    p('meshId', 'string', 'Mesh', 'Summary file path and mesh name joined with #.'),
    p('name', 'string', 'Name', 'Mesh name.'),
    p('tool', 'string', 'Tool', 'ofgpu-automesher, step_mesh, ofgpu-generate-mesh or ofgpu-convert-mesh — the four words the writers stamp.', false, en(['ofgpu-automesher', 'step_mesh', 'ofgpu-generate-mesh', 'ofgpu-convert-mesh'])),
    p('caseDir', 'string', 'Case directory', 'Workspace-relative case directory.', false, wp),
    p('configPath', 'string', 'Config path', 'Config file used, when known.', true, wp),
    p('tag', 'string', 'Tag', 'Tag suffix, when one was used.', true),
    p('nCells', 'integer', 'Cells', 'Cell count.', true),
    p('nPoints', 'integer', 'Points', 'Point count.', true),
    p('nInternalFaces', 'integer', 'Internal faces', 'Internal face count.', true),
    p('nBoundaryFaces', 'integer', 'Boundary faces', 'Boundary face count.', true),
    p('nRegions', 'integer', 'Regions', 'Disjoint cell region count.', true),
    p('totalSeconds', 'double', 'Total seconds', 'Wall seconds the mesher spent.', true, { valueType: 'seconds' }),
    p('stoppedAfter', 'string', 'Stopped after', 'Stage the mesher stopped after; null on a full mesh.', true),
    p('runId', 'string', 'Run', 'Run that wrote the summary; null on every row today, since no .meshSummary.json exists yet.', true),
    p('writtenAt', 'timestamp', 'Written at', 'When the summary was written.', true),
  ],
  ontologyVersion: '0.1.0',
}

/** One row per patch, written by the importer's mesh fold (D-p). boundaryType
 *  stays a plain string: the polyMesh reader passes through whatever the
 *  boundary file says, so closing the enum would refuse a real mesh. */
const MeshPatch: ObjectTypeDef = {
  apiName: 'MeshPatch', displayName: 'Mesh patch', pluralName: 'Mesh patches',
  description: "One row per patch, written by the importer's mesh fold: a boundary patch of one mesh.",
  icon: 'patch',
  source: { projection: 'meshSummaryPatches', paths: ['**/*_summary.json', '**/constant/polyMesh/.meshSummary.json'] },
  primaryKey: 'patchId', titleKey: 'name',
  properties: [
    p('patchId', 'string', 'Patch', 'Mesh id and patch name joined with /.'),
    p('meshId', 'string', 'Mesh', 'The mesh this patch belongs to.'),
    p('name', 'string', 'Name', 'Patch name from the boundary file.'),
    p('boundaryType', 'string', 'Boundary type', "The boundary file's own type word, e.g. patch or wall; a plain string, never a closed enum.", true),
    p('nFaces', 'integer', 'Faces', 'Face count.', true),
  ],
  ontologyVersion: '0.1.0',
}

/** One report per mesh, written by the importer's mesh fold (D-p). A failed
 *  automesher run writes no file at all, so passed is true on every row that
 *  exists; the absence of a report is itself the signal. */
const MeshQualityReport: ObjectTypeDef = {
  apiName: 'MeshQualityReport', displayName: 'Mesh quality report', pluralName: 'Mesh quality reports',
  description: "One report per mesh, written by the importer's mesh fold: the G1-G7 gate verdict for one mesh.",
  icon: 'quality',
  source: { projection: 'meshSummaryQuality', paths: ['**/*_summary.json', '**/constant/polyMesh/.meshSummary.json'] },
  primaryKey: 'reportId', titleKey: 'label',
  properties: [
    p('reportId', 'string', 'Report', 'Equals the meshId it grades.'),
    p('meshId', 'string', 'Mesh', 'The mesh graded.'),
    p('label', 'string', 'Label', 'G1-G7 for the mesh name.'),
    p('passed', 'boolean', 'Passed', 'Whether every gate passed; true on every row that exists.', true),
    p('failedGates', 'array', 'Failed gates', 'The seven gate names, exactly: G1 (positive volume), G2 (closure), G3 (one cell region), G4 (non-orthogonality), G5 (thickness), G6 (gradient conditioning), G7 (addressing).', false, arr('string')),
    p('minVolume', 'double', 'Min volume', 'Smallest cell volume (G1).', true),
    p('minVolumeCell', 'integer', 'Min volume cell', 'Cell with the smallest volume.', true),
    p('maxClosure', 'double', 'Max closure', 'Worst closure gap (G2).', true),
    p('maxClosureCell', 'integer', 'Max closure cell', 'Cell with the worst closure.', true),
    p('nRegions', 'integer', 'Regions', 'Disjoint cell region count (G3).', true),
    p('regionSizes', 'array', 'Region sizes', 'Cells per region.', false, arr('integer')),
    p('maxNonOrthDeg', 'double', 'Max non-orthogonality', 'Worst face non-orthogonality in degrees (G4).', true),
    p('meanNonOrthDeg', 'double', 'Mean non-orthogonality', 'Mean face non-orthogonality in degrees.', true),
    p('nNonOrthOverReport', 'integer', 'Non-orthogonality over report', 'Faces above the reporting threshold.', true),
    p('minThicknessRatio', 'double', 'Min thickness ratio', 'Worst layer thickness ratio (G5).', true),
    p('minThicknessCell', 'integer', 'Min thickness cell', 'Cell with the worst thickness ratio.', true),
    p('maxCond', 'double', 'Max conditioning', 'Worst gradient conditioning (G6).', true),
    p('maxCondCell', 'integer', 'Max conditioning cell', 'Cell with the worst conditioning.', true),
    p('nDuplicateFaces', 'integer', 'Duplicate faces', 'Duplicate face count (G7).', true),
    p('lduOrdered', 'boolean', 'LDU ordered', 'Whether the addressing is LDU ordered (G7).', true),
  ],
  ontologyVersion: '0.1.0',
}

/** One regions.json manifest: how a mesh is split into regions. */
const RegionLayout: ObjectTypeDef = {
  apiName: 'RegionLayout', displayName: 'Region layout', pluralName: 'Region layouts',
  description: 'One regions.json manifest: which regions a mesh splits into and how they join.',
  icon: 'layout',
  source: { projection: 'regionsManifest', paths: ['**/regions.json'] },
  primaryKey: 'layoutId', titleKey: 'name',
  properties: [
    p('layoutId', 'string', 'Layout', 'Workspace-relative path of the regions.json.', false, wp),
    p('name', 'string', 'Name', 'Layout name.'),
    p('version', 'integer', 'Version', 'Manifest version; the reader throws unless it is 1.'),
    p('units', 'string', 'Units', 'Length units.', true),
    p('tolerance', 'double', 'Tolerance', 'Snap tolerance in metres.', true, { valueType: 'metres' }),
    p('sourceTool', 'string', 'Source tool', 'Tool that wrote the manifest.', true),
    p('sourceVersion', 'string', 'Source version', 'Version of that tool.', true),
    p('sourceGeometry', 'string', 'Source geometry', 'Geometry file the regions came from.', true, wp),
    p('sourceConfig', 'string', 'Source config', 'Config used to split the mesh.', true, wp),
    p('nRegions', 'integer', 'Regions', 'Region count.'),
    p('nInterfaces', 'integer', 'Interfaces', 'Interface count.'),
  ],
  ontologyVersion: '0.1.0',
}

/** One region of a layout. kind is nullable: four dieStack regions carry none
 *  and the reader leaves it undefined. */
const Region: ObjectTypeDef = {
  apiName: 'Region', displayName: 'Region', pluralName: 'Regions',
  description: 'One region of a layout: a fluid or solid block with its own polyMesh and material.',
  icon: 'region',
  source: { projection: 'regionsManifestRegions', paths: ['**/regions.json'] },
  primaryKey: 'regionId', titleKey: 'name',
  properties: [
    p('regionId', 'string', 'Region', 'Layout id and region name joined with #.'),
    p('layoutId', 'string', 'Layout', 'The layout declaring this region.'),
    p('name', 'string', 'Name', 'Region name.'),
    p('kind', 'string', 'Kind', 'fluid or solid; null when the manifest declares none.', true, en(['fluid', 'solid'])),
    p('polyMeshPath', 'string', 'PolyMesh path', 'Workspace-relative polyMesh directory.', false, wp),
    p('materialName', 'string', 'Material', 'Material name; the writer only warns when absent.', true),
    p('cellCount', 'integer', 'Cells', 'Cell count.', true),
  ],
  ontologyVersion: '0.1.0',
}

/** A pair of regions joined across a conformal interface. The per-side facts
 *  (patch, faces, which side) live on the joins link, not here (D-g). */
const Interface: ObjectTypeDef = {
  apiName: 'Interface', displayName: 'Interface', pluralName: 'Interfaces',
  description: 'A pair of regions joined across a conformal interface, with its tolerance and contact resistance.',
  icon: 'interface',
  source: { projection: 'regionsManifestInterfaces', paths: ['**/regions.json'] },
  primaryKey: 'interfaceId', titleKey: 'name',
  properties: [
    p('interfaceId', 'string', 'Interface', 'Layout id and both patch names joined with # and |.'),
    p('layoutId', 'string', 'Layout', 'The layout declaring this interface.'),
    p('name', 'string', 'Name', 'a_to_b / b_to_a.'),
    p('tolerance', 'double', 'Tolerance', 'Interface tolerance in metres.', true, { valueType: 'metres' }),
    p('rc', 'double', 'Contact resistance', 'Contact resistance across the interface.', true),
    p('kappaLayers', 'json', 'Kappa layers', 'The thickness and kappa stack, verbatim.', true),
  ],
  ontologyVersion: '0.1.0',
}

/** One assistant conversation. The messages[], ui[] and toolCalls[] arrays are
 *  not properties: toolCalls becomes the ToolCall type and the 1.2 MB of message
 *  history stays on disk, referenced. */
const Session: ObjectTypeDef = {
  apiName: 'Session', displayName: 'Session', pluralName: 'Sessions',
  description: 'One assistant conversation: model, settings and counters, with the history itself kept on disk.',
  icon: 'session',
  source: { projection: 'sessions', paths: ['gui/sessions/s_*.json'] },
  primaryKey: 'sessionId', titleKey: 'title',
  properties: [
    p('sessionId', 'string', 'Session', 's_<base36> id.'),
    p('title', 'string', 'Title', 'First user line, truncated.'),
    p('model', 'string', 'Model', 'Model name.'),
    p('createdAt', 'timestamp', 'Created at', 'ISO-8601 creation time.'),
    p('updatedAt', 'timestamp', 'Updated at', 'ISO-8601 last write.'),
    p('autoApprove', 'string', 'Auto approve', 'none, reads or all.', false, en(['none', 'reads', 'all'])),
    p('effort', 'string', 'Effort', 'low, medium, high, xhigh or max.', false, en(['low', 'medium', 'high', 'xhigh', 'max'])),
    p('locale', 'string', 'Locale', 'ko or en.', false, en(['ko', 'en'])),
    p('notifyOnRunEnd', 'boolean', 'Notify on run end', 'Whether run completion notifies.'),
    p('nMessages', 'integer', 'Messages', 'Message count; the messages stay on disk.'),
    p('nToolCalls', 'integer', 'Tool calls', 'Tool call count.'),
    p('casePath', 'string', 'Case path', 'Case the conversation was about; absent from all 80 files on disk.', true, wp),
    p('allowedTools', 'array', 'Allowed tools', 'Remembered tool approvals.', false, arr('string')),
  ],
  ontologyVersion: '0.1.0',
}

/** One tool invocation inside a session. startedAt/endedAt are epoch ms on disk
 *  while Run.startedAt is ISO-8601; timestamp covers both and the importer
 *  normalises. */
const ToolCall: ObjectTypeDef = {
  apiName: 'ToolCall', displayName: 'Tool call', pluralName: 'Tool calls',
  description: 'One tool invocation inside a session: name, policy, status and its one-line summary.',
  icon: 'tool',
  source: { projection: 'sessionToolCalls', paths: ['gui/sessions/s_*.json'] },
  primaryKey: 'toolUseId', titleKey: 'summary',
  properties: [
    p('toolUseId', 'string', 'Tool use', 'The call id.'),
    p('sessionId', 'string', 'Session', 'Session holding the call.'),
    p('name', 'string', 'Name', 'Tool name.'),
    p('summary', 'string', 'Summary', 'One-line human summary for the card.'),
    p('policy', 'string', 'Policy', 'auto, ask or never.', false, en(['auto', 'ask', 'never'])),
    p('status', 'string', 'Status', 'pending, awaiting_approval, running, ok, error, denied or cancelled.', false, en(['pending', 'awaiting_approval', 'running', 'ok', 'error', 'denied', 'cancelled'])),
    p('runId', 'string', 'Run', 'Run the call started, when it did (139 of 524 rows).', true),
    p('startedAt', 'timestamp', 'Started at', 'Epoch ms on disk; null when unknown.', true),
    p('endedAt', 'timestamp', 'Ended at', 'Epoch ms on disk.', true),
    p('error', 'string', 'Error', 'Error line when the call failed.', true),
    p('resultPreview', 'string', 'Result preview', 'Truncated result preview.', true),
    p('input', 'json', 'Input', 'Arguments verbatim.'),
  ],
  ontologyVersion: '0.1.0',
}

export const OBJECT_TYPES: ObjectTypeDef[] = [Run, Case, Driver, Commit, Mesh, MeshPatch, MeshQualityReport, RegionLayout, Region, Interface, Session, ToolCall]
