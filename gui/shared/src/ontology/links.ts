// gui/shared/src/ontology/links.ts — the fourteen link types. Sides follow D-h:
// from.apiName reads on the from object, to.apiName on the to object. There is
// no producedBy and no second Mesh-Run edge: the edge is usesMesh (D-q).
import type { LinkTypeDef, PropertyDef } from './types.js'

const fk = (onType: string, property: string) => ({ kind: 'foreignKey' as const, onType, property })
const V = '0.1.0'

const executed: LinkTypeDef = {
  apiName: 'executed', displayName: 'Executed binary', cardinality: 'MANY_TO_ONE',
  from: { apiName: 'executed', displayName: 'Executions', objectType: 'Run' },
  to: { apiName: 'runs', displayName: 'Runs', objectType: 'Driver' },
  backing: fk('Run', 'binary'), ontologyVersion: V,
}

const runs: LinkTypeDef = {
  apiName: 'runs', displayName: 'Case run', cardinality: 'MANY_TO_ONE',
  from: { apiName: 'case', displayName: 'Case', objectType: 'Run' },
  to: { apiName: 'runs', displayName: 'Runs', objectType: 'Case' },
  backing: fk('Run', 'casePath'), ontologyVersion: V,
}

const atCommit: LinkTypeDef = {
  apiName: 'atCommit', displayName: 'At commit', cardinality: 'MANY_TO_ONE',
  // Does not exist today (DM:132): both foreign keys are the properties N0 adds,
  // so every row is null until N0 lands. PL:35 calls this the single most
  // valuable missing edge, and the reason Run.gitSha is declared before its writer.
  from: { apiName: 'atCommit', displayName: 'Commit', objectType: 'Run' },
  to: { apiName: 'runs', displayName: 'Runs', objectType: 'Commit' },
  backing: fk('Run', 'gitSha'), ontologyVersion: V,
}

const usesMesh: LinkTypeDef = {
  apiName: 'usesMesh', displayName: 'Uses mesh', cardinality: 'MANY_TO_ONE',
  // Does not exist today (DM:132): Run.meshId is written by N0 and is null until
  // then. One fact — this run used this mesh — one edge, one backing (D-q).
  from: { apiName: 'mesh', displayName: 'Mesh', objectType: 'Run' },
  to: { apiName: 'runs', displayName: 'Runs', objectType: 'Mesh' },
  backing: fk('Run', 'meshId'), ontologyVersion: V,
}

const hasPatch: LinkTypeDef = {
  apiName: 'hasPatch', displayName: 'Has patch', cardinality: 'ONE_TO_MANY',
  from: { apiName: 'patches', displayName: 'Patches', objectType: 'Mesh' },
  to: { apiName: 'mesh', displayName: 'Mesh', objectType: 'MeshPatch' },
  backing: fk('MeshPatch', 'meshId'), ontologyVersion: V,
}

const gradedBy: LinkTypeDef = {
  apiName: 'gradedBy', displayName: 'Graded by', cardinality: 'ONE_TO_ONE',
  // OURS: Palantir's one-to-one cardinality page 404s and is UNVERIFIED (R:157);
  // the shape — a foreign key with a unique index on the FK column — is ours.
  from: { apiName: 'qualityReport', displayName: 'Quality report', objectType: 'Mesh' },
  to: { apiName: 'mesh', displayName: 'Mesh', objectType: 'MeshQualityReport' },
  backing: fk('MeshQualityReport', 'meshId'), ontologyVersion: V,
}

const contains: LinkTypeDef = {
  apiName: 'contains', displayName: 'Contains', cardinality: 'ONE_TO_MANY',
  from: { apiName: 'regions', displayName: 'Regions', objectType: 'RegionLayout' },
  to: { apiName: 'layout', displayName: 'Layout', objectType: 'Region' },
  backing: fk('Region', 'layoutId'), ontologyVersion: V,
}

const declares: LinkTypeDef = {
  apiName: 'declares', displayName: 'Declares', cardinality: 'ONE_TO_MANY',
  from: { apiName: 'interfaces', displayName: 'Interfaces', objectType: 'RegionLayout' },
  to: { apiName: 'layout', displayName: 'Layout', objectType: 'Interface' },
  backing: fk('Interface', 'layoutId'), ontologyVersion: V,
}

const joins: LinkTypeDef = {
  apiName: 'joins', displayName: 'Joins', cardinality: 'MANY_TO_MANY',
  // Exactly the three facts that belong to one side of the join (D-g, §2.3 cap).
  // The importer writes two rows per interface.
  from: { apiName: 'regions', displayName: 'Regions', objectType: 'Interface' },
  to: { apiName: 'interfaces', displayName: 'Interfaces', objectType: 'Region' },
  backing: { kind: 'joinTable', projection: 'regionsManifestInterfaceSides' },
  properties: [
    { apiName: 'side', displayName: 'Side', baseType: 'string', nullable: false, description: 'Which of the two sides of the interface this region is.', valueType: 'enum', enumValues: ['a', 'b'] },
    { apiName: 'patch', displayName: 'Patch', baseType: 'string', nullable: false, description: 'The patch name on this side.' },
    { apiName: 'faces', displayName: 'Faces', baseType: 'integer', nullable: true, description: 'Faces on this side.' },
  ] as PropertyDef[],
  ontologyVersion: V,
}

const usesLayout: LinkTypeDef = {
  apiName: 'usesLayout', displayName: 'Uses layout', cardinality: 'MANY_TO_ONE',
  from: { apiName: 'layout', displayName: 'Layout', objectType: 'Case' },
  to: { apiName: 'cases', displayName: 'Cases', objectType: 'RegionLayout' },
  backing: fk('Case', 'regionsManifest'), ontologyVersion: V,
}

const belongsTo: LinkTypeDef = {
  apiName: 'belongsTo', displayName: 'Belongs to', cardinality: 'MANY_TO_ONE',
  from: { apiName: 'session', displayName: 'Session', objectType: 'ToolCall' },
  to: { apiName: 'toolCalls', displayName: 'Tool calls', objectType: 'Session' },
  backing: fk('ToolCall', 'sessionId'), ontologyVersion: V,
}

const started: LinkTypeDef = {
  apiName: 'started', displayName: 'Started', cardinality: 'MANY_TO_ONE',
  // MANY_TO_ONE, not one-to-one: 139 tool-call rows name 61 distinct runs.
  from: { apiName: 'run', displayName: 'Run', objectType: 'ToolCall' },
  to: { apiName: 'toolCalls', displayName: 'Tool calls', objectType: 'Run' },
  backing: fk('ToolCall', 'runId'), ontologyVersion: V,
}

const touched: LinkTypeDef = {
  apiName: 'touched', displayName: 'Touched', cardinality: 'MANY_TO_MANY',
  // A join table for the same reason: 68 session.runs[] entries over 61 distinct runs.
  from: { apiName: 'runs', displayName: 'Runs', objectType: 'Session' },
  to: { apiName: 'sessions', displayName: 'Sessions', objectType: 'Run' },
  backing: { kind: 'joinTable', projection: 'sessionRuns' },
  ontologyVersion: V,
}

const attachedTo: LinkTypeDef = {
  apiName: 'attachedTo', displayName: 'Attached to', cardinality: 'MANY_TO_MANY',
  // OURS and UNVERIFIED (fact sheet §6.1: attachments are uncovered by the research).
  // One link type points at ONE subject type — the LinkSide shape has no polymorphism —
  // and the subject today is the Session the file arrived on; N6's upload route
  // creates this edge. Many-to-many because one screenshot can be evidence on
  // more than one subject (R:155).
  from: { apiName: 'attachedTo', displayName: 'Subjects', objectType: 'Attachment' },
  to: { apiName: 'attachments', displayName: 'Attachments', objectType: 'Session' },
  backing: { kind: 'joinTable', projection: 'attachedTo' },
  properties: [
    { apiName: 'role', displayName: 'Role', baseType: 'string', nullable: false, description: 'Why the file is attached.', valueType: 'enum', enumValues: ['evidence', 'request', 'result', 'reference'] },
  ],
  ontologyVersion: V,
}

export const LINK_TYPES: LinkTypeDef[] = [executed, runs, atCommit, usesMesh, hasPatch, gradedBy, contains, declares, joins, usesLayout, belongsTo, started, touched, attachedTo]
