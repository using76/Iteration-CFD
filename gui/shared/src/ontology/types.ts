// gui/shared/src/ontology/types.ts — the single owner of the object-type and
// link-type vocabulary (facts-aip-contract.md §1.1, §2.1, verbatim). The
// registry in ./registry.ts validates literals of these shapes at runtime.

export type ObjectTypeApiName = string   // PascalCase, /^[A-Za-z][A-Za-z0-9]{0,99}$/
export type PropertyApiName   = string   // camelCase,  /^[a-z][A-Za-z0-9]{0,99}$/

export type BaseType =
  | 'string' | 'integer' | 'long' | 'double' | 'boolean'
  | 'timestamp'          // ISO-8601 on the wire, epoch ms in SQLite
  | 'date'
  | 'json'               // opaque blob stored verbatim; never queried by field
  | 'struct'             // grouped scalars, one designated main field
  | 'array'              // of any non-array BaseType
  | 'attachmentRef'      // -> Attachment.attachmentId, see §6

export interface PropertyDef {
  apiName: PropertyApiName
  displayName: string
  baseType: BaseType
  items?: BaseType                                                    // iff baseType === 'array'
  fields?: Record<string, { baseType: BaseType; main?: true }>        // iff baseType === 'struct'
  /** Semantic wrapper; narrows the base type, never widens it. */
  valueType?: 'workspacePath' | 'gitSha' | 'sha256' | 'uuid' | 'url'
            | 'enum' | 'seconds' | 'metres' | 'kelvin'
  enumValues?: string[]          // required iff valueType === 'enum'
  nullable: boolean              // no optionals: every property is present, possibly null
  description: string            // shown in search results (R:142)
  derived?: { function: string } // computed on read; never written by an action
}

export interface ObjectTypeDef {
  apiName: ObjectTypeApiName
  displayName: string
  pluralName: string
  description: string
  icon: string                                       // UI only
  source: { projection: string; paths: string[] }    // exactly one projection; see ADOPT-3
  primaryKey: PropertyApiName
  titleKey: PropertyApiName
  properties: PropertyDef[]
  actionCreatedOnly?: true       // then source.projection === ""
  ontologyVersion: string        // the version this type last changed in
}

export type Cardinality = 'ONE_TO_ONE' | 'ONE_TO_MANY' | 'MANY_TO_ONE' | 'MANY_TO_MANY'

/**
 * One end of a link. `from.apiName` is the accessor on a `from.objectType` object that reaches the
 * `to` side; `to.apiName` is the accessor on a `to.objectType` object that reaches the `from` side.
 * So `atCommit` reads `run.atCommit` from the Run and `commit.runs` from the Commit (R:159-160).
 */
export interface LinkSide {
  apiName: string            // camelCase; the accessor on THIS side's object type (see D-h)
  displayName: string
  objectType: ObjectTypeApiName
}

export interface LinkTypeDef {
  apiName: string            // camelCase on the wire (`atCommit`); kebab in prose (`at-commit`)
  displayName: string
  from: LinkSide
  to: LinkSide
  cardinality: Cardinality
  backing:
    | { kind: 'foreignKey';  onType: ObjectTypeApiName; property: PropertyApiName }
    | { kind: 'joinTable';   projection: string }
    | { kind: 'objectBacked'; joinObjectType: ObjectTypeApiName }
  properties?: PropertyDef[] // only with joinTable; max 3; scalars only (§2.3)
  ordered?: true             // properties must then contain an integer `index`
  ontologyVersion: string
}

// From RESEARCH.md R:141 / facts ADOPT-4 — the nine, and only these nine.
export const RESERVED_API_NAMES = ['ontology', 'object', 'property', 'link', 'relation', 'rid', 'primaryKey', 'typeId', 'ontologyObject'] as const
