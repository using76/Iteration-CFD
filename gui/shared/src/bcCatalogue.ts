// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The solver's boundary-condition catalogue (SPEC-LIT 120, docs/17 BC-22): every public type name and its
// one status. The data is bcCatalogue.data.ts, which rust/src/field/catalogue_json.rs generates from
// IMPLEMENTED_BC_NAMES and rust/src/field/catalogue.rs, together with docs/schema/bc-catalogue-1.json;
// server/src/registry/sync.test.ts holds the three to each other. The catalogue lists no per-name entry
// keys: a JSONC case's per-field conditions and their required members are docs/schema/case-1.json's.
import { BC_CATALOGUE_DATA } from './bcCatalogue.data'

export type BcStatus = 'implemented' | 'alias' | 'planned' | 'owned' | 'refused'

/** One catalogue row. A trailing `*` in `name` marks a refused prefix family. */
export interface BcCatalogueEntry {
  family: string
  name: string
  status: BcStatus
  /** alias: the implemented name it is read as. */
  to?: string
  /** planned / owned: the plan unit that implements it (BC-NN, CMP-NN, CHR-NN). */
  unit?: string
  /** refused: a phrase of the reason the solver's refusal prints. */
  reason?: string
  /** The field the solver is asked on; `psi` is any field. */
  field: string
  extendedBy: string[]
  probes: string[]
}

export interface BcCatalogue {
  $comment: string
  version: 1
  counts: Record<BcStatus, number>
  /** Every name the solver's BcKind::from_name accepts, in IMPLEMENTED_BC_NAMES order. */
  implemented: string[]
  entries: BcCatalogueEntry[]
}

export const BC_CATALOGUE: BcCatalogue = BC_CATALOGUE_DATA

/** The names a foam case directory's 0/ files may use as a condition type. */
export const IMPLEMENTED_BC_TYPES: readonly string[] = BC_CATALOGUE.implemented

/** The row a type name falls under: the exact name, else the refused prefix family it starts with; null when the catalogue does not know it. */
export function bcCatalogueEntry(name: string): BcCatalogueEntry | null {
  const exact = BC_CATALOGUE.entries.find((e) => e.name === name)
  if (exact) return exact
  return BC_CATALOGUE.entries.find((e) => e.name.endsWith('*') && name.startsWith(e.name.slice(0, -1))) ?? null
}

/** One clause saying what the solver does with a type name. */
export function describeBcType(name: string): string {
  const e = bcCatalogueEntry(name)
  if (!e) return `${name} is not a boundary-condition type the solver knows`
  switch (e.status) {
    case 'implemented':
      return `${name} is implemented by the solver`
    case 'alias':
      return `${name} is read by the solver as ${e.to ?? '?'}`
    case 'planned':
    case 'owned':
      return `${name} is not implemented yet (planned in ${e.unit ?? '?'})`
    case 'refused':
      return `${name} is refused by the solver (${e.reason ?? 'by name'})`
  }
}
