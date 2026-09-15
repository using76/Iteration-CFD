// gui/shared/src/ontology/registry.dc.ts — the data-centre registry: the shipped
// declarations plus docs/13 section 3's ten levels, one version covering both.
import { ACTION_TYPES, type ActionTypeDef } from './actions.js'
import { LINK_TYPES } from './links.js'
import { OBJECT_TYPES } from './objects.js'
import { type LinkTypeDef, type ObjectTypeDef } from './types.js'
import { DC_LINK_TYPES } from './links.dc.js'
import { DC_OBJECT_TYPES } from './objects.dc.js'
import { validateDcRules } from './rules.dc.js'
import { buildRegistry, ONTOLOGY_VERSION, OntologyError, type OntologyInput, type OntologyRegistry } from './registry.js'

/** The data-centre ontology is the shipped ontology plus docs/13 section 3's ten levels; one
 *  version covers both, and this unit does not bump it. */
export const DC_ONTOLOGY_VERSION = ONTOLOGY_VERSION

export interface DcRegistryInput { objects?: ObjectTypeDef[]; links?: LinkTypeDef[]; actions?: ActionTypeDef[] }

/** Builds a registry over the shipped declarations plus the data-centre ones, running the shipped
 *  validator and then the four rules of docs/13 section 3. Throws OntologyError on any problem of
 *  severity 'error'. The `extra` argument exists only so a failing fixture can be passed in and
 *  refused; no production caller ever passes one. */
export function buildDcRegistry(extra: DcRegistryInput = {}): OntologyRegistry {
  const input: OntologyInput = {
    version: DC_ONTOLOGY_VERSION,
    objects: [...OBJECT_TYPES, ...DC_OBJECT_TYPES, ...(extra.objects ?? [])],
    links: [...LINK_TYPES, ...DC_LINK_TYPES, ...(extra.links ?? [])],
    actions: [...ACTION_TYPES, ...(extra.actions ?? [])],
  }
  const dc = validateDcRules(input)
  if (dc.some((x) => x.severity === 'error')) throw new OntologyError(dc)
  return buildRegistry(input)
}

/** The shipped data-centre ontology, built at module load. */
export const DC_ONTOLOGY: OntologyRegistry = buildDcRegistry()
