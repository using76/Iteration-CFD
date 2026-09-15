// gui/shared/src/ontology/seed/dc.concepts.ts — the fifteen Concept rows of
// the data-centre seed, and the links that leave them. Every row's source is
// docs/13-data-centre-axis.md: its section 4 is the only statement in this
// repository of what is and is not modelled.
import { conceptId, dcLink, dcRow, type SeedLink, type SeedObject } from './dc.types.js'

const SRC = 'docs/13-data-centre-axis.md'

const C = (slug: string, title: string, status: string, oneLine: string): SeedObject => {
  const id = conceptId(slug)
  return dcRow('Concept', id, SRC, { slug: id, title, oneLine, status })
}

export const DC_CONCEPTS: readonly SeedObject[] = Object.freeze([
  C('buoyancy', 'Buoyancy', 'modelled', 'Air made lighter by heat rises; the solver carries it as a density-ratio body force, not a Boussinesq linearisation.'),
  C('turbulence', 'Turbulence', 'modelled', 'Unresolved eddies mix momentum and heat; the data-centre driver models them one way only, with standard k-epsilon.'),
  C('air-management', 'Air management', 'modelled', 'How much of the cold air reaches a rack inlet before it is spoiled, expressed as three indices rather than a picture.'),
  C('recirculation', 'Recirculation', 'modelled', 'Hot air returning to a rack inlet instead of leaving through the return; the solver reports it as a return index above 100 per cent.'),
  C('bypass', 'Bypass', 'modelled', 'Cold air leaving through the return without passing a rack; the solver reports it as a return index below 100 per cent.'),
  C('stratification', 'Stratification', 'absent', 'A stably layered room resists vertical mixing; the solver solves the field but reports no layer height and no stability number.'),
  C('mixed-convection', 'Mixed convection', 'absent', 'Where buoyancy and shear are comparable a wall function is least defensible; nothing in the data-centre driver computes that ratio.'),
  C('it-heat-load', 'IT heat load', 'modelled', 'A rack is a stated power released into a box of cells and a stated volume flow; it heats the air but does not move it.'),
  C('condensation', 'Condensation', 'refused', 'Water leaving the air as liquid; the solver reports supersaturated cells and refuses by name to model the phase change.'),
  C('dehumidification', 'Dehumidification', 'absent', 'Moisture removed at a cold coil; there is no coil in this model, so the latent split has nowhere to be accounted.'),
  C('containment', 'Containment', 'absent', 'A physical barrier between the cold and hot aisles; a whole patch can carry a very large loss coefficient, an aisle roof cannot.'),
  C('leakage', 'Leakage', 'absent', 'Air escaping through cable cut-outs, tile gaps and door gaps; only a whole side of the room can be an opening here.'),
  C('thermal-mass', 'Thermal mass', 'absent', 'The heat the room and its contents can absorb before the air warms; a steady solver has no time in which to absorb it.'),
  C('free-cooling', 'Free cooling', 'absent', 'Cooling without mechanical refrigeration, bounded by the highest supply temperature the rack inlets still tolerate; the sweep that would find it is not run.'),
  C('capture', 'Capture', 'absent', 'What share of a rack\'s intake comes from a cooling source rather than from the room; it needs one passive tracer per source.'),
])

// expressedBy is n:n and not required: leakage and thermal-mass point at no
// equation this seed carries, on purpose. refusedBy exists only for a concept
// whose status is refused, and exactly one is.
export const DC_CONCEPT_LINKS: readonly SeedLink[] = Object.freeze([
  dcLink('Concept', 'expressedBy', conceptId('buoyancy'), 'EQ-C4', SRC),
  dcLink('Concept', 'expressedBy', conceptId('buoyancy'), 'EQ-B7', SRC),
  dcLink('Concept', 'expressedBy', conceptId('turbulence'), 'EQ-T1', SRC),
  dcLink('Concept', 'expressedBy', conceptId('turbulence'), 'EQ-T9', SRC),
  dcLink('Concept', 'expressedBy', conceptId('air-management'), 'EQ-D-RCI', SRC),
  dcLink('Concept', 'expressedBy', conceptId('air-management'), 'EQ-D-RTI', SRC),
  dcLink('Concept', 'expressedBy', conceptId('air-management'), 'EQ-D-SHI', SRC),
  dcLink('Concept', 'expressedBy', conceptId('recirculation'), 'EQ-D-RTI', SRC),
  dcLink('Concept', 'expressedBy', conceptId('bypass'), 'EQ-D-RTI', SRC),
  dcLink('Concept', 'expressedBy', conceptId('stratification'), 'EQ-B2', SRC),
  dcLink('Concept', 'expressedBy', conceptId('mixed-convection'), 'EQ-B4', SRC),
  dcLink('Concept', 'expressedBy', conceptId('it-heat-load'), 'EQ-M1', SRC),
  dcLink('Concept', 'expressedBy', conceptId('it-heat-load'), 'EQ-M3', SRC),
  dcLink('Concept', 'expressedBy', conceptId('condensation'), 'EQ-P6', SRC),
  dcLink('Concept', 'expressedBy', conceptId('dehumidification'), 'EQ-M11', SRC),
  dcLink('Concept', 'expressedBy', conceptId('containment'), 'EQ-M7', SRC),
  dcLink('Concept', 'expressedBy', conceptId('free-cooling'), 'EQ-D-PUE', SRC),
  // capture has no expressedBy link on purpose: its absence is already carried
  // by the CAP-SPECIES row, and one fact has one carrier.
  dcLink('Concept', 'refusedBy', conceptId('condensation'), 'refuse_condensation', SRC),
])
