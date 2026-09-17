// gui/shared/src/ontology/seed/dc.metrics.ts — the twenty-four MetricDef rows
// of the data-centre seed: the metric catalogue, with each row's status pinned
// to the binary by the grep probes of DC_METRIC_ANCHORS, and the links that
// carry a computed metric to the Capability that provides it. A metric that is
// not computed says why in its own reason, and never carries a number this
// project could not have produced (docs/13, section 3).
import { dcRow, dcLink, type Anchor, type SeedLink, type SeedObject } from './dc.types.js'

const SRC_CODE = 'rust/src/dcmetrics.rs'       // computed and refused rows
const SRC_DOCS = 'docs/13-data-centre-axis.md' // absent rows: section 4's named absences
const SRC = 'rust/src/dcmetrics.rs'            // the links

const M = (apiName: string, sourcePath: string, props: Record<string, unknown>): SeedObject =>
  dcRow('MetricDef', apiName, sourcePath, { apiName, ...props })

export const DC_METRICS: readonly SeedObject[] = Object.freeze([
  M('RCI_HI', SRC_CODE, { unit: '%', range: 'at most 100', ideal: '100', status: 'computed',
    equationId: 'EQ-D-RCI', definitionSource: 'Herrlin:ASHRAE-Trans-111(2):2005' }),
  M('RCI_LO', SRC_CODE, { unit: '%', range: 'at most 100', ideal: '100', status: 'computed',
    equationId: 'EQ-D-RCI', definitionSource: 'Herrlin:ASHRAE-Trans-111(2):2005' }),
  M('RTI', SRC_CODE, { unit: '%', range: '0 and above', ideal: '100', status: 'computed',
    equationId: 'EQ-D-RTI', definitionSource: 'Herrlin:ASHRAE-Trans-114(2):2008' }),
  M('SHI', SRC_CODE, { unit: '-', range: '0 to 1', ideal: '0', status: 'computed',
    equationId: 'EQ-D-SHI', definitionSource: 'Sharma:AIAA-2002-3091:2002' }),
  M('RHI', SRC_CODE, { unit: '-', range: '0 to 1', ideal: '1', status: 'computed',
    equationId: 'EQ-D-SHI', definitionSource: 'Sharma:AIAA-2002-3091:2002' }),
  M('FAN_SHAFT_POWER', SRC_CODE, { unit: 'W', range: '0 and above', ideal: 'no single value',
    status: 'computed', equationId: 'EQ-M6', definitionSource: 'doc:speclit' }),
  M('DELTA_T', SRC_CODE, { unit: 'K', range: 'unbounded', ideal: 'no single value',
    status: 'computed', equationId: null, definitionSource: 'doc:lbnl-self-benchmarking-2009' }),
  M('AE', SRC_CODE, { unit: 'W/cfm', range: '0 and above',
    ideal: '0.5 or below is LBNL better practice, context and not a gate',
    status: 'computed', equationId: null, clauseId: 'LBNL:self-benchmarking:2009 / metric A4' }),
  M('PUE', SRC_CODE, { unit: 'kWh/kWh', range: '1 and above', ideal: '1', status: 'refused',
    equationId: null, clauseId: 'CENELEC:FprEN-50600-4-2:2016 / eq. 1',
    reason: 'A facility annual-energy ratio over a coincident twelve months. A steady room model has no facility boundary, no annual period and no non-electric inputs, so the solver refuses it by name and prints its three honest inputs instead. The current edition of the international standard is behind a paywall and no number from it is quoted.' }),
  M('pPUE', SRC_CODE, { unit: 'kWh/kWh', range: '1 and above', ideal: '1', status: 'refused',
    equationId: null, definitionSource: 'doc:fpren-50600-4-2-2016',
    reason: 'The same refusal as the full ratio, over a declared subsystem boundary; the boundary and metering rules are in text this project has not bought.' }),
  M('DCiE', SRC_CODE, { unit: '%', range: '0 to 100', ideal: '100', status: 'refused',
    equationId: null, definitionSource: 'doc:lbnl-self-benchmarking-2009',
    reason: 'The reciprocal of a ratio this solver refuses to compute, so it is refused for the same reason.' }),
  M('MLC', SRC_CODE, { unit: '-', range: '0 to 1', ideal: 'no single value', status: 'refused',
    equationId: null, clauseId: 'ASHRAE:90.4:2022 / Table 6.5',
    reason: 'An annualised mechanical load component computed over eight thousand seven hundred and sixty weather-hour bins at four load levels. What a room model can feed it is fan shaft power at the converged operating point and the highest supply temperature the inlets still tolerate; the maxima it is compared against are in a paywalled table.' }),
  M('ELC', SRC_CODE, { unit: '-', range: '0 to 1', ideal: 'no single value', status: 'refused',
    equationId: null, definitionSource: 'doc:ashrae-90-4-2019-addendum-h',
    reason: 'A worst-case electrical loss fraction summed over the power chain at four load levels. It is an electrical-design quantity with no fluid in it at all.' }),
  M('CI_COLD', SRC_DOCS, { unit: '%', range: '0 to 100', ideal: '100', status: 'absent',
    equationId: null, definitionSource: 'doc:aivc-capture-index-record',
    reason: 'The share of a rack\'s intake that comes from local cooling sources, computed by tracking passive tracer concentrations. It needs one transported scalar per supply source; `rust/src/species.rs` has the machinery and no driver builds it, so the first file to change is `rust/src/bin/datacentre.rs`.' }),
  M('CI_HOT', SRC_DOCS, { unit: '%', range: '0 to 100', ideal: '100', status: 'absent',
    equationId: null, definitionSource: 'doc:aivc-capture-index-record',
    reason: 'The share of a rack\'s exhaust captured by local extracts, by the same tracer method and blocked by the same absence.' }),
  M('BR', SRC_DOCS, { unit: '-', range: '0 to 1', ideal: '0', status: 'absent',
    equationId: null, definitionSource: 'Tozer:ASHRAE-Trans-115(1):2009',
    reason: 'Algebraic in the four temperatures the patch means already produce, but the exact assignment of supply, return and exhaust differs between derived statements of it, and the primary is paywalled. Read the primary before coding it, and print which convention was used.' }),
  M('RR', SRC_DOCS, { unit: '-', range: '0 to 1', ideal: '0', status: 'absent',
    equationId: null, definitionSource: 'Tozer:ASHRAE-Trans-115(1):2009',
    reason: 'The same four temperatures and the same unresolved convention as the bypass ratio.' }),
  M('RH_DELTA', SRC_DOCS, { unit: '% RH', range: 'unbounded', ideal: 'no single value',
    status: 'absent', equationId: null, definitionSource: 'doc:lbnl-self-benchmarking-2009',
    reason: 'Supply and return relative humidity are computed into device buffers on every humid run and are never printed or written; the difference has nowhere to come from until they are.' }),
  M('WUE', SRC_DOCS, { unit: 'L/kWh', range: '0 and above', ideal: '0', status: 'absent',
    equationId: null, definitionSource: 'doc:tgg-wp35-wue',
    reason: 'Annual site water over annual IT energy. A room model has no water in it.' }),
  M('WUE_SOURCE', SRC_DOCS, { unit: 'L/kWh', range: '0 and above', ideal: '0', status: 'absent',
    equationId: null, definitionSource: 'doc:tgg-wp35-wue',
    reason: 'Site water plus the water the source energy consumed, which needs an energy-water intensity factor for the grid the site sits on.' }),
  M('CUE', SRC_DOCS, { unit: 'kgCO2e/kWh', range: '0 and above', ideal: '0', status: 'absent',
    equationId: null, definitionSource: 'TGG:WP32:1',
    reason: 'Total carbon over IT energy, which is the site\'s emission factor times a ratio this solver refuses to compute.' }),
  M('ERF', SRC_DOCS, { unit: '-', range: '0 to 1', ideal: 'no single value', status: 'absent',
    equationId: null, definitionSource: 'CENELEC:EN-50600-4-6:2020',
    reason: 'Exported reuse energy over total energy. The defining equation as printed in the standard could not be read: the public preview had no extractable text. Confirm it against the standard before anything computes it.' }),
  M('REF', SRC_DOCS, { unit: '-', range: '0 to 1', ideal: 'no single value', status: 'absent',
    equationId: null, definitionSource: 'CENELEC:EN-50600-4-3:2019',
    reason: 'Renewable energy with guarantees of origin over total energy; a procurement fact, not a fluid one.' }),
  M('CER', SRC_DOCS, { unit: '-', range: '0 and above', ideal: 'no single value', status: 'absent',
    equationId: null, definitionSource: 'CENELEC:EN-50600-4-7:2020',
    reason: 'Heat removed over cooling-system energy, both over the same period. It is the only facility ratio a room model touches at all, and only through the heat removed.' }),
])

// The grep probes that pin each metric's status to the binary: the catalogue
// test asserts a row's status is 'computed' IF AND ONLY IF its probe is found
// in the tree, so drift in either direction is a named failure, not a quiet
// lie. shaft_power and delta_t carry their full signatures because the bare
// names occur more than once in their modules.
export const DC_METRIC_ANCHORS: Record<string, Anchor> = Object.freeze({
  RCI_HI: { file: 'rust/src/dcmetrics.rs', symbol: 'pub fn rci_hi(' },
  RCI_LO: { file: 'rust/src/dcmetrics.rs', symbol: 'pub fn rci_lo(' },
  RTI: { file: 'rust/src/dcmetrics.rs', symbol: 'pub fn rti(' },
  SHI: { file: 'rust/src/dcmetrics.rs', symbol: 'pub fn shi_rhi(' },
  RHI: { file: 'rust/src/dcmetrics.rs', symbol: 'pub fn shi_rhi(' },
  FAN_SHAFT_POWER: { file: 'rust/src/fan.rs', symbol: 'pub fn shaft_power(&self, q_dev: Scalar)' },
  DELTA_T: { file: 'rust/src/dcmetrics.rs', symbol: 'pub fn delta_t(t_return: Scalar, t_supply: Scalar)' },
  AE: { file: 'rust/src/dcmetrics.rs', symbol: 'pub fn airflow_efficiency(' },
})

// computedBy is O1's `computes` link named from the metric side; defines is
// seeded only where it is genuinely one-to-one — RCI and SHI/RHI carry the
// relation as MetricDef.equationId instead, one equation there defining two
// metrics.
export const DC_METRIC_LINKS: readonly SeedLink[] = Object.freeze([
  dcLink('MetricDef', 'computedBy', 'RCI_HI', 'CAP-DCMETRICS', SRC),
  dcLink('MetricDef', 'computedBy', 'RCI_LO', 'CAP-DCMETRICS', SRC),
  dcLink('MetricDef', 'computedBy', 'RTI', 'CAP-DCMETRICS', SRC),
  dcLink('MetricDef', 'computedBy', 'SHI', 'CAP-DCMETRICS', SRC),
  dcLink('MetricDef', 'computedBy', 'RHI', 'CAP-DCMETRICS', SRC),
  dcLink('MetricDef', 'computedBy', 'DELTA_T', 'CAP-DCMETRICS', SRC),
  dcLink('MetricDef', 'computedBy', 'AE', 'CAP-DCMETRICS', SRC),
  dcLink('MetricDef', 'computedBy', 'FAN_SHAFT_POWER', 'CAP-FAN', SRC),
  dcLink('Equation', 'defines', 'EQ-D-RTI', 'RTI', SRC),
  dcLink('Equation', 'defines', 'EQ-M6', 'FAN_SHAFT_POWER', SRC),
])
