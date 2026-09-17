// gui/shared/src/ontology/seed/dc.types.ts — the shapes, the closed lists, the
// row and link helpers, and the anchor table of the data-centre seed. O2 is
// data as code: frozen typed rows plus a validator that refuses a malformed
// row by name. No storage, no I/O; the only run-time effect is a throw
// (docs/13 section 6, row O2).

// The shape a seeded row has. Structurally assignable to N2's `ObjectInput`
// (gui/server/src/ontology/store.ts) minus its optional `importedAt`, which
// the store fills from its own clock at import time. `shared` may not import
// from `server`, so this is re-declared, not imported; keep it assignable.
export interface SeedObject {
  readonly type: string
  readonly id: string
  readonly props: Readonly<Record<string, unknown>>
  readonly sourcePath: string
}

// A link named by its ACCESSOR, never by the link type's apiName: O1 owns the
// apiNames and docs/13 section 3 makes only the accessors binding.
// `resolveSeedLink` turns an accessor into a LinkTypeDef and a direction, the
// way N2's `resolveLinkSide` does server-side.
export interface SeedLink {
  readonly fromType: string
  readonly accessor: string
  readonly fromId: string
  readonly toId: string
  readonly sourcePath: string
}

export interface DcSeed {
  readonly objects: readonly SeedObject[]
  readonly links: readonly SeedLink[]
}

export interface DcSeedProblem {
  readonly code: DcSeedProblemCode
  readonly objectType: string
  readonly id: string
  readonly property: string | null
  readonly message: string
}

export type DcSeedProblemCode =
  | 'UNKNOWN_TYPE' | 'UNKNOWN_PROPERTY' | 'MISSING_PROPERTY' | 'BAD_ENUM'
  | 'BAD_PRIMARY_KEY' | 'DUPLICATE_ID' | 'BAD_SOURCE' | 'BAD_SPEC_SECTION'
  | 'L1_TYPE_SEEDED' | 'SHEET_LABEL_ID' | 'UNRESOLVED_LINK'
  | 'VALUE_WITHOUT_SOURCE' | 'UNKNOWN_PUBLIC_SOURCE' | 'REASON_REQUIRED'
  | 'DANGLING_COMPUTED_BY'

// A probe that proves a stored `file:line` still points where it says: the
// test greps `symbol` in `file`, requires exactly one hit, and requires the
// Capability's `anchor` to name that line. A grep survives an insertion above
// it; a hard-coded line does not.
export interface Anchor { readonly file: string; readonly symbol: string }

// The corpus tables O2 may never seed. Only an extractor writes them, and only
// C5's approved import writes a typed row (docs/13 section 2, the one rule).
export const DC_L1_TYPES: readonly string[] = Object.freeze([
  'document', 'chunk', 'object_chunk', 'candidate_object', 'candidate_link', 'unmapped_span',
])

// Workspace-relative, forward-slash paths a seeded row or link may name as its
// source. Every entry is confirmed to exist in this repository; the validator
// refuses anything else, and so does the seed test. Never a scratchpad path,
// never absolute, never containing a dot-dot segment.
export const DC_SEED_SOURCES: readonly string[] = Object.freeze([
  'docs/13-data-centre-axis.md',
  'rust/SPEC-LIT.md',
  'rust/src/dcmetrics.rs',
  'rust/src/fan.rs',
  'rust/src/psychro.rs',
  'rust/src/momentum.rs',
  'rust/src/simple.rs',
  'rust/src/species.rs',
  'rust/src/s2s.rs',
  'rust/src/cht.rs',
  'rust/src/scalar_transport.rs',
  'rust/src/io/contract.rs',
  'rust/src/io/case_dc.rs',
  'rust/src/bin/datacentre.rs',
  'rust/src/bin/lowmach.rs',
  'cases/coldAisle.dc.jsonc',
  'gui/shared/src/registry.ts',
])

// Every value a specSection property may hold. Each entry is a real heading of
// rust/SPEC-LIT.md, checked by the seed test with the regex it builds there. A
// parenthesised paragraph label, such as the one the physics sheet carries for
// the affinity laws, is NOT a heading: cite the heading above it instead.
export const DC_SPEC_SECTIONS: readonly string[] = Object.freeze([
  'S5', 'S6', 'S6.4', 'S9', 'S13.4', 'S18', 'S19', 'S25', 'S26', 'S29.3', 'S44', 'S47', 'S49',
  'S52', 'S52.5', 'S52.9', 'S53', 'S53.4', 'S53.5', 'S53.6', 'S54', 'S54.3', 'S54.5', 'S54.7',
  'S55', 'S55.1', 'S55.2', 'S55.3', 'S55.4', 'S55.5', 'S55.6', 'S55.7', 'S69',
])

// The document ids a citation or a publicSource may name, and the only ones
// (docs/13 section 5). `url` holds the in-repo path for this project's own
// documents and is empty for an external document whose address was not kept
// with the verification; `fetched` is when the entry's availability was last
// verified. The 2011 TC 9.9 whitepaper mirror is deliberately absent: docs/13
// section 4 says it is not even fetched, and listing it would claim a fetch
// that did not happen.
export interface PublicSource {
  readonly id: string
  readonly title: string
  readonly url: string
  readonly licence: string
  readonly fetched: string
  readonly verification: 'VERIFIED' | 'PARTIAL' | 'UNVERIFIED'
}

export const DC_PUBLIC_SOURCE_IDS: readonly PublicSource[] = Object.freeze([
  { id: 'doc:speclit', title: 'rust/SPEC-LIT.md, this project\'s own specification', url: 'rust/SPEC-LIT.md', licence: 'ours', fetched: '2026-09-15', verification: 'VERIFIED' },
  { id: 'doc:coldaisle-case', title: 'cases/coldAisle.dc.jsonc, this project\'s own worked case', url: 'cases/coldAisle.dc.jsonc', licence: 'ours', fetched: '2026-09-15', verification: 'VERIFIED' },
  { id: 'doc:ashrae-journal-2022-05', title: 'Quirk, Davidson & Schmidt, "ASHRAE\'s Data Center Thermal Guidelines - Air-Cooled Evolution", ASHRAE Journal, May 2022, pp. 54-58', url: '', licence: '(c) 2022 ASHRAE, posted publicly by ASHRAE; numeric values only are reported, never its text', fetched: '2026-09-15', verification: 'VERIFIED' },
  { id: 'doc:ashrae-90-4-2022-fact-sheet', title: 'ANSI/ASHRAE Standard 90.4-2022 fact sheet, ASHRAE Government Affairs', url: '', licence: '(c) ASHRAE, free public download', fetched: '2026-09-15', verification: 'VERIFIED' },
  { id: 'doc:ashrae-90-4-2019-addendum-h', title: 'ANSI/ASHRAE Addendum h to Standard 90.4-2019, ANSI-approved 2022-11-08', url: '', licence: '(c) 2022 ASHRAE, ISSN 1041-2336; tables not reproduced', fetched: '2026-09-15', verification: 'VERIFIED' },
  { id: 'doc:fpren-50600-4-2-2016', title: 'FprEN 50600-4-2:2016, the public final draft for vote', url: '', licence: 'CENELEC copyright; the formula is cited as a formula, the text is not reproduced', fetched: '2026-09-15', verification: 'VERIFIED' },
  { id: 'doc:iec-webstore-111538', title: 'the IEC webstore catalogue page for ISO/IEC 30134-2:2026', url: '', licence: 'catalogue metadata only', fetched: '2026-09-15', verification: 'VERIFIED' },
  { id: 'doc:tgg-wp35-wue', title: 'The Green Grid WP#35, Water Usage Effectiveness', url: '', licence: 'The Green Grid copyright, free download', fetched: '2026-09-15', verification: 'VERIFIED' },
  { id: 'doc:tgg-wp49-pue', title: 'The Green Grid WP#49, PUE: A Comprehensive Examination of the Metric', url: '', licence: 'The Green Grid copyright, free download', fetched: '2026-09-15', verification: 'PARTIAL' },
  { id: 'doc:lbnl-self-benchmarking-2009', title: 'Mathew, Ganguly, Greenberg & Sartor, Self-benchmarking Guide for Data Centers, LBNL for NYSERDA, 13 July 2009', url: '', licence: 'prepared under US Government sponsorship; quotable', fetched: '2026-09-15', verification: 'VERIFIED' },
  { id: 'doc:jrc-coc-2024', title: 'JRC136986, 2024 Best Practice Guidelines for the EU Code of Conduct on Data Centre Energy Efficiency', url: '', licence: 'CC BY 4.0', fetched: '2026-09-15', verification: 'VERIFIED' },
  { id: 'doc:jrc-coc-2025', title: 'JRC141521, the 2025 edition, DOI 10.2760/9449356', url: '', licence: 'CC BY 4.0', fetched: '2026-09-15', verification: 'VERIFIED' },
  { id: 'doc:kuzay-2022-data-in-brief', title: 'Kuzay et al., Data in Brief 45 (2022) 108587, DOI 10.1016/j.dib.2022.108587', url: '', licence: 'CC BY 4.0', fetched: '2026-09-15', verification: 'VERIFIED' },
  { id: 'doc:wibron-2019-energies', title: 'Wibron, Ljung & Lundstrom, Energies 12(8) 1473 (2019), DOI 10.3390/en12081473', url: '', licence: 'CC BY 4.0, full text not reachable', fetched: '2026-09-15', verification: 'PARTIAL' },
  { id: 'doc:aivc-capture-index-record', title: 'the AIVC bibliographic record for Shrivastava & VanGilder, ASHRAE Transactions 113(1) (2007) 126-136', url: '', licence: 'bibliographic record; the paper itself is ASHRAE-copyrighted and was not fetched', fetched: '2026-09-15', verification: 'VERIFIED' },
  { id: 'doc:uptime-tier-topology-abstract', title: 'the public abstract of Uptime Institute Tier Standard: Topology', url: '', licence: '(c) 2013-2026 Uptime Institute, all rights reserved; abstract only', fetched: '2026-09-15', verification: 'PARTIAL' },
  { id: 'doc:tia-942-c-white-paper', title: 'the public TIA white paper on ANSI/TIA-942-C, May 2024', url: '', licence: 'TIA copyright, public white paper', fetched: '2026-09-15', verification: 'PARTIAL' },
  { id: 'doc:hvac-best-ashrae-tc99', title: 'a public secondary summary of the ASHRAE TC 9.9 guidelines', url: '', licence: 'third-party web page, all rights reserved', fetched: '2026-09-15', verification: 'PARTIAL' },
])

// Every physics-sheet equation id this seed does NOT carry. An Equation is
// seeded only if something else in this seed points at it (a Concept
// expressedBy link, a Model equationIds entry, or a MetricDef equationId); an
// orphan equation is an unverifiable claim, a named deferral is an auditable
// one. EQ-T16 (buoyancy production G_b) is deferred for a second, stated
// reason: whether the data-centre driver's k-epsilon enables it was not
// established, and seeding it on kEpsilon's equationIds would assert that it
// was.
export const DC_SEED_DEFERRED_EQUATION_IDS: readonly string[] = Object.freeze([
  'EQ-C1', 'EQ-C2', 'EQ-C3', 'EQ-C4b', 'EQ-C5', 'EQ-C7', 'EQ-C8', 'EQ-C9', 'EQ-C10', 'EQ-C11', 'EQ-C12', 'EQ-C13',
  'EQ-T2', 'EQ-T3', 'EQ-T4', 'EQ-T5', 'EQ-T6', 'EQ-T7', 'EQ-T8', 'EQ-T10', 'EQ-T11', 'EQ-T11b', 'EQ-T12', 'EQ-T13', 'EQ-T14', 'EQ-T15', 'EQ-T16',
  'EQ-B1', 'EQ-B1a', 'EQ-B3', 'EQ-B5', 'EQ-B6', 'EQ-B8',
  'EQ-P1', 'EQ-P2', 'EQ-P3', 'EQ-P4', 'EQ-P5', 'EQ-P5b', 'EQ-P7', 'EQ-P8',
  'EQ-M2', 'EQ-M4b', 'EQ-M7b', 'EQ-M9', 'EQ-M10', 'EQ-M12', 'EQ-M13', 'EQ-M14', 'EQ-M15',
  'EQ-N1', 'EQ-N2', 'EQ-N3', 'EQ-N4', 'EQ-N5', 'EQ-N6',
  'EQ-TM1', 'EQ-TM2', 'EQ-TM3', 'EQ-TM4', 'EQ-TM5', 'EQ-TM6',
  'EQ-D1', 'EQ-D2', 'EQ-D3', 'EQ-D4', 'EQ-D5', 'EQ-D6', 'EQ-D7',
])

// One call per row: the row's primary-key property is filled from the same
// variable as its id, so the two can never drift.
export const dcRow = (type: string, id: string, sourcePath: string,
                      props: Record<string, unknown>): SeedObject =>
  ({ type, id, sourcePath, props: Object.freeze({ ...props }) })

export const dcLink = (fromType: string, accessor: string, fromId: string,
                       toId: string, sourcePath: string): SeedLink =>
  ({ fromType, accessor, fromId, toId, sourcePath })

export const conceptId = (slug: string): string => `concept:${slug}`

export const clauseId = (standardId: string, locator: string): string => `${standardId} / ${locator}`

export const sym = (symbol: string, unit: string, meaning: string) => ({ symbol, unit, meaning })

// The anchor probes behind every Capability row: file plus a symbol that
// occurs on exactly one line of it. The test greps these and requires the
// row's `anchor` to name the line the symbol sits on, so a row's stored
// file:line drifts into a loud failure instead of a quiet lie. Lines were
// re-derived from this tree on the day the seed was written.
export const DC_CAPABILITY_ANCHORS: Record<string, Anchor> = Object.freeze({
  'CAP-FLOW': { file: 'rust/src/simple.rs', symbol: 'pub fn correct_outer(' },
  'CAP-BUOY': { file: 'rust/src/momentum.rs', symbol: 'pub struct BuoyancyCoeffs' },
  'CAP-ENERGY': { file: 'rust/src/scalar_transport.rs', symbol: 'pub struct ScalarTransport<\'m> {' },
  'CAP-PSYCHRO': { file: 'rust/src/psychro.rs', symbol: 'pub fn virtual_temperature(t: Scalar, yv: Scalar)' },
  'CAP-FAN': { file: 'rust/src/fan.rs', symbol: 'pub enum CurveKind' },
  'CAP-JUMP': { file: 'rust/src/fan.rs', symbol: 'pub enum PorousJump' },
  'CAP-DCMETRICS': { file: 'rust/src/dcmetrics.rs', symbol: 'pub fn rci_hi(' },
  'CAP-REFUSE': { file: 'rust/src/io/contract.rs', symbol: 'pub fn unsupported<T>(' },
  'CAP-TURB-DC': { file: 'rust/src/bin/datacentre.rs', symbol: 'KEpsilon::new(' },
  'refuse_baffle_insertion': { file: 'rust/src/fan.rs', symbol: 'pub fn refuse_baffle_insertion' },
  'refuse_capacitance_fft': { file: 'rust/src/fan.rs', symbol: 'pub fn refuse_capacitance_fft' },
  'refuse_condensation': { file: 'rust/src/psychro.rs', symbol: 'pub fn refuse_condensation' },
  'refuse_wet_bulb_field': { file: 'rust/src/psychro.rs', symbol: 'pub fn refuse_wet_bulb_field' },
  'refuse_scattered_rack': { file: 'rust/src/dcmetrics.rs', symbol: 'pub fn refuse_scattered_rack' },
  'CAP-SPECIES': { file: 'rust/src/species.rs', symbol: 'pub struct Species<\'m> {' },
  'CAP-RADIATION-DC': { file: 'rust/src/s2s.rs', symbol: 'pub struct S2s<\'m> {' },
  'CAP-CHT': { file: 'rust/src/cht.rs', symbol: 'pub struct Cht' },
  'CAP-YPLUS': { file: 'rust/src/bin/lowmach.rs', symbol: 'let mut yplus_by_patch' },
  'CAP-GATE-WALLVALID': { file: 'rust/src/bin/datacentre.rs', symbol: 'WallFunctionCoeffs::default()' },
  'CAP-FIELDOUT-DC': { file: 'rust/src/bin/datacentre.rs', symbol: 'fn write_csv(' },
  'CAP-TRANSIENT-DC': { file: 'rust/src/io/case_dc.rs', symbol: 'pub struct DcRun' },
  'CAP-SWEEP-DC': { file: 'rust/src/io/case_dc.rs', symbol: 'supply_temperature_sweep' },
  'CAP-COIL': { file: 'rust/src/io/case_dc.rs', symbol: 'pub struct DcFan' },
})
