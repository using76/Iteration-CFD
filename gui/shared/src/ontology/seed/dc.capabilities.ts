// gui/shared/src/ontology/seed/dc.capabilities.ts — the fifteen Model rows,
// the twenty-three Capability rows and the implementedBy links of the
// data-centre seed. A Model constant never appears on a case and a case value
// never appears on a Model (docs/13 section 3, rule 1). Line numbers in the
// anchors were re-derived from this tree on 2026-09-15; the tests grep the
// symbols, not the numbers.
import { dcLink, dcRow, type SeedLink, type SeedObject } from './dc.types.js'

const M = (apiName: string, family: string, constants: Record<string, unknown>,
           equationIds: string[], specSection: string, sourcePath: string): SeedObject =>
  dcRow('Model', apiName, sourcePath, { apiName, family, constants, equationIds, specSection })

export const DC_MODELS: readonly SeedObject[] = Object.freeze([
  M('kEpsilon', 'turbulence', { C_mu: 0.09, C_1: 1.44, C_2: 1.92, sigma_k: 1.0, sigma_eps: 1.3 },
    ['EQ-T1'], 'S6', 'rust/src/bin/datacentre.rs'),
  M('standard', 'wallTreatment', { kappa: 0.41, E: 9.8, yPlusLam: 11.53 }, ['EQ-T9'], 'S6.4', 'rust/src/bin/datacentre.rs'),
  M('constantPressure', 'fanCurve', {}, ['EQ-M4', 'EQ-M5'], 'S52.5', 'rust/src/fan.rs'),
  M('quadratic', 'fanCurve', {}, ['EQ-M4', 'EQ-M5'], 'S52.5', 'rust/src/fan.rs'),
  M('table', 'fanCurve', {}, ['EQ-M4', 'EQ-M5'], 'S52.5', 'rust/src/fan.rs'),
  M('K', 'tile', {}, ['EQ-M7'], 'S53.4', 'rust/src/fan.rs'),
  M('openAreaRatio', 'tile', {}, ['EQ-M7', 'EQ-M8'], 'S53.4', 'rust/src/fan.rs'),
  M('darcyForchheimer', 'tile', {}, ['EQ-M7'], 'S53.4', 'rust/src/fan.rs'),
  M('A1', 'ashraeClass', { tLoAllowableC: 15, tLoRecommendedC: 18, tHiRecommendedC: 27, tHiAllowableC: 32 }, ['EQ-D-RCI'], 'S55.1', 'rust/src/dcmetrics.rs'),
  M('A2', 'ashraeClass', { tLoAllowableC: 10, tLoRecommendedC: 18, tHiRecommendedC: 27, tHiAllowableC: 35 }, ['EQ-D-RCI'], 'S55.1', 'rust/src/dcmetrics.rs'),
  M('A3', 'ashraeClass', { tLoAllowableC: 5, tLoRecommendedC: 18, tHiRecommendedC: 27, tHiAllowableC: 40 }, ['EQ-D-RCI'], 'S55.1', 'rust/src/dcmetrics.rs'),
  M('A4', 'ashraeClass', { tLoAllowableC: 5, tLoRecommendedC: 18, tHiRecommendedC: 27, tHiAllowableC: 45 }, ['EQ-D-RCI'], 'S55.1', 'rust/src/dcmetrics.rs'),
  // H1 is the one class whose recommended band is NOT 18 to 27 C: it is 18 to
  // 22 C, which is why a hard-coded 18/27 envelope was structurally wrong
  // before the five-class change landed. The bands are the row's constants.
  M('H1', 'ashraeClass', { tLoAllowableC: 5, tLoRecommendedC: 18, tHiRecommendedC: 22, tHiAllowableC: 25 }, ['EQ-D-RCI'], 'S55.1', 'rust/src/dcmetrics.rs'),
  M('thirds', 'rciSamples', { heightFractions: [0.1667, 0.5, 0.8333] }, ['EQ-D-RCI'], 'S55.1', 'rust/src/dcmetrics.rs'),
  // faces is mesh-dependent: the report says so, and the lowering builds the
  // sample set from cell centres inside the inlet-sample box, so the
  // documented word for it is inexact.
  M('faces', 'rciSamples', {}, ['EQ-D-RCI'], 'S55.1', 'rust/src/dcmetrics.rs'),
])

const CAP = (tag: string, kind: string, module: string, anchor: string, specSection: string,
             reason: string | null, permissiveFallback: string | null): SeedObject =>
  dcRow('Capability', tag, module, { tag, kind, module, anchor, specSection, reason, permissiveFallback })

// provides — nine. A reason is permitted on a provides row; it is written only
// where it would otherwise carry a fact the tree would lose.
export const DC_CAPABILITIES: readonly SeedObject[] = Object.freeze([
  CAP('CAP-FLOW', 'provides', 'rust/src/simple.rs', 'rust/src/simple.rs:692', 'S5', null, null),
  CAP('CAP-BUOY', 'provides', 'rust/src/momentum.rs', 'rust/src/momentum.rs:116', 'S9',
    'The body force is the density-ratio form, exact for an ideal gas at constant pressure, not a Boussinesq linearisation; the reference temperature and gravity come from the case\'s air block.', null),
  CAP('CAP-ENERGY', 'provides', 'rust/src/scalar_transport.rs', 'rust/src/scalar_transport.rs:222', 'S26',
    'The data-centre driver transports temperature as a plain scalar with a turbulent Prandtl number; `rust/src/energy.rs`, and with it the Jayatilleke thermal wall function, is not wired into it, so a wall rule gets a fixed-value temperature and nothing else.', null),
  CAP('CAP-PSYCHRO', 'provides', 'rust/src/psychro.rs', 'rust/src/psychro.rs:216', 'S54',
    'Humidity is one transported vapour fraction; the psychrometric state is computed on the device and supersaturation is reported with a cell count and a worst excess, never clipped. The real-gas enhancement factor is named and not implemented, and the resulting bias is printed.', null),
  CAP('CAP-FAN', 'provides', 'rust/src/fan.rs', 'rust/src/fan.rs:102', 'S52',
    'One curve per whole patch, corrected for density and speed at every evaluation. A curve with fewer than two points, a non-increasing flow column, a rising pressure branch, a non-positive maximum pressure or flow, a non-positive curve density or speed, or an efficiency outside the half-open unit interval is refused by name in `FanCurve::validate`. A fan condition on any field but pressure is structurally impossible: only the fan module rewrites the triple.', null),
  CAP('CAP-JUMP', 'provides', 'rust/src/fan.rs', 'rust/src/fan.rs:638', 'S53',
    'A data-centre case can only produce the boundary-patch form; the internal-face form exists and no case key reaches it. A jump condition on a non-pressure field is structurally impossible, and the model gets the flow rate right and the near-tile jet wrong, which the report says on every run that has a jump.', null),
  CAP('CAP-DCMETRICS', 'provides', 'rust/src/dcmetrics.rs', 'rust/src/dcmetrics.rs:180', 'S55',
    'The reductions are deterministic and fixed-partition. An ASHRAE class the solver does not know, a sample set it does not know, and a metric patch that is unknown or has no faces are each refused by name with the menu printed.', null),
  CAP('CAP-REFUSE', 'provides', 'rust/src/io/contract.rs', 'rust/src/io/contract.rs:102', 'S13.4',
    'Recognised and implemented is used; recognised and not implemented is an error naming the setting and the menu; not recognised is an error naming the setting. One switch downgrades the rejection to a warning and prints what it substituted, once per distinct setting.', null),
  CAP('CAP-TURB-DC', 'provides', 'rust/src/bin/datacentre.rs', 'rust/src/bin/datacentre.rs:497', 'S6',
    'The driver builds standard k-epsilon with standard wall functions and no other model: the turbulence model, the wall treatment, the inlet turbulence intensity and length scale, and the convection scheme are all hard-coded and none is settable from a case. Realizable, RNG, SST, Spalart-Allmaras, DES and LES all exist in the crate and none is reachable from a `.dc.jsonc`.', null),
  // refuses — five. Every tag is a refuse_ function that exists in the module
  // it names, which is what the acceptance criterion counts.
  CAP('refuse_baffle_insertion', 'refuses', 'rust/src/fan.rs', 'rust/src/fan.rs:654', 'S53.5',
    'Splitting an existing internal face into a coincident pair of boundary faces is a topology mutation this solver does not perform. The two routes that exist are to emit the coincident pair at mesh-generation time, or to model the plenum as a separate region and use the boundary form. A jump on an ordinary internal face needs no baffle; what it cannot do is make the scalars jump, which is the only thing a baffle adds. The case key exists so that this refusal can fire.',
    'Under the permissive switch the refusal returns an empty internal-face jump, and the caller **discards it**: the tile falls through to its ordinary lowering and becomes a boundary porous jump on the same patch, with the scalars continuous. The behaviour is safe; the comment above the call site describes it wrongly, and D1 fixes that comment.'),
  CAP('refuse_capacitance_fft', 'refuses', 'rust/src/fan.rs', 'rust/src/fan.rs:674', 'S52.9',
    'A fan patch makes a face neither uniformly Dirichlet nor uniformly Neumann, and a jump makes the coefficient non-constant, so the direct Fourier pressure path is not available on a room that has either. The rank-one correction that would put it back is named and not implemented.',
    'pbicgstab'),
  CAP('refuse_condensation', 'refuses', 'rust/src/psychro.rs', 'rust/src/psychro.rs:297', 'S54.5',
    'A saturation-constrained source with its own inner iteration is a different model. What is offered instead: supersaturation is reported with a cell count and the worst excess, and the case may set a barometric pressure, virtual temperature, a fan supply relative humidity and a tile plenum relative humidity. No case key reaches this refusal today, so it is documented rather than fired.', null),
  CAP('refuse_wet_bulb_field', 'refuses', 'rust/src/psychro.rs', 'rust/src/psychro.rs:320', 'S54.5',
    'A per-cell root find with a data-dependent trip count is not capturable as a fixed graph. What is offered instead: the report\'s host-side wet bulb, the dew point and the relative humidity. No case key reaches this refusal today.', null),
  CAP('refuse_scattered_rack', 'refuses', 'rust/src/dcmetrics.rs', 'rust/src/dcmetrics.rs:422', 'S55.5',
    'Reducing over a scattered index list is a segmented reduction, which is deterministic but is not this project\'s single reduction shape, and no second reduction kernel is added for one metric. What is offered instead: give each rack its own patch, which is also what makes the per-rack heat flows reportable one by one.', null),
  // absent — nine, each naming the first file that would have to change.
  // docs/13 section 4 requires a named absence to be a row, never a gap.
  CAP('CAP-SPECIES', 'absent', 'rust/src/species.rs', 'rust/src/species.rs:169', 'S19',
    'Multi-species transport exists in the crate and is constructed only in tests and the validation binary; no solver driver builds it. A capture index needs one transported tracer per supply source and one per extract, so the first file to change is `rust/src/bin/datacentre.rs`, which today builds exactly one transported scalar for the vapour fraction.', null),
  CAP('CAP-RADIATION-DC', 'absent', 'rust/src/s2s.rs', 'rust/src/s2s.rs:1982', 'S49',
    'Surface-to-surface radiation exists in the crate and is constructed only in tests and the validation binary. Rack-to-rack and rack-to-wall radiant exchange is therefore absent from every data-centre run; the first files to change are `rust/src/bin/datacentre.rs` and a radiation key in the case format.', null),
  CAP('CAP-CHT', 'absent', 'rust/src/cht.rs', 'rust/src/cht.rs:2104', 'S47',
    'Conjugate heat transfer is a separate driver with a separate case format and is not composable with a data-centre case. A rack as a solid, with its own conduction and its own surface, cannot be expressed here.', null),
  CAP('CAP-YPLUS', 'absent', 'rust/src/bin/lowmach.rs', 'rust/src/bin/lowmach.rs:2269', 'S6.4',
    'Per-patch minimum, mean and maximum wall distance in wall units is computed and printed by the low-Mach driver and is absent from the data-centre driver. D5 ports it and must not re-derive it; the first file to change is `rust/src/bin/datacentre.rs`.', null),
  CAP('CAP-GATE-WALLVALID', 'absent', 'rust/src/bin/datacentre.rs', 'rust/src/bin/datacentre.rs:503', 'S6.4',
    'Nothing computes the ratio of buoyancy to inertia at a wall, so no run says whether its wall function was valid. A hall lives in exactly the band where that question decides the answer: about 0.18 in a 2 m/s aisle and about 18 in a stalled one. The first file to change is `rust/src/bin/datacentre.rs`, beside the wall-function coefficients it passes unconditionally.', null),
  CAP('CAP-FIELDOUT-DC', 'absent', 'rust/src/bin/datacentre.rs', 'rust/src/bin/datacentre.rs:832', 'S44',
    'The driver writes a single comma-separated snapshot of the final state and no field at all: no mesh, no volume data, no restart. A customer cannot be shown the cold aisle. The first file to change is `rust/src/bin/datacentre.rs`; the specification\'s own output block is not wired into the case format.', null),
  CAP('CAP-TRANSIENT-DC', 'absent', 'rust/src/io/case_dc.rs', 'rust/src/io/case_dc.rs:379', 'S55',
    'The data-centre run is steady by construction: a fixed outer-iteration count, a steady turbulence control, and a run block with no time step and no end time. A cooling-unit failure, a load ramp and a ride-through cannot be asked for. The first file to change is `rust/src/io/case_dc.rs`.', null),
  CAP('CAP-SWEEP-DC', 'absent', 'rust/src/io/case_dc.rs', 'rust/src/io/case_dc.rs:351', 'S55.4',
    'The supply-temperature sweep key is parsed and never read, so the free-cooling ceiling is absent from every report. This is the defect class the case format exists to prevent, live in the shipped format. D4 implements it; the first files to change are `rust/src/io/case_dc.rs`, where the key is read, and `rust/src/bin/datacentre.rs`, where the ceiling is written.', null),
  CAP('CAP-COIL', 'absent', 'rust/src/io/case_dc.rs', 'rust/src/io/case_dc.rs:165', 'S52',
    'The only cooling device the format carries is a fan patch with a pressure-flow curve. There is no coil, no sensible and latent split, no return-air control and no supply-temperature set point, so a real air handler cannot be expressed and its dehumidification cannot be accounted. The first file to change is `rust/src/io/case_dc.rs`.', null),
])

// Every model is implemented by a capability: 2 + 3 + 3 + 7 = fifteen.
export const DC_CAPABILITY_LINKS: readonly SeedLink[] = Object.freeze([
  dcLink('Model', 'implementedBy', 'kEpsilon', 'CAP-TURB-DC', 'rust/src/bin/datacentre.rs'),
  dcLink('Model', 'implementedBy', 'standard', 'CAP-TURB-DC', 'rust/src/bin/datacentre.rs'),
  dcLink('Model', 'implementedBy', 'constantPressure', 'CAP-FAN', 'rust/src/fan.rs'),
  dcLink('Model', 'implementedBy', 'quadratic', 'CAP-FAN', 'rust/src/fan.rs'),
  dcLink('Model', 'implementedBy', 'table', 'CAP-FAN', 'rust/src/fan.rs'),
  dcLink('Model', 'implementedBy', 'K', 'CAP-JUMP', 'rust/src/fan.rs'),
  dcLink('Model', 'implementedBy', 'openAreaRatio', 'CAP-JUMP', 'rust/src/fan.rs'),
  dcLink('Model', 'implementedBy', 'darcyForchheimer', 'CAP-JUMP', 'rust/src/fan.rs'),
  dcLink('Model', 'implementedBy', 'A1', 'CAP-DCMETRICS', 'rust/src/dcmetrics.rs'),
  dcLink('Model', 'implementedBy', 'A2', 'CAP-DCMETRICS', 'rust/src/dcmetrics.rs'),
  dcLink('Model', 'implementedBy', 'A3', 'CAP-DCMETRICS', 'rust/src/dcmetrics.rs'),
  dcLink('Model', 'implementedBy', 'A4', 'CAP-DCMETRICS', 'rust/src/dcmetrics.rs'),
  dcLink('Model', 'implementedBy', 'H1', 'CAP-DCMETRICS', 'rust/src/dcmetrics.rs'),
  dcLink('Model', 'implementedBy', 'thirds', 'CAP-DCMETRICS', 'rust/src/dcmetrics.rs'),
  dcLink('Model', 'implementedBy', 'faces', 'CAP-DCMETRICS', 'rust/src/dcmetrics.rs'),
])
