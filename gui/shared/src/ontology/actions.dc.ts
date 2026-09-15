// gui/shared/src/ontology/actions.dc.ts — the eight data-centre actions of the
// data-centre axis, five case decisions here and three judgements beside them.
// Each is an edit a human approves: the class, a rack's airflow, a fan curve
// scaled to a room-only model, a tile's K scaled for partial coverage, the run
// budget, accepting a run's closure, assessing a number, asserting compliance.
// Types only from actions.js, so the only runtime edge is actions.js importing
// this file's DC_ACTION_TYPES — no module cycle at load.
import type { ActionTypeDef } from './actions.js'

export const SET_ASHRAE_CLASS: ActionTypeDef = {
  apiName: 'setAshraeClass',
  displayName: 'Set the ASHRAE class',
  description:
    'Name the ASHRAE TC 9.9 equipment class a data-centre case is judged against, and optionally the ' +
    'rack-inlet sample set. The class sets the ALLOWABLE band that RCI normalises the excess by, so two ' +
    'cases differing only in the class report different RCI_HI and RCI_LO.',
  parameters: [
    { apiName: 'dcCaseId', displayName: 'Case', required: true, default: null,
      description: 'The DcCase primary key: the sha256 of the case file bytes, 64 lowercase hex',
      type: { t: 'string', pattern: '^[0-9a-f]{64}$' } },
    { apiName: 'ashraeClass', displayName: 'Class', required: true, default: null,
      description: 'ASHRAE TC 9.9 5th ed. equipment class. Allowable dry-bulb: A1 15-32, A2 10-35, A3 5-40, A4 5-45, H1 5-25 C',
      type: { t: 'enum', values: ['A1', 'A2', 'A3', 'A4', 'H1'] } },
    { apiName: 'rciSamples', displayName: 'RCI sample set', required: false, default: null,
      description: 'thirds takes three points per rack and is mesh-independent; faces takes every inlet face and is mesh-dependent. null leaves the case as it is',
      type: { t: 'enum', values: ['thirds', 'faces'] } },
    { apiName: 'rationale', displayName: 'Rationale', required: true, default: null,
      description: 'Why this class, in one sentence. It is recorded in the edit log, not on the case',
      type: { t: 'string', minLength: 8, maxLength: 400 } },
  ],
  prepare: 'setAshraeClass',
  rules: [
    { rule: 'modifyObject', objectType: 'DcCase',
      target: { from: 'parameter', parameter: 'dcCaseId' },
      properties: {
        ashraeClass: { from: 'parameter', parameter: 'ashraeClass' },
        rciSamples:  { from: 'prepared',  key: 'rciSamples' } } },
  ],
  functionRule: null,
  criteria: [
    { id: 'dcRowExists', severity: 'block',
      message: 'no {{type}} {{id}} in the mirror; import the case first',
      params: { objectType: 'DcCase', idParam: 'dcCaseId' } },
    { id: 'dcClassIsNew', severity: 'warn',
      message: 'DcCase {{id}} is already class {{class}}; this proposal changes nothing',
      params: {} },
  ],
  permission: { submitters: ['user', 'agent'], requiresApproval: true, policy: 'ask' },
  sideEffects: [],
  maxEdits: 2,
  ontologyVersion: '0.1.0',
}

export const SET_RACK_AIRFLOW: ActionTypeDef = {
  apiName: 'setRackAirflow',
  displayName: 'Set a rack airflow',
  description:
    'Set the volumetric airflow of one rack and record the basis it came from. Nothing in a rack ' +
    'schedule contains this number, and it is the denominator of dT_equipment = Q_IT / (mdot cp), ' +
    'which RTI is measured against.',
  parameters: [
    { apiName: 'rackId', displayName: 'Rack', required: true, default: null,
      description: 'The DcRack primary key: the case sha256, a hash sign, then the rack name',
      type: { t: 'string', maxLength: 200 } },
    { apiName: 'flow', displayName: 'Airflow', required: true, default: null,
      description: 'Rack airflow in cubic metres per second, greater than zero',
      type: { t: 'double', min: 0 } },
    { apiName: 'basis', displayName: 'Basis', required: true, default: null,
      description: 'Where the number came from: a vendor sheet, the 10 CFM per kW rule of thumb, a measurement, or an assumed rise across the rack',
      type: { t: 'enum', values: ['vendor', '10CFM_per_kW', 'measured', 'assumedDeltaT'] } },
    { apiName: 'rationale', displayName: 'Rationale', required: true, default: null,
      description: 'Why this airflow, in one sentence',
      type: { t: 'string', minLength: 8, maxLength: 400 } },
  ],
  prepare: 'setRackAirflow',
  rules: [
    { rule: 'modifyObject', objectType: 'DcRack',
      target: { from: 'parameter', parameter: 'rackId' },
      properties: {
        flow: { from: 'parameter', parameter: 'flow' } } },
  ],
  functionRule: null,
  criteria: [
    { id: 'dcRowExists', severity: 'block',
      message: 'no {{type}} {{id}} in the mirror; import the case first',
      params: { objectType: 'DcRack', idParam: 'rackId' } },
    { id: 'dcValueIsPositive', severity: 'block',
      message: '{{param}} is {{value}}; dT_equipment divides by the rack airflow, so it must be greater than zero',
      params: { param: 'flow' } },
    { id: 'dcFlowChangeIsLarge', severity: 'warn',
      message: 'rack {{id}}: flow {{before}} -> {{after}} m3/s changes dT_equipment, and RTI is measured against it',
      params: { factor: 2 } },
  ],
  permission: { submitters: ['user', 'agent'], requiresApproval: true, policy: 'ask' },
  sideEffects: [],
  maxEdits: 2,
  ontologyVersion: '0.1.0',
}

export const SCALE_FAN_CURVE: ActionTypeDef = {
  apiName: 'scaleFanCurve',
  displayName: 'Scale a fan curve',
  description:
    'Rescale a fan curve shut-off pressure and free delivery for a room-only model, recording the ' +
    'curve it was scaled from. A manufacturer CRAH curve includes plenum, coil, duct and rack ' +
    'resistance that this model does not contain; using it unscaled, or using a scaled value on a ' +
    'real job, are both wrong, so both ends are recorded.',
  parameters: [
    { apiName: 'fanId', displayName: 'Fan', required: true, default: null,
      description: 'The DcFan primary key: the case sha256, a hash sign, then the patch name',
      type: { t: 'string', maxLength: 200 } },
    { apiName: 'fromDpMax', displayName: 'From shut-off pressure', required: true, default: null,
      description: 'The dpMax now on the row, in Pa. It must match, or the proposal is refused',
      type: { t: 'double' } },
    { apiName: 'fromQMax', displayName: 'From free delivery', required: true, default: null,
      description: 'The qMax now on the row, in m3/s. It must match, or the proposal is refused',
      type: { t: 'double' } },
    { apiName: 'dpMax', displayName: 'Shut-off pressure', required: true, default: null,
      description: 'The scaled shut-off pressure in Pa, greater than zero',
      type: { t: 'double', min: 0 } },
    { apiName: 'qMax', displayName: 'Free delivery', required: true, default: null,
      description: 'The scaled free delivery in m3/s, greater than zero',
      type: { t: 'double', min: 0 } },
    { apiName: 'reason', displayName: 'Reason', required: true, default: null,
      description: 'What was taken out of the curve, in one sentence: which resistances this model does not carry',
      type: { t: 'string', minLength: 8, maxLength: 400 } },
  ],
  prepare: 'scaleFanCurve',
  rules: [
    { rule: 'modifyObject', objectType: 'DcFan',
      target: { from: 'parameter', parameter: 'fanId' },
      properties: {
        dpMax: { from: 'parameter', parameter: 'dpMax' },
        qMax:  { from: 'parameter', parameter: 'qMax' } } },
  ],
  functionRule: null,
  criteria: [
    { id: 'dcRowExists', severity: 'block',
      message: 'no {{type}} {{id}} in the mirror; import the case first',
      params: { objectType: 'DcFan', idParam: 'fanId' } },
    { id: 'dcCurveIsScalable', severity: 'block',
      message: 'fan {{id}} carries a {{curveType}} curve, which has no shut-off pressure to scale; edit the case file',
      params: {} },
    { id: 'dcFromCurveMatches', severity: 'block',
      message: 'fan {{id}} has dpMax {{actualDp}} Pa and qMax {{actualQ}} m3/s, not {{fromDp}} / {{fromQ}}; the curve you are scaling from is not the one on the row',
      params: {} },
    { id: 'dcValueIsPositive', severity: 'block',
      message: '{{param}} is {{value}}; a fan curve needs a positive shut-off pressure',
      params: { param: 'dpMax' } },
  ],
  permission: { submitters: ['user', 'agent'], requiresApproval: true, policy: 'ask' },
  sideEffects: [],
  maxEdits: 2,
  ontologyVersion: '0.1.0',
}

export const SCALE_TILE_LOSS: ActionTypeDef = {
  apiName: 'scaleTileLoss',
  displayName: 'Scale a tile loss coefficient',
  description:
    'Set a perforated-tile loss coefficient scaled for partial floor coverage, and record the ' +
    'arithmetic. Real tiles cover part of the patch they act on, and the same pressure drop at the ' +
    'same total flow needs K scaled by the area ratio squared. The proposal is refused unless the K ' +
    'it states is that product.',
  parameters: [
    { apiName: 'tileId', displayName: 'Tile', required: true, default: null,
      description: 'The DcTile primary key: the case sha256, a hash sign, then the patch name',
      type: { t: 'string', maxLength: 200 } },
    { apiName: 'kTile', displayName: 'Tile loss coefficient', required: true, default: null,
      description: 'The published loss coefficient of the tile itself, at its own open-area ratio',
      type: { t: 'double', min: 0 } },
    { apiName: 'patchArea', displayName: 'Patch area', required: true, default: null,
      description: 'Area of the patch the porous jump acts on, in m2',
      type: { t: 'double', min: 0 } },
    { apiName: 'tileArea', displayName: 'Open tile area', required: true, default: null,
      description: 'Total area of the tiles on that patch, in m2. It cannot exceed the patch area',
      type: { t: 'double', min: 0 } },
    { apiName: 'k', displayName: 'Scaled loss coefficient', required: true, default: null,
      description: 'The scaled coefficient to store: kTile times the square of patchArea over tileArea',
      type: { t: 'double', min: 0 } },
    { apiName: 'rationale', displayName: 'Rationale', required: true, default: null,
      description: 'Where the tile coefficient and the two areas came from, in one sentence',
      type: { t: 'string', minLength: 8, maxLength: 400 } },
  ],
  prepare: 'scaleTileLoss',
  rules: [
    { rule: 'modifyObject', objectType: 'DcTile',
      target: { from: 'parameter', parameter: 'tileId' },
      properties: {
        k:       { from: 'parameter', parameter: 'k' },
        kSource: { from: 'static', value: 'stated' } } },
  ],
  functionRule: null,
  criteria: [
    { id: 'dcRowExists', severity: 'block',
      message: 'no {{type}} {{id}} in the mirror; import the case first',
      params: { objectType: 'DcTile', idParam: 'tileId' } },
    { id: 'dcTileAreaFitsPatch', severity: 'block',
      message: 'tileArea {{tileArea}} m2 is not inside patchArea {{patchArea}} m2',
      params: {} },
    { id: 'dcAreaScalingMatches', severity: 'block',
      message: 'K {{k}} is not {{kTile}} x ({{patchArea}}/{{tileArea}})^2 = {{expected}}',
      params: { relTol: 0.001 } },
    { id: 'dcTileIsOpenAreaParameterised', severity: 'warn',
      message: 'tile {{id}} states openAreaRatio {{ratio}}; after this edit the row carries both parameterisations and kSource says which one the case used',
      params: {} },
  ],
  permission: { submitters: ['user', 'agent'], requiresApproval: true, policy: 'ask' },
  sideEffects: [],
  maxEdits: 2,
  ontologyVersion: '0.1.0',
}

export const SET_RUN_BUDGET: ActionTypeDef = {
  apiName: 'setRunBudget',
  displayName: 'Set the run budget and its stop rule',
  description:
    'Set how many outer iterations a data-centre case runs for, and record the closure a human is ' +
    'willing to stop at. The shipped driver runs a fixed budget with no convergence test, so ' +
    'choosing the budget IS choosing when to stop; the criterion records the intent, and no run ' +
    'reads it yet.',
  parameters: [
    { apiName: 'dcCaseId', displayName: 'Case', required: true, default: null,
      description: 'The DcCase primary key: the sha256 of the case file bytes, 64 lowercase hex',
      type: { t: 'string', pattern: '^[0-9a-f]{64}$' } },
    { apiName: 'iterations', displayName: 'Iterations', required: true, default: null,
      description: 'Outer iterations, at least one',
      type: { t: 'integer', min: 1 } },
    { apiName: 'continuityRatioMax', displayName: 'Continuity closure', required: false, default: null,
      description: 'Largest accepted net flow imbalance as a fraction of the largest opening. null records no closure target',
      type: { t: 'double', min: 0 } },
    { apiName: 'operatingPointGapPctMax', displayName: 'Operating-point gap', required: false, default: null,
      description: 'Largest accepted gap, in per cent, between a fan gathered flow and its patch flux',
      type: { t: 'double', min: 0 } },
    { apiName: 'consecutiveReports', displayName: 'Consecutive reports', required: false, default: null,
      description: 'How many consecutive reports must hold both thresholds before a run may stop',
      type: { t: 'integer', min: 1 } },
    { apiName: 'divergenceAction', displayName: 'On divergence', required: false, default: 'stopWithNamedError',
      description: 'What a diverged run should do. A diverged run stopping with a named error is the house rule',
      type: { t: 'enum', values: ['stopWithNamedError', 'continue'] } },
    { apiName: 'rationale', displayName: 'Rationale', required: true, default: null,
      description: 'Why this budget and these thresholds, in one sentence',
      type: { t: 'string', minLength: 8, maxLength: 400 } },
  ],
  prepare: 'setRunBudget',
  rules: [
    { rule: 'modifyObject', objectType: 'DcCase',
      target: { from: 'parameter', parameter: 'dcCaseId' },
      properties: { run: { from: 'prepared', key: 'runBlock' } } },
    { rule: 'createOrModifyObject', objectType: 'ConvergenceCriterion',
      primaryKey: { from: 'prepared', key: 'stopCriterionId' },
      properties: {
        dcCaseId:                { from: 'parameter', parameter: 'dcCaseId' },
        continuityRatioMax:      { from: 'parameter', parameter: 'continuityRatioMax' },
        operatingPointGapPctMax: { from: 'parameter', parameter: 'operatingPointGapPctMax' },
        consecutiveReports:      { from: 'parameter', parameter: 'consecutiveReports' } } },
  ],
  functionRule: null,
  criteria: [
    { id: 'dcRowExists', severity: 'block',
      message: 'no {{type}} {{id}} in the mirror; import the case first',
      params: { objectType: 'DcCase', idParam: 'dcCaseId' } },
    { id: 'dcValueIsPositive', severity: 'block',
      message: '{{param}} is {{value}}; a run budget is at least one iteration',
      params: { param: 'iterations' } },
    { id: 'dcStopRuleNotEnforced', severity: 'warn',
      message: 'the data-centre driver has no stop rule yet, so this criterion records intent and no run reads it',
      params: {} },
  ],
  permission: { submitters: ['user', 'agent'], requiresApproval: true, policy: 'ask' },
  sideEffects: [],
  maxEdits: 4,
  ontologyVersion: '0.1.0',
}

export const DC_ACTION_TYPES: ActionTypeDef[] = [
  SET_ASHRAE_CLASS, SET_RACK_AIRFLOW, SCALE_FAN_CURVE, SCALE_TILE_LOSS, SET_RUN_BUDGET,
]
