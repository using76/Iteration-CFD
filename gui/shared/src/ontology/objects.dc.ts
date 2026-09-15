// gui/shared/src/ontology/objects.dc.ts — the nineteen data-centre object
// types of docs/13 section 3's ten levels, one const per type. N1's twelve
// live in objects.ts; this file builds beside them (registry.dc.ts composes
// both). Property names follow the case format (rust/src/io/case_dc.rs) and
// the D1 report; no value is written here — only the shapes that hold them.
import type { BaseType, ObjectTypeDef, PropertyDef } from './types.js'
const p = (apiName: string, baseType: BaseType, displayName: string, description: string, nullable = false, extra: Partial<PropertyDef> = {}): PropertyDef =>
  ({ apiName, displayName, baseType, nullable, description, ...extra })
const en = (values: string[]): Partial<PropertyDef> => ({ valueType: 'enum', enumValues: values })
const arr = (items: BaseType): Partial<PropertyDef> => ({ items })
const wp: Partial<PropertyDef> = { valueType: 'workspacePath' }
/** A named physical idea with no numbers: buoyancy, recirculation, bypass. */
const Concept: ObjectTypeDef = {
  apiName: 'Concept', displayName: 'Concept', pluralName: 'Concepts',
  description: 'A named physical idea of data-centre airflow with no numbers attached.', icon: 'idea',
  source: { projection: 'dcConcepts', paths: ['docs/13-data-centre-axis.md'] },
  primaryKey: 'slug', titleKey: 'title',
  properties: [
    p('slug', 'string', 'Slug', 'Primary key, concept:<kebab>, e.g. concept:recirculation.'),
    p('title', 'string', 'Title', 'The idea in two or three words.'),
    p('oneLine', 'string', 'One line', 'Our words, 160 characters or fewer; the cap is stated here and checked by no code in this unit.'),
    p('status', 'string', 'Status', 'Whether the solver models the idea, refuses it by name, or has nothing for it.', false, en(['modelled', 'refused', 'absent'])),
    p('refusedByTag', 'string', 'Refused by', 'The Capability tag behind the refusal; non-null only when status is refused.', true),
  ],
  ontologyVersion: '0.1.0',
}
/** One closed mathematical statement. Its symbol table is human-checked and lives as a json blob no query filters on. */
const Equation: ObjectTypeDef = {
  apiName: 'Equation', displayName: 'Equation', pluralName: 'Equations',
  description: 'One closed mathematical statement with its symbols, validity range and our own citation.', icon: 'equation',
  source: { projection: 'dcEquations', paths: ['docs/13-data-centre-axis.md'] },
  primaryKey: 'id', titleKey: 'name',
  properties: [
    p('id', 'string', 'Id', "Primary key, the physics sheet's id, e.g. EQ-B4 or EQ-D-RTI."),
    p('name', 'string', 'Name', 'The statement in a few words.'),
    p('latex', 'string', 'LaTeX', 'The statement itself in LaTeX.'),
    p('symbols', 'json', 'Symbols', 'Verbatim array of {symbol, unit, meaning}; human-checked, never filled by a model, and never queried by field.'),
    p('validity', 'string', 'Validity', 'The range the statement holds over.'),
    p('specSection', 'string', 'Spec section', 'The document section the statement is written in, e.g. S55.2; null when ours.', true),
    p('citation', 'string', 'Citation', 'Our own one-line bibliographic sentence plus a DOI where one exists.'),
    p('verification', 'string', 'Verification', "The fact sheet's own mark on this statement.", false, en(['VERIFIED', 'PARTIAL', 'UNVERIFIED'])),
  ],
  ontologyVersion: '0.1.0',
}
/** A choice among equations plus closures and constants. Constants live here and nowhere else — never on a case (docs/13 section 3 rule 1). */
const Model: ObjectTypeDef = {
  apiName: 'Model', displayName: 'Model', pluralName: 'Models',
  description: 'A choice among equations, closures and constants that makes a physical system computable.', icon: 'model',
  source: { projection: 'dcModels', paths: ['rust/src/models/registry.rs', 'rust/src/fan.rs', 'rust/src/dcmetrics.rs'] },
  primaryKey: 'apiName', titleKey: 'apiName',
  properties: [
    p('apiName', 'string', 'Api name', "Primary key and title; the registry name exactly as the code spells it, e.g. kEpsilon, quadratic, thirds, A1."),
    p('family', 'string', 'Family', 'What the model chooses among: turbulence, wallTreatment, fanCurve, rciSampleSet, ashraeEnvelope.'),
    p('constants', 'json', 'Constants', 'The model constants verbatim, e.g. {"C_mu": 0.09}; they live here and never on a case.'),
    p('equationIds', 'array', 'Equation ids', 'The equations this model chooses between and closes.', false, arr('string')),
    p('specSection', 'string', 'Spec section', 'The document section describing the model, when one exists.', true),
    p('capabilityTag', 'string', 'Capability tag', 'The Capability tag behind implementedBy.', true),
  ],
  ontologyVersion: '0.1.0',
}
/** What a binary can do, or refuses by name. A refusal is a first-class row whose reason is its load-bearing field. */
const Capability: ObjectTypeDef = {
  apiName: 'Capability', displayName: 'Capability', pluralName: 'Capabilities',
  description: 'What a binary can do, or refuses by name, anchored to a file and line the tests grep.', icon: 'capability',
  source: { projection: 'dcCapabilities', paths: ['rust/src/**/*.rs'] },
  primaryKey: 'tag', titleKey: 'tag',
  properties: [
    p('tag', 'string', 'Tag', 'Primary key and title: a tag for provides and absent, or the refuse function name or unsupported-note setting string for refuses.'),
    p('kind', 'string', 'Kind', 'Whether the binary provides this, refuses it by name, or has nothing for it.', false, en(['provides', 'refuses', 'absent'])),
    p('module', 'string', 'Module', 'The source file the capability lives in, e.g. rust/src/fan.rs.', false, wp),
    p('anchor', 'string', 'Anchor', 'file:line of the anchor; the stored line is for readers, the tests grep a symbol instead.'),
    p('specSection', 'string', 'Spec section', 'The document section, and the id of any check that gates this capability until a verdict type exists.', true),
    p('reason', 'string', 'Reason', 'The load-bearing field of a refusal: why the binary says no. Required when kind is not provides.', true),
    p('permissiveFallback', 'string', 'Permissive fallback', 'What the permissive mode substitutes, where it substitutes anything.', true),
    p('driverName', 'string', 'Driver name', 'The driver behind offeredBy; null until a later unit fills it.', true),
  ],
  ontologyVersion: '0.1.0',
}
/** One concrete problem: the .dc.jsonc document. The case is keyed on the sha256 of the file's bytes, not on its path. */
const DcCase: ObjectTypeDef = {
  apiName: 'DcCase', displayName: 'Dc case', pluralName: 'Dc cases',
  description: 'One concrete data-centre problem: the case document, keyed on the hash of its bytes.', icon: 'case',
  source: { projection: 'dcCases', paths: ['cases/**/*.dc.jsonc'] },
  primaryKey: 'dcCaseId', titleKey: 'name',
  properties: [
    p('dcCaseId', 'string', 'Dc case', 'Primary key: the 64 lowercase hex sha256 of the file bytes.', false, { valueType: 'sha256' }),
    p('name', 'string', 'Name', "The case's own name key; the path is not the name (a case file named plume.jsonc may be named anything)."),
    p('workspacePath', 'string', 'Workspace path', 'Workspace-relative path of the case document.', false, wp),
    p('schemaVersion', 'string', 'Schema version', 'The schema key of the document; null when the case carries none.', true),
    p('roomBounds', 'json', 'Room bounds', 'The room bounds verbatim: {min:[x,y,z], max:[x,y,z]}.'),
    p('roomCells', 'array', 'Room cells', 'The three cell counts, order preserved.', false, arr('integer')),
    p('boundaries', 'json', 'Boundaries', 'The six boundary names verbatim under xMin xMax yMin yMax zMin zMax; these six are the only patch names a case may use.'),
    p('grading', 'json', 'Grading', 'The per-axis grading block verbatim.', true),
    p('air', 'json', 'Air', 'The air block verbatim, defaults not filled in — a default the file did not write is not a fact the file stated.'),
    p('metrics', 'json', 'Metrics', 'The metrics block verbatim minus supplyTemperatureSweep, a key the solver parses and never reads.'),
    p('ashraeClass', 'string', 'Ashrae class', 'The class the case names; absent means null, never a default.', true, en(['A1', 'A2', 'A3', 'A4', 'H1'])),
    p('rciSamples', 'string', 'Rci samples', 'Which rack-inlet sample set the case asks for; absent means null.', true, en(['thirds', 'faces'])),
    p('supplyPatch', 'string', 'Supply patch', 'The patch the supply temperature is measured on; a name, not a link — there is no patch object type.'),
    p('returnPatch', 'string', 'Return patch', 'The patch the return temperature is measured on; a name, not a link.'),
    p('run', 'json', 'Run block', 'The run block verbatim: {iterations, reportEvery, initialTemperature}.'),
    p('numerics', 'json', 'Numerics', 'The numerics block verbatim.'),
    p('humidity', 'json', 'Humidity', 'The humidity block verbatim; absent on a dry case.', true),
    p('nFans', 'integer', 'Fan count', 'How many fans[] entries the document carries — a count, not the rows.'),
    p('nTiles', 'integer', 'Tile count', 'How many tiles[] entries the document carries.'),
    p('nRacks', 'integer', 'Rack count', 'How many racks[] entries the document carries.'),
    p('nPatchRules', 'integer', 'Patch rule count', 'How many patches[] entries the document carries.'),
    p('variantOfCaseId', 'string', 'Variant of', 'The case this one is a variant of; the foreign key behind variantOf.', true, { valueType: 'sha256' }),
    p('variantEditSummary', 'string', 'Variant edit summary', 'One line naming what the variant changed; the full edit set lives in the edit log.', true),
  ],
  ontologyVersion: '0.1.0',
}
/** One entry of the case's fans[] array. */
const DcFan: ObjectTypeDef = {
  apiName: 'DcFan', displayName: 'Dc fan', pluralName: 'Dc fans',
  description: 'One fan patch of a data-centre case, after the fans[] entries of the case format.', icon: 'fan',
  source: { projection: 'dcCaseFans', paths: ['cases/**/*.dc.jsonc'] },
  primaryKey: 'fanId', titleKey: 'patch',
  properties: [
    p('fanId', 'string', 'Fan', "Primary key: dcCaseId + '#' + patch."),
    p('dcCaseId', 'string', 'Dc case', 'The case this fan belongs to; the foreign key behind hasFan.', false, { valueType: 'sha256' }),
    p('patch', 'string', 'Patch', 'One of the six boundary names; the case format, not this declaration, checks which.'),
    p('direction', 'string', 'Direction', 'Which way the machine blows; the case must say, there is no default.', false, en(['inflow', 'outflow'])),
    p('curveType', 'string', 'Curve type', 'The pressure-flow curve family; a case may write constant or constantPressure and the solver reads both as one kind.', false, en(['constantPressure', 'constant', 'quadratic', 'table'])),
    p('dpMax', 'double', 'Dp max', 'The shut-off pressure rise, or the constant rise; absent under a table curve.', true),
    p('qMax', 'double', 'Q max', 'Free delivery; the case key spells it QMax with a capital Q, the property is qMax.', true),
    p('curvePoints', 'json', 'Curve points', 'The manufacturer table verbatim: [[Q, dp], ...].', true),
    p('rhoCurve', 'double', 'Rho curve', "The measurement density the curve was stated at; absent means null, never the driver's default.", true),
    p('speedCurve', 'double', 'Speed curve', 'The curve speed the points were measured at.', true),
    p('speed', 'double', 'Speed', 'The speed being run.', true),
    p('efficiency', 'double', 'Efficiency', 'The efficiency that divides the shaft power.', true),
    p('ambientPressure', 'double', 'Ambient pressure', 'Static pressure on the far side, gauge.', true),
    p('relaxation', 'double', 'Relaxation', 'Under-relaxation of the operating point.', true),
    p('supplyTemperature', 'double', 'Supply temperature', 'Supply air temperature at an inflow fan.', true),
    p('supplyRelativeHumidity', 'double', 'Supply relative humidity', 'Supply relative humidity, 0 to 1.', true),
  ],
  ontologyVersion: '0.1.0',
}
/** One entry of the case's tiles[] array. */
const DcTile: ObjectTypeDef = {
  apiName: 'DcTile', displayName: 'Dc tile', pluralName: 'Dc tiles',
  description: 'One perforated tile, grille or filter of a data-centre case, after the tiles[] entries of the case format.', icon: 'tile',
  source: { projection: 'dcCaseTiles', paths: ['cases/**/*.dc.jsonc'] },
  primaryKey: 'tileId', titleKey: 'patch',
  properties: [
    p('tileId', 'string', 'Tile', "Primary key: dcCaseId + '#' + patch."),
    p('dcCaseId', 'string', 'Dc case', 'The case this tile belongs to; the foreign key behind hasTile.', false, { valueType: 'sha256' }),
    p('patch', 'string', 'Patch', 'One of the six boundary names.'),
    p('k', 'double', 'K', 'The loss coefficient on the approach velocity; the case key spells it K with a capital K.', true),
    p('openAreaRatio', 'double', 'Open area ratio', "The tile's open-area fraction, the alternative parameterisation.", true),
    p('alpha', 'double', 'Alpha', 'The permeability of the Darcy-Forchheimer triple.', true),
    p('c2', 'double', 'C2', 'The inertial resistance factor; the case key spells it C2.', true),
    p('thickness', 'double', 'Thickness', 'The sheet thickness, a parameter rather than a mesh dimension.', true),
    p('baffle', 'boolean', 'Baffle', 'A request that exists only so the refusal of internal baffles can fire; never true in a case that runs.', true),
    p('plenumPressure', 'double', 'Plenum pressure', 'The static pressure behind the tile, gauge.'),
    p('plenumTemperature', 'double', 'Plenum temperature', 'The plenum air temperature behind the tile.'),
    p('plenumRelativeHumidity', 'double', 'Plenum relative humidity', 'The plenum relative humidity, 0 to 1.', true),
    p('kSource', 'string', 'K source', 'Which of the three parameterisations the case used; no conversion is ever done on this side of the boundary.', true, en(['stated', 'derivedFromOpenArea', 'darcyForchheimer'])),
  ],
  ontologyVersion: '0.1.0',
}
/** One entry of the case's racks[] array. */
const DcRack: ObjectTypeDef = {
  apiName: 'DcRack', displayName: 'Dc rack', pluralName: 'Dc racks',
  description: 'One heat-release rack of a data-centre case with its stated flow and its inlet sample box.', icon: 'rack',
  source: { projection: 'dcCaseRacks', paths: ['cases/**/*.dc.jsonc'] },
  primaryKey: 'rackId', titleKey: 'name',
  properties: [
    p('rackId', 'string', 'Rack', "Primary key: dcCaseId + '#' + name."),
    p('dcCaseId', 'string', 'Dc case', 'The case this rack belongs to; the foreign key behind hasRack.', false, { valueType: 'sha256' }),
    p('name', 'string', 'Name', "The rack's name, also the key the CSV report uses."),
    p('zone', 'json', 'Zone', 'The heat-release box verbatim: {min, max}, metres.'),
    p('power', 'double', 'Power', 'The IT power, watts.'),
    p('flow', 'double', 'Flow', "The stated volumetric flow, m^3/s: the denominator of dT equipment and therefore what sets RTI. Changing it is O4's setRackAirflow action; no flow basis is declared because a basis nobody stated is not a fact."),
    p('inletSamples', 'json', 'Inlet samples', 'The box the inlet temperature is sampled over, verbatim: {min, max}.'),
  ],
  ontologyVersion: '0.1.0',
}
/** One per run: the whole of what the metrics section computed. */
const DcMetricReport: ObjectTypeDef = {
  apiName: 'DcMetricReport', displayName: 'Dc metric report', pluralName: 'Dc metric reports',
  description: 'Everything one data-centre run computed, with the provenance every number carries.', icon: 'report',
  source: { projection: 'dcMetricReports', paths: ['gui/runs/*/dc-report.json'] },
  primaryKey: 'reportId', titleKey: 'reportId',
  properties: [
    p('reportId', 'string', 'Report', 'Primary key and title; it equals the runId.'),
    p('runId', 'string', 'Run', 'The run that produced this report; the foreign key behind measuredIn.'),
    p('dcCaseId', 'string', 'Dc case', 'The hash the document states and the fold recomputed; the foreign key behind reportsCase.', false, { valueType: 'sha256' }),
    p('casePath', 'string', 'Case path', 'Workspace-relative path of the case the run executed.', false, wp),
    p('gitSha', 'string', 'Git sha', 'The commit the run executed; a result without it is refused by the importer, so it is not nullable.', false, { valueType: 'gitSha' }),
    p('gitDirty', 'boolean', 'Git dirty', 'True when tracked content differed from the commit at launch.', true),
    p('nCells', 'long', 'Cells', 'The cell count of the mesh the run used.'),
    p('iterations', 'integer', 'Iterations', 'Outer iterations the run performed.'),
    p('continuityRatio', 'double', 'Continuity ratio', 'Net flow over largest opening: the convergence evidence of the whole axis.'),
    p('ashraeClass', 'string', 'Ashrae class', 'The class the run actually used.', false, en(['A1', 'A2', 'A3', 'A4', 'H1'])),
    p('rciSamples', 'string', 'Rci samples', 'The sample set the run actually used.', false, en(['thirds', 'faces'])),
    p('nSamples', 'integer', 'Samples', 'The sample count, reported because RCI depends on it.'),
    p('dtMeasured', 'boolean', 'D T measured', 'Whether the equipment temperature rise was measured across the racks or derived from IT heat and a stated flow; they are not the same measurement.'),
    p('rciHi', 'double', 'Rci hi', 'The over-temperature index, per cent.'),
    p('rciLo', 'double', 'Rci lo', 'The under-temperature index, per cent.'),
    p('rti', 'double', 'Rti', 'The return temperature index, per cent.'),
    p('shi', 'double', 'Shi', 'The supply heat index, per cent.'),
    p('rhi', 'double', 'Rhi', 'The return heat index, per cent.'),
    p('tSupply', 'double', 'Supply temperature', 'The flux-weighted supply patch mean.'),
    p('tReturn', 'double', 'Return temperature', 'The flux-weighted return patch mean.'),
    p('dtEquipment', 'double', 'Equipment temperature rise', 'The rise the report states, measured or derived.'),
    p('tInletMax', 'double', 'Max inlet temperature', 'The hottest rack-inlet sample.'),
    p('fanPower', 'double', 'Fan power', 'Total fan shaft power at the converged operating points, watts.'),
    p('itHeat', 'double', 'It heat', 'Total IT heat, watts.'),
    p('freeCoolingCeiling', 'double', 'Free-cooling ceiling', 'The highest supply temperature holding RCI_HI at 100 per cent; null means no sweep was run, never a guess.', true),
    p('startedAt', 'timestamp', 'Started at', 'When the run launched.'),
    p('endedAt', 'timestamp', 'Ended at', 'When the run ended.'),
    p('wallSeconds', 'double', 'Wall seconds', 'Wall-clock duration, seconds.'),
  ],
  ontologyVersion: '0.1.0',
}
/** One per fan, per run: where the fan curve crossed the system curve. */
const FanOperatingPoint: ObjectTypeDef = {
  apiName: 'FanOperatingPoint', displayName: 'Fan operating point', pluralName: 'Fan operating points',
  description: 'The operating point one fan settled at during one run.', icon: 'point',
  source: { projection: 'dcFanOperatingPoints', paths: ['gui/runs/*/dc-report.json'] },
  primaryKey: 'pointId', titleKey: 'pointId',
  properties: [
    p('pointId', 'string', 'Point', "Primary key and title: runId + '#' + patch."),
    p('runId', 'string', 'Run', 'The run that produced this point; the foreign key behind hasFanPoint.'),
    p('fanId', 'string', 'Fan', "The fan this point belongs to; the foreign key behind pointOfFan. Computable as dcCaseId + '#' + patch and null until a later unit fills it.", true),
    p('patch', 'string', 'Patch', 'The patch the fan draws on.'),
    p('q', 'double', 'Q', 'Volumetric flow at the operating point, m^3/s.'),
    p('dp', 'double', 'Dp', 'Pressure rise at the operating point, Pa.'),
    p('shaftPower', 'double', 'Shaft power', 'Shaft power at the operating point, watts.'),
    p('outerResidualPct', 'double', 'Outer residual', "The gap between the fan's own gathered flow and the patch flux at the last corrector; printed only when a fan exists.", true),
  ],
  ontologyVersion: '0.1.0',
}
/** One per run: the convergence evidence. */
const PatchFlowBalance: ObjectTypeDef = {
  apiName: 'PatchFlowBalance', displayName: 'Patch flow balance', pluralName: 'Patch flow balances',
  description: 'The per-patch flow balance of one run: the convergence evidence of the whole axis.', icon: 'balance',
  source: { projection: 'dcPatchFlowBalances', paths: ['gui/runs/*/dc-report.json'] },
  primaryKey: 'balanceId', titleKey: 'balanceId',
  properties: [
    p('balanceId', 'string', 'Balance', 'Primary key and title; it equals the runId.'),
    p('runId', 'string', 'Run', 'The run that produced this balance; the foreign key behind hasFlowBalance.'),
    p('perPatch', 'json', 'Per patch', 'The per-patch flow array verbatim and in order.'),
    p('net', 'double', 'Net', 'Copied from the document, never summed from the per-patch numbers: those are rounded for printing while the closure was computed on the device at full precision.'),
    p('largestOpening', 'double', 'Largest opening', 'Copied from the document for the same reason as net.'),
    p('netOverLargest', 'double', 'Net over largest', 'Copied from the document for the same reason as net.'),
  ],
  ontologyVersion: '0.1.0',
}
/** One per rack, per run: the inlet temperature the metrics sampled. */
const RackInletTemperature: ObjectTypeDef = {
  apiName: 'RackInletTemperature', displayName: 'Rack inlet temperature', pluralName: 'Rack inlet temperatures',
  description: 'The sampled inlet temperature of one rack during one run.', icon: 'thermometer',
  source: { projection: 'dcRackInletTemperatures', paths: ['gui/runs/*/dc-report.json'] },
  primaryKey: 'measurementId', titleKey: 'measurementId',
  properties: [
    p('measurementId', 'string', 'Measurement', "Primary key and title: runId + '#' + rackName."),
    p('runId', 'string', 'Run', 'The run that produced this measurement; the foreign key behind hasRackInlet.'),
    p('rackId', 'string', 'Rack', "The rack measured; the foreign key behind inletOfRack. Computable as dcCaseId + '#' + rackName and null until a later unit fills it.", true),
    p('rackName', 'string', 'Rack name', "The rack's name in the case."),
    p('meanInletT', 'double', 'Mean inlet temperature', 'The sampled mean inlet temperature.'),
    p('sampleCount', 'integer', 'Sample count', 'How many samples the mean is over; null until a driver emits a per-rack count.', true),
    p('metricApiName', 'string', 'Metric', 'The metric this number is a value of; the foreign key behind valueOf.', true),
  ],
  ontologyVersion: '0.1.0',
}
/** One per caveat kind, per run: a model limitation the driver printed. */
const ModelCaveat: ObjectTypeDef = {
  apiName: 'ModelCaveat', displayName: 'Model caveat', pluralName: 'Model caveats',
  description: 'One model limitation a run printed, with the text it printed it in.', icon: 'caveat',
  source: { projection: 'dcModelCaveats', paths: ['gui/runs/*/dc-report.json'] },
  primaryKey: 'caveatId', titleKey: 'caveatId',
  properties: [
    p('caveatId', 'string', 'Caveat', "Primary key and title: runId + '#' + kind."),
    p('runId', 'string', 'Run', 'The run that printed this caveat; the foreign key behind caveatedBy.'),
    p('kind', 'string', 'Kind', 'Which caveat of the driver it is; a plain string until the vocabulary is fixed by a schema output.'),
    p('text', 'string', 'Text', 'The caveat as the driver printed it.'),
    p('specRef', 'string', 'Spec reference', 'The document section the caveat points at, e.g. S54.5.', true),
  ],
  ontologyVersion: '0.1.0',
}
/** The definition of a metric, not a value: how the chat answers what can be computed. */
const MetricDef: ObjectTypeDef = {
  apiName: 'MetricDef', displayName: 'Metric definition', pluralName: 'Metric definitions',
  description: 'The definition of one metric with its status: computed, refused by name, or absent.', icon: 'metric',
  source: { projection: 'dcMetricDefs', paths: ['docs/13-data-centre-axis.md'] },
  primaryKey: 'apiName', titleKey: 'apiName',
  properties: [
    p('apiName', 'string', 'Api name', 'Primary key and title, e.g. RCI_HI, RTI, PUE.'),
    p('unit', 'string', 'Unit', 'The unit the metric is stated in.'),
    p('range', 'string', 'Range', 'The range the metric takes.'),
    p('ideal', 'string', 'Ideal', 'The value a well-run room scores.'),
    p('status', 'string', 'Status', 'Whether the solver computes it, refuses it by name, or has nothing for it.', false, en(['computed', 'refused', 'absent'])),
    p('reason', 'string', 'Reason', 'Why a refused or absent metric is refused or absent; required whenever status is not computed.', true),
    p('equationId', 'string', 'Equation', 'The equation the definition cites; the definition itself is the defines link.', true),
    p('definitionSource', 'string', 'Definition source', 'A document id naming where the definition was read from, when a public one exists.', true),
    p('clauseId', 'string', 'Clause', 'The standard clause the definition sits in; the foreign key behind definedIn.', true),
    p('computedByTag', 'string', 'Computed by', 'The capability that computes it; the foreign key behind computes.', true),
    p('identityGate', 'string', 'Identity gate', 'The id of the gate that checks the metric identity, e.g. 55-A.', true),
  ],
  ontologyVersion: '0.1.0',
}
/** An external standard, and what it costs to read it. */
const Standard: ObjectTypeDef = {
  apiName: 'Standard', displayName: 'Standard', pluralName: 'Standards',
  description: 'An external standard document, with the licence and purchase note a user needs to read it.', icon: 'standard',
  source: { projection: 'dcStandards', paths: ['docs/13-data-centre-axis.md'] },
  primaryKey: 'standardId', titleKey: 'title',
  properties: [
    p('standardId', 'string', 'Standard', "Primary key: issuer + ':' + number + ':' + edition, e.g. ASHRAE:TC9.9:5."),
    p('issuer', 'string', 'Issuer', 'Who publishes the standard.'),
    p('number', 'string', 'Number', 'The standard number.'),
    p('edition', 'string', 'Edition', 'Which edition.'),
    p('title', 'string', 'Title', 'The standard title.'),
    p('licence', 'string', 'Licence', 'How the standard may be read; this and the access note are how a user learns what to buy.'),
    p('isPaywalled', 'boolean', 'Paywalled', 'True when the text cannot be read without paying.'),
    p('accessNote', 'string', 'Access note', 'A purchase route or publication number.', true),
    p('supersedesStandardId', 'string', 'Supersedes', 'The standard this one replaces; the foreign key behind supersedes.', true),
  ],
  ontologyVersion: '0.1.0',
}
/** A locator inside a standard: our claim about it, never its text. */
const StandardClause: ObjectTypeDef = {
  apiName: 'StandardClause', displayName: 'Standard clause', pluralName: 'Standard clauses',
  description: 'A locator inside one standard, holding our own claim about what it governs; its text is never stored.', icon: 'clause',
  source: { projection: 'dcStandardClauses', paths: ['docs/13-data-centre-axis.md'] },
  primaryKey: 'clauseId', titleKey: 'locator',
  properties: [
    p('clauseId', 'string', 'Clause', "Primary key: standardId + ' / ' + locator, spaced slash."),
    p('standardId', 'string', 'Standard', 'The standard this clause is inside; the foreign key behind clauseOf.'),
    p('locator', 'string', 'Locator', 'Where inside the standard, e.g. Table 3 or 5.3.1.'),
    p('claim', 'string', 'Claim', 'Our own one-line claim about what the clause governs; never the clause wording.'),
    p('publicValue', 'string', 'Public value', 'Stated only when a public source states it.', true),
    p('publicSource', 'string', 'Public source', 'The document id of that public source; a public value without one is a defect the seed gate refuses.', true),
    p('verification', 'string', 'Verification', "The fact sheet's own mark on this clause.", false, en(['VERIFIED', 'PARTIAL', 'UNVERIFIED'])),
  ],
  ontologyVersion: '0.1.0',
}
/** A threshold a human set; it enters the ontology only by an approved action. */
const AcceptanceCriterion: ObjectTypeDef = {
  apiName: 'AcceptanceCriterion', displayName: 'Acceptance criterion', pluralName: 'Acceptance criteria',
  description: 'A numeric threshold a human set for design acceptance; created by an approved action, never by an import.', icon: 'criterion',
  source: { projection: '', paths: [] }, actionCreatedOnly: true,
  primaryKey: 'criterionId', titleKey: 'criterionId',
  properties: [
    p('criterionId', 'string', 'Criterion', 'Primary key and title, a uuid.', false, { valueType: 'uuid' }),
    p('metricApiName', 'string', 'Metric', 'The metric the threshold constrains; the foreign key behind constrainedBy.'),
    p('operator', 'string', 'Operator', 'The comparison, in words so the words can become a tool enum unescaped.', false, en(['lt', 'lte', 'gt', 'gte', 'eq', 'ne'])),
    p('threshold', 'double', 'Threshold', 'The number the metric is compared against.'),
    p('clauseId', 'string', 'Clause', 'The standard clause the threshold rests on; every threshold names its clause, and this is the foreign key behind citesClause.', true),
    p('setBy', 'string', 'Set by', 'The principal id of the human who set it.'),
    p('setAt', 'timestamp', 'Set at', 'When it was set.'),
    p('rationale', 'string', 'Rationale', 'Why the human set this threshold.'),
  ],
  ontologyVersion: '0.1.0',
}
/** Design acceptance: the word, the measured number, and who asserted it. */
const AcceptanceVerdict: ObjectTypeDef = {
  apiName: 'AcceptanceVerdict', displayName: 'Acceptance verdict', pluralName: 'Acceptance verdicts',
  description: 'A design-acceptance verdict against one criterion, asserted by a human principal.', icon: 'verdict',
  source: { projection: '', paths: [] }, actionCreatedOnly: true,
  primaryKey: 'verdictId', titleKey: 'verdictId',
  properties: [
    p('verdictId', 'string', 'Verdict', "Primary key and title: reportId + '#' + criterionId."),
    p('reportId', 'string', 'Report', 'The report assessed; the foreign key behind assessedBy.'),
    p('criterionId', 'string', 'Criterion', 'The criterion assessed against; the foreign key behind assessedAgainst.', false, { valueType: 'uuid' }),
    p('word', 'string', 'Word', 'One of four words, disjoint from the gate-verdict words.', false, en(['MEETS', 'MARGINAL', 'FAILS', 'NOT-ASSESSED'])),
    p('measured', 'double', 'Measured', 'The measured value, null on a not-assessed verdict.', true),
    p('threshold', 'double', 'Threshold', 'The threshold as it stood at assessment time.'),
    p('margin', 'double', 'Margin', 'How far the measurement sat from the threshold, null when not assessed.', true),
    p('assertedBy', 'string', 'Asserted by', 'The principal id; an agent may propose and only a human may apply the compliance action.'),
    p('assertedByKind', 'string', 'Asserted by kind', 'Which kind of principal asserted it.', false, en(['human', 'agent'])),
    p('assertedAt', 'timestamp', 'Asserted at', 'When it was asserted.'),
    p('notAssessedReason', 'string', 'Not-assessed reason', 'Why the criterion was not assessed, on a not-assessed verdict.', true),
  ],
  ontologyVersion: '0.1.0',
}
/** When a run is allowed to stop. Declared so the shape exists; whether any run carries one is the user's decision. */
const ConvergenceCriterion: ObjectTypeDef = {
  apiName: 'ConvergenceCriterion', displayName: 'Convergence criterion', pluralName: 'Convergence criteria',
  description: 'The stop rule of one data-centre case, declared so the shape exists; a run carries one only after the user approves it.', icon: 'stop',
  source: { projection: 'dcConvergenceCriteria', paths: ['cases/**/*.dc.jsonc'] },
  primaryKey: 'stopCriterionId', titleKey: 'stopCriterionId',
  properties: [
    p('stopCriterionId', 'string', 'Stop criterion', "Primary key and title: dcCaseId + '#stop'."),
    p('dcCaseId', 'string', 'Dc case', 'The case the rule belongs to; the foreign key behind hasStopCriterion.', false, { valueType: 'sha256' }),
    p('continuityRatioMax', 'double', 'Continuity ratio max', 'The largest net-over-largest the stop accepts.'),
    p('operatingPointGapPctMax', 'double', 'Operating point gap max', 'The largest fan operating-point gap, per cent, the stop accepts.'),
    p('consecutiveReports', 'integer', 'Consecutive reports', 'How many consecutive reports must satisfy both bounds.'),
    p('setBy', 'string', 'Set by', 'The principal who set the rule, when one has.', true),
    p('setAt', 'timestamp', 'Set at', 'When it was set.', true),
    p('rationale', 'string', 'Rationale', 'Why the rule holds these numbers.', true),
  ],
  ontologyVersion: '0.1.0',
}
export const DC_OBJECT_TYPES: ObjectTypeDef[] = [Concept, Equation, Model, Capability, DcCase, DcFan, DcTile, DcRack, DcMetricReport, FanOperatingPoint, PatchFlowBalance, RackInletTemperature, ModelCaveat, MetricDef, Standard, StandardClause, AcceptanceCriterion, AcceptanceVerdict, ConvergenceCriterion]
