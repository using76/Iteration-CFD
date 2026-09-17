// gui/shared/src/ontology/links.dc.ts — the thirty-three data-centre link types. Sides
// follow D-h; docs/13 section 3's accessors are binding, the api names are ours.
import type { LinkSide, LinkTypeDef } from './types.js'

const fk = (onType: string, property: string) => ({ kind: 'foreignKey' as const, onType, property })
const jt = (projection: string) => ({ kind: 'joinTable' as const, projection })
const side = (objectType: string, apiName: string, displayName: string): LinkSide => ({ apiName, displayName, objectType })
const V = '0.1.0'

const expressedBy: LinkTypeDef = {
  apiName: 'expressedBy', displayName: 'Expressed by', cardinality: 'MANY_TO_MANY',
  from: side('Concept', 'expressedBy', 'Equations'), to: side('Equation', 'concepts', 'Concepts'), backing: jt('conceptEquations'), ontologyVersion: V,
}
const refusedBy: LinkTypeDef = {
  apiName: 'refusedBy', displayName: 'Refused by', cardinality: 'MANY_TO_ONE',
  from: side('Concept', 'refusedBy', 'Capability'), to: side('Capability', 'refusesConcepts', 'Concepts refused'), backing: fk('Concept', 'refusedByTag'), ontologyVersion: V,
}
const instantiatedBy: LinkTypeDef = {
  apiName: 'instantiatedBy', displayName: 'Instantiated by', cardinality: 'MANY_TO_MANY',
  from: side('Equation', 'instantiatedBy', 'Models'), to: side('Model', 'instantiates', 'Equations'), backing: jt('equationModelInstantiations'), ontologyVersion: V,
}
// The link is the definition; MetricDef.equationId is the citation; one equation can define two metrics.
const defines: LinkTypeDef = {
  apiName: 'defines', displayName: 'Defines', cardinality: 'ONE_TO_ONE',
  from: side('Equation', 'defines', 'Metric definitions'), to: side('MetricDef', 'definedByEquation', 'Equation'), backing: jt('equationMetricDefinitions'), ontologyVersion: V,
}
const cites: LinkTypeDef = {
  apiName: 'cites', displayName: 'Cites', cardinality: 'MANY_TO_MANY',
  from: side('Equation', 'cites', 'Clauses'), to: side('StandardClause', 'citedByEquations', 'Equations'), backing: jt('equationClauseCitations'), ontologyVersion: V,
}
const implementedBy: LinkTypeDef = {
  apiName: 'implementedBy', displayName: 'Implemented by', cardinality: 'MANY_TO_ONE',
  from: side('Model', 'implementedBy', 'Capability'), to: side('Capability', 'models', 'Models'), backing: fk('Model', 'capabilityTag'), ontologyVersion: V,
}
// O2 seeds no Driver row and none of this link: the edge exists so a later unit can fill it.
const offeredBy: LinkTypeDef = {
  apiName: 'offeredBy', displayName: 'Offered by', cardinality: 'MANY_TO_ONE',
  from: side('Capability', 'offeredBy', 'Driver'), to: side('Driver', 'capabilities', 'Capabilities'), backing: fk('Capability', 'driverName'), ontologyVersion: V,
}
const computes: LinkTypeDef = {
  apiName: 'computes', displayName: 'Computes', cardinality: 'ONE_TO_MANY',
  from: side('Capability', 'computes', 'Metrics'), to: side('MetricDef', 'computedBy', 'Computing capability'), backing: fk('MetricDef', 'computedByTag'), ontologyVersion: V,
}
const selects: LinkTypeDef = {
  apiName: 'selects', displayName: 'Selects', cardinality: 'MANY_TO_MANY',
  // The link property records which key of the case document made the selection.
  from: side('DcCase', 'selects', 'Models'), to: side('Model', 'selectedBy', 'Cases'), backing: jt('dcCaseModelSelections'), ontologyVersion: V,
  properties: [{ apiName: 'key', displayName: 'Case key', baseType: 'string', nullable: false, description: 'Which key of the case document selected this model, e.g. metrics.ashraeClass.' }],
}
const hasFan: LinkTypeDef = {
  apiName: 'hasFan', displayName: 'Has fan', cardinality: 'ONE_TO_MANY',
  from: side('DcCase', 'fans', 'Fans'), to: side('DcFan', 'case', 'Case'), backing: fk('DcFan', 'dcCaseId'), ontologyVersion: V,
}
const hasTile: LinkTypeDef = {
  apiName: 'hasTile', displayName: 'Has tile', cardinality: 'ONE_TO_MANY',
  from: side('DcCase', 'tiles', 'Tiles'), to: side('DcTile', 'case', 'Case'), backing: fk('DcTile', 'dcCaseId'), ontologyVersion: V,
}
const hasRack: LinkTypeDef = {
  apiName: 'hasRack', displayName: 'Has rack', cardinality: 'ONE_TO_MANY',
  from: side('DcCase', 'racks', 'Racks'), to: side('DcRack', 'case', 'Case'), backing: fk('DcRack', 'dcCaseId'), ontologyVersion: V,
}
const variantOf: LinkTypeDef = {
  apiName: 'variantOf', displayName: 'Variant of', cardinality: 'MANY_TO_ONE',
  from: side('DcCase', 'variantOf', 'Base case'), to: side('DcCase', 'variants', 'Variants'), backing: fk('DcCase', 'variantOfCaseId'), ontologyVersion: V,
}
// A join table, not a foreign key: Run.caseId is a workspace path, a DcCase key is the file's sha256.
const runsDcCase: LinkTypeDef = {
  apiName: 'runsDcCase', displayName: 'Ran dc case', cardinality: 'MANY_TO_ONE',
  from: side('Run', 'dcCase', 'Dc case'), to: side('DcCase', 'runs', 'Runs'), backing: jt('runDcCases'), ontologyVersion: V,
}
// Zero rows until a convergence stop rule lands, and whether one ever does is the user's decision.
const stoppedBy: LinkTypeDef = {
  apiName: 'stoppedBy', displayName: 'Stopped by', cardinality: 'MANY_TO_MANY',
  from: side('Run', 'stoppedBy', 'Stop decisions'), to: side('ConvergenceCriterion', 'runs', 'Runs stopped'), backing: jt('runStopDecisions'), ontologyVersion: V,
}
const hasStopCriterion: LinkTypeDef = {
  apiName: 'hasStopCriterion', displayName: 'Has stop criterion', cardinality: 'ONE_TO_ONE',
  from: side('DcCase', 'stopCriterion', 'Stop criterion'), to: side('ConvergenceCriterion', 'case', 'Case'), backing: fk('ConvergenceCriterion', 'dcCaseId'), ontologyVersion: V,
}
const measuredIn: LinkTypeDef = {
  apiName: 'measuredIn', displayName: 'Measured in', cardinality: 'MANY_TO_ONE',
  from: side('DcMetricReport', 'measuredIn', 'Run'), to: side('Run', 'dcReports', 'Dc reports'), backing: fk('DcMetricReport', 'runId'), ontologyVersion: V,
}
const reportsCase: LinkTypeDef = {
  apiName: 'reportsCase', displayName: 'Reports case', cardinality: 'MANY_TO_ONE',
  from: side('DcMetricReport', 'case', 'Case'), to: side('DcCase', 'reports', 'Reports'), backing: fk('DcMetricReport', 'dcCaseId'), ontologyVersion: V,
}
const caveatedBy: LinkTypeDef = {
  apiName: 'caveatedBy', displayName: 'Caveated by', cardinality: 'ONE_TO_MANY',
  from: side('DcMetricReport', 'caveatedBy', 'Caveats'), to: side('ModelCaveat', 'report', 'Report'), backing: fk('ModelCaveat', 'runId'), ontologyVersion: V,
}
const hasFlowBalance: LinkTypeDef = {
  apiName: 'hasFlowBalance', displayName: 'Has flow balance', cardinality: 'ONE_TO_ONE',
  from: side('DcMetricReport', 'flowBalance', 'Flow balance'), to: side('PatchFlowBalance', 'report', 'Report'), backing: fk('PatchFlowBalance', 'runId'), ontologyVersion: V,
}
const hasFanPoint: LinkTypeDef = {
  apiName: 'hasFanPoint', displayName: 'Has fan point', cardinality: 'ONE_TO_MANY',
  from: side('DcMetricReport', 'fanPoints', 'Fan points'), to: side('FanOperatingPoint', 'report', 'Report'), backing: fk('FanOperatingPoint', 'runId'), ontologyVersion: V,
}
const hasRackInlet: LinkTypeDef = {
  apiName: 'hasRackInlet', displayName: 'Has rack inlet', cardinality: 'ONE_TO_MANY',
  from: side('DcMetricReport', 'rackInlets', 'Rack inlets'), to: side('RackInletTemperature', 'report', 'Report'), backing: fk('RackInletTemperature', 'runId'), ontologyVersion: V,
}
const pointOfFan: LinkTypeDef = {
  apiName: 'pointOfFan', displayName: 'Point of fan', cardinality: 'MANY_TO_ONE',
  from: side('FanOperatingPoint', 'fan', 'Fan'), to: side('DcFan', 'operatingPoints', 'Operating points'), backing: fk('FanOperatingPoint', 'fanId'), ontologyVersion: V,
}
const inletOfRack: LinkTypeDef = {
  apiName: 'inletOfRack', displayName: 'Inlet of rack', cardinality: 'MANY_TO_ONE',
  from: side('RackInletTemperature', 'rack', 'Rack'), to: side('DcRack', 'inletTemperatures', 'Inlet temperatures'), backing: fk('RackInletTemperature', 'rackId'), ontologyVersion: V,
}
const valueOf: LinkTypeDef = {
  apiName: 'valueOf', displayName: 'Value of', cardinality: 'MANY_TO_ONE',
  from: side('RackInletTemperature', 'valueOf', 'Metric'), to: side('MetricDef', 'rackInletValues', 'Rack inlet values'), backing: fk('RackInletTemperature', 'metricApiName'), ontologyVersion: V,
}
const assessedBy: LinkTypeDef = {
  apiName: 'assessedBy', displayName: 'Assessed by', cardinality: 'ONE_TO_MANY',
  from: side('DcMetricReport', 'assessedBy', 'Verdicts'), to: side('AcceptanceVerdict', 'about', 'Report'), backing: fk('AcceptanceVerdict', 'reportId'), ontologyVersion: V,
}
const assessedAgainst: LinkTypeDef = {
  apiName: 'assessedAgainst', displayName: 'Assessed against', cardinality: 'MANY_TO_ONE',
  from: side('AcceptanceVerdict', 'assessedAgainst', 'Criterion'), to: side('AcceptanceCriterion', 'verdicts', 'Verdicts'), backing: fk('AcceptanceVerdict', 'criterionId'), ontologyVersion: V,
}
const constrainedBy: LinkTypeDef = {
  apiName: 'constrainedBy', displayName: 'Constrained by', cardinality: 'ONE_TO_MANY',
  from: side('MetricDef', 'constrainedBy', 'Criteria'), to: side('AcceptanceCriterion', 'metric', 'Metric'), backing: fk('AcceptanceCriterion', 'metricApiName'), ontologyVersion: V,
}
const citesClause: LinkTypeDef = {
  apiName: 'citesClause', displayName: 'Cites clause', cardinality: 'MANY_TO_ONE',
  from: side('AcceptanceCriterion', 'citesClause', 'Clause'), to: side('StandardClause', 'criteria', 'Criteria'), backing: fk('AcceptanceCriterion', 'clauseId'), ontologyVersion: V,
}
const definedIn: LinkTypeDef = {
  apiName: 'definedIn', displayName: 'Defined in', cardinality: 'MANY_TO_ONE',
  from: side('MetricDef', 'definedIn', 'Clause'), to: side('StandardClause', 'definesMetrics', 'Metrics defined'), backing: fk('MetricDef', 'clauseId'), ontologyVersion: V,
}
const clauseOf: LinkTypeDef = {
  apiName: 'clauseOf', displayName: 'Clause of', cardinality: 'MANY_TO_ONE',
  from: side('StandardClause', 'standard', 'Standard'), to: side('Standard', 'clauses', 'Clauses'), backing: fk('StandardClause', 'standardId'), ontologyVersion: V,
}
const supersedes: LinkTypeDef = {
  apiName: 'supersedes', displayName: 'Supersedes', cardinality: 'MANY_TO_ONE',
  from: side('Standard', 'supersedes', 'Superseded edition'), to: side('Standard', 'supersededBy', 'Supersedes'), backing: fk('Standard', 'supersedesStandardId'), ontologyVersion: V,
}
const equivalentTo: LinkTypeDef = {
  apiName: 'equivalentTo', displayName: 'Equivalent to', cardinality: 'MANY_TO_MANY',
  from: side('Standard', 'equivalentTo', 'Equivalent standards'), to: side('Standard', 'equivalents', 'Equivalents'), backing: jt('standardEquivalences'), ontologyVersion: V,
}

export const DC_LINK_TYPES: LinkTypeDef[] = [expressedBy, refusedBy, instantiatedBy, defines, cites, implementedBy, offeredBy, computes, selects, hasFan, hasTile, hasRack, variantOf, runsDcCase, stoppedBy, hasStopCriterion, measuredIn, reportsCase, caveatedBy, hasFlowBalance, hasFanPoint, hasRackInlet, pointOfFan, inletOfRack, valueOf, assessedBy, assessedAgainst, constrainedBy, citesClause, definedIn, clauseOf, supersedes, equivalentTo]
