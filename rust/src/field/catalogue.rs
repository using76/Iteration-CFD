// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT §120 as data: the boundary-condition catalogue - every public
//! boundary-condition type name and its one status.
//!
//! The type NAMES are taken by name only from the public user guides
//! (<https://www.openfoam.com/documentation/guides/latest/doc/guide-bcs.html>
//! and <https://doc.cfd.direct/openfoam/user-guide-v12/boundaries>); no
//! description, formula or source code was taken from either, and no
//! OpenFOAM source was read. The tests that hold this table to the code are
//! in `field.rs` ([`crate::field::BcKind::from_name`] is the reader it
//! describes).

/// The one status a catalogue name can hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Status {
    /// In `IMPLEMENTED_BC_NAMES` and accepted by `BcKind::from_name` on `Entry::field`.
    Implemented,
    /// Accepted with a printed mapping, not in `IMPLEMENTED_BC_NAMES`; `to` is in it.
    Alias { to: &'static str },
    /// Refused today; the PREC BC unit that implements it (`BC-NN`).
    Planned { unit: &'static str },
    /// Refused today; the other stream's unit that implements it (`CMP-NN` or `CHR-NN`).
    Owned { unit: &'static str },
    /// Refused permanently by name; `key` is a phrase the refusal's note must contain.
    Refused { key: &'static str },
}

/// One catalogue row: a name, its status, and what the reader does with it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Entry {
    pub family: &'static str,
    /// The type name; a trailing `*` marks a prefix family (only for `Refused`).
    pub name: &'static str,
    pub status: Status,
    /// The field `from_name` is asked on (it matters only for the field-restricted names).
    pub field: &'static str,
    /// Units that extend an `Implemented` name later (empty for every other status).
    pub extended_by: &'static [&'static str],
    /// The concrete names a `*` entry is probed with; empty means "probe `name` itself".
    pub probes: &'static [&'static str],
}

/// Every public boundary-condition type name, in §120's order: the
/// implemented names in `IMPLEMENTED_BC_NAMES`' order first, then the two
/// aliases, then the planned names by their unit, the owned names by their
/// stream, and the permanent refusals last.
pub(crate) const CATALOGUE: &[Entry] = &[
    // ---- implemented (95), in IMPLEMENTED_BC_NAMES' order -----------------
    Entry { family: "basic", name: "fixedValue", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "basic", name: "noSlip", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "basic", name: "zeroGradient", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "basic", name: "fixedGradient", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "basic", name: "uniformFixedGradient", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "basic", name: "mixed", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "freestream", name: "freestream", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "freestream", name: "freestreamVelocity", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "freestream", name: "freestreamPressure", status: Status::Implemented, field: "psi", extended_by: &["BC-20"], probes: &[] },
    Entry { family: "basic", name: "calculated", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "basic", name: "empty", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "basic", name: "symmetry", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "basic", name: "symmetryPlane", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "basic", name: "slip", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "basic", name: "wedge", status: Status::Implemented, field: "psi", extended_by: &["BC-03", "BC-04"], probes: &[] },
    Entry { family: "coupled", name: "cyclic", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "coupled", name: "cyclicSlip", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "coupled", name: "processor", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "inlet-outlet", name: "inletOutlet", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "inlet-outlet", name: "turbulentIntensityKineticEnergyInlet", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "inlet-outlet", name: "turbulentMixingLengthDissipationRateInlet", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "inlet-outlet", name: "turbulentMixingLengthFrequencyInlet", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "inlet-outlet", name: "pressureInletOutletVelocity", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "inlet-outlet", name: "fixedFluxPressure", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "inlet-outlet", name: "totalPressure", status: Status::Implemented, field: "psi", extended_by: &["CMP-16"], probes: &[] },
    Entry { family: "inlet-outlet", name: "flowRateInletVelocity", status: Status::Implemented, field: "psi", extended_by: &["CMP-16"], probes: &[] },
    Entry { family: "wall", name: "movingWallVelocity", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "wall function", name: "nutkWallFunction", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "wall function", name: "nutUWallFunction", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "wall function", name: "nutLowReWallFunction", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "wall function", name: "epsilonWallFunction", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "wall function", name: "omegaWallFunction", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "wall function", name: "kqRWallFunction", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "wall function", name: "kLowReWallFunction", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "wall function", name: "thermalWallFunction", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "wall function", name: "nutkRoughWallFunction", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "wall function", name: "nutURoughWallFunction", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "wall function", name: "wernerWengleWallFunction", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "thermal", name: "fixedFluxTemperature", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "VOF", name: "constantAlphaContactAngle", status: Status::Implemented, field: "alpha.water", extended_by: &[], probes: &[] },
    Entry { family: "VOF", name: "dynamicAlphaContactAngle", status: Status::Implemented, field: "alpha.water", extended_by: &[], probes: &[] },
    Entry { family: "thermal", name: "coupledTemperature", status: Status::Implemented, field: "T", extended_by: &[], probes: &[] },
    Entry { family: "thermal", name: "thermalContactResistance", status: Status::Implemented, field: "T", extended_by: &[], probes: &[] },
    Entry { family: "radiation", name: "greyDiffusiveRadiationViewFactor", status: Status::Implemented, field: "T", extended_by: &[], probes: &[] },
    Entry { family: "radiation", name: "s2sWall", status: Status::Implemented, field: "T", extended_by: &[], probes: &[] },
    Entry { family: "jump", name: "fanPressure", status: Status::Implemented, field: "p", extended_by: &[], probes: &[] },
    Entry { family: "jump", name: "fan", status: Status::Implemented, field: "p", extended_by: &["BC-17"], probes: &[] },
    Entry { family: "jump", name: "porousJumpPressure", status: Status::Implemented, field: "p", extended_by: &[], probes: &[] },
    Entry { family: "jump", name: "porousBafflePressure", status: Status::Implemented, field: "p", extended_by: &["BC-17"], probes: &[] },
    // SPEC-LIT 120.1: the Function1 conditions of the time-varying family.
    Entry { family: "time-varying", name: "uniformFixedValue", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "time-varying", name: "uniformInletOutlet", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "time-varying", name: "uniformTotalPressure", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "time-varying", name: "fixedProfile", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    // SPEC-LIT 120.3: the wall and slip family.
    Entry { family: "basic", name: "extrapolatedCalculated", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "basic", name: "directionMixed", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "wall", name: "rotatingWallVelocity", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "wall", name: "translatingWallVelocity", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "wall", name: "partialSlip", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "wall", name: "fixedNormalSlip", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "wall", name: "fixedShearStress", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    // SPEC-LIT 120.4: the two outlets. `outletInlet` is (120.4a)'s reverse
    // flux switch; `advective` is (120.4b)-(120.4d)'s Robin outlet, reached
    // only where the caller refreshes it every step.
    Entry { family: "outlet", name: "outletInlet", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "outlet", name: "advective", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    // SPEC-LIT 120.5: the velocity-inlet family. The three fixed-value
    // profiles ride `fixedValue` with a per-face value the setup computes;
    // the five pressure-driven kinds rewrite their triple from the flux
    // every outer iteration, where the caller opts in.
    Entry { family: "velocity inlet", name: "surfaceNormalFixedValue", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "velocity inlet", name: "cylindricalInletVelocity", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "velocity inlet", name: "swirlFlowRateInletVelocity", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "velocity inlet", name: "pressureInletVelocity", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "velocity inlet", name: "pressureInletUniformVelocity", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "velocity inlet", name: "pressureDirectedInletVelocity", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "velocity inlet", name: "pressureDirectedInletOutletVelocity", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "velocity inlet", name: "pressureInletOutletParSlipVelocity", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    // SPEC-LIT 120.6: the Outlets II family - four patch-mean rewrites and
    // an extrapolated pressure gradient, refreshed every outer iteration
    // where the caller opts in with `BcInputs::patch_means`.
    Entry { family: "outlet", name: "fixedMean", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "outlet", name: "fixedMeanOutletInlet", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "outlet", name: "fixedFluxExtrapolatedPressure", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "outlet", name: "outletMappedUniformInlet", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "outlet", name: "matchedFlowRateOutletVelocity", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    // SPEC-LIT 120.7: the buoyant pressure family - a static pressure at a
    // patch, written in the solved `p_rgh` from the caller's `PrghInputs`.
    Entry { family: "buoyant pressure", name: "prghPressure", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "buoyant pressure", name: "prghTotalPressure", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "buoyant pressure", name: "prghTotalHydrostaticPressure", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "buoyant pressure", name: "uniformDensityHydrostaticPressure", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    // SPEC-LIT 120.8: the thermal family - three Robin triples Energy
    // rewrites from the face's current conductance every outer iteration,
    // and `turbulentHeatFluxTemperature`, which rides fixedFluxTemperature.
    Entry { family: "thermal", name: "externalWallHeatFluxTemperature", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "thermal", name: "turbulentHeatFluxTemperature", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "thermal", name: "lumpedMassTemperature", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "thermal", name: "totalFlowRateAdvectiveDiffusive", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    // SPEC-LIT 120.9: the atmospheric family - four Richards & Hoxey inlet
    // profiles computed once at setup, and two wall functions `wfNutWall`
    // writes.
    Entry { family: "atmospheric", name: "atmBoundaryLayerInletVelocity", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "atmospheric", name: "atmBoundaryLayerInletK", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "atmospheric", name: "atmBoundaryLayerInletEpsilon", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "atmospheric", name: "atmBoundaryLayerInletOmega", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "atmospheric", name: "nutkAtmRoughWallFunction", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "atmospheric", name: "nutUBlendedWallFunction", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    // SPEC-LIT 120.10: the recycled inflow - a fixedValue refreshed from the
    // offset plane every outer iteration, and a seeded fluctuation drawn
    // every time step.
    Entry { family: "mapped inflow", name: "mapped", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "mapped inflow", name: "mappedFixedValue", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "mapped inflow", name: "turbulentInlet", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    // SPEC-LIT 120.11: the inflow database, in space at setup and in time on
    // the device clock.
    Entry { family: "mapped inflow", name: "timeVaryingMappedFixedValue", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    // SPEC-LIT 120.2: the two cyclic jumps, on the pressure only - the jump
    // source enters the pressure assembly and no other.
    Entry { family: "jump", name: "fixedJump", status: Status::Implemented, field: "p", extended_by: &[], probes: &[] },
    Entry { family: "jump", name: "uniformJump", status: Status::Implemented, field: "p", extended_by: &[], probes: &[] },
    // ---- alias (5) --------------------------------------------------------
    Entry { family: "thermal", name: "compressible::alphatJayatillekeWallFunction", status: Status::Alias { to: "thermalWallFunction" }, field: "T", extended_by: &[], probes: &[] },
    Entry { family: "thermal", name: "compressible::turbulentTemperatureCoupledBaffleMixed", status: Status::Alias { to: "coupledTemperature" }, field: "T", extended_by: &[], probes: &[] },
    // SPEC-LIT 120.3: `nutUWallFunction` already applies Spalding's formula.
    Entry { family: "wall function", name: "nutUSpaldingWallFunction", status: Status::Alias { to: "nutUWallFunction" }, field: "psi", extended_by: &[], probes: &[] },
    // SPEC-LIT 120.8: the two alphat aliases. Read as `calculated` on a field
    // named `alphat`, refused naming `alphat` on any other.
    Entry { family: "thermal", name: "alphatWallFunction", status: Status::Alias { to: "calculated" }, field: "alphat", extended_by: &[], probes: &[] },
    Entry { family: "thermal", name: "compressible::alphatWallFunction", status: Status::Alias { to: "calculated" }, field: "alphat", extended_by: &[], probes: &[] },
    // ---- planned (10), by unit --------------------------------------------
    Entry { family: "synthetic inflow", name: "turbulentDFSEMInlet", status: Status::Planned { unit: "BC-16" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "synthetic inflow", name: "turbulentDigitalFilterInlet", status: Status::Planned { unit: "BC-16" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "coupled", name: "cyclicAMI", status: Status::Planned { unit: "BC-19" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "compressible", name: "inletOutletTotalTemperature", status: Status::Planned { unit: "BC-20" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "compressible", name: "supersonicFreestream", status: Status::Planned { unit: "BC-20" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "compressible", name: "fixedPressureCompressibleDensity", status: Status::Planned { unit: "BC-20" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "VOF", name: "variableHeightFlowRate", status: Status::Planned { unit: "BC-21" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "VOF", name: "variableHeightFlowRateInletVelocity", status: Status::Planned { unit: "BC-21" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "VOF", name: "outletPhaseMeanVelocity", status: Status::Planned { unit: "BC-21" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "VOF", name: "phaseHydrostaticPressure", status: Status::Planned { unit: "BC-21" }, field: "psi", extended_by: &[], probes: &[] },
    // ---- owned (10), by stream --------------------------------------------
    Entry { family: "compressible", name: "totalTemperature", status: Status::Owned { unit: "CMP-16" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "compressible", name: "waveTransmissive", status: Status::Owned { unit: "CMP-17" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "compressible", name: "characteristicFarfield", status: Status::Owned { unit: "CMP-18" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "compressible", name: "farfield", status: Status::Owned { unit: "CMP-18" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "radiation", name: "MarshakRadiation", status: Status::Owned { unit: "CHR-23" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "radiation", name: "MarshakRadiationFixedTemperature", status: Status::Owned { unit: "CHR-23" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "radiation", name: "greyDiffusiveRadiation", status: Status::Owned { unit: "CHR-23" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "radiation", name: "wideBandDiffusiveRadiation", status: Status::Owned { unit: "CHR-23" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "species", name: "semiPermeableBaffleMassFraction", status: Status::Owned { unit: "CHR-29" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "species", name: "specieTransfer", status: Status::Owned { unit: "CHR-29" }, field: "psi", extended_by: &[], probes: &[] },
    // ---- refused (10), permanently ----------------------------------------
    Entry { family: "refused", name: "coded*", status: Status::Refused { key: "loads no user code" }, field: "psi", extended_by: &[], probes: &["codedFixedValue", "codedMixed"] },
    Entry { family: "refused", name: "#codeStream", status: Status::Refused { key: "loads no user code" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "refused", name: "cyclicACMI", status: Status::Refused { key: "partly overlap" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "refused", name: "cyclicRepeatAMI", status: Status::Refused { key: "repeated around a sector" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "refused", name: "nonConformal*", status: Status::Refused { key: "intersects its two sides while the case runs" }, field: "psi", extended_by: &[], probes: &["nonConformalCyclic", "nonConformalError", "nonConformalMappedWall"] },
    Entry { family: "refused", name: "v2WallFunction", status: Status::Refused { key: "v2-f turbulence model" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "refused", name: "fWallFunction", status: Status::Refused { key: "v2-f turbulence model" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "refused", name: "wave*", status: Status::Refused { key: "no wave model" }, field: "psi", extended_by: &[], probes: &["waveAlpha", "waveVelocity"] },
    Entry { family: "refused", name: "activeBaffleVelocity", status: Status::Refused { key: "change of mesh topology" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "refused", name: "compressible::turbulentTemperatureRadCoupledMixed", status: Status::Refused { key: "never both" }, field: "T", extended_by: &[], probes: &[] },
];
