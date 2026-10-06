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
    // ---- implemented (49), in IMPLEMENTED_BC_NAMES' order -----------------
    Entry { family: "basic", name: "fixedValue", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "basic", name: "noSlip", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "basic", name: "zeroGradient", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "basic", name: "fixedGradient", status: Status::Implemented, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "basic", name: "uniformFixedGradient", status: Status::Implemented, field: "psi", extended_by: &["BC-06"], probes: &[] },
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
    Entry { family: "inlet-outlet", name: "flowRateInletVelocity", status: Status::Implemented, field: "psi", extended_by: &["BC-06", "CMP-16"], probes: &[] },
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
    // ---- alias (2) --------------------------------------------------------
    Entry { family: "thermal", name: "compressible::alphatJayatillekeWallFunction", status: Status::Alias { to: "thermalWallFunction" }, field: "T", extended_by: &[], probes: &[] },
    Entry { family: "thermal", name: "compressible::turbulentTemperatureCoupledBaffleMixed", status: Status::Alias { to: "coupledTemperature" }, field: "T", extended_by: &[], probes: &[] },
    // ---- planned (59), by unit --------------------------------------------
    Entry { family: "time-varying", name: "uniformFixedValue", status: Status::Planned { unit: "BC-06" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "time-varying", name: "uniformInletOutlet", status: Status::Planned { unit: "BC-06" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "time-varying", name: "uniformTotalPressure", status: Status::Planned { unit: "BC-06" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "time-varying", name: "fixedProfile", status: Status::Planned { unit: "BC-06" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "basic", name: "extrapolatedCalculated", status: Status::Planned { unit: "BC-07" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "basic", name: "directionMixed", status: Status::Planned { unit: "BC-07" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "wall", name: "rotatingWallVelocity", status: Status::Planned { unit: "BC-07" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "wall", name: "translatingWallVelocity", status: Status::Planned { unit: "BC-07" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "wall", name: "partialSlip", status: Status::Planned { unit: "BC-07" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "wall", name: "fixedNormalSlip", status: Status::Planned { unit: "BC-07" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "wall", name: "fixedShearStress", status: Status::Planned { unit: "BC-07" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "wall function", name: "nutUSpaldingWallFunction", status: Status::Planned { unit: "BC-07" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "velocity inlet", name: "surfaceNormalFixedValue", status: Status::Planned { unit: "BC-08" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "velocity inlet", name: "cylindricalInletVelocity", status: Status::Planned { unit: "BC-08" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "velocity inlet", name: "swirlFlowRateInletVelocity", status: Status::Planned { unit: "BC-08" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "velocity inlet", name: "pressureInletVelocity", status: Status::Planned { unit: "BC-08" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "velocity inlet", name: "pressureInletUniformVelocity", status: Status::Planned { unit: "BC-08" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "velocity inlet", name: "pressureDirectedInletVelocity", status: Status::Planned { unit: "BC-08" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "velocity inlet", name: "pressureDirectedInletOutletVelocity", status: Status::Planned { unit: "BC-08" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "velocity inlet", name: "pressureInletOutletParSlipVelocity", status: Status::Planned { unit: "BC-08" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "outlet", name: "outletInlet", status: Status::Planned { unit: "BC-09" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "outlet", name: "advective", status: Status::Planned { unit: "BC-09" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "outlet", name: "fixedMean", status: Status::Planned { unit: "BC-10" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "outlet", name: "fixedMeanOutletInlet", status: Status::Planned { unit: "BC-10" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "outlet", name: "fixedFluxExtrapolatedPressure", status: Status::Planned { unit: "BC-10" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "outlet", name: "outletMappedUniformInlet", status: Status::Planned { unit: "BC-10" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "outlet", name: "matchedFlowRateOutletVelocity", status: Status::Planned { unit: "BC-10" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "buoyant pressure", name: "prghPressure", status: Status::Planned { unit: "BC-11" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "buoyant pressure", name: "prghTotalPressure", status: Status::Planned { unit: "BC-11" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "buoyant pressure", name: "prghTotalHydrostaticPressure", status: Status::Planned { unit: "BC-11" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "buoyant pressure", name: "uniformDensityHydrostaticPressure", status: Status::Planned { unit: "BC-11" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "thermal", name: "externalWallHeatFluxTemperature", status: Status::Planned { unit: "BC-12" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "thermal", name: "turbulentHeatFluxTemperature", status: Status::Planned { unit: "BC-12" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "thermal", name: "lumpedMassTemperature", status: Status::Planned { unit: "BC-12" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "thermal", name: "totalFlowRateAdvectiveDiffusive", status: Status::Planned { unit: "BC-12" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "thermal", name: "alphatWallFunction", status: Status::Planned { unit: "BC-12" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "thermal", name: "compressible::alphatWallFunction", status: Status::Planned { unit: "BC-12" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "atmospheric", name: "atmBoundaryLayerInletVelocity", status: Status::Planned { unit: "BC-13" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "atmospheric", name: "atmBoundaryLayerInletK", status: Status::Planned { unit: "BC-13" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "atmospheric", name: "atmBoundaryLayerInletEpsilon", status: Status::Planned { unit: "BC-13" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "atmospheric", name: "atmBoundaryLayerInletOmega", status: Status::Planned { unit: "BC-13" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "atmospheric", name: "nutkAtmRoughWallFunction", status: Status::Planned { unit: "BC-13" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "atmospheric", name: "nutUBlendedWallFunction", status: Status::Planned { unit: "BC-13" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "mapped inflow", name: "mapped", status: Status::Planned { unit: "BC-14" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "mapped inflow", name: "mappedFixedValue", status: Status::Planned { unit: "BC-14" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "mapped inflow", name: "turbulentInlet", status: Status::Planned { unit: "BC-14" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "mapped inflow", name: "timeVaryingMappedFixedValue", status: Status::Planned { unit: "BC-15" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "synthetic inflow", name: "turbulentDFSEMInlet", status: Status::Planned { unit: "BC-16" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "synthetic inflow", name: "turbulentDigitalFilterInlet", status: Status::Planned { unit: "BC-16" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "jump", name: "fixedJump", status: Status::Planned { unit: "BC-17" }, field: "psi", extended_by: &[], probes: &[] },
    Entry { family: "jump", name: "uniformJump", status: Status::Planned { unit: "BC-17" }, field: "psi", extended_by: &[], probes: &[] },
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
