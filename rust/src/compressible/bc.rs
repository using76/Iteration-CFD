// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The compressible inlets of SPEC-LIT §126.1 and §126.3, as built in
//! §126.8 (docs/17 CMP-16): `totalPressure` with `gamma`,
//! `totalTemperature`, and the mass-flow inlet (`flowRateInletVelocity`
//! with `massFlowRate`).
//!
//! ```text
//! T_b = T_0 - |u_b|^2 / (2 c_p)                                            (126.1)
//! p_b = p_0 + (p_0 + p_op) expm1( gamma/(gamma-1) log1p( -|u_b|^2 / (2 c_p T_0) ) )
//!                                                                          (126.3)
//! u_b = - ( mdot / sum_f rho_f |Sf| ) n_f ,  rho_f = psi_f (p_b,f + p_op)  (126.4)
//! ```
//!
//! Each is a rewrite of the section-4 triple on its patch's faces, once per
//! outer iteration, from the latest state ([`CompressibleBcs::apply`]); the
//! assemblies gain no branch. (126.1) and (126.3) hold on an inflow face
//! (`u_b . Sf < 0`, Carlson's Table 1 convention); an outflow face is
//! zero-gradient (`fr = 0`, `g = 0`).
//!
//! [`lower`] takes the three field files as read and returns them with each
//! compressible kind replaced by a static placeholder (`fixedValue` at
//! `p_0`, `T_0` or zero velocity) that the shared setup of `field_setup.rs`
//! builds as it always has, plus the [`BcRule`]s that rewrite those faces.
//! The shared setup is not touched, so the incompressible `totalPressure`
//! (no `gamma`) of every other driver is what it was, bit for bit.
//!
//! The outlets of SPEC-LIT §126.4-§126.5, as built in §126.9 (docs/17
//! CMP-17): the supersonic switch of every fixed-`p` outlet, and
//! `waveTransmissive` on `p` or `T`:
//!
//! ```text
//! p_b = p_set  if u_n,P / c_P < 1  (fr = 1),  extrapolated otherwise      (126.5)
//! d phi/dt + w d phi/dn = K (phi_inf - phi),  w = u_n,P + c_P on p, u_n,P else,
//!   K = sigma c_P (1 - M_n^2) / lInf                                       (126.6)
//! fr = (1 + K dt) / (1 + alpha + K dt),  ref = (phi_b^o + K dt phi_inf) / (1 + K dt)
//!                                                                          (126.7)
//! ```
//!
//! (126.7) is BC-09's Robin rewrite (SPEC-LIT (120.4d) of §120.4), whose
//! kernel takes the wave speed and the rate as face-field inputs:
//! [`CompressibleOutlets`] fills them from the owner cell and calls
//! `field_ops::advective_robin_scalar` with a kind mask of its own, so no
//! `BcKind` is added. A `waveTransmissive` entry is built as a
//! `zeroGradient` placeholder; a fixed-`p` entry is built as it is read.
//!
//! The characteristic far-field of SPEC-LIT §126.6, as built in §126.10
//! (docs/17 CMP-18): `characteristicFarfield` (JSONC `farfield`) on `U`,
//! `p` and `T` together, Carlson's Riemann invariants with the owner's
//! state inside and the stated free stream outside:
//!
//! ```text
//! R+ = U_i + 2 c_i/(gamma-1),  R- = U_o - 2 c_o/(gamma-1)                  (126.8)
//!   supersonic outflow: R- from the interior; supersonic inflow: R+ from
//!   the free stream                                                       (126.9)
//! U_b = (R+ + R-)/2,  c_b = (gamma-1)(R+ - R-)/4                           (126.10)
//! u_b, s_b from the interior if U_b > 0, else from the free stream         (126.11)
//! rho_b = (c_b^2/(gamma s_b))^(1/(gamma-1)),  p_b = rho_b c_b^2/gamma,
//!   T_b = p_b/(rho_b R_s)                                                  (126.12)
//! ```
//!
//! [`CompressibleFarfields`] runs one kernel that rewrites all three
//! triples (`fr = 1`) per outer iteration; [`farfield_face`] is its f64
//! host twin. A far-field entry is built as a `fixedValue` placeholder at
//! its stated free stream.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §126 intro, §126.1, §126.3 - the equations and
//!     the inflow convention; §126.8 - this file as built, its contract;
//!     §126.4, §126.5 and §126.9 - the outlets and their contract;
//!     §120.4 - the Robin form (120.4d) the outlets reuse; §126.6 and
//!     §126.10 - the far-field and its contract
//!   Carlson (2011) §2.2, eqs. (8)-(17) - the characteristic far-field,
//!     which (126.8)-(126.12) restate
//!   Carlson (2011) §2.4, eq. (25) - the static-pressure outlet and its
//!     supersonic extrapolation, which (126.5) restates
//!   NACA Report 1135 (Ames Research Staff, 1953), eqs. (43)-(45): the
//!     isentropic total-to-static relations (126.1)-(126.2) restate
//!   Carlson, "Inflow/Outflow Boundary Conditions with Application to
//!     FUN3D", NASA/TM-2011-217181 (2011), eq. (48) and Table 1 - the
//!     adiabatic inflow statement and the sign of an inflow face
//! Boundary-condition and dictionary NAMES follow the public user guides by
//! name only; no OpenFOAM source was opened.
//! No GPL-licensed source was consulted.

use std::collections::BTreeMap;

use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::field_ops::{self, FieldKernels};
use crate::io::fields::{PatchFieldSpec, RawScalarField, RawVectorField};
use crate::field::BcKind;
use crate::mesh::{GpuMesh, HostMesh};
use crate::timescheme::DdtCoeffs;
use crate::{Label, Scalar, Vec3};

use super::thermo::PerfectGas;
use super::CompressibleState;

#[cfg(test)]
mod tests {
    //! CMP-16's gate table of SPEC-LIT §126.8, written once in
    //! `tests/shared/compressible_bc_cases.rs` and included here and by
    //! `tests/compressible_bc.rs`, which runs the same tests in both builds.
    use crate as ofgpu;

    include!("../../tests/shared/compressible_bc_cases.rs");
}

/// The one-block width of `cmpBcMassFlowInlet` (`OFGPU_CMP_BC_BLOCK`).
const MASS_FLOW_BLOCK: u32 = 256;

/// Which of the three conditions a [`BcRule`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleKind {
    /// (126.3) on `p`: `totalPressure` with `gamma`.
    TotalPressure,
    /// (126.1) on `T`: `totalTemperature`.
    TotalTemperature,
    /// (126.4) on `U`: `flowRateInletVelocity` with `massFlowRate`.
    MassFlowInlet,
}

/// One patch's compressible condition: the faces `[start, start + size)` of
/// the flattened boundary arrays, and its stated value.
#[derive(Debug, Clone, PartialEq)]
pub struct BcRule {
    pub kind: RuleKind,
    pub patch: String,
    pub start: usize,
    pub size: usize,
    /// `p_0` (Pa, in the gauge of `p`), `T_0` (K) or `mdot` (kg/s).
    pub value: f64,
    /// [`RuleKind::TotalPressure`] only: the same patch's `totalTemperature`
    /// `T_0`, or 0, in which case each face takes `T_0 = T_b + |u_b|^2/(2 c_p)`
    /// (§126.8). 0 on every other kind.
    pub t0: f64,
}

/// [`lower`]'s result: the fields the shared setup builds, and the rules.
#[derive(Debug, Clone)]
pub struct Lowered {
    pub u: RawVectorField,
    pub p: RawScalarField,
    pub t: RawScalarField,
    pub rules: Vec<BcRule>,
    /// SPEC-LIT §126.9: the outlets, one per mesh patch that carries one.
    pub outlets: Vec<OutletRule>,
    /// SPEC-LIT §126.10: the far-fields, one per mesh patch whose `U`, `p`
    /// and `T` all carry `characteristicFarfield` (or `farfield`).
    pub farfields: Vec<FarfieldRule>,
}

/// A refusal naming the field and the patch key.
fn refuse(field: &str, key: &str, why: String) -> Error {
    Error::Refused(format!("{field} patch {key}: {why}"))
}

/// One number of a patch entry, in f64 whatever the build: the last token
/// that parses, as `PatchFieldSpec::number` reads `uniform 1e5` or a
/// dimensioned entry. `None` when the key is absent.
fn number64(spec: &PatchFieldSpec, field: &str, key: &str, entry: &str) -> Result<Option<f64>> {
    let Some(raw) = spec.extra.get(entry) else {
        return Ok(None);
    };
    match raw.split_whitespace().filter_map(|t| t.parse::<f64>().ok()).next_back() {
        Some(v) if v.is_finite() => Ok(Some(v)),
        _ => Err(refuse(field, key, format!("{entry}: \"{raw}\" is not a finite number"))),
    }
}

fn required64(spec: &PatchFieldSpec, field: &str, key: &str, entry: &str, kind: &str) -> Result<f64> {
    number64(spec, field, key, entry)?.ok_or_else(|| {
        refuse(field, key, format!("{kind} needs a `{entry}` entry and the patch has none (SPEC-LIT §126.8)"))
    })
}

/// A stated `gamma` must be the gas's: the gas law carries one.
fn check_gamma(spec: &PatchFieldSpec, field: &str, key: &str, kind: &str, gas: &PerfectGas) -> Result<Option<f64>> {
    let g = number64(spec, field, key, "gamma")?;
    if let Some(g) = g {
        if (g - gas.gamma).abs() > 1e-12 * gas.gamma {
            return Err(refuse(
                field,
                key,
                format!(
                    "{kind} states gamma {g}, but the gas's gamma is {}; the gas law carries one \
                     ratio of specific heats, so the patch must state the same (SPEC-LIT §126.8)",
                    gas.gamma
                ),
            ));
        }
    }
    Ok(g)
}

/// The placeholder a compressible kind is built as before its first rewrite.
fn placeholder_scalar(v: f64) -> PatchFieldSpec {
    PatchFieldSpec { type_name: "fixedValue".into(), value: vec![v as Scalar], ..PatchFieldSpec::default() }
}

/// Check one scalar entry and, if it is a compressible kind, return its
/// placeholder and its stated value. `field` is `p` or `T`.
fn scalar_kind(field: &str, key: &str, spec: &PatchFieldSpec, gas: &PerfectGas) -> Result<Option<(RuleKind, f64)>> {
    match (field, spec.type_name.as_str()) {
        ("p", "totalPressure") => {
            if check_gamma(spec, field, key, "totalPressure", gas)?.is_none() {
                return Err(refuse(
                    field,
                    key,
                    "totalPressure without `gamma` is the incompressible kinematic condition \
                     (p0 - |U|^2/2), which a gas does not obey; ofgpu-compressible needs the \
                     compressible form, totalPressure with `gamma` (SPEC-LIT §126.1)"
                        .into(),
                ));
            }
            let p0 = required64(spec, field, key, "p0", "totalPressure")?;
            if !(p0 + gas.p_op > 0.0) {
                return Err(refuse(
                    field,
                    key,
                    format!("totalPressure p0 {p0} + pOperating {} is not above 0 Pa (SPEC-LIT §122.3)", gas.p_op),
                ));
            }
            Ok(Some((RuleKind::TotalPressure, p0)))
        }
        ("T", "totalTemperature") => {
            check_gamma(spec, field, key, "totalTemperature", gas)?;
            let t0 = required64(spec, field, key, "T0", "totalTemperature")?;
            if !(t0 > 0.0) {
                return Err(refuse(field, key, format!("totalTemperature T0 {t0} is not above 0 K (SPEC-LIT §122.3)")));
            }
            Ok(Some((RuleKind::TotalTemperature, t0)))
        }
        (_, "totalPressure" | "totalTemperature" | "flowRateInletVelocity") => Err(refuse(
            field,
            key,
            format!(
                "{} is a condition on {}, not on {field} (SPEC-LIT §126.8)",
                spec.type_name,
                match spec.type_name.as_str() {
                    "totalPressure" => "p",
                    "totalTemperature" => "T",
                    _ => "U",
                }
            ),
        )),
        _ => Ok(None),
    }
}

/// The mass-flow inlet on `U`: `flowRateInletVelocity` with `massFlowRate`.
fn vector_kind(key: &str, spec: &PatchFieldSpec) -> Result<Option<f64>> {
    match spec.type_name.as_str() {
        "flowRateInletVelocity" => {
            if spec.extra.contains_key("volumetricFlowRate") {
                return Err(refuse(
                    "U",
                    key,
                    "flowRateInletVelocity with volumetricFlowRate: a gas inlet states its mass \
                     flow, so ofgpu-compressible takes massFlowRate only (SPEC-LIT §126.3)"
                        .into(),
                ));
            }
            let m = required64(spec, "U", key, "massFlowRate", "flowRateInletVelocity")?;
            if !(m > 0.0) {
                return Err(refuse(
                    "U",
                    key,
                    format!("flowRateInletVelocity massFlowRate {m} is not above 0 kg/s: (126.4) is an inlet"),
                ));
            }
            Ok(Some(m))
        }
        "totalPressure" | "totalTemperature" => Err(refuse(
            "U",
            key,
            format!("{} is a condition on {}, not on U (SPEC-LIT §126.8)", spec.type_name,
                if spec.type_name == "totalPressure" { "p" } else { "T" }),
        )),
        _ => Ok(None),
    }
}

/// Which outlet an [`OutletRule`] is (SPEC-LIT §126.9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutletKind {
    /// (126.5) on `p`: every `fixedValue` `p` patch as read.
    SupersonicSwitch,
    /// (126.6)-(126.7) on `p` or `T`: `waveTransmissive`.
    WaveTransmissive,
}

/// The field an [`OutletRule`] rewrites.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutletField {
    P,
    T,
}

/// One patch's outlet: the faces `[start, start + size)` and, for
/// `waveTransmissive`, `lInf` (0 when not stated, so `K = 0`), `sigma` and
/// `fieldInf` (both 0 then). Every value is read in f64.
#[derive(Debug, Clone, PartialEq)]
pub struct OutletRule {
    pub kind: OutletKind,
    pub field: OutletField,
    pub patch: String,
    pub start: usize,
    pub size: usize,
    pub l_inf: f64,
    pub sigma: f64,
    pub field_inf: f64,
}

/// A `waveTransmissive` entry's `(lInf, sigma, fieldInf)`, `None` for any
/// other type. Refused by name: a stated `gamma` that is not the gas's, a
/// `phi`/`rho`/`psi` naming another field (the condition reads the solver's
/// own state), `lInf <= 0`, `lInf` without `sigma` (no default is recalled,
/// SPEC-LIT §126.5) or without `fieldInf`, `sigma <= 0`, and `sigma` or
/// `fieldInf` without `lInf` (read by nothing, since `K = 0` then).
fn wave_entry(field: &str, key: &str, spec: &PatchFieldSpec, gas: &PerfectGas) -> Result<Option<(f64, f64, f64)>> {
    if spec.type_name != "waveTransmissive" {
        return Ok(None);
    }
    check_gamma(spec, field, key, "waveTransmissive", gas)?;
    let own: [(&str, &[&str]); 3] = [("phi", &["phi"]), ("rho", &["rho"]), ("psi", &["thermo:psi", "psi"])];
    for (entry, names) in own {
        if let Some(v) = spec.extra.get(entry) {
            if !names.contains(&v.trim()) {
                return Err(refuse(
                    field,
                    key,
                    format!(
                        "waveTransmissive names {entry} \"{v}\", but (126.6) is formed from the solver's own u, T and gas, so only {} is accepted (SPEC-LIT §126.9)",
                        names.join(" or ")
                    ),
                ));
            }
        }
    }
    let l_inf = number64(spec, field, key, "lInf")?;
    let sigma = number64(spec, field, key, "sigma")?;
    let Some(l) = l_inf else {
        for entry in ["sigma", "fieldInf"] {
            if spec.extra.contains_key(entry) {
                return Err(refuse(
                    field,
                    key,
                    format!(
                        "waveTransmissive states {entry} without lInf; K = 0 when lInf is not stated (126.6), so {entry} would be read by nothing. State lInf, or remove {entry} (SPEC-LIT §126.9)"
                    ),
                ));
            }
        }
        return Ok(Some((0.0, 0.0, 0.0)));
    };
    if !(l > 0.0) {
        return Err(refuse(field, key, format!("waveTransmissive lInf {l} is not above 0 m (SPEC-LIT §126.9)")));
    }
    let Some(sg) = sigma else {
        return Err(refuse(
            field,
            key,
            "waveTransmissive states lInf and no sigma; (126.6)'s K = sigma c (1 - M_n^2) / lInf has no recalled default for sigma, so it is a required entry when lInf is stated (SPEC-LIT §126.5)"
                .into(),
        ));
    };
    if !(sg > 0.0) {
        return Err(refuse(field, key, format!("waveTransmissive sigma {sg} is not above 0 (SPEC-LIT §126.9)")));
    }
    let fi = required64(spec, field, key, "fieldInf", "waveTransmissive with lInf")?;
    Ok(Some((l, sg, fi)))
}

/// SPEC-LIT §126.10: whether `type_name` is the characteristic far-field,
/// in either spelling (`farfield` is the JSONC kind, §122.6).
fn is_farfield(type_name: &str) -> bool {
    matches!(type_name, "characteristicFarfield" | "farfield")
}

/// One patch's characteristic far-field (SPEC-LIT §126.10): the faces
/// `[start, start + size)` and the stated free stream, widened to f64 from
/// the `value` entries of `U`, `p` (gauge, in the datum of `p`) and `T`.
#[derive(Debug, Clone, PartialEq)]
pub struct FarfieldRule {
    pub patch: String,
    pub start: usize,
    pub size: usize,
    pub u: [f64; 3],
    pub p: f64,
    pub t: f64,
}

/// A far-field entry's one stated value (its free-stream component): a
/// `uniform` value is required, since (126.8) reads one free stream.
fn farfield_scalar(field: &str, key: &str, spec: &PatchFieldSpec, gas: &PerfectGas) -> Result<f64> {
    check_gamma(spec, field, key, &spec.type_name, gas)?;
    match spec.value.as_slice() {
        [v] if v.is_finite() => Ok(super::w64(*v)),
        _ => Err(refuse(
            field,
            key,
            format!(
                "{} needs one uniform `value`, the free stream's {field} (SPEC-LIT §126.10)",
                spec.type_name
            ),
        )),
    }
}

fn farfield_vector(key: &str, spec: &PatchFieldSpec, gas: &PerfectGas) -> Result<[f64; 3]> {
    check_gamma(spec, "U", key, &spec.type_name, gas)?;
    match spec.value_v.as_slice() {
        [v] if v.x.is_finite() && v.y.is_finite() && v.z.is_finite() => Ok([v.x, v.y, v.z].map(super::w64)),
        _ => Err(refuse(
            "U",
            key,
            format!("{} needs one uniform `value`, the free stream's U (SPEC-LIT §126.10)", spec.type_name),
        )),
    }
}

/// Which of (126.9)-(126.11)'s four cases a far-field face is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FarfieldBranch {
    SubsonicInflow,
    SubsonicOutflow,
    SupersonicInflow,
    SupersonicOutflow,
}

/// The far-field face state of [`farfield_face`]: `u_b`, the gauge `p_b`
/// (`p_abs,b - p_op`) and `T_b`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FarfieldFace {
    pub u: [f64; 3],
    pub p: f64,
    pub t: f64,
    pub branch: FarfieldBranch,
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The host twin of `cmpBcFarfield`, in f64 and in the kernel's operation
/// order: (126.8)-(126.12) of SPEC-LIT 126.6 for the unit outward normal
/// `n`, the owner's `u_i`, gauge `p_i` and `T_i`, and the free stream
/// `u_o`, gauge `p_o` and `T_o`. `c = sqrt(gamma R_s T)` and
/// `rho = (p + p_op)/(R_s T)` on both sides (§126 intro, (122.10)).
#[allow(clippy::too_many_arguments)]
pub fn farfield_face(
    gas: &PerfectGas,
    n: [f64; 3],
    u_i: [f64; 3],
    p_i: f64,
    t_i: f64,
    u_o: [f64; 3],
    p_o: f64,
    t_o: f64,
) -> FarfieldFace {
    let (gamma, rs, pop) = (gas.gamma, gas.r_s(), gas.p_op);
    let (gm1, tg, ex, grs) = (gamma - 1.0, 2.0 / (gamma - 1.0), 1.0 / (gamma - 1.0), gamma * rs);
    let co = (grs * t_o).sqrt();
    let pao = p_o + pop;
    let so = pao / (pao / (rs * t_o)).powf(gamma);
    let ui = dot3(u_i, n);
    let ci = (grs * t_i).sqrt();
    let uo = dot3(u_o, n);
    let (mut rp, mut rm) = (ui + tg * ci, uo - tg * co);
    let sup = ui.abs() / ci >= 1.0;
    if ui > 0.0 && sup {
        rm = ui - tg * ci;
    }
    if ui < 0.0 && sup {
        rp = uo + tg * co;
    }
    let ub = 0.5 * (rp + rm);
    let cb = gm1 * (rp - rm) * 0.25;
    let (u, sb) = if ub > 0.0 {
        let d = ub - ui;
        let pa = p_i + pop;
        let rhoi = pa / (rs * t_i);
        ([u_i[0] + d * n[0], u_i[1] + d * n[1], u_i[2] + d * n[2]], pa / rhoi.powf(gamma))
    } else {
        let d = ub - uo;
        ([u_o[0] + d * n[0], u_o[1] + d * n[1], u_o[2] + d * n[2]], so)
    };
    let cb2 = cb * cb;
    let rhob = (cb2 / (gamma * sb)).powf(ex);
    let pab = rhob * cb2 / gamma;
    let branch = match (sup, ui > 0.0, ui < 0.0, ub > 0.0) {
        (true, true, _, _) => FarfieldBranch::SupersonicOutflow,
        (true, _, true, _) => FarfieldBranch::SupersonicInflow,
        (_, _, _, true) => FarfieldBranch::SubsonicOutflow,
        _ => FarfieldBranch::SubsonicInflow,
    };
    FarfieldFace { u, p: pab - pop, t: pab / (rhob * rs), branch }
}

/// SPEC-LIT §126.8: the three field files as read, with every compressible
/// kind replaced by its placeholder, and one [`BcRule`] per mesh patch that
/// carries one. Refused by name: `totalPressure` without `gamma`, a stated
/// `gamma` that is not the gas's, a missing `p0`/`T0`/`massFlowRate`,
/// `volumetricFlowRate`, a kind on the wrong field, and a mass-flow inlet
/// whose `p` is not `zeroGradient` (§126.3).
pub fn lower(
    u: &RawVectorField,
    p: &RawScalarField,
    t: &RawScalarField,
    hm: &HostMesh,
    gas: &PerfectGas,
) -> Result<Lowered> {
    let mut out = Lowered { u: u.clone(), p: p.clone(), t: t.clone(), rules: Vec::new(), outlets: Vec::new(), farfields: Vec::new() };

    // Every entry, by its own key (patterns too): validate and replace.
    let lower_scalar = |field: &str, b: &mut BTreeMap<String, PatchFieldSpec>| -> Result<()> {
        for (key, spec) in b.iter_mut() {
            if let Some((_, v)) = scalar_kind(field, key, spec, gas)? {
                *spec = placeholder_scalar(v);
            } else if is_farfield(&spec.type_name) {
                // §126.10: fixed at the free stream until the first rewrite.
                *spec = placeholder_scalar(farfield_scalar(field, key, spec, gas)?);
            } else if wave_entry(field, key, spec, gas)?.is_some() {
                // §126.9: zero-gradient until the first rewrite, so phi_b^o
                // starts at the owner cell's value.
                *spec = PatchFieldSpec { type_name: "zeroGradient".into(), ..PatchFieldSpec::default() };
            }
        }
        Ok(())
    };
    lower_scalar("p", &mut out.p.boundary)?;
    lower_scalar("T", &mut out.t.boundary)?;
    for (key, spec) in out.u.boundary.iter_mut() {
        if spec.type_name == "waveTransmissive" {
            return Err(refuse(
                "U",
                key,
                "waveTransmissive on U is not built: (126.6) is built on p and T (SPEC-LIT §126.9); U takes zeroGradient at a non-reflecting outlet"
                    .into(),
            ));
        }
        if is_farfield(&spec.type_name) {
            let v = farfield_vector(key, spec, gas)?;
            *spec = PatchFieldSpec {
                type_name: "fixedValue".into(),
                value_v: vec![Vec3::new(v[0] as Scalar, v[1] as Scalar, v[2] as Scalar)],
                ..PatchFieldSpec::default()
            };
            continue;
        }
        if vector_kind(key, spec)?.is_some() {
            *spec = PatchFieldSpec {
                type_name: "fixedValue".into(),
                value_v: vec![Vec3::new(0.0, 0.0, 0.0)],
                ..PatchFieldSpec::default()
            };
        }
        // A scalar kind on U is refused by vector_kind; a vector kind on p or
        // T by scalar_kind.
    }

    // One rule per mesh patch, from the entry that governs it.
    for patch in &hm.patches {
        let name = patch.name.as_str();
        let (start, size) = (patch.start, patch.size);
        if let Some(spec) = u.spec(name)? {
            if let Some(mdot) = vector_kind(name, spec)? {
                let zg = p.spec(name)?.is_some_and(|s| s.type_name == "zeroGradient");
                if !zg {
                    return Err(refuse(
                        "p",
                        name,
                        "a mass-flow inlet (U flowRateInletVelocity with massFlowRate) takes p \
                         zero-gradient (SPEC-LIT §126.3); set p to zeroGradient on the patch"
                            .into(),
                    ));
                }
                out.rules.push(BcRule { kind: RuleKind::MassFlowInlet, patch: name.into(), start, size, value: mdot, t0: 0.0 });
            }
        }
        let t_rule = match t.spec(name)? {
            Some(spec) => scalar_kind("T", name, spec, gas)?,
            None => None,
        };
        if let Some((kind, t0)) = t_rule {
            out.rules.push(BcRule { kind, patch: name.into(), start, size, value: t0, t0: 0.0 });
        }
        if let Some(spec) = p.spec(name)? {
            if let Some((kind, p0)) = scalar_kind("p", name, spec, gas)? {
                let t0 = t_rule.map_or(0.0, |(_, t0)| t0);
                out.rules.push(BcRule { kind, patch: name.into(), start, size, value: p0, t0 });
            }
        }
        // §126.10: the far-field decides U, p and T together, from one
        // free stream, so all three carry it or none does.
        let (su, sp, st) = (u.spec(name)?, p.spec(name)?, t.spec(name)?);
        let ff = [su.map(|s| s.type_name.as_str()), sp.map(|s| s.type_name.as_str()), st.map(|s| s.type_name.as_str())]
            .map(|t| t.is_some_and(is_farfield));
        if ff.iter().any(|&b| b) {
            if let (true, true, true, Some(su), Some(sp), Some(st)) = (ff[0], ff[1], ff[2], su, sp, st) {
                let rule = FarfieldRule {
                    patch: name.into(),
                    start,
                    size,
                    u: farfield_vector(name, su, gas)?,
                    p: farfield_scalar("p", name, sp, gas)?,
                    t: farfield_scalar("T", name, st, gas)?,
                };
                if !(rule.t > 0.0) {
                    return Err(refuse("T", name, format!("the far-field's free-stream T {} is not above 0 K (SPEC-LIT §122.3)", rule.t)));
                }
                if !(rule.p + gas.p_op > 0.0) {
                    return Err(refuse(
                        "p",
                        name,
                        format!("the far-field's free-stream p {} + pOperating {} is not above 0 Pa (SPEC-LIT §122.3)", rule.p, gas.p_op),
                    ));
                }
                out.farfields.push(rule);
            } else {
                let on: Vec<&str> = ["U", "p", "T"].iter().zip(ff).filter(|(_, b)| *b).map(|(f, _)| *f).collect();
                let field = ["U", "p", "T"].iter().zip(ff).find(|(_, b)| !*b).map_or("U", |(f, _)| *f);
                return Err(refuse(
                    field,
                    name,
                    format!(
                        "characteristicFarfield is on {} only; (126.8)-(126.12) decide U, p and T together from one free stream, so the patch states it on all three (SPEC-LIT §126.10)",
                        on.join(" and ")
                    ),
                ));
            }
        }
        // §126.9: the outlets. A fixed p is (126.5)'s static-pressure
        // outlet; waveTransmissive is (126.6) on p or T.
        for (field, raw) in [(OutletField::P, p), (OutletField::T, t)] {
            let fname = if field == OutletField::P { "p" } else { "T" };
            let Some(spec) = raw.spec(name)? else { continue };
            let rule = |kind, (l_inf, sigma, field_inf)| OutletRule {
                kind,
                field,
                patch: name.into(),
                start,
                size,
                l_inf,
                sigma,
                field_inf,
            };
            if let Some(w) = wave_entry(fname, name, spec, gas)? {
                out.outlets.push(rule(OutletKind::WaveTransmissive, w));
            } else if field == OutletField::P && spec.type_name == "fixedValue" {
                out.outlets.push(rule(OutletKind::SupersonicSwitch, (0.0, 0.0, 0.0)));
            }
        }
    }
    Ok(out)
}

/// The three kernels and the run's rules (SPEC-LIT §126.8).
pub struct CompressibleBcs {
    rules: Vec<BcRule>,
    total_pressure: CudaFunction,
    total_temperature: CudaFunction,
    mass_flow: CudaFunction,
    /// `fl(2 c_p)`, `fl(gamma/(gamma-1))`, `fl(p_op)`, and `p_op` in f64.
    cp2: Scalar,
    ge: Scalar,
    pop: Scalar,
    p_op: f64,
}

impl CompressibleBcs {
    /// Load the kernels for `rules` on `m`; a rule whose faces leave the
    /// boundary arrays is refused.
    pub fn new(gpu: &Gpu, m: &GpuMesh, gas: &PerfectGas, rules: Vec<BcRule>) -> Result<Self> {
        for r in &rules {
            if r.start + r.size > m.n_boundary_faces {
                return Err(Error::Refused(format!(
                    "compressible::bc: patch {} faces {}..{} leave the {} boundary faces",
                    r.patch,
                    r.start,
                    r.start + r.size,
                    m.n_boundary_faces
                )));
            }
        }
        let k = KernelSet::new(gpu, crate::kernels::COMPRESSIBLE)?;
        Ok(Self {
            rules,
            total_pressure: k.func("cmpBcTotalPressure")?,
            total_temperature: k.func("cmpBcTotalTemperature")?,
            mass_flow: k.func("cmpBcMassFlowInlet")?,
            cp2: (2.0 * gas.cp()) as Scalar,
            ge: (gas.gamma / (gas.gamma - 1.0)) as Scalar,
            pop: gas.p_op as Scalar,
            p_op: gas.p_op,
        })
    }

    pub fn rules(&self) -> &[BcRule] {
        &self.rules
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    fn any(&self, kind: RuleKind) -> bool {
        self.rules.iter().any(|r| r.kind == kind && r.size > 0)
    }

    /// One rewrite of every rule's triple, then the face values of each
    /// field a rule touched: the mass-flow inlets first (from `psi_b` and
    /// `p_b`), then `U`'s faces; the total temperatures (from `u_b`), then
    /// `T`'s; the total pressures (from `u_b` and, without a `T_0`, `T_b`),
    /// then `p`'s. The caller re-forms the boundary gas state.
    pub fn apply(&self, gpu: &Gpu, fldk: &FieldKernels, s: &mut CompressibleState, m: &GpuMesh) -> Result<()> {
        for r in self.rules.iter().filter(|r| r.kind == RuleKind::MassFlowInlet && r.size > 0) {
            let (start, n) = (r.start as Label, r.size as Label);
            let (mdot, pop) = (r.value as Scalar, self.pop);
            let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (MASS_FLOW_BLOCK, 1, 1), shared_mem_bytes: 0 };
            let f = self.mass_flow.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut s.u.fr)
                    .arg(&mut s.u.ref_value)
                    .arg(&mut s.u.ref_grad)
                    .arg(&s.psi.bf)
                    .arg(&s.p.bf)
                    .arg(&m.b_sf)
                    .arg(&m.b_mag_sf)
                    .arg(&start)
                    .arg(&n)
                    .arg(&mdot)
                    .arg(&pop)
                    .launch(cfg)?;
            }
        }
        if self.any(RuleKind::MassFlowInlet) {
            field_ops::correct_boundary_conditions_vector(gpu, fldk, &mut s.u, m)?;
        }

        for r in self.rules.iter().filter(|r| r.kind == RuleKind::TotalTemperature && r.size > 0) {
            let (start, n) = (r.start as Label, r.size as Label);
            let (t0, cp2) = (r.value as Scalar, self.cp2);
            let f = self.total_temperature.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut s.t.fr)
                    .arg(&mut s.t.ref_value)
                    .arg(&mut s.t.ref_grad)
                    .arg(&s.u.bf)
                    .arg(&m.b_sf)
                    .arg(&start)
                    .arg(&n)
                    .arg(&t0)
                    .arg(&cp2)
                    .launch(cfg_for(r.size))?;
            }
        }
        if self.any(RuleKind::TotalTemperature) {
            field_ops::correct_boundary_conditions(gpu, fldk, &mut s.t, m)?;
        }

        for r in self.rules.iter().filter(|r| r.kind == RuleKind::TotalPressure && r.size > 0) {
            let (start, n) = (r.start as Label, r.size as Label);
            let p0 = r.value as Scalar;
            let pt = (r.value + self.p_op) as Scalar;
            let (t0, cp2, ge) = (r.t0 as Scalar, self.cp2, self.ge);
            let f = self.total_pressure.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut s.p.fr)
                    .arg(&mut s.p.ref_value)
                    .arg(&mut s.p.ref_grad)
                    .arg(&s.u.bf)
                    .arg(&s.t.bf)
                    .arg(&m.b_sf)
                    .arg(&start)
                    .arg(&n)
                    .arg(&p0)
                    .arg(&pt)
                    .arg(&t0)
                    .arg(&cp2)
                    .arg(&ge)
                    .launch(cfg_for(r.size))?;
            }
        }
        if self.any(RuleKind::TotalPressure) {
            field_ops::correct_boundary_conditions(gpu, fldk, &mut s.p, m)?;
        }
        Ok(())
    }
}

/// One field's `waveTransmissive` faces (SPEC-LIT §126.9): the kind mask
/// BC-09's kernel matches (`BcKind::Advective` on the rule's faces, -1
/// elsewhere), the wave speed and the rate (126.6), `phi_inf`, and the face
/// values of the two previous steps. Every buffer covers the whole boundary.
struct WaveFaces {
    mask: DevBuf<Label>,
    w: DevBuf<Scalar>,
    k_relax: DevBuf<Scalar>,
    phi_inf: DevBuf<Scalar>,
    b_old: DevBuf<Scalar>,
    b_old_old: DevBuf<Scalar>,
}

impl WaveFaces {
    fn new(gpu: &Gpu, nbf: usize, rules: &[&OutletRule]) -> Result<Self> {
        let mut mask = vec![-1 as Label; nbf];
        let mut inf = vec![0.0 as Scalar; nbf];
        for r in rules {
            for f in r.start..r.start + r.size {
                mask[f] = BcKind::Advective as Label;
                inf[f] = r.field_inf as Scalar;
            }
        }
        // A zero-length upload still needs a buffer the launchers accept.
        let n = nbf.max(1);
        mask.resize(n, -1);
        inf.resize(n, 0.0);
        Ok(Self {
            mask: gpu.upload(&mask)?,
            w: gpu.zeros(n)?,
            k_relax: gpu.zeros(n)?,
            phi_inf: gpu.upload(&inf)?,
            b_old: gpu.zeros(n)?,
            b_old_old: gpu.zeros(n)?,
        })
    }
}

/// The time coefficients (126.7) is formed with in one outer iteration
/// (SPEC-LIT §126.9): the run's own `(a_n, a_0, a_00)` of a transient run
/// (`Ddt::face_coeffs`, Euler or backward) with no local step, or, under
/// `localEuler`, `(1, -1, 0)` with the per-cell `r_P = 1/dt_P` by which
/// `cmpBcWaveSpeed` divides `w` and `K`.
#[derive(Clone, Copy)]
pub struct WaveTime<'a> {
    pub c: DdtCoeffs,
    pub r_delta_t: Option<&'a DevBuf<Scalar>>,
}

/// The two outlet kernels and the run's outlet rules (SPEC-LIT §126.9).
pub struct CompressibleOutlets {
    rules: Vec<OutletRule>,
    switch: CudaFunction,
    speed: CudaFunction,
    /// `fl(gamma R_s)`.
    grs: Scalar,
    p_wave: Option<WaveFaces>,
    t_wave: Option<WaveFaces>,
}

impl CompressibleOutlets {
    /// Load the kernels for `rules` on `m`; a rule whose faces leave the
    /// boundary arrays is refused.
    pub fn new(gpu: &Gpu, m: &GpuMesh, gas: &PerfectGas, rules: Vec<OutletRule>) -> Result<Self> {
        let nbf = m.n_boundary_faces;
        for r in &rules {
            if r.start + r.size > nbf {
                return Err(Error::Refused(format!(
                    "compressible::bc: outlet patch {} faces {}..{} leave the {nbf} boundary faces",
                    r.patch,
                    r.start,
                    r.start + r.size
                )));
            }
        }
        let wave = |field: OutletField| -> Result<Option<WaveFaces>> {
            let of: Vec<&OutletRule> =
                rules.iter().filter(|r| r.kind == OutletKind::WaveTransmissive && r.field == field).collect();
            if of.is_empty() {
                Ok(None)
            } else {
                WaveFaces::new(gpu, nbf, &of).map(Some)
            }
        };
        let (p_wave, t_wave) = (wave(OutletField::P)?, wave(OutletField::T)?);
        let k = KernelSet::new(gpu, crate::kernels::COMPRESSIBLE)?;
        Ok(Self {
            switch: k.func("cmpBcSupersonicSwitch")?,
            speed: k.func("cmpBcWaveSpeed")?,
            grs: (gas.gamma * gas.r_s()) as Scalar,
            rules,
            p_wave,
            t_wave,
        })
    }

    pub fn rules(&self) -> &[OutletRule] {
        &self.rules
    }

    /// Whether any rule is `waveTransmissive`, which needs a time step.
    pub fn has_wave(&self) -> bool {
        self.p_wave.is_some() || self.t_wave.is_some()
    }

    /// `phi_b^o = phi_b^oo = phi_b` from the face values as they stand: the
    /// start of a run (`Compressible::initialise`).
    pub fn store_initial(&mut self, gpu: &Gpu, fldk: &FieldKernels, s: &CompressibleState) -> Result<()> {
        for (wf, bf) in [(&mut self.p_wave, &s.p.bf), (&mut self.t_wave, &s.t.bf)] {
            if let Some(wf) = wf {
                let n = s.p.n_boundary_faces;
                field_ops::copy_field(gpu, fldk, &mut wf.b_old, bf, n)?;
                field_ops::copy_field(gpu, fldk, &mut wf.b_old_old, bf, n)?;
            }
        }
        Ok(())
    }

    /// `phi_b^oo <- phi_b^o`, `phi_b^o <- phi_b`, once at the top of every
    /// time step (§126.5: "refreshed once per step").
    pub fn store_old_time(&mut self, gpu: &Gpu, fldk: &FieldKernels, s: &CompressibleState) -> Result<()> {
        for (wf, bf) in [(&mut self.p_wave, &s.p.bf), (&mut self.t_wave, &s.t.bf)] {
            if let Some(wf) = wf {
                let n = s.p.n_boundary_faces;
                field_ops::copy_field(gpu, fldk, &mut wf.b_old_old, &wf.b_old, n)?;
                field_ops::copy_field(gpu, fldk, &mut wf.b_old, bf, n)?;
            }
        }
        Ok(())
    }

    /// One rewrite of every outlet's triple from the owner cells' `u` and
    /// `T`, then the face values of each field a rule touched: the switches
    /// (126.5), then `p`'s faces; with `time`, the `waveTransmissive` faces
    /// of `p` (126.6) with `w = u_n + c`, then of `T` with `w = u_n`, each
    /// by BC-09's Robin kernel and followed by that field's evaluation.
    /// Without `time` (the start of a run) only the switches run. The
    /// caller re-forms the boundary gas state.
    pub fn apply(
        &mut self,
        gpu: &Gpu,
        fldk: &FieldKernels,
        s: &mut CompressibleState,
        m: &GpuMesh,
        time: Option<WaveTime<'_>>,
    ) -> Result<()> {
        let mut any_switch = false;
        for r in self.rules.iter().filter(|r| r.kind == OutletKind::SupersonicSwitch && r.size > 0) {
            let (start, n, grs) = (r.start as Label, r.size as Label, self.grs);
            let f = self.switch.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut s.p.fr)
                    .arg(&mut s.p.ref_grad)
                    .arg(&s.u.f)
                    .arg(&s.t.f)
                    .arg(&m.b_face_cells)
                    .arg(&m.b_sf)
                    .arg(&m.b_mag_sf)
                    .arg(&start)
                    .arg(&n)
                    .arg(&grs)
                    .launch(cfg_for(r.size))?;
            }
            any_switch = true;
        }
        if any_switch {
            field_ops::correct_boundary_conditions(gpu, fldk, &mut s.p, m)?;
        }
        let Some(time) = time else {
            return Ok(());
        };
        let nbf = m.n_boundary_faces;
        let (local, rdt) = match time.r_delta_t {
            Some(r) => (1 as Label, r),
            None => (0 as Label, &m.v),
        };
        for field in [OutletField::P, OutletField::T] {
            let wf = match field {
                OutletField::P => &mut self.p_wave,
                OutletField::T => &mut self.t_wave,
            };
            let Some(wf) = wf else { continue };
            let acoustic: Label = if field == OutletField::P { 1 } else { 0 };
            for r in self.rules.iter().filter(|r| r.kind == OutletKind::WaveTransmissive && r.field == field && r.size > 0) {
                let (start, n, grs) = (r.start as Label, r.size as Label, self.grs);
                let sigma_by_l: Scalar = if r.l_inf > 0.0 { (r.sigma / r.l_inf) as Scalar } else { 0.0 };
                let f = self.speed.clone();
                unsafe {
                    gpu.stream()
                        .launch_builder(&f)
                        .arg(&mut wf.w)
                        .arg(&mut wf.k_relax)
                        .arg(&s.u.f)
                        .arg(&s.t.f)
                        .arg(&m.b_face_cells)
                        .arg(&m.b_sf)
                        .arg(&m.b_mag_sf)
                        .arg(rdt)
                        .arg(&local)
                        .arg(&acoustic)
                        .arg(&start)
                        .arg(&n)
                        .arg(&grs)
                        .arg(&sigma_by_l)
                        .launch(cfg_for(r.size))?;
                }
            }
            let fld = match field {
                OutletField::P => &mut s.p,
                OutletField::T => &mut s.t,
            };
            field_ops::advective_robin_scalar(
                gpu,
                fldk,
                &mut fld.fr,
                &mut fld.ref_value,
                &mut fld.ref_grad,
                &wf.mask,
                BcKind::Advective,
                &wf.w,
                &wf.k_relax,
                &wf.phi_inf,
                &wf.b_old,
                &wf.b_old_old,
                &m.b_delta_coeffs,
                time.c,
                nbf,
            )?;
            field_ops::correct_boundary_conditions(gpu, fldk, fld, m)?;
        }
        Ok(())
    }
}

/// The far-field kernel and the run's far-field rules (SPEC-LIT §126.10).
pub struct CompressibleFarfields {
    rules: Vec<FarfieldRule>,
    farfield: CudaFunction,
    /// Per rule: `fl(u_o)`, `fl(c_o)` and `fl(s_o)`, formed in f64.
    free: Vec<([Scalar; 3], Scalar, Scalar)>,
    /// `fl(gamma)`, `fl(gamma - 1)`, `fl(2/(gamma-1))`, `fl(1/(gamma-1))`,
    /// `fl(R_s)`, `fl(gamma R_s)`, `fl(p_op)`.
    gas: [Scalar; 7],
}

impl CompressibleFarfields {
    /// Load the kernel for `rules` on `m`; a rule whose faces leave the
    /// boundary arrays is refused.
    pub fn new(gpu: &Gpu, m: &GpuMesh, gas: &PerfectGas, rules: Vec<FarfieldRule>) -> Result<Self> {
        let nbf = m.n_boundary_faces;
        for r in &rules {
            if r.start + r.size > nbf {
                return Err(Error::Refused(format!(
                    "compressible::bc: far-field patch {} faces {}..{} leave the {nbf} boundary faces",
                    r.patch,
                    r.start,
                    r.start + r.size
                )));
            }
        }
        let (gamma, rs, pop) = (gas.gamma, gas.r_s(), gas.p_op);
        let free = rules
            .iter()
            .map(|r| {
                let pa = r.p + pop;
                let so = pa / (pa / (rs * r.t)).powf(gamma);
                (r.u.map(|v| v as Scalar), (gamma * rs * r.t).sqrt() as Scalar, so as Scalar)
            })
            .collect();
        let k = KernelSet::new(gpu, crate::kernels::COMPRESSIBLE)?;
        Ok(Self {
            farfield: k.func("cmpBcFarfield")?,
            free,
            gas: [gamma, gamma - 1.0, 2.0 / (gamma - 1.0), 1.0 / (gamma - 1.0), rs, gamma * rs, pop].map(|v| v as Scalar),
            rules,
        })
    }

    pub fn rules(&self) -> &[FarfieldRule] {
        &self.rules
    }

    /// One rewrite of every far-field face's three triples from the owner
    /// cells' `u`, `p` and `T` (all read before any is written), then the
    /// face values of `U`, `p` and `T`. The caller re-forms the boundary
    /// gas state.
    pub fn apply(&self, gpu: &Gpu, fldk: &FieldKernels, s: &mut CompressibleState, m: &GpuMesh) -> Result<()> {
        let [gamma, gm1, tg, ex, rs, grs, pop] = self.gas;
        let mut any = false;
        for (r, (uo, co, so)) in self.rules.iter().zip(&self.free).filter(|(r, _)| r.size > 0) {
            let (start, n) = (r.start as Label, r.size as Label);
            let f = self.farfield.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut s.u.fr)
                    .arg(&mut s.u.ref_value)
                    .arg(&mut s.u.ref_grad)
                    .arg(&mut s.p.fr)
                    .arg(&mut s.p.ref_value)
                    .arg(&mut s.p.ref_grad)
                    .arg(&mut s.t.fr)
                    .arg(&mut s.t.ref_value)
                    .arg(&mut s.t.ref_grad)
                    .arg(&s.u.f)
                    .arg(&s.p.f)
                    .arg(&s.t.f)
                    .arg(&m.b_face_cells)
                    .arg(&m.b_sf)
                    .arg(&m.b_mag_sf)
                    .arg(&start)
                    .arg(&n)
                    .arg(&uo[0])
                    .arg(&uo[1])
                    .arg(&uo[2])
                    .arg(co)
                    .arg(so)
                    .arg(&gamma)
                    .arg(&gm1)
                    .arg(&tg)
                    .arg(&ex)
                    .arg(&rs)
                    .arg(&grs)
                    .arg(&pop)
                    .launch(cfg_for(r.size))?;
            }
            any = true;
        }
        if any {
            field_ops::correct_boundary_conditions_vector(gpu, fldk, &mut s.u, m)?;
            field_ops::correct_boundary_conditions(gpu, fldk, &mut s.p, m)?;
            field_ops::correct_boundary_conditions(gpu, fldk, &mut s.t, m)?;
        }
        Ok(())
    }
}
