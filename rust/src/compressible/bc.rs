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
//! Written from:
//!   ofgpu `SPEC-LIT.md` §126 intro, §126.1, §126.3 - the equations and
//!     the inflow convention; §126.8 - this file as built, its contract
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

use crate::device::{cfg_for, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::field_ops::{self, FieldKernels};
use crate::io::fields::{PatchFieldSpec, RawScalarField, RawVectorField};
use crate::mesh::{GpuMesh, HostMesh};
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
    let mut out = Lowered { u: u.clone(), p: p.clone(), t: t.clone(), rules: Vec::new() };

    // Every entry, by its own key (patterns too): validate and replace.
    let lower_scalar = |field: &str, b: &mut BTreeMap<String, PatchFieldSpec>| -> Result<()> {
        for (key, spec) in b.iter_mut() {
            if let Some((_, v)) = scalar_kind(field, key, spec, gas)? {
                *spec = placeholder_scalar(v);
            }
        }
        Ok(())
    };
    lower_scalar("p", &mut out.p.boundary)?;
    lower_scalar("T", &mut out.t.boundary)?;
    for (key, spec) in out.u.boundary.iter_mut() {
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
