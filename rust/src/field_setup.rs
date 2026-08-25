// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Turning a parsed case field into a device field, and back again.
//!
//! Written from:
//!   SPEC-LIT.md section 4 (the single mixed-form boundary representation,
//!   which is our own design)
//!   SPEC-LIT.md section 13.4 - a setting the solver cannot honour must fail
//!     loudly; `-permissive` is the one escape hatch
//!   SPEC-LIT.md sections 15.2 and 15.5 - `nutLowRe` is `nu_t = 0`, and each
//!     field's own patch type decides whether it gets a wall treatment
//!   SPEC-LIT.md section 15.6 - the case's `C_mu` must reach the wall
//!     treatment as well as the model
//!   Launder & Spalding, *Comput. Methods Appl. Mech. Eng.* 3 (1974) 269-289
//!     - `k = 3/2 (I |U|)^2` and `epsilon = C_mu^{3/4} k^{3/2}/L`, the
//!     equilibrium relations the turbulence inlet conditions are named after
//!   Wilcox, *Turbulence Modeling for CFD* - `omega = k^{1/2}/(C_mu^{1/4} L)`,
//!     the same relation written for the specific dissipation rate
//! No GPL-licensed source was consulted.
//!
//! # What this file is for
//!
//! [`crate::io::fields`] gives a [`RawScalarField`] / [`RawVectorField`]: the
//! internal values plus, per patch, a *named* boundary condition with whatever
//! entries that name happens to carry. [`crate::field`] wants something quite
//! different: flat per-boundary-face arrays holding the Robin triple
//!
//! ```text
//! psi_b = fr*ref_value + (1 - fr)*(psi_P + ref_grad/Delta_b)
//! ```
//!
//! and nothing else. This module is the translation between the two, and it is
//! the *only* place in the solver where a boundary-condition name is looked at.
//! Once a field is on the device, every kernel sees the triple.
//!
//! # Why a named condition can always be reduced to the triple
//!
//! Because each named condition simply fixes where it sits on the Robin line:
//!
//! | Condition | `fr` | `ref_value` | `ref_grad` |
//! |---|---|---|---|
//! | `fixedValue` | 1 | the value | 0 |
//! | `zeroGradient` | 0 | 0 | 0 |
//! | `fixedGradient` | 0 | 0 | the gradient |
//! | `mixed` | as given | as given | as given |
//! | `inletOutlet` | recomputed each iteration from the face flux | inletValue | 0 |
//! | `calculated` | 1 | the value the file carried | 0 |
//! | `turbulentIntensityKineticEnergyInlet` | flux-switched | `3/2 (I \|U\|)^2` | 0 |
//! | `turbulentMixingLengthDissipationRateInlet` | flux-switched | `C_mu^{3/4} k^{3/2}/L` | 0 |
//! | `turbulentMixingLengthFrequencyInlet` | flux-switched | `k^{1/2}/(C_mu^{1/4} L)` | 0 |
//! | `pressureInletOutletVelocity` | flux-switched | `n (n.U)` | 0 |
//! | `fixedFluxPressure` | 0 | 0 | the gradient that carries the prescribed flux |
//! | `totalPressure` | 1 | `p0 - \|U\|^2/2` on inflow, `p0` on outflow | 0 |
//! | `movingWallVelocity` | 1 | the wall velocity, normal component removed | 0 |
//! | `flowRateInletVelocity` | 1 | `-n Q/A` | 0 |
//!
//! # When the derived conditions are evaluated
//!
//! *DESIGN.* The five conditions that read another field - the two turbulence
//! inlets, `totalPressure`, `pressureInletOutletVelocity` - are evaluated
//! **once, here, from the case's own initial fields**, and thereafter only
//! their value FRACTION is refreshed, from the face flux, by
//! [`update_inlet_outlet`] and its kernel twin. That is exact wherever the
//! condition is used in practice: an inlet at which `U` is a fixed value has
//! a `k`, an `epsilon` and an `omega` that do not change either. It is *not*
//! exact at an inlet whose velocity the solution itself sets, and this
//! paragraph, not a comment in a kernel, is where that is recorded. Making
//! them dynamic is a per-iteration kernel over the boundary and nothing in the
//! design here stands in its way.
//!
//! `symmetry`, `empty` and `cyclic` are not points on that line at all - they
//! are topology, and the kernels branch on [`BcKind`] for them. That is why
//! `bc_kind` is stored alongside the triple rather than being thrown away
//! after setup.
//!
//! # The two rules that are easy to get wrong
//!
//! 1. **The mesh wins on `empty`, `cyclic` and `symmetry`.** A field file may
//!    well say `fixedValue` on a patch the mesh declares `empty` - writing
//!    `zeroGradient` there is common too, and both are meaningless. Honour the
//!    topology; a Dirichlet condition on a 2-D front plane would pollute the
//!    solution with a boundary that does not physically exist.
//!
//! 2. **A patch missing from the field file is not an error.** It gets
//!    `Calculated` holding the internal value of its own cell, which is what a
//!    field written by a code that only lists non-default patches implies.
//!    Failing here would reject perfectly ordinary cases.

use std::collections::BTreeMap;

use crate::device::Gpu;
use crate::error::{Error, Result};
use crate::field::{BcKind, GpuScalarField, GpuVectorField};
use crate::io::fields::{PatchFieldSpec, RawScalarField, RawVectorField};
use crate::mesh::{GpuMesh, HostMesh, PatchKind};
use crate::{Label, Scalar, Vec3};

// ==========================================================================
//  Host-side staging
// ==========================================================================

/// The boundary triple for one field, on the host, before upload.
///
/// Built patch by patch and uploaded once. Keeping it as plain `Vec`s rather
/// than writing into device memory piecemeal means one transfer per array
/// instead of one per patch, and it makes the whole thing testable without a
/// GPU.
#[derive(Debug, Clone)]
pub struct BoundaryTriple<T> {
    pub fr: Vec<Scalar>,
    pub ref_value: Vec<T>,
    pub ref_grad: Vec<T>,
    pub kind: Vec<Label>,
    /// The evaluated face value, so a field can be written back out without a
    /// device round trip if nothing has run yet.
    pub value: Vec<T>,
}

impl<T: Copy + Default> BoundaryTriple<T> {
    fn new(n: usize) -> Self {
        Self {
            fr: vec![0.0; n],
            ref_value: vec![T::default(); n],
            ref_grad: vec![T::default(); n],
            kind: vec![BcKind::ZeroGradient as Label; n],
            value: vec![T::default(); n],
        }
    }
}

// ==========================================================================
//  What a derived condition needs from the rest of the case
// ==========================================================================

/// The other fields a boundary condition may be defined in terms of.
///
/// `turbulentIntensityKineticEnergyInlet` is `3/2 (I |U|)^2` and cannot be
/// built without `U`; the mixing-length inlets cannot be built without `k`.
/// Rather than have those conditions reach for a global, the caller says what
/// it has - and a condition that needs something absent produces an error
/// naming the field it wanted, which is the only honest answer.
///
/// All slices are indexed by **flattened boundary face**, the same indexing
/// `HostMesh::b_face_cells` uses.
#[derive(Default, Clone, Copy)]
pub struct BcInputs<'a> {
    /// Evaluated boundary velocity.
    pub u_b: Option<&'a [Vec3]>,
    /// Evaluated boundary `k`.
    pub k_b: Option<&'a [Scalar]>,
    /// Boundary flux, positive OUT of the domain. Decides inflow from outflow
    /// for `totalPressure`; the flux-switched conditions get their fraction
    /// from [`update_inlet_outlet`] instead.
    pub phi_b: Option<&'a [Scalar]>,
    /// `C_mu` **as the case set it**, not the built-in 0.09 - SPEC-LIT §15.6.
    /// `None` means the case did not override it.
    pub cmu: Option<Scalar>,
}

impl<'a> BcInputs<'a> {
    /// `C_mu`, defaulting to the Launder & Spalding value.
    fn cmu(&self) -> Scalar {
        self.cmu.unwrap_or(0.09)
    }
}

/// The per-face geometry and cross-field data one patch's conditions see.
struct PatchAux<'a> {
    /// `[n]` unit outward normals.
    normal: Vec<Vec3>,
    /// Total patch area, for `flowRateInletVelocity`.
    total_area: Scalar,
    /// Where this patch starts in the flattened boundary arrays.
    at: usize,
    inputs: BcInputs<'a>,
}

impl<'a> PatchAux<'a> {
    fn build(m: &HostMesh, p: &crate::mesh::PatchInfo, inputs: BcInputs<'a>) -> Self {
        let mut normal = Vec::with_capacity(p.size);
        let mut total_area: Scalar = 0.0;

        for i in 0..p.size {
            // `.get`, not indexing: a synthetic mesh in a test may carry no
            // face areas, and a condition that does not use them must still
            // build. The ones that DO use them report a zero-area patch.
            let sf = m.b_sf.get(p.start + i).copied().unwrap_or_default();
            let a = sf.dot(sf).sqrt();
            total_area += a;
            normal.push(if a > 0.0 { sf * (1.0 / a) } else { Vec3::ZERO });
        }

        Self {
            normal,
            total_area,
            at: p.start,
            inputs,
        }
    }

    fn u(&self, i: usize, patch: &str, wanted_by: &str) -> Result<Vec3> {
        match self.inputs.u_b {
            Some(u) => Ok(u.get(self.at + i).copied().unwrap_or_default()),
            None => Err(needs(patch, wanted_by, "U")),
        }
    }

    fn k(&self, i: usize, patch: &str, wanted_by: &str) -> Result<Scalar> {
        match self.inputs.k_b {
            Some(k) => Ok(k.get(self.at + i).copied().unwrap_or(0.0)),
            None => Err(needs(patch, wanted_by, "k")),
        }
    }

    /// Sf points OUT, so an inward flux is negative. With no flux loaded, the
    /// velocity's own normal component decides; with neither, outflow.
    fn is_inflow(&self, i: usize) -> bool {
        if let Some(phi) = self.inputs.phi_b {
            return phi.get(self.at + i).copied().unwrap_or(0.0) < 0.0;
        }
        if let Some(u) = self.inputs.u_b {
            let un = u.get(self.at + i).copied().unwrap_or_default();
            return un.dot(self.normal[i]) < 0.0;
        }
        false
    }
}

fn needs(patch: &str, condition: &str, field: &str) -> Error {
    Error::Field {
        field: patch.to_string(),
        msg: format!(
            "{condition} is defined in terms of `{field}`, and this setup was              given no `{field}` to evaluate it from"
        ),
    }
}

/// What the mesh insists on, regardless of what the field file says.
///
/// Returns `None` when the patch is an ordinary one and the field file's
/// opinion should stand.
fn topology_override(kind: PatchKind) -> Option<BcKind> {
    match kind {
        PatchKind::Empty => Some(BcKind::Empty),
        PatchKind::Cyclic | PatchKind::Processor => Some(BcKind::Cyclic),
        PatchKind::Symmetry => Some(BcKind::Symmetry),
        PatchKind::Wall | PatchKind::Generic => None,
    }
}

/// Expand a patch entry that may be uniform, empty, or per-face.
///
/// A uniform entry is written once in the file and applies to every face; an
/// absent one means the condition does not use it. Both are normal, so neither
/// is an error - but a list of the *wrong* non-zero length is a real mistake
/// in the case and is reported.
fn spread<T: Copy>(src: &[T], n: usize, fill: T, what: &str, patch: &str) -> Result<Vec<T>> {
    match src.len() {
        0 => Ok(vec![fill; n]),
        1 => Ok(vec![src[0]; n]),
        k if k == n => Ok(src.to_vec()),
        k => Err(Error::Field {
            field: patch.to_string(),
            msg: format!("{what} has {k} values but the patch has {n} faces"),
        }),
    }
}

// ==========================================================================
//  Scalar fields
// ==========================================================================

/// Reduce one patch's named condition to the Robin triple.
///
/// `internal` is the wall-adjacent cell value for each face of this patch,
/// used only as the fallback for `Calculated` when the file carried no
/// `value` entry.
#[allow(clippy::too_many_arguments)]
fn scalar_patch(
    spec: Option<&PatchFieldSpec>,
    n: usize,
    patch: &str,
    field: &str,
    mesh_kind: PatchKind,
    internal: &[Scalar],
    aux: &PatchAux,
    out: &mut BoundaryTriple<Scalar>,
    at: usize,
) -> Result<()> {
    // Topology first: a field file cannot argue with an empty patch.
    if let Some(forced) = topology_override(mesh_kind) {
        for i in 0..n {
            out.kind[at + i] = forced as Label;
            out.fr[at + i] = 0.0;
            out.ref_value[at + i] = 0.0;
            out.ref_grad[at + i] = 0.0;
            out.value[at + i] = internal[i];
        }
        return Ok(());
    }

    let Some(spec) = spec else {
        // Not named in the file. Hold the internal value.
        for i in 0..n {
            out.kind[at + i] = BcKind::Calculated as Label;
            out.fr[at + i] = 1.0;
            out.ref_value[at + i] = internal[i];
            out.ref_grad[at + i] = 0.0;
            out.value[at + i] = internal[i];
        }
        return Ok(());
    };

    let kind = BcKind::from_name(&spec.type_name, field, patch)?;

    let value = spread(&spec.value, n, 0.0, "value", patch)?;
    let grad = spread(&spec.gradient, n, 0.0, "gradient", patch)?;
    let rvalue = spread(&spec.ref_value, n, 0.0, "refValue", patch)?;
    let rgrad = spread(&spec.ref_gradient, n, 0.0, "refGradient", patch)?;
    let inlet = spread(&spec.inlet_value, n, 0.0, "inletValue", patch)?;
    let vfrac = spread(&spec.value_fraction, n, 1.0, "valueFraction", patch)?;

    // The entries only one condition each knows about. Read here, once per
    // patch, so that a missing or unreadable one is reported against the
    // condition that wanted it (SPEC-LIT 13.4) rather than defaulted.
    let intensity = match kind {
        BcKind::TurbulentIntensityKineticEnergyInlet => Some(spec.required_number(
            "turbulentIntensity",
            patch,
            "turbulentIntensityKineticEnergyInlet",
        )?),
        _ => None,
    };
    let mixing_length = match kind {
        BcKind::TurbulentMixingLengthDissipationRateInlet
        | BcKind::TurbulentMixingLengthFrequencyInlet => {
            let l = spec.required_number("mixingLength", patch, &spec.type_name)?;
            if !(l > 0.0) {
                return Err(Error::Field {
                    field: patch.to_string(),
                    msg: format!("{}: mixingLength must be positive, not {l}", spec.type_name),
                });
            }
            Some(l)
        }
        _ => None,
    };
    let p0 = match kind {
        BcKind::TotalPressure => Some(spec.required_number("p0", patch, "totalPressure")?),
        _ => None,
    };

    let cmu = aux.inputs.cmu();

    for i in 0..n {
        let (fr, rv, rg) = match kind {
            BcKind::FixedValue => (1.0, value[i], 0.0),
            BcKind::FixedGradient => (0.0, 0.0, grad[i]),
            BcKind::Mixed => (vfrac[i], rvalue[i], rgrad[i]),

            // The fraction is rewritten every iteration from the face flux
            // (inflow -> Dirichlet at inletValue, outflow -> zeroGradient).
            // Seeding it as inflow is the safe start: it holds a sensible
            // value on the first assembly, before any flux exists.
            BcKind::InletOutlet => (1.0, inlet[i], 0.0),

            // k = 3/2 (I |U|)^2 - the definition of the turbulence intensity
            // I as the ratio of the r.m.s. velocity fluctuation to the mean,
            // with the three components equal (Launder & Spalding 1974).
            // Flux-switched, exactly as inletOutlet is.
            BcKind::TurbulentIntensityKineticEnergyInlet => {
                let i_t = intensity.unwrap_or(0.0);
                let mag_u = aux
                    .u(i, patch, "turbulentIntensityKineticEnergyInlet")?
                    .mag();
                let iu = i_t * mag_u;
                (1.0, 1.5 * iu * iu, 0.0)
            }

            // epsilon = C_mu^{3/4} k^{3/2}/L: the equilibrium dissipation of
            // an eddy of size L carrying energy k (Launder & Spalding 1974).
            BcKind::TurbulentMixingLengthDissipationRateInlet => {
                let l = mixing_length.unwrap_or(1.0);
                let kb = aux
                    .k(i, patch, "turbulentMixingLengthDissipationRateInlet")?
                    .max(0.0);
                (1.0, cmu.powf(0.75) * kb * kb.sqrt() / l, 0.0)
            }

            // omega = epsilon/(C_mu k) applied to the same relation, i.e.
            // k^{1/2}/(C_mu^{1/4} L) (Wilcox).
            BcKind::TurbulentMixingLengthFrequencyInlet => {
                let l = mixing_length.unwrap_or(1.0);
                let kb = aux
                    .k(i, patch, "turbulentMixingLengthFrequencyInlet")?
                    .max(0.0);
                (1.0, kb.sqrt() / (cmu.powf(0.25) * l), 0.0)
            }

            // A fixed-gradient pressure: the surface-normal gradient is
            // whatever makes the boundary flux come out as prescribed. On a
            // wall with no body force that gradient is zero, which is the
            // `gradient` entry's default and the case this condition exists
            // for; a body-force case supplies the gradient explicitly.
            BcKind::FixedFluxPressure => (0.0, 0.0, grad[i]),

            // Kinematic total pressure: p0 = p + |U|^2/2 where the flow comes
            // in, and p = p0 where it leaves carrying its kinetic energy with
            // it. Bernoulli, evaluated from the initial velocity - see the
            // module header on when the derived conditions are evaluated.
            BcKind::TotalPressure => {
                let p_total = p0.unwrap_or(0.0);
                if aux.is_inflow(i) {
                    let mag = aux.u(i, patch, "totalPressure")?.mag();
                    (1.0, p_total - 0.5 * mag * mag, 0.0)
                } else {
                    (1.0, p_total, 0.0)
                }
            }

            // Written by a model, not solved for. Seed with whatever the file
            // carried so a restart is continuous.
            BcKind::Calculated => (
                1.0,
                if spec.value.is_empty() {
                    internal[i]
                } else {
                    value[i]
                },
                0.0,
            ),

            // Wall functions rewrite the triple themselves once y+ is known.
            // Until then they behave as the condition they degenerate to:
            // nut and k are zeroGradient-like, epsilon and omega are fixed at
            // the wall-adjacent cell, which the wall-function kernel sets.
            // The rough nut variants degenerate the same way - Ks/Cs only
            // matter once the wall-function kernel runs (SPEC-LIT 15.3/29.2).
            BcKind::NutkWallFunction
            | BcKind::NutUWallFunction
            | BcKind::NutkRoughWallFunction
            | BcKind::NutURoughWallFunction
            | BcKind::EpsilonWallFunction
            | BcKind::OmegaWallFunction => (
                1.0,
                if spec.value.is_empty() {
                    internal[i]
                } else {
                    value[i]
                },
                0.0,
            ),

            // SPEC-LIT 15.2: nu_t,w = 0, and that is the whole model. Pinned
            // here as well as in `turbNutBoundary`, so a nut field that never
            // reaches a turbulence model still reads zero at the wall.
            BcKind::NutLowReWallFunction => (1.0, 0.0, 0.0),

            BcKind::KqRWallFunction | BcKind::KLowReWallFunction | BcKind::ZeroGradient => {
                (0.0, 0.0, 0.0)
            }

            // SPEC-LIT 29.3: the Jayatilleke thermal wall function rewrites
            // the triple itself once u_tau and y+ are known - see
            // `EnergyWallData::update` in `src/energy.rs`. Until then it
            // behaves as a plain fixedValue at the wall temperature `T_w` the
            // file's `value` entry supplied, the same "degenerate until the
            // kernel runs" convention the block above uses.
            BcKind::ThermalWallFunction => (
                1.0,
                if spec.value.is_empty() {
                    internal[i]
                } else {
                    value[i]
                },
                0.0,
            ),

            // Vector-only conditions on a scalar field. Naming one is a
            // mistake in the case, not something to guess at.
            BcKind::PressureInletOutletVelocity
            | BcKind::MovingWallVelocity
            | BcKind::FlowRateInletVelocity => {
                return Err(Error::Field {
                    field: field.to_string(),
                    msg: format!(
                        "{patch}: `{}` is a vector condition and {field} is a scalar field",
                        spec.type_name
                    ),
                })
            }

            // Reached only if the mesh disagreed with the field file about
            // topology, which topology_override already handled above.
            BcKind::Empty | BcKind::Symmetry | BcKind::Cyclic => (0.0, 0.0, 0.0),
        };

        out.kind[at + i] = kind as Label;
        out.fr[at + i] = fr;
        out.ref_value[at + i] = rv;
        out.ref_grad[at + i] = rg;
        // psi_b at Delta -> the internal cell value, which is the best guess
        // available before the mesh's delta coefficients are applied.
        out.value[at + i] = fr * rv + (1.0 - fr) * internal[i];
    }

    Ok(())
}

/// Build the host-side boundary triple for a scalar field.
///
/// Every condition that is defined only in terms of the field itself works
/// through this; the ones defined in terms of `U` or `k` need
/// [`scalar_boundary_with`].
pub fn scalar_boundary(
    raw: &RawScalarField,
    m: &HostMesh,
    internal: &[Scalar],
) -> Result<BoundaryTriple<Scalar>> {
    scalar_boundary_with(raw, m, internal, BcInputs::default())
}

/// [`scalar_boundary`], told what else the case has loaded.
pub fn scalar_boundary_with(
    raw: &RawScalarField,
    m: &HostMesh,
    internal: &[Scalar],
    inputs: BcInputs,
) -> Result<BoundaryTriple<Scalar>> {
    let mut out = BoundaryTriple::new(m.n_boundary_faces);

    for p in &m.patches {
        // The wall-adjacent cell value for each face of this patch.
        let cells: Vec<Scalar> = (0..p.size)
            .map(|i| {
                let c = m.b_face_cells[p.start + i] as usize;
                internal.get(c).copied().unwrap_or(0.0)
            })
            .collect();

        let aux = PatchAux::build(m, p, inputs);

        scalar_patch(
            raw.spec(&p.name)?,
            p.size,
            &p.name,
            &raw.name,
            p.kind,
            &cells,
            &aux,
            &mut out,
            p.start,
        )?;
    }

    Ok(out)
}

/// Read a parsed scalar field onto the device.
pub fn upload_scalar(
    gpu: &Gpu,
    g: &GpuMesh,
    m: &HostMesh,
    raw: &RawScalarField,
) -> Result<GpuScalarField> {
    let internal = crate::io::fields::expand_scalars(&raw.internal, m.n_cells, &raw.name)?;
    let b = scalar_boundary(raw, m, &internal)?;

    Ok(GpuScalarField {
        name: raw.name.clone(),
        n_cells: g.n_cells,
        n_boundary_faces: g.n_boundary_faces,
        f: gpu.upload(&internal)?,
        f0: gpu.upload(&internal)?,
        // Both old levels start on the initial field; leaving psi^{n-2} at
        // zero would make the first BDF2 step differentiate a discontinuity
        // that never happened.
        f00: gpu.upload(&internal)?,
        bf: gpu.upload(&b.value)?,
        fr: gpu.upload(&b.fr)?,
        ref_value: gpu.upload(&b.ref_value)?,
        ref_grad: gpu.upload(&b.ref_grad)?,
        bc_kind: gpu.upload(&b.kind)?,
    })
}

/// Bring a device scalar field back into a writable case field.
///
/// The patch *types* come from `template`, not from `bc_kind`: the case file
/// says `nutkWallFunction`, and writing back the word `calculated` because
/// that is how the kernel treats it would quietly change the case. Only the
/// numbers are taken from the device.
pub fn download_scalar(
    gpu: &Gpu,
    g: &GpuMesh,
    f: &GpuScalarField,
    template: &RawScalarField,
) -> Result<RawScalarField> {
    let internal = gpu.download(&f.f)?;
    let bvals = gpu.download(&f.bf)?;

    // Resolved per patch, so a case written with a `".*"` pattern comes back
    // out as one explicit entry per patch rather than as a pattern nothing
    // filled in.
    let mut boundary: BTreeMap<String, PatchFieldSpec> = BTreeMap::new();
    for p in &g.patches {
        let mut spec = template
            .spec(&p.name)?
            .cloned()
            .unwrap_or_else(|| PatchFieldSpec {
                type_name: "calculated".to_string(),
                ..Default::default()
            });
        spec.value = bvals[p.start..p.start + p.size].to_vec();
        spec.value_v.clear();
        boundary.insert(p.name.clone(), spec);
    }

    Ok(RawScalarField {
        name: f.name.clone(),
        dimensions: template.dimensions.clone(),
        internal,
        boundary,
        boundary_patterns: Vec::new(),
    })
}

// ==========================================================================
//  Vector fields
// ==========================================================================

#[allow(clippy::too_many_arguments)]
fn vector_patch(
    spec: Option<&PatchFieldSpec>,
    n: usize,
    patch: &str,
    field: &str,
    mesh_kind: PatchKind,
    internal: &[Vec3],
    aux: &PatchAux,
    out: &mut BoundaryTriple<Vec3>,
    at: usize,
) -> Result<()> {
    let zero = Vec3::default();

    if let Some(forced) = topology_override(mesh_kind) {
        for i in 0..n {
            out.kind[at + i] = forced as Label;
            out.fr[at + i] = 0.0;
            out.ref_value[at + i] = zero;
            out.ref_grad[at + i] = zero;
            // For symmetry the normal component is reflected out by the
            // boundary-correction kernel, which needs the face normal; the
            // internal value is the right seed until it runs.
            out.value[at + i] = internal[i];
        }
        return Ok(());
    }

    let Some(spec) = spec else {
        for i in 0..n {
            out.kind[at + i] = BcKind::Calculated as Label;
            out.fr[at + i] = 1.0;
            out.ref_value[at + i] = internal[i];
            out.ref_grad[at + i] = zero;
            out.value[at + i] = internal[i];
        }
        return Ok(());
    };

    let kind = BcKind::from_name(&spec.type_name, field, patch)?;

    // `noSlip` carries no value entry - the whole point of the name is that
    // the value is zero - so it must not fall through to the internal value.
    let no_slip = spec.type_name == "noSlip";

    let value = spread(&spec.value_v, n, zero, "value", patch)?;
    let grad = spread(&spec.gradient_v, n, zero, "gradient", patch)?;
    let rvalue = spread(&spec.ref_value_v, n, zero, "refValue", patch)?;
    let rgrad = spread(&spec.ref_gradient_v, n, zero, "refGradient", patch)?;
    let inlet = spread(&spec.inlet_value_v, n, zero, "inletValue", patch)?;
    let vfrac = spread(&spec.value_fraction, n, 1.0, "valueFraction", patch)?;

    // `flowRateInletVelocity` names its flow rate one of two ways. Both are a
    // volume flux here, because the whole solver is incompressible and its
    // `rho` is 1: `massFlowRate` and `volumetricFlowRate` are then the same
    // number, and a case that means otherwise has to say so with a density
    // this solver does not carry.
    let flow_rate = match kind {
        BcKind::FlowRateInletVelocity => {
            let q = match spec.number("volumetricFlowRate", patch)? {
                Some(q) => q,
                None => spec.required_number(
                    "massFlowRate",
                    patch,
                    "flowRateInletVelocity (volumetricFlowRate or massFlowRate)",
                )?,
            };
            if !(aux.total_area > 0.0) {
                return Err(Error::Field {
                    field: field.to_string(),
                    msg: format!("flowRateInletVelocity on {patch}: the patch has zero area"),
                });
            }
            Some(q)
        }
        _ => None,
    };

    // A tangential wall velocity on `pressureInletOutletVelocity` is not
    // implemented, and SPEC-LIT 13.4 says so out loud rather than dropping it.
    if kind == BcKind::PressureInletOutletVelocity {
        if let Some(t) = spec.extra.get("tangentialVelocity") {
            if t.split_whitespace().any(|w| matches!(w.parse::<f64>(), Ok(v) if v != 0.0)) {
                return crate::io::contract::unsupported(
                    &format!("{field}: boundaryField/{patch}/tangentialVelocity"),
                    t,
                    &["(0 0 0)"],
                    "a zero tangential velocity",
                    (),
                )
                .map(|_| ());
            }
        }
    }

    for i in 0..n {
        let (fr, rv, rg) = match kind {
            BcKind::FixedValue if no_slip => (1.0, zero, zero),
            BcKind::FixedValue => (1.0, value[i], zero),
            BcKind::FixedGradient => (0.0, zero, grad[i]),
            BcKind::Mixed => (vfrac[i], rvalue[i], rgrad[i]),
            BcKind::InletOutlet => (1.0, inlet[i], zero),

            // Velocity at an open boundary whose pressure is prescribed: the
            // flux sets the velocity where the flow comes IN, and the interior
            // sets it where the flow goes out. Only the normal component is
            // determined - the tangential part of an inflow is not information
            // the boundary has - so the inflow value is n (n.U) taken from the
            // velocity the case supplied. Flux-switched, so the fraction is
            // refreshed each iteration by `update_inlet_outlet`.
            BcKind::PressureInletOutletVelocity => {
                let n_hat = aux.normal[i];
                let u = if spec.value_v.is_empty() {
                    internal[i]
                } else {
                    value[i]
                };
                (1.0, n_hat * u.dot(n_hat), zero)
            }

            // The wall's own velocity, with the wall-normal component removed:
            // a wall that is moving may drag the fluid along it, but it cannot
            // push fluid through itself. On a static mesh that is the entire
            // difference from `fixedValue`, and it is not a small one - a
            // `value` with a normal component would inject mass through the
            // wall. (A moving MESH would add its own normal flux here; ofgpu
            // has no mesh motion, so there is none to add.)
            BcKind::MovingWallVelocity => {
                let n_hat = aux.normal[i];
                let u = value[i];
                (1.0, u - n_hat * u.dot(n_hat), zero)
            }

            // U = -n Q/A: the uniform normal velocity that carries the
            // prescribed volumetric flow rate through the patch. Minus,
            // because Sf points out of the domain and an inlet flows in.
            BcKind::FlowRateInletVelocity => {
                let q = flow_rate.unwrap_or(0.0);
                (1.0, aux.normal[i] * (-q / aux.total_area), zero)
            }

            BcKind::Calculated => (
                1.0,
                if spec.value_v.is_empty() {
                    internal[i]
                } else {
                    value[i]
                },
                zero,
            ),

            // Scalar-only conditions on a vector field: the turbulence inlets,
            // the two pressure conditions, and every wall function. Naming one
            // here is a mistake in the case.
            BcKind::TurbulentIntensityKineticEnergyInlet
            | BcKind::TurbulentMixingLengthDissipationRateInlet
            | BcKind::TurbulentMixingLengthFrequencyInlet
            | BcKind::FixedFluxPressure
            | BcKind::TotalPressure
            | BcKind::NutkWallFunction
            | BcKind::NutUWallFunction
            | BcKind::NutkRoughWallFunction
            | BcKind::NutURoughWallFunction
            | BcKind::NutLowReWallFunction
            | BcKind::EpsilonWallFunction
            | BcKind::OmegaWallFunction
            | BcKind::KqRWallFunction
            | BcKind::KLowReWallFunction
            | BcKind::ThermalWallFunction => {
                return Err(Error::Field {
                    field: field.to_string(),
                    msg: format!(
                        "{patch}: `{}` is a scalar condition and {field} is a vector field",
                        spec.type_name
                    ),
                })
            }

            BcKind::ZeroGradient | BcKind::Empty | BcKind::Symmetry | BcKind::Cyclic => {
                (0.0, zero, zero)
            }
        };

        out.kind[at + i] = kind as Label;
        out.fr[at + i] = fr;
        out.ref_value[at + i] = rv;
        out.ref_grad[at + i] = rg;
        out.value[at + i] = rv * fr + internal[i] * (1.0 - fr);
    }

    Ok(())
}

pub fn vector_boundary(
    raw: &RawVectorField,
    m: &HostMesh,
    internal: &[Vec3],
) -> Result<BoundaryTriple<Vec3>> {
    vector_boundary_with(raw, m, internal, BcInputs::default())
}

/// [`vector_boundary`], told what else the case has loaded.
pub fn vector_boundary_with(
    raw: &RawVectorField,
    m: &HostMesh,
    internal: &[Vec3],
    inputs: BcInputs,
) -> Result<BoundaryTriple<Vec3>> {
    let mut out = BoundaryTriple::new(m.n_boundary_faces);

    for p in &m.patches {
        let cells: Vec<Vec3> = (0..p.size)
            .map(|i| {
                let c = m.b_face_cells[p.start + i] as usize;
                internal.get(c).copied().unwrap_or_default()
            })
            .collect();

        let aux = PatchAux::build(m, p, inputs);

        vector_patch(
            raw.spec(&p.name)?,
            p.size,
            &p.name,
            &raw.name,
            p.kind,
            &cells,
            &aux,
            &mut out,
            p.start,
        )?;
    }

    Ok(out)
}

pub fn upload_vector(
    gpu: &Gpu,
    g: &GpuMesh,
    m: &HostMesh,
    raw: &RawVectorField,
) -> Result<GpuVectorField> {
    let internal = crate::io::fields::expand_vectors(&raw.internal, m.n_cells, &raw.name)?;
    let b = vector_boundary(raw, m, &internal)?;

    Ok(GpuVectorField {
        name: raw.name.clone(),
        n_cells: g.n_cells,
        n_boundary_faces: g.n_boundary_faces,
        f: gpu.upload(&internal)?,
        f0: gpu.upload(&internal)?,
        // Both old levels start on the initial field; leaving psi^{n-2} at
        // zero would make the first BDF2 step differentiate a discontinuity
        // that never happened.
        f00: gpu.upload(&internal)?,
        bf: gpu.upload(&b.value)?,
        fr: gpu.upload(&b.fr)?,
        ref_value: gpu.upload(&b.ref_value)?,
        ref_grad: gpu.upload(&b.ref_grad)?,
        bc_kind: gpu.upload(&b.kind)?,
    })
}

pub fn download_vector(
    gpu: &Gpu,
    g: &GpuMesh,
    f: &GpuVectorField,
    template: &RawVectorField,
) -> Result<RawVectorField> {
    let internal = gpu.download(&f.f)?;
    let bvals = gpu.download(&f.bf)?;

    let mut boundary: BTreeMap<String, PatchFieldSpec> = BTreeMap::new();
    for p in &g.patches {
        let mut spec = template
            .spec(&p.name)?
            .cloned()
            .unwrap_or_else(|| PatchFieldSpec {
                type_name: "calculated".to_string(),
                ..Default::default()
            });
        spec.value_v = bvals[p.start..p.start + p.size].to_vec();
        spec.value.clear();
        boundary.insert(p.name.clone(), spec);
    }

    Ok(RawVectorField {
        name: f.name.clone(),
        dimensions: template.dimensions.clone(),
        internal,
        boundary,
        boundary_patterns: Vec::new(),
    })
}

// ==========================================================================
//  In-place setup, used by the drivers
// ==========================================================================
//
// The drivers construct a field with `GpuScalarField::zeros`, hand it to a
// model that wants to own it, and only then have the case file to fill it
// from. So these fill an EXISTING field rather than returning a new one -
// the same translation as `upload_scalar`, written into buffers that already
// exist. No reallocation, and a model keeps the field it was given.

/// Fill an existing device scalar field from a parsed case field.
pub fn setup_scalar_field(
    gpu: &Gpu,
    f: &mut GpuScalarField,
    raw: &RawScalarField,
    m: &HostMesh,
) -> Result<()> {
    let internal = crate::io::fields::expand_scalars(&raw.internal, m.n_cells, &raw.name)?;
    let b = scalar_boundary(raw, m, &internal)?;

    gpu.write(&mut f.f, &internal)?;
    gpu.write(&mut f.f0, &internal)?;
    gpu.write(&mut f.f00, &internal)?;
    gpu.write(&mut f.bf, &b.value)?;
    gpu.write(&mut f.fr, &b.fr)?;
    gpu.write(&mut f.ref_value, &b.ref_value)?;
    gpu.write(&mut f.ref_grad, &b.ref_grad)?;
    gpu.write(&mut f.bc_kind, &b.kind)?;
    Ok(())
}

/// [`setup_scalar_field`], told what else the case has loaded, for the
/// conditions defined in terms of another field.
pub fn setup_scalar_field_with(
    gpu: &Gpu,
    f: &mut GpuScalarField,
    raw: &RawScalarField,
    m: &HostMesh,
    inputs: BcInputs,
) -> Result<()> {
    let internal = crate::io::fields::expand_scalars(&raw.internal, m.n_cells, &raw.name)?;
    let b = scalar_boundary_with(raw, m, &internal, inputs)?;

    gpu.write(&mut f.f, &internal)?;
    gpu.write(&mut f.f0, &internal)?;
    gpu.write(&mut f.f00, &internal)?;
    gpu.write(&mut f.bf, &b.value)?;
    gpu.write(&mut f.fr, &b.fr)?;
    gpu.write(&mut f.ref_value, &b.ref_value)?;
    gpu.write(&mut f.ref_grad, &b.ref_grad)?;
    gpu.write(&mut f.bc_kind, &b.kind)?;
    Ok(())
}

/// [`setup_vector_field`], told what else the case has loaded.
pub fn setup_vector_field_with(
    gpu: &Gpu,
    f: &mut GpuVectorField,
    raw: &RawVectorField,
    m: &HostMesh,
    inputs: BcInputs,
) -> Result<()> {
    let internal = crate::io::fields::expand_vectors(&raw.internal, m.n_cells, &raw.name)?;
    let b = vector_boundary_with(raw, m, &internal, inputs)?;

    gpu.write(&mut f.f, &internal)?;
    gpu.write(&mut f.f0, &internal)?;
    gpu.write(&mut f.f00, &internal)?;
    gpu.write(&mut f.bf, &b.value)?;
    gpu.write(&mut f.fr, &b.fr)?;
    gpu.write(&mut f.ref_value, &b.ref_value)?;
    gpu.write(&mut f.ref_grad, &b.ref_grad)?;
    gpu.write(&mut f.bc_kind, &b.kind)?;
    Ok(())
}

/// Fill an existing device vector field from a parsed case field.
pub fn setup_vector_field(
    gpu: &Gpu,
    f: &mut GpuVectorField,
    raw: &RawVectorField,
    m: &HostMesh,
) -> Result<()> {
    let internal = crate::io::fields::expand_vectors(&raw.internal, m.n_cells, &raw.name)?;
    let b = vector_boundary(raw, m, &internal)?;

    gpu.write(&mut f.f, &internal)?;
    gpu.write(&mut f.f0, &internal)?;
    gpu.write(&mut f.f00, &internal)?;
    gpu.write(&mut f.bf, &b.value)?;
    gpu.write(&mut f.fr, &b.fr)?;
    gpu.write(&mut f.ref_value, &b.ref_value)?;
    gpu.write(&mut f.ref_grad, &b.ref_grad)?;
    gpu.write(&mut f.bc_kind, &b.kind)?;
    Ok(())
}

/// Copy a device scalar field's numbers back into a case field for writing.
///
/// Only the NUMBERS. `out` keeps its own patch types, because those came from
/// the case the run started with: a patch written as `nutkWallFunction` must
/// still say `nutkWallFunction` afterwards, even though the solver treats it
/// as a value it computes. Writing back the solver's view of a patch would
/// quietly rewrite the case on every output.
pub fn harvest_scalar_field(
    gpu: &Gpu,
    out: &mut RawScalarField,
    f: &GpuScalarField,
    m: &HostMesh,
) -> Result<()> {
    out.internal = gpu.download(&f.f)?;
    let bvals = gpu.download(&f.bf)?;

    // A pattern key is expanded into one entry per patch it governs, so the
    // written file names every patch explicitly and can be read back as the
    // start time of another run.
    let expanded = expand_patterns(&out.boundary, &out.boundary_patterns, m)?;
    if let Some(b) = expanded {
        out.boundary = b;
        out.boundary_patterns.clear();
    }

    for p in &m.patches {
        if let Some(spec) = out.boundary.get_mut(&p.name) {
            spec.value = bvals[p.start..p.start + p.size].to_vec();
        }
    }
    Ok(())
}

/// [`harvest_scalar_field`] for a vector field.
pub fn harvest_vector_field(
    gpu: &Gpu,
    out: &mut RawVectorField,
    f: &GpuVectorField,
    m: &HostMesh,
) -> Result<()> {
    out.internal = gpu.download(&f.f)?;
    let bvals = gpu.download(&f.bf)?;

    let expanded = expand_patterns(&out.boundary, &out.boundary_patterns, m)?;
    if let Some(b) = expanded {
        out.boundary = b;
        out.boundary_patterns.clear();
    }

    for p in &m.patches {
        if let Some(spec) = out.boundary.get_mut(&p.name) {
            spec.value_v = bvals[p.start..p.start + p.size].to_vec();
        }
    }
    Ok(())
}

/// One flag per boundary face, computed from `raw`'s OWN patch types by
/// `test`.
///
/// **SPEC-LIT §15.5, and a correctness requirement.** The decision must come
/// from each field's own patch type, never from another field's. Deriving
/// `nut`'s treatment from `epsilon`'s produces two opposite silent failures: a
/// `fixedValue 0` on `nut` (the correct low-Re setup) overwritten by a wall
/// function because `epsilon` asked for one, and a `nutkWallFunction` left
/// inert because `epsilon` did not. Both give a plausible field and a wrong
/// wall shear.
///
/// A patch the field file does not name gets `false`.
pub fn faces_where(
    raw: &RawScalarField,
    m: &HostMesh,
    test: fn(BcKind) -> bool,
) -> Result<Vec<bool>> {
    let mut flags = vec![false; m.n_boundary_faces];

    for p in &m.patches {
        let Some(spec) = raw.spec(&p.name)? else {
            continue;
        };
        if !test(BcKind::from_name(&spec.type_name, &raw.name, &p.name)?) {
            continue;
        }
        for i in 0..p.size {
            flags[p.start + i] = true;
        }
    }

    Ok(flags)
}

/// Which faces constrain their wall-adjacent CELL, from `epsilon`'s or
/// `omega`'s own patch types (SPEC-LIT §15.5).
pub fn wall_cell_faces(raw: &RawScalarField, m: &HostMesh) -> Result<Vec<bool>> {
    faces_where(raw, m, BcKind::constrains_wall_cell)
}

/// Which faces get a wall value for `nu_t`, from `nut`'s own patch types.
///
/// `nutLowReWallFunction` is deliberately NOT one of them - SPEC-LIT §15.2.
pub fn nut_wall_faces(raw: &RawScalarField, m: &HostMesh) -> Result<Vec<bool>> {
    faces_where(raw, m, BcKind::is_nut_wall_function)
}

/// The old single-flag entry point, kept for callers that have only the
/// dissipation field to hand.
///
/// It answers the `epsilon`/`omega` question - which wall CELLS are
/// constrained - and nothing else. Anything asking about `nu_t` must use
/// [`nut_wall_faces`] and `nut`'s own file.
pub fn wall_function_faces(raw: &RawScalarField, m: &HostMesh) -> Result<Vec<bool>> {
    wall_cell_faces(raw, m)
}

/// Expand every pattern key into one entry per patch it governs.
///
/// `None` when there is nothing to expand, so the common case allocates
/// nothing.
fn expand_patterns(
    boundary: &BTreeMap<String, PatchFieldSpec>,
    patterns: &[String],
    m: &HostMesh,
) -> Result<Option<BTreeMap<String, PatchFieldSpec>>> {
    if patterns.is_empty() {
        return Ok(None);
    }

    let mut out: BTreeMap<String, PatchFieldSpec> = BTreeMap::new();
    for p in &m.patches {
        if let Some(key) = crate::io::fields::governing_key(boundary, patterns, &p.name)? {
            if let Some(spec) = boundary.get(&key) {
                out.insert(p.name.clone(), spec.clone());
            }
        }
    }

    Ok(Some(out))
}

/// Which faces get which wall treatment - **one answer per field, from that
/// field's own patch types** (SPEC-LIT §15.5).
///
/// The two sets are genuinely different and deriving one from the other is a
/// physics bug in both directions:
///
/// * `nut = fixedValue 0` (or `nutLowReWallFunction`) with
///   `epsilon = epsilonWallFunction` is the correct low-Re setup with a
///   constrained wall cell. Deriving `nut`'s treatment from `epsilon`'s puts a
///   wall function on `nu_t` that the case explicitly refused, and
///   overpredicts the wall shear stress.
/// * `nut = nutkWallFunction` with `epsilon = fixedValue` is a perfectly legal
///   case. Deriving `nut`'s treatment from `epsilon`'s leaves the wall
///   function inert and underpredicts it.
///
/// Both give a plausible field, which is why neither is noticed.
#[derive(Debug, Clone, Default)]
pub struct WallFaces {
    /// From `epsilon`'s or `omega`'s own patch types: the faces whose
    /// wall-adjacent CELL is pinned to the near-wall relation.
    pub constrained_cells: Vec<bool>,
    /// From `nut`'s own patch types: the faces that get a wall value for
    /// `nu_t`. `nutLowReWallFunction` is not one of them - SPEC-LIT §15.2.
    pub nut: Vec<bool>,
}

impl WallFaces {
    /// Read both sets from the case's own field files.
    ///
    /// `nut` is optional because a case may not carry one. When it is absent
    /// there is no `nut` patch type to ask, so the dissipation field's answer
    /// is used - and a line is printed saying so, because that IS the
    /// derivation this type exists to prevent and it should not happen
    /// quietly.
    pub fn from_case(
        dissipation: &RawScalarField,
        nut: Option<&RawScalarField>,
        m: &HostMesh,
    ) -> Result<Self> {
        let constrained_cells = wall_cell_faces(dissipation, m)?;

        let nut = match nut {
            Some(raw) => nut_wall_faces(raw, m)?,
            None => {
                if constrained_cells.iter().any(|b| *b) {
                    crate::io::contract::warn_once(
                        "nut: no field file",
                        &format!(
                            "no `nut` field in the start time, so nu_t's wall treatment is \
                             being taken from `{}`'s patch types. Add a `nut` file to say \
                             what the wall should do (SPEC-LIT 15.5).",
                            dissipation.name
                        ),
                    );
                }
                constrained_cells.clone()
            }
        };

        Ok(Self {
            constrained_cells,
            nut,
        })
    }

    /// No wall functions anywhere - what a passive-scalar transport equation
    /// wants.
    pub fn none(n_boundary_faces: usize) -> Self {
        Self {
            constrained_cells: vec![false; n_boundary_faces],
            nut: vec![false; n_boundary_faces],
        }
    }
}

/// Per-face roughness for `nut`'s wall function - SPEC-LIT §15.3/§29.2.
///
/// Read from `nut`'s OWN patch entries, the same file [`WallFaces::nut`]
/// reads its flag from - never from another field's, for the SPEC-LIT §15.5
/// reason [`WallFaces`] documents. Every vector has `n_boundary_faces`
/// entries; a face that is not a rough wall function carries `Ks = 0`, which
/// is what makes `E_eff = E` there and the rough kernels collapse to the
/// smooth ones to round-off (the §22 gate).
#[derive(Debug, Clone)]
pub struct NutRoughness {
    /// True where the face's `nu_t` comes from the velocity-based `nutU`
    /// family rather than `nutk`'s `k`-based one (SPEC-LIT §15.1). Only
    /// meaningful where [`WallFaces::nut`] is also set for that face.
    pub u_based: Vec<bool>,
    /// Sand-grain height, m. Zero on every smooth face.
    pub ks: Vec<Scalar>,
    /// The roughness constant. Meaningful only where `ks > 0`; `0.5` is the
    /// SPEC-LIT §15.3 default for uniform sand, filled in here so a rough
    /// patch that omits `Cs` still gets it rather than a stray zero.
    pub cs: Vec<Scalar>,
}

impl NutRoughness {
    /// Every face smooth: `nu_t`'s own patch type still decides whether it
    /// gets a wall value at all, through [`WallFaces::nut`]; this only ever
    /// widens what a face already asked for.
    pub fn none(n_boundary_faces: usize) -> Self {
        Self {
            u_based: vec![false; n_boundary_faces],
            ks: vec![0.0; n_boundary_faces],
            cs: vec![0.5; n_boundary_faces],
        }
    }

    /// Read from `nut`'s own patch types, or [`Self::none`] when the case has
    /// no `nut` file - [`WallFaces::from_case`] already warns about that
    /// absence once, on the shared cause, so this does not warn again.
    pub fn from_case(nut: Option<&RawScalarField>, m: &HostMesh) -> Result<Self> {
        let Some(raw) = nut else {
            return Ok(Self::none(m.n_boundary_faces));
        };

        let mut u_based = vec![false; m.n_boundary_faces];
        let mut ks = vec![0.0 as Scalar; m.n_boundary_faces];
        let mut cs = vec![0.5 as Scalar; m.n_boundary_faces];

        for p in &m.patches {
            let Some(spec) = raw.spec(&p.name)? else {
                continue;
            };
            let kind = BcKind::from_name(&spec.type_name, &raw.name, &p.name)?;
            if !kind.is_nut_wall_function() {
                continue;
            }

            let u_flag = kind.is_nut_velocity_based();
            // A rough type without `Ks` is the §13.4 error its name promised:
            // discarding the roughness would make it a smooth wall wearing a
            // rough wall's name (SPEC-LIT §15.3).
            let (k_val, c_val) = if kind.is_nut_rough() {
                (
                    spec.required_number("Ks", &p.name, &spec.type_name)?,
                    spec.number("Cs", &p.name)?.unwrap_or(0.5),
                )
            } else {
                (0.0, 0.5)
            };

            for i in 0..p.size {
                u_based[p.start + i] = u_flag;
                ks[p.start + i] = k_val;
                cs[p.start + i] = c_val;
            }
        }

        Ok(Self { u_based, ks, cs })
    }
}

// ==========================================================================
//  SPEC-LIT §29.1: wallTreatment preset expansion (route a, OpenFOAM)
// ==========================================================================

/// SPEC-LIT §29.1 route (a): fill in the `wallTreatment` DEFAULT on every
/// wall patch `raw` left out entirely - "explicit per-field patch types
/// override per patch" means a patch [`RawScalarField::spec`] already finds
/// (exact key or matching pattern) is untouched, since that is by
/// construction the more specific of the two. Only `nut`/`k`/`epsilon`/
/// `omega`/`T` have a preset row; `field` naming anything else is a no-op.
///
/// The synthesised entry carries no `value` - every wall-function [`BcKind`]
/// this can produce already falls back to the wall cell's own internal value
/// when `value` is empty (see [`scalar_patch`]), the same default an
/// explicit entry with no `value` would get.
pub fn apply_wall_treatment_defaults(
    raw: &mut RawScalarField,
    field: &str,
    m: &HostMesh,
    treatment: crate::io::case::WallTreatment,
    roughness: Option<crate::io::case::Roughness>,
) -> Result<()> {
    use crate::io::case::WallTreatment;

    let type_name = match field {
        "nut" => treatment.nut_type(),
        "k" => treatment.k_type(),
        "epsilon" => treatment.epsilon_type(),
        "omega" => treatment.omega_type(),
        "T" => match treatment.thermal_type() {
            Some(t) => t,
            // SPEC-LIT §29.3: `lowRe` pins the molecular resistance the
            // resolved mesh already provides - nothing to default here.
            None => return Ok(()),
        },
        _ => return Ok(()),
    };

    for p in &m.patches {
        if p.kind != PatchKind::Wall {
            continue;
        }
        if raw.spec(&p.name)?.is_some() {
            continue;
        }

        let mut s = PatchFieldSpec {
            type_name: type_name.to_string(),
            ..Default::default()
        };
        if field == "nut" && treatment == WallTreatment::Rough {
            if let Some(r) = roughness {
                s.extra.insert("Ks".to_string(), r.ks.to_string());
                s.extra.insert("Cs".to_string(), r.cs.to_string());
            }
        }
        raw.boundary.insert(p.name.clone(), s);
    }

    Ok(())
}

// ==========================================================================
//  SPEC-LIT §29.1 point 3: the wall-row consistency contract
// ==========================================================================

/// Whether a field's OWN patch type declares "the wall model constrains this
/// cell" (`Full`) or "the mesh resolves the sublayer, no wall model"
/// (`LowRe`). There is no third variant here - `None` (see the `family_of`
/// functions below) is not a third family, it is "this field has no opinion"
/// (a plain `fixedValue`, an inlet condition reused unchanged on a wall...),
/// and SPEC-LIT §15.5 is explicit that the fields are independent: a field
/// with no opinion never conflicts with one that has one. This check only
/// catches two fields that DO both have an opinion and disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WallRowFamily {
    Full,
    LowRe,
}

fn nut_family(k: BcKind) -> Option<WallRowFamily> {
    if k.is_nut_wall_function() {
        Some(WallRowFamily::Full)
    } else if k.is_nut_low_re() {
        Some(WallRowFamily::LowRe)
    } else {
        None
    }
}

fn k_family(k: BcKind) -> Option<WallRowFamily> {
    match k {
        BcKind::KqRWallFunction => Some(WallRowFamily::Full),
        BcKind::KLowReWallFunction => Some(WallRowFamily::LowRe),
        _ => None,
    }
}

/// `epsilon`'s and `omega`'s own patch types can only ever DECLARE `Full`
/// (`epsilonWallFunction`/`omegaWallFunction` - SPEC-LIT §6.4); there is no
/// distinct low-Re-named condition for either. SPEC-LIT §29.1's `lowRe` row
/// completes them with plain `zeroGradient`, which reads as "no opinion"
/// here, not a second family - exactly the completion the table asks for.
fn eps_omega_family(k: BcKind) -> Option<WallRowFamily> {
    if k.constrains_wall_cell() {
        Some(WallRowFamily::Full)
    } else {
        None
    }
}

/// A human name for the `BcKind`s the wall-row check talks about, for its
/// diagnostics - and for writing a `-permissive` substitution back into a
/// [`PatchFieldSpec::type_name`].
pub fn bc_kind_name(k: BcKind) -> &'static str {
    match k {
        BcKind::NutkWallFunction => "nutkWallFunction",
        BcKind::NutUWallFunction => "nutUWallFunction",
        BcKind::NutkRoughWallFunction => "nutkRoughWallFunction",
        BcKind::NutURoughWallFunction => "nutURoughWallFunction",
        BcKind::NutLowReWallFunction => "nutLowReWallFunction",
        BcKind::KqRWallFunction => "kqRWallFunction",
        BcKind::KLowReWallFunction => "kLowReWallFunction",
        BcKind::EpsilonWallFunction => "epsilonWallFunction",
        BcKind::OmegaWallFunction => "omegaWallFunction",
        BcKind::ZeroGradient => "zeroGradient",
        BcKind::FixedValue => "fixedValue",
        BcKind::Calculated => "calculated",
        _ => "(other)",
    }
}

/// The type name `field` would need for `family` - SPEC-LIT §29.1's table,
/// restricted to the fields this check covers.
fn completion_for(field: &str, family: WallRowFamily) -> &'static str {
    match (field, family) {
        ("nut", WallRowFamily::Full) => "nutkWallFunction",
        ("nut", WallRowFamily::LowRe) => "nutLowReWallFunction",
        ("k", WallRowFamily::Full) => "kqRWallFunction",
        ("k", WallRowFamily::LowRe) => "kLowReWallFunction",
        ("epsilon", WallRowFamily::Full) => "epsilonWallFunction",
        ("epsilon", WallRowFamily::LowRe) => "zeroGradient",
        ("omega", WallRowFamily::Full) => "omegaWallFunction",
        ("omega", WallRowFamily::LowRe) => "zeroGradient",
        _ => "zeroGradient",
    }
}

fn family_kind(field: &str, family: WallRowFamily) -> BcKind {
    match (field, family) {
        ("nut", WallRowFamily::Full) => BcKind::NutkWallFunction,
        ("nut", WallRowFamily::LowRe) => BcKind::NutLowReWallFunction,
        ("k", WallRowFamily::Full) => BcKind::KqRWallFunction,
        ("k", WallRowFamily::LowRe) => BcKind::KLowReWallFunction,
        ("epsilon", WallRowFamily::Full) => BcKind::EpsilonWallFunction,
        ("epsilon", WallRowFamily::LowRe) => BcKind::ZeroGradient,
        ("omega", WallRowFamily::Full) => BcKind::OmegaWallFunction,
        _ => BcKind::ZeroGradient,
    }
}

/// One wall patch's four turbulence-field patch types, as resolved from
/// whichever route selected them (an explicit per-field entry, a per-patch
/// `treatment` override, or a case-level `wallTreatment` preset). `None`
/// where the case does not solve that field at all - `omega` under
/// k-epsilon, say; a model that solves neither has nothing for this check to
/// disagree about.
#[derive(Debug, Clone, Copy, Default)]
pub struct WallRow {
    pub nut: Option<BcKind>,
    pub k: Option<BcKind>,
    pub epsilon: Option<BcKind>,
    pub omega: Option<BcKind>,
}

impl WallRow {
    fn entries(self) -> [(&'static str, Option<BcKind>, fn(BcKind) -> Option<WallRowFamily>); 4] {
        [
            ("nut", self.nut, nut_family as fn(BcKind) -> Option<WallRowFamily>),
            ("k", self.k, k_family),
            ("epsilon", self.epsilon, eps_omega_family),
            ("omega", self.omega, eps_omega_family),
        ]
    }
}

/// SPEC-LIT §29.1 point 3, the consistency contract: whatever route selected
/// them, the four per-field types on one wall patch must belong to one row of
/// the §29.1 table (`nut` may be any of `nutk`/`nutU`/`rough` in the shared
/// "full" row). A mixed row is a §13.4 error naming the patch, the offending
/// pair and the consistent completions; `-permissive` substitutes the row
/// implied by the field that spoke first, in `nut`/`k`/`epsilon`/`omega`
/// order (`nut`'s own choice, in the common case where `nut` has one), and
/// prints the substitution.
///
/// Returns the row unchanged where it was already consistent (the common
/// case, since every preset-expanded row is consistent by construction); the
/// corrected row otherwise, under `-permissive`.
/// [`validate_wall_row`] for the OpenFOAM-directory route.
///
/// The JSONC path runs the SPEC-LIT §29.1 consistency contract during
/// lowering; a case read from `0/` field files must meet the same contract,
/// or the contradiction the contract exists for (`nutLowReWallFunction` on
/// `nut` with `epsilonWallFunction` on `epsilon`) runs silently - which is
/// exactly how it shipped the first time, caught by a supervisor functional
/// check rather than by the test suite. Builds each WALL patch's row from
/// the raw fields' own patch types, validates it, and writes a
/// `-permissive` correction back into the raw specs so the corrected type
/// is what the solver actually assembles.
///
/// Fields the case does not carry are `None` and take no part - a k-epsilon
/// case has nothing to say about `omega`.
pub fn validate_wall_rows(
    patches: &[crate::mesh::PatchInfo],
    mut nut: Option<&mut RawScalarField>,
    mut k: Option<&mut RawScalarField>,
    mut epsilon: Option<&mut RawScalarField>,
    mut omega: Option<&mut RawScalarField>,
) -> Result<()> {
    use crate::mesh::PatchKind;

    for p in patches {
        if p.kind != PatchKind::Wall {
            continue;
        }
        let pname = p.name.as_str();

        let kind_of = |f: &Option<&mut RawScalarField>, field: &str| -> Result<Option<BcKind>> {
            match f {
                Some(rf) => match rf.boundary.get(pname) {
                    Some(spec) => Ok(Some(BcKind::from_name(&spec.type_name, field, pname)?)),
                    None => Ok(None),
                },
                None => Ok(None),
            }
        };

        let row = WallRow {
            nut: kind_of(&nut, "nut")?,
            k: kind_of(&k, "k")?,
            epsilon: kind_of(&epsilon, "epsilon")?,
            omega: kind_of(&omega, "omega")?,
        };
        let corrected = validate_wall_row(pname, row)?;

        let mut write_back =
            |f: &mut Option<&mut RawScalarField>, before: Option<BcKind>, after: Option<BcKind>| {
                if before == after {
                    return;
                }
                if let (Some(rf), Some(kk)) = (f.as_deref_mut(), after) {
                    if let Some(spec) = rf.boundary.get_mut(pname) {
                        spec.type_name = bc_kind_name(kk).to_string();
                    }
                }
            };
        write_back(&mut nut, row.nut, corrected.nut);
        write_back(&mut k, row.k, corrected.k);
        write_back(&mut epsilon, row.epsilon, corrected.epsilon);
        write_back(&mut omega, row.omega, corrected.omega);
    }
    Ok(())
}

pub fn validate_wall_row(patch: &str, row: WallRow) -> Result<WallRow> {
    let entries = row.entries();

    // The reference family: the first field, in nut/k/epsilon/omega order,
    // that actually declares one. Nothing declaring one means this patch is
    // not participating in the wall-function family at all, as far as this
    // check is concerned.
    let mut reference: Option<(&'static str, WallRowFamily)> = None;
    for (name, kind, family_of) in entries {
        if let Some(f) = kind.and_then(family_of) {
            reference = Some((name, f));
            break;
        }
    }
    let Some((reference_field, reference_family)) = reference else {
        return Ok(row);
    };

    // The first OTHER field that disagrees with the reference.
    let mut offender: Option<(&'static str, BcKind)> = None;
    for (name, kind, family_of) in entries {
        if name == reference_field {
            continue;
        }
        if let Some(k) = kind {
            if family_of(k).is_some_and(|f| f != reference_family) {
                offender = Some((name, k));
                break;
            }
        }
    }
    let Some((offender_field, offender_kind)) = offender else {
        return Ok(row);
    };

    let reference_kind = match reference_field {
        "nut" => row.nut,
        "k" => row.k,
        "epsilon" => row.epsilon,
        _ => row.omega,
    }
    .expect("the reference field carries the family it just supplied");

    let opposite = match reference_family {
        WallRowFamily::Full => WallRowFamily::LowRe,
        WallRowFamily::LowRe => WallRowFamily::Full,
    };
    let setting = format!("wall consistency/{patch}");
    let value = format!(
        "{reference_field} = {} vs {offender_field} = {}",
        bc_kind_name(reference_kind),
        bc_kind_name(offender_kind)
    );
    let completions = format!(
        "set {offender_field} to {} to match {reference_field}, or set {reference_field} to {} \
         to match {offender_field}",
        completion_for(offender_field, reference_family),
        completion_for(reference_field, opposite),
    );

    if !crate::io::contract::permissive() {
        return Err(Error::Config(format!(
            "{setting}: {value} is not one row of SPEC-LIT 29.1's wall-treatment table \
             ({completions})\n  (run with -permissive to substitute the row implied by \
             {reference_field} and continue)"
        )));
    }

    let corrected = family_kind(offender_field, reference_family);
    crate::io::contract::warn_once(
        &setting,
        &format!(
            "-permissive: {setting}: {value} is not one row of SPEC-LIT 29.1's wall-treatment \
             table; substituting {offender_field} = {} (the row implied by {reference_field})",
            bc_kind_name(corrected)
        ),
    );

    let mut out = row;
    match offender_field {
        "nut" => out.nut = Some(corrected),
        "k" => out.k = Some(corrected),
        "epsilon" => out.epsilon = Some(corrected),
        "omega" => out.omega = Some(corrected),
        _ => {}
    }
    Ok(out)
}

/// The wall constants a model needs, from the parsed case.
///
/// `kappa`, `E`, `Cmu`, `beta1` and the derived `y_plus_lam` are
/// already one struct by the time [`crate::io::case`] has read them, so this
/// is a copy. It exists so a driver names its intent rather than reaching into
/// the controls struct, and so the two can diverge later without touching
/// every driver.
pub fn wall_coeffs_from_case(
    w: &crate::io::case::WallFunctionCoeffs,
) -> crate::io::case::WallFunctionCoeffs {
    w.clone()
}

// ==========================================================================
//  Flux
// ==========================================================================

/// Build the face flux from the cell velocity: `phi_f = interpolate(U)_f . Sf`.
///
/// The fallback for a case with no `phi` file. It is a *starting* flux, not a
/// conservative one: linear interpolation of a cell velocity does not satisfy
/// `sum_f phi_f = 0` per cell, and it will not until either a pressure
/// correction or the potential-flow solve in [`crate::potential_flow`] has
/// run. [`max_div_phi`] is there to say how far off it is; on a real case
/// expect something like 1e-2, not 1e-15.
///
/// Done on the host because it runs once, before the time loop.
pub fn compute_phi_from_u(
    gpu: &Gpu,
    phi: &mut crate::field::GpuSurfaceScalarField,
    u: &GpuVectorField,
    m: &HostMesh,
) -> Result<()> {
    let uc = gpu.download(&u.f)?;
    let ub = gpu.download(&u.bf)?;

    let mut internal = vec![0.0 as Scalar; m.n_internal_faces];
    for f in 0..m.n_internal_faces {
        let o = m.owner[f] as usize;
        let n = m.neighbour[f] as usize;
        let w = m.weights[f];
        let uf = uc[o] * w + uc[n] * (1.0 - w);
        internal[f] = uf.dot(m.sf[f]);
    }

    let mut boundary = vec![0.0 as Scalar; m.n_boundary_faces];
    for bf in 0..m.n_boundary_faces {
        boundary[bf] = ub[bf].dot(m.b_sf[bf]);
    }

    gpu.write(&mut phi.f, &internal)?;
    gpu.write(&mut phi.bf, &boundary)?;
    Ok(())
}

/// Copy the conservative face flux back into a case field for writing.
///
/// Unlike [`harvest_scalar_field`] this INVENTS the boundary entries, because
/// `phi` has none to inherit: it is not a field the case supplies, it is the
/// one the pressure equation produces. Every patch gets an explicit entry, so
/// the file names every patch and can be read back as the start time of
/// another run.
///
/// *DESIGN.* An `empty` patch is written as `empty`, which carries no `value`.
/// Its flux is identically zero - `Sf` there has no component the 2-D solution
/// resolves - so nothing is lost, and an `empty` patch carrying a `calculated`
/// face value is not a file another tool would accept. Every other patch is
/// `calculated` with its face values, cyclic patches included: a cyclic couple
/// carries real flux and dropping it would defeat the round trip.
///
/// The result satisfies `Σ_f (±phi_f) = 0` per cell to whatever accuracy the
/// last pressure solve reached, which is the property a restart needs and
/// neither `interpolate(U)·Sf` nor a potential-flow seed has - SPEC-LIT §5.1.
pub fn harvest_surface_scalar_field(
    gpu: &Gpu,
    out: &mut RawScalarField,
    phi: &crate::field::GpuSurfaceScalarField,
    m: &HostMesh,
) -> Result<()> {
    out.internal = gpu.download(&phi.f)?;
    let bvals = gpu.download(&phi.bf)?;

    // Written fresh every time: a stale entry from a previous harvest would
    // survive into a file whose mesh had changed under it.
    out.boundary.clear();
    out.boundary_patterns.clear();

    for p in &m.patches {
        let mut spec = PatchFieldSpec::default();
        if p.kind == PatchKind::Empty {
            spec.type_name = "empty".to_string();
        } else {
            spec.type_name = "calculated".to_string();
            spec.value = bvals
                .get(p.start..p.start + p.size)
                .map(<[Scalar]>::to_vec)
                .unwrap_or_default();
        }
        out.boundary.insert(p.name.clone(), spec);
    }

    // m^3/s: a VOLUMETRIC flux, because this solver's momentum equation is the
    // incompressible one and `phi` never carries a density.
    if out.dimensions.is_empty() {
        out.dimensions = "[0 3 -1 0 0 0 0]".to_string();
    }
    Ok(())
}

/// The worst continuity error in the mesh, read back from the device.
///
/// See [`max_div_phi_host`] for why this number matters.
pub fn max_div_phi(
    gpu: &Gpu,
    phi: &crate::field::GpuSurfaceScalarField,
    m: &HostMesh,
) -> Result<Scalar> {
    let internal = gpu.download(&phi.f)?;
    let boundary = gpu.download(&phi.bf)?;
    Ok(max_div_phi_host(&internal, &boundary, m))
}

/// Recompute the `inletOutlet` value fraction from the current face flux.
///
/// `inletOutlet` is Dirichlet where the flow comes IN and zero-gradient where
/// it goes out - which is the only stable thing to do at an open boundary,
/// since prescribing a value on outflow fights the interior solution. The
/// switch is the sign of the face flux, so it has to be redone whenever the
/// flux changes.
///
/// The convention: `phi_b` is positive OUT of the domain, so inflow is
/// `phi_b < 0`. A face with exactly zero flux is treated as outflow, which
/// leaves it zero-gradient and lets the interior decide.
///
/// Host-side, for setup. The in-loop version is
/// [`crate::field_ops::update_inlet_outlet`], which is a kernel.
pub fn update_inlet_outlet(
    gpu: &Gpu,
    f: &mut GpuScalarField,
    phi: &crate::field::GpuSurfaceScalarField,
    m: &HostMesh,
) -> Result<()> {
    let kinds = gpu.download(&f.bc_kind)?;
    let phib = gpu.download(&phi.bf)?;
    let mut fr = gpu.download(&f.fr)?;

    for bf in 0..m.n_boundary_faces {
        // The whole flux-switched block, not just `inletOutlet`: the
        // turbulence inlets and `pressureInletOutletVelocity` switch on the
        // same test and differ only in the value they switch TO.
        if (crate::field::FLUX_SWITCHED_FIRST..=crate::field::FLUX_SWITCHED_LAST)
            .contains(&kinds[bf])
        {
            fr[bf] = if phib[bf] < 0.0 { 1.0 } else { 0.0 };
        }
    }

    gpu.write(&mut f.fr, &fr)
}

// ==========================================================================
//  Flux diagnostics
// ==========================================================================

/// The worst continuity error in the mesh: `max_P |sum_f (+/-) phi_f|`.
///
/// A discretely conservative flux field satisfies `sum_f (+/-) phi_f = 0` in
/// every cell, exactly, to round-off. It is not enough for it to be *close*:
/// the pressure equation is `div(rAUf grad p) = div(phiHbyA)`, so whatever
/// this number is becomes a spurious source that the pressure solve then
/// faithfully propagates through the whole domain.
///
/// The sign convention is the one the mesh carries: a face's `phi` points from
/// its owner to its neighbour, so it leaves the owner and enters the
/// neighbour. Boundary faces always point out of the domain.
///
/// Absolute, not normalised - the caller knows the flux scale of its own
/// problem and a relative measure here would hide a small-velocity case.
pub fn max_div_phi_host(internal: &[Scalar], boundary: &[Scalar], m: &HostMesh) -> Scalar {
    let mut div = vec![0.0 as Scalar; m.n_cells];

    for f in 0..m.n_internal_faces.min(internal.len()) {
        let o = m.owner[f] as usize;
        let n = m.neighbour[f] as usize;
        div[o] += internal[f];
        div[n] -= internal[f];
    }

    for bf in 0..m.n_boundary_faces.min(boundary.len()) {
        let c = m.b_face_cells[bf] as usize;
        div[c] += boundary[bf];
    }

    div.iter().fold(0.0 as Scalar, |a, d| a.max(d.abs()))
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::PatchInfo;

    /// Two cells, two boundary patches of one face each.
    fn mesh() -> HostMesh {
        HostMesh {
            n_cells: 2,
            n_internal_faces: 1,
            n_boundary_faces: 2,
            owner: vec![0],
            neighbour: vec![1],
            b_face_cells: vec![0, 1],
            patches: vec![
                PatchInfo {
                    name: "inlet".into(),
                    type_name: "patch".into(),
                    kind: PatchKind::Generic,
                    start: 0,
                    size: 1,
                    nbr_patch: None,
                },
                PatchInfo {
                    name: "front".into(),
                    type_name: "empty".into(),
                    kind: PatchKind::Empty,
                    start: 1,
                    size: 1,
                    nbr_patch: None,
                },
            ],
            ..Default::default()
        }
    }

    fn spec(t: &str) -> PatchFieldSpec {
        PatchFieldSpec {
            type_name: t.into(),
            ..Default::default()
        }
    }

    /// The two-cell mesh with real face areas, for the conditions that need
    /// a normal or a patch area. `inlet` faces -x, `front` faces +x.
    fn mesh_with_areas() -> HostMesh {
        let mut m = mesh();
        m.b_sf = vec![Vec3::new(-2.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)];
        m
    }

    fn scalar_field(patch: &str, s: PatchFieldSpec) -> RawScalarField {
        let mut raw = RawScalarField {
            name: "psi".into(),
            ..Default::default()
        };
        raw.boundary.insert(patch.into(), s);
        raw
    }

    fn vector_field(patch: &str, s: PatchFieldSpec) -> RawVectorField {
        let mut raw = RawVectorField {
            name: "U".into(),
            ..Default::default()
        };
        raw.boundary.insert(patch.into(), s);
        raw
    }

    // ------------------------------------------------------------------
    //  SPEC-LIT 13.4 - an unrecognised name is an error, not a Dirichlet
    // ------------------------------------------------------------------

    /// The failure this whole task exists to remove: `garbageBC` used to run
    /// to completion, as a hard Dirichlet at whatever `value` held, and give
    /// a field indistinguishable from a real one.
    #[test]
    fn an_unrecognised_bc_name_is_an_error_that_names_it() {
        let _guard = crate::io::contract::permissive_test_guard();
        crate::io::contract::set_permissive(false);

        let mut sp = spec("garbageBC");
        sp.value = vec![1.0];
        let raw = scalar_field("inlet", sp);

        let e = scalar_boundary(&raw, &mesh(), &[0.0, 0.0])
            .expect_err("an unknown boundary condition must not run");
        let msg = e.to_string();
        assert!(msg.contains("garbageBC"), "{msg}");
        assert!(msg.contains("inlet"), "{msg}");
        // The menu, so the user can fix it without reading the source.
        assert!(msg.contains("zeroGradient"), "{msg}");
    }

    /// Every name the reader claims to implement must reach a DIFFERENT
    /// condition from the silent-Dirichlet fallback it used to get.
    #[test]
    fn every_bc_name_round_trips_to_the_condition_it_names() {
        use crate::field::IMPLEMENTED_BC_NAMES;

        for name in IMPLEMENTED_BC_NAMES {
            let k = BcKind::from_name(name, "psi", "inlet")
                .unwrap_or_else(|e| panic!("{name} is on the menu but rejected: {e}"));

            // The one thing that must never happen: a name on the menu
            // mapping to `Calculated` unless it IS `calculated`.
            if *name != "calculated" {
                assert_ne!(
                    k,
                    BcKind::Calculated,
                    "{name} still falls through to the silent Dirichlet"
                );
            }
        }

        // And the mappings that carry physics, spelled out.
        let k = |n: &str| BcKind::from_name(n, "psi", "p").expect("known");
        assert_eq!(k("calculated"), BcKind::Calculated);
        assert_eq!(k("zeroGradient"), BcKind::ZeroGradient);
        assert_eq!(k("noSlip"), BcKind::FixedValue);
        assert_eq!(
            k("turbulentIntensityKineticEnergyInlet"),
            BcKind::TurbulentIntensityKineticEnergyInlet
        );
        assert_eq!(k("fixedFluxPressure"), BcKind::FixedFluxPressure);
        assert_eq!(k("totalPressure"), BcKind::TotalPressure);
        assert_eq!(k("movingWallVelocity"), BcKind::MovingWallVelocity);
        assert_eq!(k("flowRateInletVelocity"), BcKind::FlowRateInletVelocity);
        assert_eq!(k("nutLowReWallFunction"), BcKind::NutLowReWallFunction);
        assert_eq!(k("thermalWallFunction"), BcKind::ThermalWallFunction);
    }

    /// SPEC-LIT §29.3: OpenFOAM's `alphat` name for the Jayatilleke wall
    /// function is accepted as an alias for `thermalWallFunction`, not
    /// rejected - and the substitution is reported through `warn_once`,
    /// never silently.
    #[test]
    fn the_alphat_jayatilleke_name_is_accepted_as_an_alias_and_reported() {
        let k = BcKind::from_name(
            "compressible::alphatJayatillekeWallFunction",
            "T",
            "wall4",
        )
        .expect("the alias must be accepted, not rejected");
        assert_eq!(k, BcKind::ThermalWallFunction);
    }

    // ------------------------------------------------------------------
    //  The derived conditions
    // ------------------------------------------------------------------

    /// `k = 3/2 (I |U|)^2`, and flux-switched so an outflow is zero-gradient.
    #[test]
    fn turbulence_intensity_inlet_is_three_halves_i_u_squared() {
        let mut sp = spec("turbulentIntensityKineticEnergyInlet");
        sp.extra.insert("turbulentIntensity".into(), "0.05".into());
        let raw = scalar_field("inlet", sp);

        let u_b = [Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO];
        let b = scalar_boundary_with(
            &raw,
            &mesh_with_areas(),
            &[0.0, 0.0],
            BcInputs {
                u_b: Some(&u_b),
                ..Default::default()
            },
        )
        .expect("builds");

        let want = 1.5 * (0.05 * 10.0f64 as Scalar).powi(2);
        assert!((b.ref_value[0] - want).abs() < 1e-12, "{}", b.ref_value[0]);
        assert_eq!(b.kind[0], BcKind::TurbulentIntensityKineticEnergyInlet as Label);
        assert!(BcKind::TurbulentIntensityKineticEnergyInlet.is_flux_switched());
    }

    /// Without a `turbulentIntensity` entry the condition is not defined, and
    /// defaulting it would put an invented turbulence level on the inlet.
    #[test]
    fn a_turbulence_inlet_without_its_entry_is_an_error() {
        let raw = scalar_field("inlet", spec("turbulentIntensityKineticEnergyInlet"));
        let u_b = [Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO];
        assert!(scalar_boundary_with(
            &raw,
            &mesh_with_areas(),
            &[0.0, 0.0],
            BcInputs {
                u_b: Some(&u_b),
                ..Default::default()
            },
        )
        .is_err());
    }

    /// `epsilon = C_mu^{3/4} k^{3/2}/L`, with the CASE's C_mu - SPEC-LIT 15.6.
    #[test]
    fn mixing_length_inlet_uses_the_cases_cmu() {
        let mut sp = spec("turbulentMixingLengthDissipationRateInlet");
        sp.extra.insert("mixingLength".into(), "0.01".into());
        let raw = scalar_field("inlet", sp);

        let k_b = [0.375 as Scalar, 0.0];
        let build = |cmu: Scalar| {
            scalar_boundary_with(
                &raw,
                &mesh_with_areas(),
                &[0.0, 0.0],
                BcInputs {
                    k_b: Some(&k_b),
                    cmu: Some(cmu),
                    ..Default::default()
                },
            )
            .expect("builds")
            .ref_value[0]
        };

        let want = (0.09 as Scalar).powf(0.75) * 0.375 * (0.375 as Scalar).sqrt() / 0.01;
        assert!((build(0.09) - want).abs() < 1e-10);
        // A case that overrides C_mu must move this too.
        assert!(build(0.12) > build(0.09));
    }

    /// `U = -n Q/A`.
    #[test]
    fn flow_rate_inlet_carries_the_prescribed_volume_flux() {
        let mut sp = spec("flowRateInletVelocity");
        sp.extra.insert("volumetricFlowRate".into(), "0.5".into());
        let raw = vector_field("inlet", sp);

        let m = mesh_with_areas();
        let b = vector_boundary(&raw, &m, &[Vec3::ZERO, Vec3::ZERO]).expect("builds");

        // |Sf| = 2, n = (-1,0,0), so U = -n Q/A = (+0.25, 0, 0): into the
        // domain through a patch whose outward normal is -x.
        assert!((b.ref_value[0].x - 0.25).abs() < 1e-12, "{:?}", b.ref_value[0]);
        assert_eq!(b.fr[0], 1.0);
        // The flux it actually carries is Q, inward.
        let flux = b.value[0].dot(m.b_sf[0]);
        assert!((flux + 0.5).abs() < 1e-12, "flux {flux}");
    }

    /// A moving wall drags the fluid along itself and cannot push fluid
    /// through itself: the normal component of the prescribed value goes.
    #[test]
    fn moving_wall_velocity_drops_the_normal_component() {
        let mut sp = spec("movingWallVelocity");
        sp.value_v = vec![Vec3::new(3.0, 1.0, 0.0)];
        let raw = vector_field("inlet", sp);

        let b = vector_boundary(&raw, &mesh_with_areas(), &[Vec3::ZERO, Vec3::ZERO])
            .expect("builds");

        // n = (-1,0,0): the x component is normal and must vanish.
        assert!(b.ref_value[0].x.abs() < 1e-12, "{:?}", b.ref_value[0]);
        assert!((b.ref_value[0].y - 1.0).abs() < 1e-12);
    }

    /// `p = p0 - |U|^2/2` where the flow comes in, `p0` where it leaves.
    #[test]
    fn total_pressure_is_bernoulli_on_inflow_and_p0_on_outflow() {
        let mut sp = spec("totalPressure");
        sp.extra.insert("p0".into(), "uniform 10".into());
        let mut raw = scalar_field("inlet", sp.clone());
        raw.boundary.insert("front".into(), sp);

        // Face 0 flows IN (phi < 0), face 1 flows out.
        let u_b = [Vec3::new(2.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)];
        let phi_b = [-4.0 as Scalar, 4.0];

        let mut m = mesh_with_areas();
        // `front` is `empty` in the shared mesh, and topology beats the field
        // file; make it an ordinary patch so the outflow branch is reached.
        m.patches[1].kind = PatchKind::Generic;

        let b = scalar_boundary_with(
            &raw,
            &m,
            &[0.0, 0.0],
            BcInputs {
                u_b: Some(&u_b),
                phi_b: Some(&phi_b),
                ..Default::default()
            },
        )
        .expect("builds");

        assert!((b.ref_value[0] - (10.0 - 2.0)).abs() < 1e-12, "{}", b.ref_value[0]);
        assert!((b.ref_value[1] - 10.0).abs() < 1e-12, "{}", b.ref_value[1]);
    }

    /// A condition defined in terms of another field must say so rather than
    /// invent a value when that field is not there.
    #[test]
    fn a_derived_condition_without_its_input_is_an_error() {
        let mut sp = spec("turbulentIntensityKineticEnergyInlet");
        sp.extra.insert("turbulentIntensity".into(), "0.05".into());
        let raw = scalar_field("inlet", sp);

        let e = scalar_boundary(&raw, &mesh_with_areas(), &[0.0, 0.0])
            .expect_err("no U was supplied");
        assert!(e.to_string().contains("U"), "{e}");
    }

    // ------------------------------------------------------------------
    //  Pattern keys
    // ------------------------------------------------------------------

    /// A `".*"` boundaryField must give exactly what writing every patch out
    /// by hand gives. Before this, it matched nothing and every patch fell
    /// through to the default arm.
    #[test]
    fn a_regex_boundary_field_equals_the_explicit_one() {
        let m = mesh_with_areas();

        let mut explicit = RawScalarField {
            name: "psi".into(),
            ..Default::default()
        };
        let mut sp = spec("fixedValue");
        sp.value = vec![7.5];
        explicit.boundary.insert("inlet".into(), sp.clone());
        explicit.boundary.insert("front".into(), sp.clone());

        let mut pattern = RawScalarField {
            name: "psi".into(),
            ..Default::default()
        };
        pattern.boundary.insert("\".*\"".into(), sp);
        pattern.boundary_patterns.push("\".*\"".into());

        let a = scalar_boundary(&explicit, &m, &[1.0, 2.0]).expect("explicit");
        let b = scalar_boundary(&pattern, &m, &[1.0, 2.0]).expect("pattern");

        assert_eq!(a.fr, b.fr);
        assert_eq!(a.ref_value, b.ref_value);
        assert_eq!(a.ref_grad, b.ref_grad);
        assert_eq!(a.kind, b.kind);
        assert_eq!(a.value, b.value);
    }

    /// Exact wins over a pattern, whatever order they are written in.
    #[test]
    fn an_exact_patch_key_beats_a_pattern_that_also_matches() {
        let m = mesh_with_areas();

        let mut raw = RawScalarField {
            name: "psi".into(),
            ..Default::default()
        };
        let mut wild = spec("fixedValue");
        wild.value = vec![1.0];
        raw.boundary.insert("\".*\"".into(), wild);
        raw.boundary_patterns.push("\".*\"".into());

        let mut exact = spec("fixedValue");
        exact.value = vec![99.0];
        raw.boundary.insert("inlet".into(), exact);

        let b = scalar_boundary(&raw, &m, &[0.0, 0.0]).expect("builds");
        assert_eq!(b.ref_value[0], 99.0, "the exact `inlet` entry must win");
    }

    // ------------------------------------------------------------------
    //  SPEC-LIT 15.2 and 15.5 - the wall
    // ------------------------------------------------------------------

    /// `nutLowRe` means nu_t = 0 at the wall. `nutk` means it is not.
    #[test]
    fn nut_low_re_is_zero_at_the_wall_and_nutk_is_not() {
        let m = mesh_with_areas();

        let mut low = spec("nutLowReWallFunction");
        low.value = vec![0.004];
        let raw_low = scalar_field("inlet", low);
        let b = scalar_boundary(&raw_low, &m, &[0.003, 0.0]).expect("builds");
        assert_eq!(b.fr[0], 1.0);
        assert_eq!(b.ref_value[0], 0.0, "nutLowRe must pin nu_t to zero");
        assert_eq!(b.value[0], 0.0);

        let mut nutk = spec("nutkWallFunction");
        nutk.value = vec![0.004];
        let raw_nutk = scalar_field("inlet", nutk);
        let b = scalar_boundary(&raw_nutk, &m, &[0.003, 0.0]).expect("builds");
        assert_ne!(b.ref_value[0], 0.0, "nutk must not be pinned to zero");

        // And the face sets follow the same rule: nutLowRe is NOT a wall
        // function face, so no wall value is ever written onto it.
        assert!(nut_wall_faces(&raw_low, &m).expect("flags")[0] == false);
        assert!(nut_wall_faces(&raw_nutk, &m).expect("flags")[0]);
    }

    /// SPEC-LIT 15.5: each field decides for itself. The resolved-sublayer
    /// setup - `nut` low-Re, `epsilon` with a wall function - must give a
    /// constrained epsilon cell and NO wall value for nu_t.
    #[test]
    fn each_field_owns_its_own_wall_decision() {
        let m = mesh_with_areas();

        let raw_nut = scalar_field("inlet", spec("nutLowReWallFunction"));
        let raw_eps = scalar_field("inlet", spec("epsilonWallFunction"));

        let faces = WallFaces::from_case(&raw_eps, Some(&raw_nut), &m).expect("flags");
        assert!(faces.constrained_cells[0], "epsilon asked for a wall cell");
        assert!(!faces.nut[0], "nut refused a wall function and must be obeyed");

        // The opposite case, and the opposite failure.
        let raw_nut = scalar_field("inlet", spec("nutkWallFunction"));
        let raw_eps = scalar_field("inlet", spec("fixedValue"));
        let faces = WallFaces::from_case(&raw_eps, Some(&raw_nut), &m).expect("flags");
        assert!(!faces.constrained_cells[0]);
        assert!(faces.nut[0], "nutk must not be left inert by epsilon");
    }

    // ------------------------------------------------------------------
    //  Rough walls - SPEC-LIT §15.3, completed by §29.2
    // ------------------------------------------------------------------

    /// "A rough type WITHOUT `Ks` is the §13.4 error it is today, now naming
    /// the missing key" - the error must name `Ks`.
    #[test]
    fn nutk_rough_wall_function_without_ks_is_an_error_naming_it() {
        let m = mesh();
        let raw = scalar_field("inlet", spec("nutkRoughWallFunction"));
        let err = NutRoughness::from_case(Some(&raw), &m).expect_err("Ks is missing");
        let msg = format!("{err}");
        assert!(msg.contains("Ks"), "error does not name the missing `Ks` entry: {msg}");
    }

    /// `Ks` is read, and `Cs` defaults to `0.5` (SPEC-LIT §15.3's "uniform
    /// sand" default) when the patch does not give one.
    #[test]
    fn nutk_rough_wall_function_reads_ks_and_defaults_cs() {
        let m = mesh();
        let mut s = spec("nutkRoughWallFunction");
        s.extra.insert("Ks".into(), "0.001".into());
        let raw = scalar_field("inlet", s);

        let r = NutRoughness::from_case(Some(&raw), &m).expect("builds");
        assert_eq!(r.ks[0], 0.001);
        assert_eq!(r.cs[0], 0.5);
        assert!(!r.u_based[0], "nutk is the k-based family");
    }

    /// An explicit `Cs` overrides the default.
    #[test]
    fn nutk_rough_wall_function_reads_an_explicit_cs() {
        let m = mesh();
        let mut s = spec("nutkRoughWallFunction");
        s.extra.insert("Ks".into(), "0.002".into());
        s.extra.insert("Cs".into(), "0.8".into());
        let raw = scalar_field("inlet", s);

        let r = NutRoughness::from_case(Some(&raw), &m).expect("builds");
        assert_eq!(r.ks[0], 0.002);
        assert_eq!(r.cs[0], 0.8);
    }

    /// `nutURoughWallFunction` is the velocity-based family - SPEC-LIT §15.1
    /// completed by §29.2 - and `NutRoughness` must say so per face.
    #[test]
    fn nutu_rough_wall_function_is_marked_velocity_based() {
        let m = mesh();
        let mut s = spec("nutURoughWallFunction");
        s.extra.insert("Ks".into(), "0.001".into());
        let raw = scalar_field("inlet", s);

        let r = NutRoughness::from_case(Some(&raw), &m).expect("builds");
        assert!(r.u_based[0]);
        assert_eq!(r.ks[0], 0.001);
    }

    /// The smooth `nutk`/`nutU` wall functions carry `Ks = 0` - what makes
    /// [`crate::wallfunctions::WallData`] collapse the rough kernels to the
    /// smooth ones without a separate code path.
    #[test]
    fn smooth_nut_wall_functions_carry_zero_roughness() {
        let m = mesh();
        for name in ["nutkWallFunction", "nutUWallFunction"] {
            let raw = scalar_field("inlet", spec(name));
            let r = NutRoughness::from_case(Some(&raw), &m).expect("builds");
            assert_eq!(r.ks[0], 0.0, "{name} must carry Ks = 0");
        }
    }

    /// No `nut` file at all: every face is smooth, same as
    /// [`NutRoughness::none`].
    #[test]
    fn no_nut_file_means_no_roughness_anywhere() {
        let m = mesh();
        let r = NutRoughness::from_case(None, &m).expect("builds");
        assert_eq!(r.ks, vec![0.0; m.n_boundary_faces]);
        assert_eq!(r.cs, vec![0.5; m.n_boundary_faces]);
        assert_eq!(r.u_based, vec![false; m.n_boundary_faces]);
    }

    /// The atmospheric rough wall function is a different profile
    /// (Monin-Obukhov), not a discarded parameter: it is an error naming the
    /// two rough functions this solver does implement.
    #[test]
    fn nutk_atm_rough_wall_function_is_an_error_naming_the_two() {
        let err = BcKind::from_name("nutkAtmRoughWallFunction", "nut", "inlet")
            .expect_err("not implemented");
        let msg = format!("{err}");
        assert!(msg.contains("nutkRoughWallFunction"), "{msg}");
        assert!(msg.contains("nutURoughWallFunction"), "{msg}");
    }

    // ------------------------------------------------------------------
    //  SPEC-LIT §29.1: wallTreatment preset expansion (route a)
    // ------------------------------------------------------------------

    /// One `inlet` patch (Generic) and one `wall1` patch (Wall) - what the
    /// preset-expansion tests need that [`mesh`] does not have.
    fn mesh_with_wall() -> HostMesh {
        HostMesh {
            n_cells: 2,
            n_internal_faces: 1,
            n_boundary_faces: 2,
            owner: vec![0],
            neighbour: vec![1],
            b_face_cells: vec![0, 1],
            patches: vec![
                PatchInfo {
                    name: "inlet".into(),
                    type_name: "patch".into(),
                    kind: PatchKind::Generic,
                    start: 0,
                    size: 1,
                    nbr_patch: None,
                },
                PatchInfo {
                    name: "wall1".into(),
                    type_name: "wall".into(),
                    kind: PatchKind::Wall,
                    start: 1,
                    size: 1,
                    nbr_patch: None,
                },
            ],
            ..Default::default()
        }
    }

    #[test]
    fn preset_fills_in_a_wall_patch_the_field_left_out() {
        let m = mesh_with_wall();
        let mut raw = RawScalarField {
            name: "nut".into(),
            ..Default::default()
        };
        apply_wall_treatment_defaults(
            &mut raw,
            "nut",
            &m,
            crate::io::case::WallTreatment::Standard,
            None,
        )
        .expect("applies");
        assert_eq!(raw.spec("wall1").unwrap().unwrap().type_name, "nutkWallFunction");
        // A non-wall patch never gets a default from this preset.
        assert!(raw.spec("inlet").unwrap().is_none());
    }

    /// Route (a)'s precedence: an explicit per-field entry already on the
    /// patch is never overwritten by the preset default.
    #[test]
    fn preset_never_overrides_an_explicit_patch_entry() {
        let m = mesh_with_wall();
        let mut raw = scalar_field("wall1", spec("nutUWallFunction"));
        raw.name = "nut".into();
        apply_wall_treatment_defaults(
            &mut raw,
            "nut",
            &m,
            crate::io::case::WallTreatment::Standard,
            None,
        )
        .expect("applies");
        assert_eq!(raw.spec("wall1").unwrap().unwrap().type_name, "nutUWallFunction");
    }

    #[test]
    fn each_field_gets_its_own_row_entry() {
        let m = mesh_with_wall();
        for (field, treatment, want) in [
            ("nut", crate::io::case::WallTreatment::Spalding, "nutUWallFunction"),
            ("k", crate::io::case::WallTreatment::LowRe, "kLowReWallFunction"),
            ("epsilon", crate::io::case::WallTreatment::LowRe, "zeroGradient"),
            ("omega", crate::io::case::WallTreatment::Standard, "omegaWallFunction"),
        ] {
            let mut raw = RawScalarField {
                name: field.into(),
                ..Default::default()
            };
            apply_wall_treatment_defaults(&mut raw, field, &m, treatment, None).expect("applies");
            assert_eq!(raw.spec("wall1").unwrap().unwrap().type_name, want, "{field}/{treatment:?}");
        }
    }

    #[test]
    fn rough_preset_writes_ks_and_cs_onto_the_synthesised_entry() {
        let m = mesh_with_wall();
        let mut raw = RawScalarField {
            name: "nut".into(),
            ..Default::default()
        };
        let rough = crate::io::case::Roughness { ks: 0.002, cs: 0.7 };
        apply_wall_treatment_defaults(
            &mut raw,
            "nut",
            &m,
            crate::io::case::WallTreatment::Rough,
            Some(rough),
        )
        .expect("applies");
        let s = raw.spec("wall1").unwrap().unwrap();
        assert_eq!(s.type_name, "nutkRoughWallFunction");
        assert_eq!(s.extra.get("Ks").unwrap(), "0.002");
        assert_eq!(s.extra.get("Cs").unwrap(), "0.7");
    }

    /// §29.3: `lowRe` pins the molecular resistance already there - `T`
    /// gets no default from this preset at all.
    #[test]
    fn low_re_writes_no_thermal_default() {
        let m = mesh_with_wall();
        let mut raw = RawScalarField {
            name: "T".into(),
            ..Default::default()
        };
        apply_wall_treatment_defaults(
            &mut raw,
            "T",
            &m,
            crate::io::case::WallTreatment::LowRe,
            None,
        )
        .expect("applies");
        assert!(raw.spec("wall1").unwrap().is_none());
    }

    #[test]
    fn every_other_preset_defaults_t_to_the_thermal_wall_function() {
        let m = mesh_with_wall();
        for wt in [
            crate::io::case::WallTreatment::Standard,
            crate::io::case::WallTreatment::Spalding,
            crate::io::case::WallTreatment::Rough,
        ] {
            let mut raw = RawScalarField {
                name: "T".into(),
                ..Default::default()
            };
            apply_wall_treatment_defaults(&mut raw, "T", &m, wt, None).expect("applies");
            assert_eq!(raw.spec("wall1").unwrap().unwrap().type_name, "thermalWallFunction");
        }
    }

    // ------------------------------------------------------------------
    //  SPEC-LIT §29.1 point 3: the wall-row consistency contract
    // ------------------------------------------------------------------

    /// The running example from SPEC-LIT §29.1 itself: `nutLowReWallFunction`
    /// together with `epsilonWallFunction` is a contradiction, and strict
    /// mode refuses it naming both types and the two ways to fix it.
    #[test]
    fn nut_low_re_with_epsilon_wall_function_is_the_named_contradiction() {
        let _g = crate::io::contract::permissive_test_guard();
        crate::io::contract::set_permissive(false);

        let row = WallRow {
            nut: Some(BcKind::NutLowReWallFunction),
            k: None,
            epsilon: Some(BcKind::EpsilonWallFunction),
            omega: None,
        };
        let err = validate_wall_row("wall1", row).expect_err("must be refused");
        let msg = err.to_string();
        assert!(msg.contains("wall1"), "{msg}");
        assert!(msg.contains("nutLowReWallFunction"), "{msg}");
        assert!(msg.contains("epsilonWallFunction"), "{msg}");
        assert!(msg.contains("zeroGradient"), "{msg} (epsilon's lowRe completion)");
        assert!(msg.contains("nutkWallFunction"), "{msg} (nut's full completion)");
    }

    /// `-permissive` substitutes the row implied by `nut` and says so.
    #[test]
    fn permissive_substitutes_the_row_implied_by_nut() {
        let _g = crate::io::contract::permissive_test_guard();
        crate::io::contract::reset_warnings();
        crate::io::contract::set_permissive(true);

        let row = WallRow {
            nut: Some(BcKind::NutLowReWallFunction),
            k: None,
            epsilon: Some(BcKind::EpsilonWallFunction),
            omega: None,
        };
        let corrected = validate_wall_row("wall1", row).expect("permissive resolves it");
        assert_eq!(corrected.nut, Some(BcKind::NutLowReWallFunction));
        assert_eq!(corrected.epsilon, Some(BcKind::ZeroGradient));

        crate::io::contract::set_permissive(false);
    }

    /// Every preset row is internally consistent - the common case must never
    /// be flagged.
    #[test]
    fn every_preset_row_is_already_consistent() {
        for wt in [
            crate::io::case::WallTreatment::Standard,
            crate::io::case::WallTreatment::Spalding,
            crate::io::case::WallTreatment::Rough,
            crate::io::case::WallTreatment::LowRe,
        ] {
            let nut = BcKind::from_name(wt.nut_type(), "nut", "wall1").unwrap();
            let k = BcKind::from_name(wt.k_type(), "k", "wall1").unwrap();
            let epsilon = BcKind::from_name(wt.epsilon_type(), "epsilon", "wall1").unwrap();
            let omega = BcKind::from_name(wt.omega_type(), "omega", "wall1").unwrap();

            let row = WallRow {
                nut: Some(nut),
                k: Some(k),
                epsilon: Some(epsilon),
                omega: Some(omega),
            };
            let corrected = validate_wall_row("wall1", row).unwrap_or_else(|e| {
                panic!("{:?} row flagged as inconsistent: {e}", wt)
            });
            assert_eq!(corrected.nut, row.nut);
            assert_eq!(corrected.k, row.k);
            assert_eq!(corrected.epsilon, row.epsilon);
            assert_eq!(corrected.omega, row.omega);
        }
    }

    /// A field with no opinion (`fixedValue`) never conflicts with one that
    /// has one - SPEC-LIT §15.5's independence, preserved by this check.
    #[test]
    fn a_field_with_no_opinion_never_conflicts() {
        let row = WallRow {
            nut: Some(BcKind::NutkWallFunction),
            k: None,
            epsilon: Some(BcKind::FixedValue),
            omega: None,
        };
        let corrected = validate_wall_row("wall1", row).expect("no conflict");
        assert_eq!(corrected.epsilon, Some(BcKind::FixedValue));
    }

    #[test]
    fn fixed_value_is_fr_one() {
        let mut raw = RawScalarField {
            name: "T".into(),
            ..Default::default()
        };
        let mut s = spec("fixedValue");
        s.value = vec![900.0];
        raw.boundary.insert("inlet".into(), s);

        let b = scalar_boundary(&raw, &mesh(), &[300.0, 300.0]).unwrap();
        assert_eq!(b.fr[0], 1.0);
        assert_eq!(b.ref_value[0], 900.0);
        assert_eq!(b.ref_grad[0], 0.0);
        assert_eq!(b.value[0], 900.0);
    }

    #[test]
    fn zero_gradient_is_fr_zero() {
        let mut raw = RawScalarField {
            name: "T".into(),
            ..Default::default()
        };
        raw.boundary.insert("inlet".into(), spec("zeroGradient"));

        let b = scalar_boundary(&raw, &mesh(), &[300.0, 300.0]).unwrap();
        assert_eq!(b.fr[0], 0.0);
        assert_eq!(b.ref_grad[0], 0.0);
        // psi_b = psi_P when the gradient is zero.
        assert_eq!(b.value[0], 300.0);
    }

    #[test]
    fn fixed_gradient_keeps_the_gradient() {
        let mut raw = RawScalarField {
            name: "T".into(),
            ..Default::default()
        };
        let mut s = spec("fixedGradient");
        s.gradient = vec![5.0];
        raw.boundary.insert("inlet".into(), s);

        let b = scalar_boundary(&raw, &mesh(), &[300.0, 300.0]).unwrap();
        assert_eq!(b.fr[0], 0.0);
        assert_eq!(b.ref_grad[0], 5.0);
    }

    #[test]
    fn mixed_interpolates_between_the_two() {
        let mut raw = RawScalarField {
            name: "T".into(),
            ..Default::default()
        };
        let mut s = spec("mixed");
        s.ref_value = vec![900.0];
        s.ref_gradient = vec![0.0];
        s.value_fraction = vec![0.25];
        raw.boundary.insert("inlet".into(), s);

        let b = scalar_boundary(&raw, &mesh(), &[300.0, 300.0]).unwrap();
        assert_eq!(b.fr[0], 0.25);
        // 0.25*900 + 0.75*300
        assert!((b.value[0] - 450.0).abs() < 1e-12);
    }

    /// The one that matters: the mesh, not the field file, decides `empty`.
    #[test]
    fn topology_beats_the_field_file() {
        let mut raw = RawScalarField {
            name: "T".into(),
            ..Default::default()
        };
        let mut s = spec("fixedValue");
        s.value = vec![1234.0];
        // A fixedValue on a patch the mesh says is empty.
        raw.boundary.insert("front".into(), s);

        let b = scalar_boundary(&raw, &mesh(), &[300.0, 300.0]).unwrap();
        assert_eq!(b.kind[1], BcKind::Empty as Label);
        assert_eq!(b.fr[1], 0.0);
        assert_ne!(b.ref_value[1], 1234.0);
    }

    #[test]
    fn a_missing_patch_holds_its_cell_value() {
        let raw = RawScalarField {
            name: "T".into(),
            ..Default::default()
        };
        let b = scalar_boundary(&raw, &mesh(), &[321.0, 300.0]).unwrap();
        assert_eq!(b.kind[0], BcKind::Calculated as Label);
        assert_eq!(b.value[0], 321.0);
    }

    #[test]
    fn no_slip_is_zero_not_the_internal_value() {
        let mut raw = RawVectorField {
            name: "U".into(),
            ..Default::default()
        };
        raw.boundary.insert("inlet".into(), spec("noSlip"));

        let internal = vec![Vec3::new(2.0, 0.0, 0.0), Vec3::default()];
        let b = vector_boundary(&raw, &mesh(), &internal).unwrap();
        assert_eq!(b.fr[0], 1.0);
        assert_eq!(b.value[0], Vec3::default());
    }

    #[test]
    fn a_uniform_entry_spreads_over_the_patch() {
        let v = spread(&[7.0], 4, 0.0, "value", "p").unwrap();
        assert_eq!(v, vec![7.0; 4]);
    }

    #[test]
    fn a_wrong_length_list_is_an_error() {
        assert!(spread(&[1.0, 2.0], 4, 0.0, "value", "p").is_err());
    }
    /// The OpenFOAM-directory route runs the same §29.1 contract as the
    /// JSONC route. Regression for the gap where only JSONC validated.
    #[test]
    fn raw_field_route_refuses_a_mixed_wall_row() {
        let _guard = crate::io::contract::permissive_test_guard();
        crate::io::contract::set_permissive(false);

        let patches = vec![crate::mesh::PatchInfo {
            name: "ceiling".into(),
            type_name: "wall".into(),
            kind: crate::mesh::PatchKind::Wall,
            start: 0,
            size: 1,
            nbr_patch: None,
        }];
        let mk = |ty: &str| {
            let mut f = RawScalarField::default();
            let mut spec = crate::io::fields::PatchFieldSpec::default();
            spec.type_name = ty.into();
            f.boundary.insert("ceiling".into(), spec);
            f
        };
        let mut nut = mk("nutLowReWallFunction");
        let mut k = mk("kqRWallFunction");
        let mut eps = mk("epsilonWallFunction");

        let err = validate_wall_rows(
            &patches,
            Some(&mut nut),
            Some(&mut k),
            Some(&mut eps),
            None,
        )
        .expect_err("the lowRe/full mix must be refused");
        let msg = format!("{err}");
        assert!(msg.contains("ceiling"), "{msg}");
        assert!(msg.contains("nutLowReWallFunction"), "{msg}");
    }

}
