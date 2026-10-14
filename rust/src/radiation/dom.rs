// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The discrete-ordinates rows of the fvDOM radiation model (SPEC-LIT
//! §136.2, §136.6): one first-order upwind row per direction - the step
//! scheme - with the boundary intensities of §136.2's table, for one
//! uniform grey medium on one mesh, absorbing and, since CHR-21, scattering
//! isotropically or linear-anisotropically.
//!
//! The row is (136.3): `diag`-side `sum_f max(Omega.S_f, 0) + beta V` with
//! `beta = kappa + sigma_s`, off-diagonals `min(Omega.S_f, 0)`, source
//! `kappa (sigma T_P^4/pi) V` plus the incoming boundary faces'
//! `-min(Omega.S_b, 0) I_b^l`, plus - from [`DomRows::add_in_scatter`] - the
//! lagged in-scattering sum `(sigma_s/4pi) (G_P + A_1 Omega_l . q_P) V`
//! (§136.2, source iteration; SPEC-LIT §136.8). It is NOT a new
//! discretisation: the convection is the crate's own Gauss upwind
//! ([`crate::fv::fvm_div_gauss`] under [`crate::fv::DivScheme::Upwind`]) of
//! the face flux `phi_f = Omega_l . S_f`, the sink is
//! [`crate::fv::fvm_sp`], the emission [`crate::fv::fvm_su`], and a
//! boundary face enters through §4's triple exactly as §136.6 specifies -
//! `f = 1, refValue = I_b^l, refGrad = 0` into the medium, `f = 0` out of
//! it - then folds through [`crate::ldu_ops::add_boundary_contributions`],
//! as P1 builds on `fvm_laplacian`.
//!
//! The quadrature is generated on the host in f64 by [`super::dom_quad`]
//! and cast to [`Scalar`] once, at construction (SPEC-LIT §136.5). Every
//! launch in [`DomRows::update_boundaries`] and [`DomRows::assemble`] is
//! pure device work - the scratch was allocated in [`DomRows::new`] - so
//! both capture into a CUDA graph (SPEC-LIT §81); the gate is
//! [`tests::dom_replays_under_capture`].
//!
//! [`DomSolve`] (CHR-20, SPEC-LIT §136.7; scattering CHR-21, §136.8) solves
//! the rows: one sweep updates the boundary intensities, assembles every
//! direction, adds the in-scattering sum from the previous sweep's `G` and
//! `q` (zero on the first sweep), solves it with the crate's own Krylov
//! solver (PBiCGStab; PCG and DIC are refused, the rows being asymmetric),
//! and sums (136.4)'s `G`, `q` and wall flux in `ofacc`
//! (§118.1); [`DomSolve::solve`] repeats the sweep until `G` stops changing
//! by `tol` or the sweep cap is met, which is reported, never accepted
//! silently. [`DomSolve::sweep`] is launches only under `fixed_iters`
//! controls, so it captures too (the same gate).
//!
//! Provenance: ORIGINAL - the row, the boundary table and the refusals are
//! this project's own, built on the FDS Technical Reference Guide's
//! radiation chapter (McGrattan et al., NIST SP 1018, US Government work)
//! for the boundary forms, via SPEC-LIT §136.
//! No GPL-licensed source was consulted.

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::field::{GpuScalarField, GpuSurfaceScalarField};
use crate::fv::{self, DivScheme, FvKernels};
use crate::ldu::GpuLduMatrix;
use crate::ldu_ops::{self, LduKernels};
use crate::mesh::{GpuMesh, HostMesh, PatchKind};
use crate::radiation::{RadiationModel, SIGMA_SB};
use crate::solver::{
    self, LinearSolverKind, Preconditioner, SolverControls, SolverKernels, SolverWorkspace,
};
use crate::{Acc, Label, Scalar, Vec3};
use cudarc::driver::{CudaFunction, PushKernelArg};

use super::dom_quad::{symmetry_axis, Quadrature};
use super::p1::WallTemperature;
use super::props;

// ==========================================================================
//  The medium and the boundary conditions
// ==========================================================================

/// The uniform grey medium the discrete-ordinates rows solve for: the
/// absorption coefficient `kappa` (1/m - `kappa = 0` IS accepted, a
/// transparent medium's row is finite, unlike P1's), the scattering
/// coefficient `sigma_s` (1/m, `>= 0`; the in-scattering sum is lagged
/// source iteration, CHR-21, SPEC-LIT §136.8), and the linear-anisotropic
/// phase function's `a1` (dimensionless, `|a1| <= 1`; 0 is isotropic).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DomMedium {
    pub kappa: f64,
    pub sigma_s: f64,
    pub a1: f64,
}

impl DomMedium {
    /// Read a participating medium [`DomRows`] can solve: the model must be
    /// [`RadiationModel::FvDom`] and the absorption a constant grey `kappa`
    /// (the WSGG sets are CHR-22's superposition of grey solves). Every
    /// refusal names the entry it is about and the § it breaks.
    pub fn from_config(c: &props::ParticipatingConfig) -> Result<Self> {
        if c.model != RadiationModel::FvDom {
            return Err(Error::Config(format!(
                "radiationModel {} is not fvDOM: radiation::dom::DomMedium \
                 reads a discrete-ordinates medium, and {} is the P1 solve's \
                 medium (SPEC-LIT §136.2)",
                c.model.name(),
                c.model.name()
            )));
        }
        let kappa = match &c.absorption {
            props::Absorption::Constant { kappa } => *kappa,
            other => {
                return Err(Error::Config(format!(
                    "absorptionEmissionModel {} is refused: the fvDOM rows of \
                     SPEC-LIT §136.2 solve one grey medium per direction, and \
                     the WSGG sets are a superposition of grey solves CHR-22 \
                     will build - not this one",
                    other.kind().name()
                )));
            }
        };
        if !kappa.is_finite() || kappa < 0.0 {
            return Err(Error::Config(format!(
                "absorptionCoefficient: {kappa} is refused - kappa must be \
                 non-negative and finite (SPEC-LIT §136.2)"
            )));
        }
        Ok(Self { kappa, sigma_s: c.scatter.sigma(), a1: c.scatter.a1() })
    }
}

/// One boundary condition of SPEC-LIT §136.2's table, as a case names it per
/// patch.
#[derive(Debug, Clone, PartialEq)]
pub enum DomCondition {
    /// A grey diffuse wall: `I_w = eps sigma T_w^4/pi + ((1 - eps)/pi) sum
    /// w_l' I_P^l' |Omega_l' . n|`, the sum over the directions into the
    /// wall (FDS eq. RTEbc's discrete form). The wall's temperature is
    /// stated the two ways [`WallTemperature`] names.
    GreyDiffuse {
        /// The grey diffuse emissivity, in `(0, 1]`.
        emissivity: f64,
        /// `T_w` from the wall's own `T` patch value, or stated.
        temperature: WallTemperature,
    },
    /// An open boundary, black at the patch's ambient temperature (FDS:
    /// "open boundaries are treated as black walls"):
    /// `I_b = sigma t_inf^4/pi`, the same for every direction.
    Open {
        /// The ambient temperature, K.
        t_inf: f64,
    },
    /// A specular symmetry plane: `I_b^l = I_P^(mirror(axis, l))`, exact
    /// only when the face normal is a coordinate axis (SPEC-LIT §136.2) -
    /// [`DomRows::new`] checks every symmetry face and refuses the first
    /// one that is off-axis.
    Symmetry,
}

/// One patch's condition: the patch it owns and how its faces behave.
#[derive(Debug, Clone, PartialEq)]
pub struct DomPatch {
    pub patch: String,
    pub condition: DomCondition,
}

/// The per-face condition kind the boundary-intensity kernel switches on.
const KIND_WALL: Label = 0;
const KIND_OPEN: Label = 1;
const KIND_SYMMETRY: Label = 2;
/// An `empty` face - it contributes nothing anywhere.
const KIND_NONE: Label = 3;

// ==========================================================================
//  The device kernels of cuda/radiation.cu
// ==========================================================================

/// The seven row kernels of `cuda/radiation.cu`, loaded once per [`DomRows`].
struct DomKernels {
    flux_internal: CudaFunction,
    flux_boundary: CudaFunction,
    refl_accum: CudaFunction,
    boundary_intensity: CudaFunction,
    set_triple: CudaFunction,
    emission: CudaFunction,
    in_scatter: CudaFunction,
}

impl DomKernels {
    fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::RADIATION)?;
        Ok(Self {
            flux_internal: k.func("domFluxInternal")?,
            flux_boundary: k.func("domFluxBoundary")?,
            refl_accum: k.func("domReflAccum")?,
            boundary_intensity: k.func("domBoundaryIntensity")?,
            set_triple: k.func("domSetTriple")?,
            emission: k.func("domEmission")?,
            in_scatter: k.func("domInScatter")?,
        })
    }
}

// ==========================================================================
//  The rows
// ==========================================================================

/// One medium, one mesh, one quadrature: the per-direction intensity fields,
/// the per-direction boundary intensities, and every scratch buffer the
/// boundary update and the assembly need, allocated once so both are launch
/// sequences only and capture into a CUDA graph (SPEC-LIT §81).
///
/// The quadrature's numbers are cast to [`Scalar`] ONCE, here at
/// construction, and handed to the kernels as scalar arguments (§136.5);
/// there is no f64/f32 mixing anywhere downstream.
pub struct DomRows {
    medium: DomMedium,
    quad: Quadrature,
    n_dirs: usize,

    n_cells: usize,
    n_internal_faces: usize,
    n_bf: usize,

    /// `[n_dirs]` the unit direction `Omega_l`, cast once.
    omega_host: Vec<[Scalar; 3]>,
    /// `[n_dirs]` its weight `w = p pi/2`, cast once.
    weight_host: Vec<Scalar>,
    /// `[n_dirs]` the mirror direction per axis: `mirror(a, l)` at
    /// `[l][a]` - §136.6's XOR map, precomputed.
    mirror_host: Vec<[usize; 3]>,

    /// `[n_bf]` the condition kind per face - the kernel's switch.
    cond_kind: DevBuf<Label>,
    /// `[n_bf]` the wall emissivity, 0 on every other face.
    eps_bf: DevBuf<Scalar>,
    /// `[n_bf]` a fixed wall's stated `T_w`, or an open boundary's ambient,
    /// 0 elsewhere.
    tw_bf: DevBuf<Scalar>,
    /// `[n_bf]` 1 where a wall states its own `T_w`, else 0.
    fixed_flag: DevBuf<Label>,
    /// `[n_bf]` a symmetry face's mirror axis, 0 elsewhere.
    sym_axis: DevBuf<Label>,

    /// `n_dirs` cell fields - one per direction, `.f` holding `I_P^l`.
    intensities: Vec<GpuScalarField>,
    /// `n_dirs` `[n_bf]` buffers - `I_b^l` of §136.6, stored for EVERY
    /// direction, incoming or not: the wall's `I_w` and the open boundary's
    /// black intensity do not depend on `l`, and the symmetry face's is
    /// `I_P^(mirror(l))`.
    boundary: Vec<DevBuf<Scalar>>,

    /// One scratch flux field `Omega_l . S_f`, reused by every direction -
    /// no per-direction face-sized storage beyond the boundary intensities
    /// the contract keeps.
    flux: GpuSurfaceScalarField,
    /// Upwind weights on the internal / boundary faces, reused per direction.
    w_if: DevBuf<Scalar>,
    bw_if: DevBuf<Scalar>,
    /// The wall rows' reflection sum, reused per update.
    refl_sum: DevBuf<Scalar>,
    /// `[n_cells]` the extinction `beta = kappa + sigma_s`, uniform -
    /// [`crate::fv::fvm_sp`]'s per-cell operand.
    beta_cells: DevBuf<Scalar>,
    /// `[n_cells]` the emission `kappa sigma T^4/pi`, rewritten per assemble,
    /// then the in-scattering sum per [`DomRows::add_in_scatter`].
    su: DevBuf<Scalar>,

    k: DomKernels,
    fvk: FvKernels,
}

impl DomRows {
    /// Build the rows for `medium` on `hm`/`m` with the quadrature `quad`,
    /// and one of §136.2's three conditions on every named patch. Every
    /// non-empty patch with at least one face must be named exactly once -
    /// an unnamed patch has no incoming intensity and is refused by name,
    /// never defaulted - and a cyclic, processor or conjugate-interface
    /// patch is refused by name in v1 (SPEC-LIT §136.6, "Who must be
    /// named"). A zero-face patch and an `empty` one may stay unnamed.
    ///
    /// Every refusal is an [`Error::Config`] naming the entry it is about
    /// and the § it breaks; nothing is substituted.
    pub fn new(
        gpu: &Gpu,
        hm: &HostMesh,
        m: &GpuMesh,
        medium: &DomMedium,
        quad: &Quadrature,
        patches: &[DomPatch],
    ) -> Result<Self> {
        let DomMedium { kappa, sigma_s, a1 } = *medium;
        // (136.3) needs a finite, non-negative beta; kappa = 0 is a
        // transparent medium and its row is fine - unlike the P1 equation,
        // whose level has nothing else to hold it.
        if !kappa.is_finite() || kappa < 0.0 {
            return Err(Error::Config(format!(
                "radiation::dom::DomMedium: kappa {kappa} is refused - kappa \
                 must be non-negative and finite (SPEC-LIT §136.2)"
            )));
        }
        if !sigma_s.is_finite() || sigma_s < 0.0 {
            return Err(Error::Config(format!(
                "radiation::dom::DomMedium: sigma_s {sigma_s} is refused - \
                 the scattering coefficient must be non-negative and finite \
                 (SPEC-LIT §136.2)"
            )));
        }
        if a1.abs() > 1.0 {
            return Err(Error::Config(format!(
                "radiation::dom::DomMedium: a1 {a1} is refused - |A1| > 1 \
                 makes the phase function Phi = 1 + A1 Omega' . Omega \
                 negative for some pair of directions (SPEC-LIT §135.1)"
            )));
        }
        let (cond, eps, tw, fixf, axis) = Self::face_conditions(hm, patches)?;

        // The quadrature, cast to Scalar once (SPEC-LIT §136.5): the
        // directions and their weights travel to the kernels as scalar
        // arguments, the mirror map as precomputed indices.
        let n_dirs = quad.len();
        let omega_host: Vec<[Scalar; 3]> = quad
            .dirs()
            .iter()
            .map(|d| [d[0] as Scalar, d[1] as Scalar, d[2] as Scalar])
            .collect();
        let weight_host: Vec<Scalar> =
            quad.weights().iter().map(|&v| v as Scalar).collect();
        let mut mirror_host = vec![[0usize; 3]; n_dirs];
        for a in 0..3 {
            for l in 0..n_dirs {
                mirror_host[l][a] = quad.mirror(a, l);
            }
        }
        let mut intensities = Vec::with_capacity(n_dirs);
        let mut boundary = Vec::with_capacity(n_dirs);
        for l in 0..n_dirs {
            intensities.push(GpuScalarField::zeros(gpu, m, &format!("I{l}"))?);
            boundary.push(gpu.zeros(hm.n_boundary_faces)?);
        }
        // beta = kappa + sigma_s, formed in f64 and cast once: with
        // sigma_s = 0 it is kappa bitwise, so a non-scattering medium's rows
        // are CHR-19's exactly.
        let beta_h = vec![(kappa + sigma_s) as Scalar; hm.n_cells];
        Ok(Self {
            medium: *medium,
            quad: quad.clone(),
            n_dirs,
            n_cells: hm.n_cells,
            n_internal_faces: m.n_internal_faces,
            n_bf: hm.n_boundary_faces,
            omega_host,
            weight_host,
            mirror_host,
            cond_kind: gpu.upload(&cond)?,
            eps_bf: gpu.upload(&eps)?,
            tw_bf: gpu.upload(&tw)?,
            fixed_flag: gpu.upload(&fixf)?,
            sym_axis: gpu.upload(&axis)?,
            intensities,
            boundary,
            flux: GpuSurfaceScalarField::zeros(gpu, m, "OmegaSf")?,
            w_if: gpu.zeros(m.n_internal_faces.max(1))?,
            bw_if: gpu.zeros(hm.n_boundary_faces.max(1))?,
            refl_sum: gpu.zeros(hm.n_boundary_faces)?,
            beta_cells: gpu.upload(&beta_h)?,
            su: gpu.zeros(hm.n_cells)?,
            k: DomKernels::new(gpu)?,
            fvk: FvKernels::new(gpu)?,
        })
    }

    /// Validate the named patches against the mesh and return the per-face
    /// condition data: the kind, the emissivity, the stated `T_w` or
    /// ambient, the fixed-`T_w` flag, and the symmetry faces' mirror axes.
    /// Each patch must exist, must not be named twice, must not be `empty`
    /// or coupled, and its numbers must be physical; every symmetry face's
    /// axis is checked against `hm.b_sf` HERE, in f64, and the first
    /// offending face is refused naming its patch (SPEC-LIT §136.2).
    fn face_conditions(
        hm: &HostMesh,
        patches: &[DomPatch],
    ) -> Result<(Vec<Label>, Vec<Scalar>, Vec<Scalar>, Vec<Label>, Vec<Label>)> {
        let n_bf = hm.n_boundary_faces;
        let mut cond = vec![KIND_NONE; n_bf];
        let mut eps = vec![0.0 as Scalar; n_bf];
        let mut tw = vec![0.0 as Scalar; n_bf];
        let mut fixf = vec![0 as Label; n_bf];
        let mut axis = vec![0 as Label; n_bf];
        let mut seen: Vec<usize> = Vec::new();
        for dp in patches {
            match &dp.condition {
                DomCondition::GreyDiffuse { emissivity, temperature } => {
                    if !emissivity.is_finite() || *emissivity <= 0.0 || *emissivity > 1.0 {
                        return Err(Error::Config(format!(
                            "DomPatch \"{}\": emissivity {emissivity} is \
                             refused - eps must lie in (0, 1] (SPEC-LIT \
                             §136.2)",
                            dp.patch
                        )));
                    }
                    if let WallTemperature::Fixed(twv) = temperature {
                        if !twv.is_finite() || *twv < 0.0 {
                            return Err(Error::Config(format!(
                                "DomPatch \"{}\": fixed T_w {twv} is refused \
                                 - a wall temperature must be non-negative \
                                 and finite (SPEC-LIT §136.2)",
                                dp.patch
                            )));
                        }
                    }
                }
                DomCondition::Open { t_inf } => {
                    if !t_inf.is_finite() || *t_inf < 0.0 {
                        return Err(Error::Config(format!(
                            "DomPatch \"{}\": t_inf {t_inf} is refused - an \
                             open boundary's ambient temperature must be \
                             non-negative and finite (SPEC-LIT §136.2)",
                            dp.patch
                        )));
                    }
                }
                DomCondition::Symmetry => {}
            }
            let (pi, p) = hm
                .patches
                .iter()
                .enumerate()
                .find(|(_, p)| p.name == dp.patch)
                .ok_or_else(|| {
                    Error::Config(format!(
                        "DomPatch: no patch named \"{}\"; the mesh has {}",
                        dp.patch,
                        hm.patches
                            .iter()
                            .map(|p| p.name.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))
                })?;
            if seen.contains(&pi) {
                return Err(Error::Config(format!(
                    "DomPatch \"{}\" is named twice - one patch carries one \
                     condition of SPEC-LIT §136.2's table",
                    dp.patch
                )));
            }
            seen.push(pi);
            if p.kind == PatchKind::Empty {
                return Err(Error::Config(format!(
                    "DomPatch \"{}\": {} face(s) of kind empty - a 2-D ghost \
                     carries no incoming intensity, so this condition would \
                     be read and then ignored (SPEC-LIT §136.6)",
                    dp.patch, p.size
                )));
            }
            if matches!(
                p.kind,
                PatchKind::Cyclic | PatchKind::Processor | PatchKind::Interface
            ) {
                return Err(Error::Config(format!(
                    "DomPatch \"{}\" is kind {} - a coupled face would need \
                     the neighbour's intensity of the same direction, which \
                     is named and not built (SPEC-LIT §136.6)",
                    dp.patch,
                    p.kind.as_str()
                )));
            }
            for bf in p.start..p.start + p.size {
                match &dp.condition {
                    DomCondition::GreyDiffuse { emissivity, temperature } => {
                        cond[bf] = KIND_WALL;
                        eps[bf] = *emissivity as Scalar;
                        if let WallTemperature::Fixed(twv) = temperature {
                            tw[bf] = *twv as Scalar;
                            fixf[bf] = 1;
                        }
                    }
                    DomCondition::Open { t_inf } => {
                        cond[bf] = KIND_OPEN;
                        tw[bf] = *t_inf as Scalar;
                    }
                    DomCondition::Symmetry => {
                        let n = hm.b_sf[bf];
                        let a = symmetry_axis([n.x as f64, n.y as f64, n.z as f64])
                            .map_err(|e| {
                                Error::Config(format!(
                                    "DomPatch \"{}\", face {bf}: {e}",
                                    dp.patch
                                ))
                            })?;
                        axis[bf] = a as Label;
                        cond[bf] = KIND_SYMMETRY;
                    }
                }
            }
        }
        // Every coupled patch is refused wherever it appears, named or not;
        // every other non-empty patch with at least one face must have been
        // named above. A zero-face patch and an empty one may stay unnamed.
        for (pi, p) in hm.patches.iter().enumerate() {
            if matches!(
                p.kind,
                PatchKind::Cyclic | PatchKind::Processor | PatchKind::Interface
            ) {
                return Err(Error::Config(format!(
                    "DomRows::new: patch \"{}\" is kind {} - a coupled face \
                     would need the neighbour's intensity of the same \
                     direction, which is named and not built (SPEC-LIT \
                     §136.6)",
                    p.name,
                    p.kind.as_str()
                )));
            }
            if p.kind == PatchKind::Empty || p.size == 0 || seen.contains(&pi) {
                continue;
            }
            return Err(Error::Config(format!(
                "DomRows::new: no condition names patch \"{}\" ({} face(s) \
                 of kind {}) - a patch with faces and no named condition has \
                 no incoming intensity, and it is refused rather than \
                 defaulted (SPEC-LIT §136.6)",
                p.name,
                p.size,
                p.kind.as_str()
            )));
        }
        Ok((cond, eps, tw, fixf, axis))
    }

    /// The step addresses one mesh only: its buffers were sized at
    /// [`DomRows::new`] and a captured graph bakes the launches' grids. A
    /// different mesh is refused by name, never silently substituted.
    fn check_mesh(
        m: &GpuMesh,
        n_cells: usize,
        n_if: usize,
        n_bf: usize,
        who: &str,
    ) -> Result<()> {
        if m.n_cells != n_cells
            || m.n_internal_faces != n_if
            || m.n_boundary_faces != n_bf
        {
            return Err(Error::Config(format!(
                "{who}: the mesh handed in ({} cells, {} internal faces, {} \
                 boundary faces) is not the one DomRows::new was built on \
                 ({} cells, {} internal faces, {} boundary faces) - the \
                 step's buffers and its captured graph address one mesh only \
                 (SPEC-LIT §81)",
                m.n_cells,
                m.n_internal_faces,
                m.n_boundary_faces,
                n_cells,
                n_if,
                n_bf
            )));
        }
        Ok(())
    }

    /// §136.2's boundary intensity of EVERY direction, from the current cell
    /// intensities and `t`: the wall's `I_w` (with its reflection sum over
    /// the directions into the wall - the boundary values lag one sweep
    /// behind the rows, as the in-scattering sum does), the open boundary's
    /// black intensity, the symmetry face's mirror image. Pure launches: no
    /// allocation, no download, no host write, so the whole step captures
    /// into a CUDA graph (SPEC-LIT §81).
    pub fn update_boundaries(
        &mut self,
        gpu: &Gpu,
        m: &GpuMesh,
        t: &GpuScalarField,
    ) -> Result<()> {
        Self::check_mesh(
            m,
            self.n_cells,
            self.n_internal_faces,
            self.n_bf,
            "DomRows::update_boundaries",
        )?;
        if self.n_bf == 0 {
            return Ok(());
        }
        let nbl = self.n_bf as Label;
        let sigma = SIGMA_SB as Scalar;
        let pi = std::f64::consts::PI as Scalar;
        // The wall rows' reflection sum, direction by direction, into one
        // reused scratch (SPEC-LIT §136.2's wall row).
        gpu.fill_zero(&mut self.refl_sum)?;
        for (l, w) in self.weight_host.iter().enumerate() {
            let [ox, oy, oz] = self.omega_host[l];
            unsafe {
                gpu.stream()
                    .launch_builder(&self.k.refl_accum)
                    .arg(&mut self.refl_sum)
                    .arg(&self.intensities[l].f)
                    .arg(&m.b_sf)
                    .arg(&m.b_face_cells)
                    .arg(&ox)
                    .arg(&oy)
                    .arg(&oz)
                    .arg(w)
                    .arg(&nbl)
                    .launch(cfg_for(self.n_bf))?;
            }
        }
        // The per-direction intensities: the wall's I_w and the open
        // boundary's black intensity are the same for every direction, the
        // symmetry face reads its mirror's cells.
        for l in 0..self.n_dirs {
            let [m0, m1, m2] = self.mirror_host[l];
            unsafe {
                gpu.stream()
                    .launch_builder(&self.k.boundary_intensity)
                    .arg(&mut self.boundary[l])
                    .arg(&self.refl_sum)
                    .arg(&self.cond_kind)
                    .arg(&self.eps_bf)
                    .arg(&self.tw_bf)
                    .arg(&self.fixed_flag)
                    .arg(&t.bf)
                    .arg(&self.sym_axis)
                    .arg(&self.intensities[m0].f)
                    .arg(&self.intensities[m1].f)
                    .arg(&self.intensities[m2].f)
                    .arg(&m.b_face_cells)
                    .arg(&m.b_mag_sf)
                    .arg(&sigma)
                    .arg(&pi)
                    .arg(&nbl)
                    .launch(cfg_for(self.n_bf))?;
            }
        }
        Ok(())
    }

    /// Zero `a`, then assemble direction `l`'s row (136.3) into it with the
    /// boundary values of the last [`DomRows::update_boundaries`]: after it,
    /// `a` is ready for [`crate::ldu_ops::add_boundary_contributions`] - the
    /// caller folds the boundary pair into `diag`/`source`, exactly as every
    /// other equation in this crate does - and then a solve. The Gauss
    /// upwind of the face flux `phi_f = Omega_l . S_f` IS (136.3)'s
    /// convection (SPEC-LIT §136.6, "The row on the crate's operators");
    /// `fvm_sp` adds the implicit sink `beta V` (`beta = kappa + sigma_s`)
    /// and `fvm_su` the emission `kappa sigma T_P^4/pi V`. A scattering
    /// medium's in-scattering sum is NOT here: it needs the lagged `G` and
    /// `q`, and [`DomRows::add_in_scatter`] adds it after this call. Pure
    /// launches: no allocation, no download, no host write (SPEC-LIT §81).
    pub fn assemble(
        &mut self,
        gpu: &Gpu,
        m: &GpuMesh,
        t: &GpuScalarField,
        l: usize,
        a: &mut GpuLduMatrix,
    ) -> Result<()> {
        Self::check_mesh(
            m,
            self.n_cells,
            self.n_internal_faces,
            self.n_bf,
            "DomRows::assemble",
        )?;
        if l >= self.n_dirs {
            return Err(Error::Config(format!(
                "DomRows::assemble: direction {l} is out of range - the \
                 quadrature has {} directions (SPEC-LIT §136.1)",
                self.n_dirs
            )));
        }
        a.zero(gpu)?;
        let [ox, oy, oz] = self.omega_host[l];
        if m.n_internal_faces > 0 {
            let nl = m.n_internal_faces as Label;
            unsafe {
                gpu.stream()
                    .launch_builder(&self.k.flux_internal)
                    .arg(&mut self.flux.f)
                    .arg(&m.sf)
                    .arg(&ox)
                    .arg(&oy)
                    .arg(&oz)
                    .arg(&nl)
                    .launch(cfg_for(m.n_internal_faces))?;
            }
        }
        if self.n_bf > 0 {
            let nbl = self.n_bf as Label;
            unsafe {
                gpu.stream()
                    .launch_builder(&self.k.flux_boundary)
                    .arg(&mut self.flux.bf)
                    .arg(&m.b_sf)
                    .arg(&ox)
                    .arg(&oy)
                    .arg(&oz)
                    .arg(&nbl)
                    .launch(cfg_for(self.n_bf))?;
                // §136.6's triple for this direction, from the boundary
                // values of the last update_boundaries.
                let i_l = &mut self.intensities[l];
                gpu.stream()
                    .launch_builder(&self.k.set_triple)
                    .arg(&mut i_l.fr)
                    .arg(&mut i_l.ref_value)
                    .arg(&self.boundary[l])
                    .arg(&self.flux.bf)
                    .arg(&nbl)
                    .launch(cfg_for(self.n_bf))?;
            }
        }
        // (136.3) on the crate's own operators: the upwind weights make
        // fvm_div_gauss write max(Omega.S_f, 0) on the diagonal side,
        // min(Omega.S_f, 0) on the off-diagonals, and put the incoming
        // faces' -min(Omega.S_b, 0) I_b^l into the triple's boundary
        // coefficients - which add_boundary_contributions folds into the
        // source below (SPEC-LIT §136.6).
        fv::div_scheme_weights(
            gpu,
            &self.fvk,
            Some(&mut self.w_if),
            Some(&mut self.bw_if),
            DivScheme::Upwind,
            &self.flux,
            &self.intensities[l],
            None,
            m,
        )?;
        fv::fvm_div_gauss(
            gpu,
            &self.fvk,
            a,
            m,
            &self.flux,
            &self.w_if,
            &self.bw_if,
            &self.intensities[l],
            1.0,
        )?;
        // The implicit extinction sink beta V, beta = kappa + sigma_s.
        fv::fvm_sp(gpu, &self.fvk, a, m, &self.beta_cells, 1.0)?;
        // The emission kappa sigma T_P^4/pi per cell; fvm_su multiplies the
        // cell volume in.
        if self.n_cells > 0 {
            let nl = self.n_cells as Label;
            let kappa = self.medium.kappa as Scalar;
            let sigma = SIGMA_SB as Scalar;
            let pi = std::f64::consts::PI as Scalar;
            unsafe {
                gpu.stream()
                    .launch_builder(&self.k.emission)
                    .arg(&mut self.su)
                    .arg(&t.f)
                    .arg(&kappa)
                    .arg(&sigma)
                    .arg(&pi)
                    .arg(&nl)
                    .launch(cfg_for(self.n_cells))?;
            }
        }
        fv::fvm_su(gpu, &self.fvk, a, m, &self.su, 1.0)?;
        Ok(())
    }

    /// The quadrature the rows were built with - f64 on the host, the
    /// transcription and moment checks' subject (SPEC-LIT §136.5).
    pub fn quadrature(&self) -> &Quadrature {
        &self.quad
    }

    /// The number of directions: 8, 24, 48, 80.
    pub fn n_dirs(&self) -> usize {
        self.n_dirs
    }

    /// Direction `l`'s intensity field - `I_P^l` in `.f`.
    pub fn intensity(&self, l: usize) -> &GpuScalarField {
        &self.intensities[l]
    }

    /// Overwrite direction `l`'s cell intensities. Setup only - a host
    /// write, never called inside a capture or the time loop ([`DomSolve`]'s
    /// sweep writes `I^l` on the device, SPEC-LIT §136.7).
    pub fn set_intensity(&mut self, gpu: &Gpu, l: usize, cells: &[Scalar]) -> Result<()> {
        gpu.write(&mut self.intensities[l].f, cells)
    }

    /// `[n_bf]` `I_b^l` of §136.6 - the values of the last
    /// [`DomRows::update_boundaries`].
    pub fn boundary_intensity(&self, l: usize) -> &DevBuf<Scalar> {
        &self.boundary[l]
    }

    /// Add (136.3)'s in-scattering sum to direction `l`'s row, which
    /// [`DomRows::assemble`] has just built in `a`: the source gains
    /// `(sigma_s/4pi) (G_P + A_1 Omega_l . q_P) V_P`, with `g` and `q` the
    /// LAGGED `G` and `q` of SPEC-LIT §136.2's source iteration (the previous
    /// sweep's, §136.8). For `linearAnisotropic`, `sum_l' w_l' Phi I^l'` is
    /// exactly `G + A_1 Omega_l . q` (§136.2), so no per-pair phase matrix
    /// exists. A non-scattering medium (`sigma_s == 0`) launches nothing, so
    /// its rows are CHR-19's bitwise. Pure launches (SPEC-LIT §81).
    pub fn add_in_scatter(
        &mut self,
        gpu: &Gpu,
        m: &GpuMesh,
        l: usize,
        g: &DevBuf<Scalar>,
        q: &DevBuf<Vec3>,
        a: &mut GpuLduMatrix,
    ) -> Result<()> {
        Self::check_mesh(
            m,
            self.n_cells,
            self.n_internal_faces,
            self.n_bf,
            "DomRows::add_in_scatter",
        )?;
        if l >= self.n_dirs {
            return Err(Error::Config(format!(
                "DomRows::add_in_scatter: direction {l} is out of range - the \
                 quadrature has {} directions (SPEC-LIT §136.1)",
                self.n_dirs
            )));
        }
        if g.len() != self.n_cells || q.len() != self.n_cells {
            return Err(Error::Config(format!(
                "DomRows::add_in_scatter: G has {} and q has {} entries, the \
                 mesh {} cells (SPEC-LIT §136.2)",
                g.len(),
                q.len(),
                self.n_cells
            )));
        }
        if self.medium.sigma_s == 0.0 || self.n_cells == 0 {
            return Ok(());
        }
        // The two coefficients in f64, cast once (§136.5's rule for every
        // number the host hands the kernels).
        let four_pi = 4.0 * std::f64::consts::PI;
        let c0 = (self.medium.sigma_s / four_pi) as Scalar;
        let c1 = (self.medium.sigma_s * self.medium.a1 / four_pi) as Scalar;
        let [ox, oy, oz] = self.omega_host[l];
        let nl = self.n_cells as Label;
        unsafe {
            gpu.stream()
                .launch_builder(&self.k.in_scatter)
                .arg(&mut self.su)
                .arg(g)
                .arg(q)
                .arg(&ox)
                .arg(&oy)
                .arg(&oz)
                .arg(&c0)
                .arg(&c1)
                .arg(&nl)
                .launch(cfg_for(self.n_cells))?;
        }
        fv::fvm_su(gpu, &self.fvk, a, m, &self.su, 1.0)
    }
}

// ==========================================================================
//  The solve (CHR-20)
// ==========================================================================

/// SPEC-LIT §136.7's default sweep tolerance on `change`: 1e-8 in f64, its
/// D-F32-1 `tol32` of 1e-5 under `single` - an f32 `G` cannot resolve a
/// change of 1e-8.
pub const DEFAULT_SWEEP_TOL: f64 = if cfg!(feature = "single") { 1e-5 } else { 1e-8 };

/// SPEC-LIT §136.7's default sweep cap (§136.2's `maxScatterIterations`).
pub const DEFAULT_MAX_SWEEPS: usize = 200;

/// The controls of one [`DomSolve`]: the per-direction linear solve, and the
/// sweep loop's tolerance on `change` and its cap (SPEC-LIT §136.7).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DomControls {
    pub linear: SolverControls,
    pub tol: f64,
    pub max_sweeps: usize,
}

impl Default for DomControls {
    fn default() -> Self {
        Self {
            linear: SolverControls::default(),
            tol: DEFAULT_SWEEP_TOL,
            max_sweeps: DEFAULT_MAX_SWEEPS,
        }
    }
}

/// What [`DomSolve::solve`] did (SPEC-LIT §136.7).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DomReport {
    /// Sweeps run, at least 1.
    pub sweeps: usize,
    /// The last sweep's `max_P |G_P - G_P^prev| / max_P |G_P|`.
    pub change: f64,
    /// `change < tol`.
    pub converged: bool,
    /// The direction solves of the LAST sweep whose
    /// [`solver::SolverPerformance::converged`] is false.
    pub unconverged_solves: usize,
}

/// The solve's four sum kernels of `cuda/radiation.cu` (§136.7 step 3).
struct DomSolveKernels {
    accum_cells: CudaFunction,
    accum_wall: CudaFunction,
    round_cells: CudaFunction,
    round_faces: CudaFunction,
}

impl DomSolveKernels {
    fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::RADIATION)?;
        Ok(Self {
            accum_cells: k.func("domAccumCells")?,
            accum_wall: k.func("domAccumWall")?,
            round_cells: k.func("domRoundCells")?,
            round_faces: k.func("domRoundFaces")?,
        })
    }
}

/// The fvDOM solve of a grey medium (SPEC-LIT §136.7), scattering by lagged
/// source iteration (§136.2, CHR-21's §136.8): the
/// rows of [`DomRows`], one matrix reused by every direction, the linear
/// solver's workspace, and (136.4)'s outputs `G`, `q` and wall flux with
/// their `ofacc` accumulators - every buffer allocated in [`DomSolve::new`],
/// so a sweep is launches only.
pub struct DomSolve {
    rows: DomRows,
    a: GpuLduMatrix,
    ws: SolverWorkspace,
    lduk: LduKernels,
    solk: SolverKernels,
    k: DomSolveKernels,
    ctrl: DomControls,

    /// `[n_cells]` `G_P` of (136.4), rounded once per sweep.
    g: DevBuf<Scalar>,
    /// `[n_cells]` `q_P` of (136.4).
    q: DevBuf<Vec3>,
    /// `[n_bf]` `q_w . n` of (136.4); 0 on empty faces.
    wall_flux: DevBuf<Scalar>,
    /// The `ofacc` accumulators: `[n_cells]`, `[3 n_cells]` (x, y, z per
    /// cell) and `[n_bf]` (§118.1).
    g_acc: DevBuf<Acc>,
    q_acc: DevBuf<Acc>,
    wf_acc: DevBuf<Acc>,
}

impl DomSolve {
    /// Build the rows ([`DomRows::new`], with all its refusals) and the
    /// solve around them. The linear controls are refused by name when they
    /// name `PCG` or the `DIC` preconditioner - the step-scheme rows are
    /// asymmetric (SPEC-LIT §136.2) - or `GAMG` as solver or preconditioner
    /// (named and not built for these rows, §136.7); so are a `tol` that is
    /// not finite and positive and a zero sweep cap.
    pub fn new(
        gpu: &Gpu,
        hm: &HostMesh,
        m: &GpuMesh,
        medium: &DomMedium,
        quad: &Quadrature,
        patches: &[DomPatch],
        ctrl: DomControls,
    ) -> Result<Self> {
        match ctrl.linear.solver {
            LinearSolverKind::PCG => {
                return Err(Error::Config(
                    "DomSolve: solver PCG is refused - the step-scheme rows are \
                     asymmetric (upwind), and conjugate gradients need a \
                     symmetric matrix (SPEC-LIT §136.2); use PBiCGStab"
                        .into(),
                ));
            }
            LinearSolverKind::Gamg => {
                return Err(Error::Config(
                    "DomSolve: solver GAMG is named and not built for the fvDOM \
                     rows (SPEC-LIT §136.7); use PBiCGStab"
                        .into(),
                ));
            }
            LinearSolverKind::PBiCGStab => {}
        }
        match ctrl.linear.precon {
            Preconditioner::Dic => {
                return Err(Error::Config(
                    "DomSolve: preconditioner DIC is refused - the step-scheme \
                     rows are asymmetric (upwind) and have no Cholesky factor \
                     (SPEC-LIT §136.2); use diagonal or DILU"
                        .into(),
                ));
            }
            Preconditioner::Gamg => {
                return Err(Error::Config(
                    "DomSolve: preconditioner GAMG is named and not built for \
                     the fvDOM rows (SPEC-LIT §136.7)"
                        .into(),
                ));
            }
            Preconditioner::None | Preconditioner::Diagonal | Preconditioner::Dilu => {}
        }
        if !ctrl.tol.is_finite() || ctrl.tol <= 0.0 {
            return Err(Error::Config(format!(
                "DomSolve: sweep tol {} is refused - the tolerance on change \
                 must be finite and positive (SPEC-LIT §136.7)",
                ctrl.tol
            )));
        }
        if ctrl.max_sweeps == 0 {
            return Err(Error::Config(
                "DomSolve: max_sweeps 0 is refused - the loop runs at least \
                 one sweep (SPEC-LIT §136.7)"
                    .into(),
            ));
        }
        let rows = DomRows::new(gpu, hm, m, medium, quad, patches)?;
        let n = hm.n_cells;
        let n_bf = hm.n_boundary_faces;
        Ok(Self {
            rows,
            a: GpuLduMatrix::new(gpu, m)?,
            ws: SolverWorkspace::for_mesh(gpu, m)?,
            lduk: LduKernels::new(gpu)?,
            solk: SolverKernels::new(gpu)?,
            k: DomSolveKernels::new(gpu)?,
            ctrl,
            g: gpu.zeros(n)?,
            q: gpu.zeros(n)?,
            wall_flux: gpu.zeros(n_bf)?,
            g_acc: gpu.zeros(n)?,
            q_acc: gpu.zeros(3 * n)?,
            wf_acc: gpu.zeros(n_bf)?,
        })
    }

    /// One sweep, SPEC-LIT §136.7 steps 1-4: the boundary intensities from
    /// the current `I^l`; every direction assembled, given the in-scattering
    /// sum of the previous sweep's `G` and `q` (§136.8; nothing when
    /// `sigma_s = 0`), folded and solved into `I^l` (warm-started from it);
    /// then (136.4)'s `G`, `q` and wall flux
    /// summed in `ofacc` from the boundary values the rows were assembled
    /// with, and rounded once. Returns the number of direction solves that
    /// did not report convergence. Launches only - no host read or write -
    /// when the linear controls are `fixed_iters`, so it captures (§81).
    pub fn sweep(&mut self, gpu: &Gpu, m: &GpuMesh, t: &GpuScalarField) -> Result<usize> {
        let (n, n_if, n_bf) = (self.rows.n_cells, self.rows.n_internal_faces, self.rows.n_bf);
        DomRows::check_mesh(m, n, n_if, n_bf, "DomSolve::sweep")?;
        // Step 1.
        self.rows.update_boundaries(gpu, m, t)?;
        // Step 2.
        let mut unconverged = 0usize;
        for l in 0..self.rows.n_dirs {
            self.rows.assemble(gpu, m, t, l, &mut self.a)?;
            // The lagged in-scattering sum from the previous sweep's G and q
            // (still in self.g/self.q: step 3 rewrites them only after every
            // direction is solved) - SPEC-LIT §136.2's source iteration.
            self.rows.add_in_scatter(gpu, m, l, &self.g, &self.q, &mut self.a)?;
            ldu_ops::add_boundary_contributions(gpu, &self.lduk, &mut self.a, m)?;
            let perf = solver::solve(
                gpu,
                &self.solk,
                &mut self.rows.intensities[l].f,
                &self.a,
                m,
                &mut self.ws,
                &self.ctrl.linear,
            )?;
            if !perf.converged {
                unconverged += 1;
            }
        }
        // Steps 3 and 4: (136.4) over the directions, in ofacc.
        gpu.fill_zero(&mut self.g_acc)?;
        gpu.fill_zero(&mut self.q_acc)?;
        gpu.fill_zero(&mut self.wf_acc)?;
        let nl = n as Label;
        let nbl = n_bf as Label;
        for l in 0..self.rows.n_dirs {
            let [ox, oy, oz] = self.rows.omega_host[l];
            let w = self.rows.weight_host[l];
            if n > 0 {
                unsafe {
                    gpu.stream()
                        .launch_builder(&self.k.accum_cells)
                        .arg(&mut self.g_acc)
                        .arg(&mut self.q_acc)
                        .arg(&self.rows.intensities[l].f)
                        .arg(&ox)
                        .arg(&oy)
                        .arg(&oz)
                        .arg(&w)
                        .arg(&nl)
                        .launch(cfg_for(n))?;
                }
            }
            if n_bf > 0 {
                unsafe {
                    gpu.stream()
                        .launch_builder(&self.k.accum_wall)
                        .arg(&mut self.wf_acc)
                        .arg(&self.rows.intensities[l].f)
                        .arg(&self.rows.boundary[l])
                        .arg(&m.b_sf)
                        .arg(&m.b_mag_sf)
                        .arg(&m.b_face_cells)
                        .arg(&self.rows.cond_kind)
                        .arg(&ox)
                        .arg(&oy)
                        .arg(&oz)
                        .arg(&w)
                        .arg(&nbl)
                        .launch(cfg_for(n_bf))?;
                }
            }
        }
        if n > 0 {
            unsafe {
                gpu.stream()
                    .launch_builder(&self.k.round_cells)
                    .arg(&mut self.g)
                    .arg(&mut self.q)
                    .arg(&self.g_acc)
                    .arg(&self.q_acc)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
        }
        if n_bf > 0 {
            unsafe {
                gpu.stream()
                    .launch_builder(&self.k.round_faces)
                    .arg(&mut self.wall_flux)
                    .arg(&self.wf_acc)
                    .arg(&nbl)
                    .launch(cfg_for(n_bf))?;
            }
        }
        Ok(unconverged)
    }

    /// Sweep until `change = max_P |G_P - G_P^prev| / max_P |G_P| < tol`
    /// (0 when both maxima are 0; the first sweep starts from `G = 0`), or
    /// until `max_sweeps` (SPEC-LIT §136.7). Reaching the cap prints one
    /// line and returns `converged: false` - never an `Err`, never silent;
    /// stopping on `change` while some direction solve of the last sweep
    /// did not report convergence prints one line too.
    /// It reads `G` back after every sweep, so it is refused inside a
    /// capture; capture [`DomSolve::sweep`] instead.
    pub fn solve(&mut self, gpu: &Gpu, m: &GpuMesh, t: &GpuScalarField) -> Result<DomReport> {
        if gpu.is_capturing() {
            return Err(Error::Config(
                "DomSolve::solve reads G back after every sweep and is refused \
                 inside a capture - use DomSolve::sweep, SPEC-LIT §81"
                    .into(),
            ));
        }
        let (n, n_if, n_bf) = (self.rows.n_cells, self.rows.n_internal_faces, self.rows.n_bf);
        DomRows::check_mesh(m, n, n_if, n_bf, "DomSolve::solve")?;
        let mut prev = vec![0.0f64; n];
        let mut report =
            DomReport { sweeps: 0, change: f64::INFINITY, converged: false, unconverged_solves: 0 };
        while report.sweeps < self.ctrl.max_sweeps {
            report.unconverged_solves = self.sweep(gpu, m, t)?;
            report.sweeps += 1;
            let g = gpu.download(&self.g)?;
            let mut d_max = 0.0f64;
            let mut g_max = 0.0f64;
            for (p, gv) in prev.iter_mut().zip(g.iter()) {
                let v = f64::from(*gv);
                d_max = d_max.max((v - *p).abs());
                g_max = g_max.max(v.abs());
                *p = v;
            }
            report.change = if d_max == 0.0 && g_max == 0.0 { 0.0 } else { d_max / g_max };
            if report.change < self.ctrl.tol {
                report.converged = true;
                if report.unconverged_solves > 0 {
                    eprintln!(
                        "DomSolve: G stopped changing (change {:e} < tol {:e}) after {} \
                         sweeps, but {} direction solve(s) of the last sweep did not \
                         report convergence (SPEC-LIT §136.7)",
                        report.change, self.ctrl.tol, report.sweeps, report.unconverged_solves
                    );
                }
                return Ok(report);
            }
        }
        eprintln!(
            "DomSolve: the sweep cap was reached - {} sweeps, change {:e} against tol {:e}, \
             {} direction solve(s) of the last sweep unconverged (SPEC-LIT §136.7)",
            report.sweeps, report.change, self.ctrl.tol, report.unconverged_solves
        );
        Ok(report)
    }

    /// `[n_cells]` `G_P` of (136.4), from the last sweep.
    pub fn g(&self) -> &DevBuf<Scalar> {
        &self.g
    }

    /// `[n_cells]` `q_P` of (136.4), from the last sweep.
    pub fn q(&self) -> &DevBuf<Vec3> {
        &self.q
    }

    /// `[n_bf]` `q_w . n` of (136.4), `n = S_b/|S_b|` outward from the
    /// medium; 0 on empty faces.
    pub fn wall_flux(&self) -> &DevBuf<Scalar> {
        &self.wall_flux
    }

    /// The rows the solve owns.
    pub fn rows(&self) -> &DomRows {
        &self.rows
    }

    /// The rows, mutably - setup only ([`DomRows::set_intensity`]).
    pub fn rows_mut(&mut self) -> &mut DomRows {
        &mut self.rows
    }

    /// The controls the solve was built with.
    pub fn controls(&self) -> &DomControls {
        &self.ctrl
    }
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blockgen::{self, BlockSpec, GradedAxis};
    use crate::radiation::dom_quad::SnOrder;

    /// The process's device, if the box has one: the convention of every
    /// GPU test in the crate.
    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    /// Mesh B: x in [0, 1] with 4 cells, y in [0, 0.5] with 2, z in
    /// [0, 0.25] with 1 - every extent a binary fraction, so under `single`
    /// the f32 geometry IS the f64 geometry. xMin is a wall, xMax an open
    /// patch, the four others symmetry planes.
    fn mesh_b(gpu: &Gpu) -> Result<(HostMesh, GpuMesh)> {
        let axis = |lo: f64, hi: f64, n: usize| GradedAxis {
            lo: lo as Scalar,
            hi: hi as Scalar,
            n,
            expansion: 1.0,
            two_sided: false,
        };
        let spec = BlockSpec {
            x: axis(0.0, 1.0, 4),
            y: axis(0.0, 0.5, 2),
            z: axis(0.0, 0.25, 1),
            patch_name: BlockSpec::default().patch_name,
            patch_type: ["wall", "patch", "symmetry", "symmetry", "symmetry", "symmetry"]
                .map(String::from),
            windows: Vec::new(),
            cyclic: Vec::new(),
        };
        let hm = blockgen::build_mesh(&spec)?;
        let m = GpuMesh::upload(gpu, &hm)?;
        Ok((hm, m))
    }

    fn medium() -> DomMedium {
        DomMedium { kappa: 2.0, sigma_s: 0.0, a1: 0.0 }
    }

    /// The six patches of mesh B, named: one grey diffuse wall, one open
    /// boundary, four symmetry planes.
    fn patches_b(eps: f64, tw: WallTemperature, t_inf: f64) -> Vec<DomPatch> {
        vec![
            DomPatch {
                patch: "xMin".into(),
                condition: DomCondition::GreyDiffuse { emissivity: eps, temperature: tw },
            },
            DomPatch { patch: "xMax".into(), condition: DomCondition::Open { t_inf } },
            DomPatch { patch: "yMin".into(), condition: DomCondition::Symmetry },
            DomPatch { patch: "yMax".into(), condition: DomCondition::Symmetry },
            DomPatch { patch: "zMin".into(), condition: DomCondition::Symmetry },
            DomPatch { patch: "zMax".into(), condition: DomCondition::Symmetry },
        ]
    }

    fn uniform_t(gpu: &Gpu, m: &GpuMesh, t_k: f64) -> Result<GpuScalarField> {
        let mut t = GpuScalarField::zeros(gpu, m, "T")?;
        gpu.write(&mut t.f, &vec![t_k as Scalar; m.n_cells])?;
        gpu.write(&mut t.bf, &vec![t_k as Scalar; m.n_boundary_faces])?;
        Ok(t)
    }

    /// The registry's gate (`src/capture/registry.rs`, row
    /// `src/radiation/dom.rs`), in three captures. First, the boundary update
    /// and the assembly of EVERY direction of S4 - one matrix per
    /// direction, so the compared state covers every direction's boundary
    /// intensities and every direction's assembled diag/upper/lower/source/
    /// internal_coeffs/boundary_coeffs - capture into a CUDA graph and
    /// replay bitwise against the per-launch path, on mesh B with
    /// non-uniform intensities. Second, one whole [`DomSolve::sweep`] with
    /// fixed-iteration linear controls (SPEC-LIT §136.7). Third, two sweeps
    /// of a scattering medium, the second reading the first's `G` and `q`
    /// through [`DomRows::add_in_scatter`] (§136.8).
    #[test]
    fn dom_replays_under_capture() {
        let Some(gpu) = gpu() else { return };

        let report = crate::capture::capture_replays_bitwise(
            &gpu,
            "the fvDOM boundary update and per-direction rows (SPEC-LIT 136.2)",
            || {
                let (hm, m) = mesh_b(&gpu)?;
                let quad = Quadrature::new(SnOrder::S4)?;
                let mut dom = DomRows::new(
                    &gpu,
                    &hm,
                    &m,
                    &medium(),
                    &quad,
                    &patches_b(0.5, WallTemperature::Fixed(1000.0), 1000.0),
                )?;
                for l in 0..dom.n_dirs() {
                    let cells: Vec<Scalar> = (0..hm.n_cells)
                        .map(|c| (500.0 + 37.0 * l as f64 + 11.0 * c as f64) as Scalar)
                        .collect();
                    dom.set_intensity(&gpu, l, &cells)?;
                }
                let matrices: Vec<GpuLduMatrix> = (0..dom.n_dirs())
                    .map(|_| GpuLduMatrix::new(&gpu, &m))
                    .collect::<Result<Vec<_>>>()?;
                let t = uniform_t(&gpu, &m, 1000.0)?;
                Ok((dom, matrices, m, t))
            },
            |(dom, matrices, m, t): &mut (
                DomRows,
                Vec<GpuLduMatrix>,
                GpuMesh,
                GpuScalarField,
            )| {
                dom.update_boundaries(&gpu, m, t)?;
                for (l, a) in matrices.iter_mut().enumerate() {
                    dom.assemble(&gpu, m, t, l, a)?;
                }
                Ok(())
            },
            |(dom, matrices, _m, _t): &(
                DomRows,
                Vec<GpuLduMatrix>,
                GpuMesh,
                GpuScalarField,
            )| {
                let mut state: crate::capture::Snapshot = Vec::new();
                let mut bnd = Vec::new();
                for l in 0..dom.n_dirs() {
                    bnd.extend(gpu.download(dom.boundary_intensity(l))?);
                }
                state.push(("dom boundary intensities", bnd));
                // One entry per coefficient kind, every direction
                // concatenated in order - the compared state covers every
                // direction's assembled row.
                for (ki, name) in [
                    "dom diag",
                    "dom upper",
                    "dom lower",
                    "dom source",
                    "dom internal coeffs",
                    "dom boundary coeffs",
                ]
                .into_iter()
                .enumerate()
                {
                    let mut v = Vec::new();
                    for a in matrices {
                        let buf = [
                            &a.diag,
                            &a.upper,
                            &a.lower,
                            &a.source,
                            &a.internal_coeffs,
                            &a.boundary_coeffs,
                        ][ki];
                        v.extend(gpu.download(buf)?);
                    }
                    state.push((name, v));
                }
                Ok(state)
            },
        )
        .expect("SPEC-LIT 81.7: the fvDOM boundary update and row assembly \
                 must capture and replay bitwise");
        println!("  DOM: {report}");

        // CHR-20 (SPEC-LIT §136.7): one whole DomSolve::sweep - boundary
        // update, every direction's assembly and fixed-iteration PBiCGStab
        // solve, and the ofacc sums of (136.4) - captures and replays
        // bitwise. The compared state is every I^l, every I_b^l, G, q and
        // the wall flux.
        let report = crate::capture::capture_replays_bitwise(
            &gpu,
            "the fvDOM sweep, its solves and the (136.4) sums (SPEC-LIT 136.7)",
            || {
                let (hm, m) = mesh_b(&gpu)?;
                let quad = Quadrature::new(SnOrder::S4)?;
                let ctrl = DomControls {
                    linear: SolverControls {
                        solver: LinearSolverKind::PBiCGStab,
                        precon: Preconditioner::Diagonal,
                        fixed_iters: true,
                        report_residuals: false,
                        max_iter: 20,
                        ..SolverControls::default()
                    },
                    ..DomControls::default()
                };
                let mut dom = DomSolve::new(
                    &gpu,
                    &hm,
                    &m,
                    &medium(),
                    &quad,
                    &patches_b(0.5, WallTemperature::Fixed(1000.0), 1000.0),
                    ctrl,
                )?;
                for l in 0..dom.rows().n_dirs() {
                    let cells: Vec<Scalar> = (0..hm.n_cells)
                        .map(|c| (500.0 + 37.0 * l as f64 + 11.0 * c as f64) as Scalar)
                        .collect();
                    dom.rows_mut().set_intensity(&gpu, l, &cells)?;
                }
                let t = uniform_t(&gpu, &m, 1000.0)?;
                Ok((dom, m, t))
            },
            |(dom, m, t): &mut (DomSolve, GpuMesh, GpuScalarField)| {
                dom.sweep(&gpu, m, t)?;
                Ok(())
            },
            |(dom, _m, _t): &(DomSolve, GpuMesh, GpuScalarField)| {
                let mut state: crate::capture::Snapshot = Vec::new();
                let (mut inten, mut bnd) = (Vec::new(), Vec::new());
                for l in 0..dom.rows().n_dirs() {
                    inten.extend(gpu.download(&dom.rows().intensity(l).f)?);
                    bnd.extend(gpu.download(dom.rows().boundary_intensity(l))?);
                }
                state.push(("dom sweep intensities", inten));
                state.push(("dom sweep boundary intensities", bnd));
                state.push(("dom sweep G", gpu.download(dom.g())?));
                let q = gpu.download(dom.q())?;
                state.push(("dom sweep q x", q.iter().map(|v| v.x).collect()));
                state.push(("dom sweep q y", q.iter().map(|v| v.y).collect()));
                state.push(("dom sweep q z", q.iter().map(|v| v.z).collect()));
                state.push(("dom sweep wall flux", gpu.download(dom.wall_flux())?));
                Ok(state)
            },
        )
        .expect("SPEC-LIT 81.7: the fvDOM sweep must capture and replay bitwise");
        println!("  DOM sweep: {report}");

        // CHR-21 (SPEC-LIT §136.8): a scattering medium's step is TWO
        // sweeps, so the second reads the first's non-zero G and q through
        // the in-scattering launches; the compared state is every I^l, G,
        // q and the wall flux.
        let report = crate::capture::capture_replays_bitwise(
            &gpu,
            "the scattering fvDOM sweeps and their in-scattering sums (SPEC-LIT 136.8)",
            || {
                let (hm, m) = mesh_b(&gpu)?;
                let quad = Quadrature::new(SnOrder::S4)?;
                let ctrl = DomControls {
                    linear: SolverControls {
                        solver: LinearSolverKind::PBiCGStab,
                        precon: Preconditioner::Diagonal,
                        fixed_iters: true,
                        report_residuals: false,
                        max_iter: 20,
                        ..SolverControls::default()
                    },
                    ..DomControls::default()
                };
                let dom = DomSolve::new(
                    &gpu,
                    &hm,
                    &m,
                    &DomMedium { kappa: 2.0, sigma_s: 1.0, a1: 0.5 },
                    &quad,
                    &patches_b(0.5, WallTemperature::Fixed(1000.0), 1000.0),
                    ctrl,
                )?;
                let t = uniform_t(&gpu, &m, 1000.0)?;
                Ok((dom, m, t))
            },
            |(dom, m, t): &mut (DomSolve, GpuMesh, GpuScalarField)| {
                dom.sweep(&gpu, m, t)?;
                dom.sweep(&gpu, m, t)?;
                Ok(())
            },
            |(dom, _m, _t): &(DomSolve, GpuMesh, GpuScalarField)| {
                let mut state: crate::capture::Snapshot = Vec::new();
                let mut inten = Vec::new();
                for l in 0..dom.rows().n_dirs() {
                    inten.extend(gpu.download(&dom.rows().intensity(l).f)?);
                }
                state.push(("dom scatter intensities", inten));
                state.push(("dom scatter G", gpu.download(dom.g())?));
                let q = gpu.download(dom.q())?;
                state.push(("dom scatter q x", q.iter().map(|v| v.x).collect()));
                state.push(("dom scatter q y", q.iter().map(|v| v.y).collect()));
                state.push(("dom scatter q z", q.iter().map(|v| v.z).collect()));
                state.push(("dom scatter wall flux", gpu.download(dom.wall_flux())?));
                Ok(state)
            },
        )
        .expect("SPEC-LIT 81.7: the scattering fvDOM sweeps must capture and replay bitwise");
        println!("  DOM scatter sweeps: {report}");
    }
}


