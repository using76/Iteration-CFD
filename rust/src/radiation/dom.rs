// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The discrete-ordinates rows of the fvDOM radiation model (SPEC-LIT
//! §136.2, §136.6): one first-order upwind row per direction - the step
//! scheme - with the boundary intensities of §136.2's table, for one
//! uniform grey NON-scattering medium on one mesh.
//!
//! The row is (136.3) with no in-scattering sum (scattering is CHR-21's and
//! is refused until then): `diag`-side `sum_f max(Omega.S_f, 0) + kappa V`,
//! off-diagonals `min(Omega.S_f, 0)`, source `kappa (sigma T_P^4/pi) V` plus
//! the incoming boundary faces' `-min(Omega.S_b, 0) I_b^l`. It is NOT a new
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
use crate::mesh::{GpuMesh, HostMesh, PatchKind};
use crate::radiation::{RadiationModel, SIGMA_SB};
use crate::{Label, Scalar};
use cudarc::driver::{CudaFunction, PushKernelArg};

use super::dom_quad::{symmetry_axis, Quadrature};
use super::p1::WallTemperature;
use super::props;

// ==========================================================================
//  The medium and the boundary conditions
// ==========================================================================

/// The uniform grey non-scattering medium the discrete-ordinates rows solve
/// for: the absorption coefficient `kappa` (1/m - `kappa = 0` IS accepted, a
/// transparent medium's row is finite, unlike P1's), the scattering
/// coefficient `sigma_s` (0 in this unit - the in-scattering sum is CHR-21's),
/// and the linear-anisotropic phase function's `a1` (dimensionless,
/// `|a1| <= 1`).
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

/// The six dom kernels of `cuda/radiation.cu`, loaded once per [`DomRows`].
struct DomKernels {
    flux_internal: CudaFunction,
    flux_boundary: CudaFunction,
    refl_accum: CudaFunction,
    boundary_intensity: CudaFunction,
    set_triple: CudaFunction,
    emission: CudaFunction,
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
    /// `[n_cells]` kappa, uniform - [`crate::fv::fvm_sp`]'s per-cell operand.
    kappa_cells: DevBuf<Scalar>,
    /// `[n_cells]` the emission `kappa sigma T^4/pi`, rewritten per assemble.
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
        if !sigma_s.is_finite() || sigma_s != 0.0 {
            return Err(Error::Config(format!(
                "radiation::dom::DomMedium: sigma_s {sigma_s} is refused - \
                 scattering until CHR-21: the in-scattering sum of (136.3) \
                 is lagged source iteration and is not built yet, so a row \
                 with beta = kappa + sigma_s and no in-scattering source \
                 would be pure extinction, a different medium (SPEC-LIT \
                 §136.6)"
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
        let kappa_h = vec![kappa as Scalar; hm.n_cells];
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
            kappa_cells: gpu.upload(&kappa_h)?,
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
    /// `fvm_sp` adds the implicit sink `kappa V` and `fvm_su` the emission
    /// `kappa sigma T_P^4/pi V`. Pure launches: no allocation, no download,
    /// no host write (SPEC-LIT §81).
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
        // The implicit absorption sink: beta V with beta = kappa, the only
        // non-zero part of beta until CHR-21 lands the scattering sum.
        fv::fvm_sp(gpu, &self.fvk, a, m, &self.kappa_cells, 1.0)?;
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
    /// write, never called inside a capture or the time loop (a real sweep
    /// writes `I^l` on the device, which is CHR-20's business).
    pub fn set_intensity(&mut self, gpu: &Gpu, l: usize, cells: &[Scalar]) -> Result<()> {
        gpu.write(&mut self.intensities[l].f, cells)
    }

    /// `[n_bf]` `I_b^l` of §136.6 - the values of the last
    /// [`DomRows::update_boundaries`].
    pub fn boundary_intensity(&self, l: usize) -> &DevBuf<Scalar> {
        &self.boundary[l]
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
    /// `src/radiation/dom.rs`): the boundary update and the assembly of
    /// EVERY direction of S4 - one matrix per direction, so the compared
    /// state covers every direction's boundary intensities and every
    /// direction's assembled diag/upper/lower/source/internal_coeffs/
    /// boundary_coeffs - capture into a CUDA graph and replay bitwise
    /// against the per-launch path, on mesh B with non-uniform intensities.
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
    }
}


