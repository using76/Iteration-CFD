// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The P1 radiation solve (SPEC-LIT §135.3-§135.7, §137.1): one scalar
//! field `G` for a uniform grey participating medium, Marshak walls, the
//! wall flux and the cell-centred radiative flux vector, and the
//! linearised source the energy equation registers.
//!
//! The equation is (135.8), `-∇·(Γ ∇G) + κ G = 4 κ σ T^4`, with the
//! diffusion coefficient (135.7), `Γ = 1/(3β - A_1 σ_s)`, and the Marshak
//! wall condition (135.10) on every patch the case names. It is assembled
//! on the crate's own operators - [`crate::fv::fvm_laplacian`] for the
//! diffusion, [`crate::fv::fvm_sp`] for the absorption, [`crate::fv::fvm_su`]
//! for the emission - and solved with PCG, which the matrix's symmetry
//! makes legal (SPEC-LIT §8.2): the laplacian's `upper` and `lower` are the
//! same kernel's writes and the boundary pass touches only `diag` and
//! `source`, so nothing in the assembly can break the equality.
//!
//! Every number the assembly needs beyond the mesh - Γ, the face products
//! (135.13), the Marshak triples (135.14), the emission, the energy-source
//! pair (137.2), the wall flux (135.15) and the gradient's scaling to
//! `q_r` (135.16) - is computed by a kernel in `cuda/radiation.cu`, one
//! thread per cell or face. No host round-trip happens between the top of
//! [`P1::solve`] and its last launch, so the whole step captures into a
//! CUDA graph (SPEC-LIT §81); the capture gate is
//! [`tests::p1_solve_replays_under_capture`].
//!
//! No GPL-licensed source was consulted.

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::energy::EnergySources;
use crate::error::{Error, Result};
use crate::field::{BcKind, GpuScalarField};
use crate::field_ops::{self, FieldKernels};
use crate::fv::{self, FvKernels};
use crate::ldu::GpuLduMatrix;
use crate::ldu_ops::{self, LduKernels};
use crate::mesh::{GpuMesh, HostMesh, PatchKind};
use crate::radiation::SIGMA_SB;
use crate::solver::{self, SolverControls, SolverKernels, SolverPerformance, SolverWorkspace};
use crate::{Label, Scalar, Vec3};
use cudarc::driver::{CudaFunction, PushKernelArg};

use super::props;

// ==========================================================================
//  The medium and its walls
// ==========================================================================

/// How a Marshak wall's temperature is stated - SPEC-LIT §135.4's two names:
/// `MarshakRadiation` takes `T_w` from the wall's own `T` patch value,
/// `MarshakRadiationFixedTemperature` states one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WallTemperature {
    /// The wall's `T` patch value, read per solve - the wall heats and cools
    /// with the field it belongs to.
    FromField,
    /// A stated wall temperature, K, constant for the run.
    Fixed(f64),
}

/// One Marshak wall: the patch it owns, its emissivity, and how `T_w` is
/// stated (SPEC-LIT §135.4).
#[derive(Debug, Clone, PartialEq)]
pub struct MarshakPatch {
    pub patch: String,
    /// The grey diffuse emissivity `ε` of (135.10), in `(0, 1]`.
    pub emissivity: f64,
    pub temperature: WallTemperature,
}

/// The uniform grey participating medium [`P1`] solves for: the absorption
/// coefficient, the scattering coefficient and the linear-anisotropic phase
/// function's `A_1`, all 1/m (dimensionless for `A_1`) and uniform.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct P1Medium {
    pub kappa: f64,
    pub sigma_s: f64,
    pub a1: f64,
}

impl P1Medium {
    /// Read a participating medium that [`P1`] can solve: the model must be
    /// [`crate::radiation::RadiationModel::P1`], the absorption a constant
    /// grey `κ` (the WSGG sets are [`super::wsgg::WsggP1`]'s, and this solve is grey), and
    /// the scattering any of §137.2's three. Every refusal names the entry
    /// it is about.
    pub fn from_config(c: &props::ParticipatingConfig) -> Result<Self> {
        if c.model != crate::radiation::RadiationModel::P1 {
            return Err(Error::Config(format!(
                "radiationModel {} is not P1: radiation::p1::P1Medium reads a \
                 P1 medium, and {} is solved by nothing yet (SPEC-LIT §135.6)",
                c.model.name(),
                c.model.name()
            )));
        }
        let kappa = match &c.absorption {
            props::Absorption::Constant { kappa } => *kappa,
            other => {
                return Err(Error::Config(format!(
                    "absorptionEmissionModel {} is refused: the P1 solve of \
                     SPEC-LIT §135.3 is grey, and the WSGG sets are a \
                     superposition of grey solves, radiation::wsgg::WsggP1 \
                     (SPEC-LIT §136.9) - not this one",
                    other.kind().name()
                )));
            }
        };
        if !kappa.is_finite() || kappa < 0.0 {
            return Err(Error::Config(format!(
                "absorptionCoefficient: {kappa} is refused - κ must be \
                 non-negative and finite (SPEC-LIT §137.2)"
            )));
        }
        Ok(Self { kappa, sigma_s: c.scatter.sigma(), a1: c.scatter.a1() })
    }
}

// ==========================================================================
//  The device kernels of cuda/radiation.cu
// ==========================================================================

/// The eight kernels of `cuda/radiation.cu`, loaded once per [`P1`].
struct RadiationKernels {
    gamma_cells: CudaFunction,
    gamma_faces: CudaFunction,
    gamma_boundary: CudaFunction,
    marshak_triple: CudaFunction,
    su_emission: CudaFunction,
    source_pair: CudaFunction,
    wall_flux: CudaFunction,
    qr_scale: CudaFunction,
    gamma_cells_particles: CudaFunction,
}

impl RadiationKernels {
    fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::RADIATION)?;
        Ok(Self {
            gamma_cells: k.func("radGammaCells")?,
            gamma_faces: k.func("radGammaFaces")?,
            gamma_boundary: k.func("radGammaBoundary")?,
            marshak_triple: k.func("radMarshakTriple")?,
            su_emission: k.func("radSuEmission")?,
            source_pair: k.func("radSourcePair")?,
            wall_flux: k.func("radWallFlux")?,
            qr_scale: k.func("radQrScale")?,
            gamma_cells_particles: k.func("radGammaCellsParticles")?,
        })
    }
}

// ==========================================================================
//  The solve
// ==========================================================================

/// One P1 medium on one mesh: the field `G`, the Marshak walls, and every
/// scratch buffer the step needs, allocated once so [`P1::solve`] launches
/// and nothing else.
///
/// `G` starts at zero everywhere. (135.8)'s solution is fixed by the
/// absorption term and the walls, so the level needs no seeding; starting
/// from the emission `4σT^4` would only be a guess with no contract
/// attached, and a guess is not a boundary condition.
pub struct P1 {
    medium: P1Medium,
    /// The emission weight `w` that multiplies `σ` everywhere the solve
    /// emits - the medium's `4κσT^4`, the Marshak wall's `4σT_w^4` and
    /// (137.2)'s source pair. 1 for a grey medium (`σ·1` is exact, so the
    /// grey solve is unchanged); WSGG's `a_i` for one grey gas of
    /// [`super::wsgg::WsggP1`] (SPEC-LIT §136.9).
    weight: Scalar,

    n_cells: usize,
    n_internal_faces: usize,
    n_bf: usize,

    /// `[n_bf]` `a = ε/(2(2-ε))` of (135.10), zero on every non-Marshak
    /// face - the kernel's per-face switch.
    a_bf: DevBuf<Scalar>,
    /// `[n_bf]` the stated `T_w` where the wall states one, 0 elsewhere.
    tw_fixed: DevBuf<Scalar>,
    /// `[n_bf]` 1 where `T_w` is [`WallTemperature::Fixed`], else 0.
    fixed_flag: DevBuf<Label>,

    /// `[n_cells]` `κ`, uniform - [`crate::fv::fvm_sp`]'s per-cell operand.
    kappa_cells: DevBuf<Scalar>,
    /// `[n_cells]` Γ, written by `radGammaCells` every solve.
    gamma_cells: DevBuf<Scalar>,
    /// `[n_if]` `Γ_f |S_f|` of (135.13).
    gamma_if: DevBuf<Scalar>,
    /// `[n_bf]` `Γ_b |S_b|` of (135.13).
    gamma_bf: DevBuf<Scalar>,
    /// `[n_cells]` (135.8)'s `4κσT^4`, written every solve.
    su: DevBuf<Scalar>,
    /// `[n_cells]` (137.2)'s explicit pair, written by
    /// [`P1::register_energy_source`].
    q_src: DevBuf<Scalar>,
    sp_src: DevBuf<Scalar>,

    /// The incident radiation the solve produces.
    g: GpuScalarField,
    /// `[n_bf]` (135.15)'s `(q·n)_b`, 0 on non-Marshak faces.
    wall_flux: DevBuf<Scalar>,
    /// `[n_cells]` (135.16)'s `q_r = -Γ ∇G`.
    qr: DevBuf<Vec3>,

    /// The matrix, zeroed and re-assembled in place every solve.
    a: GpuLduMatrix,
    ws: SolverWorkspace,

    k: RadiationKernels,
    fvk: FvKernels,
    lduk: LduKernels,
    fldk: FieldKernels,
    solk: SolverKernels,
    ctrl: SolverControls,

    /// The cloud's per-cell absorption `a_p` (1/m) and grey emission `e_p`
    /// (W/m3) of (159.7) of SPEC-LIT 159.4, `[n_cells]` each, when a Phase
    /// B dust cloud is handed in with [`P1::set_particles`]; `None` is the
    /// solve that runs without one.
    particles: Option<(DevBuf<Scalar>, DevBuf<Scalar>)>,
}

impl P1 {
    /// Build the solve for `medium` on `hm`/`m` with Marshak walls on the
    /// named patches.
    ///
    /// Every refusal is an [`Error::Config`] naming the entry it is about
    /// and the § it breaks; nothing is substituted.
    pub fn new(
        gpu: &Gpu,
        hm: &HostMesh,
        m: &GpuMesh,
        medium: &P1Medium,
        walls: &[MarshakPatch],
        ctrl: SolverControls,
    ) -> Result<Self> {
        let P1Medium { kappa, sigma_s, a1 } = *medium;
        // (135.7) needs a finite Γ and (135.8) needs the absorption to hold
        // the level: κ = 0 would leave G's level to the walls alone, which
        // is a different equation from the one §135.3 derived.
        if !kappa.is_finite() || kappa < 0.0 {
            return Err(Error::Config(format!(
                "radiation::p1::P1Medium: kappa {kappa} is refused - κ must be \
                 non-negative and finite (SPEC-LIT §135.3)"
            )));
        }
        if !sigma_s.is_finite() || sigma_s < 0.0 {
            return Err(Error::Config(format!(
                "radiation::p1::P1Medium: sigma_s {sigma_s} is refused - σ_s \
                 must be non-negative and finite (SPEC-LIT §135.3)"
            )));
        }
        if a1.abs() > 1.0 {
            return Err(Error::Config(format!(
                "radiation::p1::P1Medium: a1 {a1} is refused - |A1| > 1 makes \
                 the phase function Φ = 1 + A1 Ω'·Ω negative for some pair of \
                 directions (SPEC-LIT §135.1)"
            )));
        }
        if kappa == 0.0 {
            return Err(Error::Config(
                "radiation::p1::P1Medium: kappa 0 is refused - (135.8)'s κ G \
                 is the only term that fixes G's level inside the medium, and \
                 without it the solve is not the P1 equation (SPEC-LIT §135.3)"
                    .into(),
            ));
        }
        if walls.is_empty() {
            return Err(Error::Config(
                "radiation::p1::P1::new: no Marshak patch is named - (135.8) \
                 with no wall condition is underdetermined, and a case that \
                 means the walls to be non-participating should say so with a \
                 patch that carries another condition (SPEC-LIT §135.4)"
                    .into(),
            ));
        }
        let marshak = Self::marshak_faces(hm, walls)?;

        // The per-face Marshak data: `a`, the stated `T_w` and its switch.
        // Everything else the step needs is a kernel's output.
        let n_bf = hm.n_boundary_faces;
        let mut a_h = vec![0.0 as Scalar; n_bf];
        let mut tw_h = vec![0.0 as Scalar; n_bf];
        let mut flag_h = vec![0 as Label; n_bf];
        // Face k of `marshak` is NOT wall k: a patch may carry any number
        // of faces and the walls may be listed in any order. Every face of
        // patch `w.patch` gets THAT wall's `a`, `T_w` and flag; a zero-face
        // patch contributes nothing.
        for (faces, w) in marshak.iter().zip(walls) {
            let a = (w.emissivity / (2.0 * (2.0 - w.emissivity))) as Scalar;
            for bf in faces {
                a_h[*bf] = a;
                if let WallTemperature::Fixed(tw) = w.temperature {
                    tw_h[*bf] = tw as Scalar;
                    flag_h[*bf] = 1;
                }
            }
        }

        // The Marshak faces' triple kind: §4's Mixed, which is what makes
        // `correct_boundary_conditions` and `fvm_laplacian`'s boundary pass
        // read the triple the Marshak kernel rewrites. Every other face
        // keeps what `GpuScalarField::zeros` gave it - zeroGradient on an
        // ordinary patch or a wall (a specular symmetry plane for G), and
        // nothing at all on an `empty` one.
        let mut g = GpuScalarField::zeros(gpu, m, "G")?;
        let mut kind = gpu.download(&g.bc_kind)?;
        for faces in &marshak {
            for bf in faces {
                kind[*bf] = BcKind::Mixed as Label;
            }
        }
        gpu.write(&mut g.bc_kind, &kind)?;

        let kappa_h = vec![kappa as Scalar; hm.n_cells];
        Ok(Self {
            medium: *medium,
            weight: 1.0,
            n_cells: hm.n_cells,
            n_internal_faces: m.n_internal_faces,
            n_bf,
            a_bf: gpu.upload(&a_h)?,
            tw_fixed: gpu.upload(&tw_h)?,
            fixed_flag: gpu.upload(&flag_h)?,
            kappa_cells: gpu.upload(&kappa_h)?,
            gamma_cells: gpu.zeros(hm.n_cells)?,
            gamma_if: gpu.zeros(m.n_internal_faces.max(1))?,
            gamma_bf: gpu.zeros(n_bf.max(1))?,
            su: gpu.zeros(hm.n_cells)?,
            q_src: gpu.zeros(hm.n_cells)?,
            sp_src: gpu.zeros(hm.n_cells)?,
            wall_flux: gpu.zeros(n_bf)?,
            qr: gpu.zeros(hm.n_cells)?,
            g,
            a: GpuLduMatrix::new(gpu, m)?,
            ws: SolverWorkspace::for_mesh(gpu, m)?,
            k: RadiationKernels::new(gpu)?,
            fvk: FvKernels::new(gpu)?,
            lduk: LduKernels::new(gpu)?,
            fldk: FieldKernels::new(gpu)?,
            solk: SolverKernels::new(gpu)?,
            ctrl,
            particles: None,
        })
    }

    /// Validate the named walls against the mesh and return each wall's
    /// boundary face indices, in the caller's order: each patch must exist,
    /// must not be `empty` (a Marshak face on a 2-D ghost contributes
    /// nothing, so the condition would be read and then ignored), must not
    /// be named twice, and its emissivity and stated temperature must be
    /// physical. A zero-face patch is accepted and contributes nothing.
    fn marshak_faces(hm: &HostMesh, walls: &[MarshakPatch]) -> Result<Vec<Vec<usize>>> {
        let mut seen: Vec<usize> = Vec::new();
        let mut per_wall = Vec::with_capacity(walls.len());
        for w in walls {
            if !w.emissivity.is_finite() || w.emissivity <= 0.0 || w.emissivity > 1.0 {
                return Err(Error::Config(format!(
                    "MarshakPatch \"{}\": emissivity {} is refused - ε must lie \
                     in (0, 1] (SPEC-LIT §135.4)",
                    w.patch, w.emissivity
                )));
            }
            if let WallTemperature::Fixed(tw) = w.temperature {
                if !tw.is_finite() || tw < 0.0 {
                    return Err(Error::Config(format!(
                        "MarshakPatch \"{}\": fixed T_w {tw} is refused - a wall \
                         temperature must be non-negative and finite (SPEC-LIT \
                         §135.4)",
                        w.patch
                    )));
                }
            }
            let (pi, p) = hm
                .patches
                .iter()
                .enumerate()
                .find(|(_, p)| p.name == w.patch)
                .ok_or_else(|| {
                    Error::Config(format!(
                        "MarshakPatch: no patch named \"{}\"; the mesh has {}",
                        w.patch,
                        hm.patches.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(", ")
                    ))
                })?;
            if p.kind == PatchKind::Empty {
                return Err(Error::Config(format!(
                    "MarshakPatch \"{}\": {} face(s) of kind empty - a 2-D ghost \
                     carries no wall condition, and this one would be read and \
                     then ignored (SPEC-LIT §135.4)",
                    w.patch, p.size
                )));
            }
            // The duplicate check compares PATCH indices, never a face
            // lookup: a zero-face patch has no face to look up, and its
            // `start` may sit one past the end of `b_patch`.
            if seen.contains(&pi) {
                return Err(Error::Config(format!(
                    "MarshakPatch \"{}\" is named twice - one patch carries one \
                     wall condition (SPEC-LIT §135.4)",
                    w.patch
                )));
            }
            seen.push(pi);
            per_wall.push((p.start..p.start + p.size).collect());
        }
        Ok(per_wall)
    }
}

impl P1 {
    /// Rewrite the Marshak triples from `t` (`t.bf` for a
    /// [`WallTemperature::FromField`] wall), assemble (135.12), solve `G`,
    /// correct `G`'s boundary values, and compute (135.15)'s `(q·n)_b` and
    /// (135.16)'s `q_r`.
    ///
    /// Every launch here is pure device work - the scratch was allocated in
    /// [`P1::new`] - so the whole step captures into a CUDA graph. The one
    /// branch the capture forces is the solver entry itself:
    /// `solver::solve`'s symmetry check reads the defect back to the host,
    /// which SPEC-LIT §81.3 refuses inside a capture. The symmetry it
    /// checks is structural here - the laplacian kernel writes `upper` and
    /// `lower` in one pass, and the boundary pass touches only `diag` and
    /// `source` - so the captured path enters `solver::solve_pcg` directly
    /// WHEN the controls name PCG, which is the entry that check dispatches
    /// to. Every other combination goes through `solver::solve`, captured
    /// or not: it captures PBiCGStab itself (a non-DIC preconditioner makes
    /// no host round-trip) and refuses the rest by name - the captured path
    /// must run the method the controls name, or a replay is not the step
    /// it replays.
    pub fn solve(
        &mut self,
        gpu: &Gpu,
        m: &GpuMesh,
        t: &GpuScalarField,
    ) -> Result<SolverPerformance> {
        // The step addresses one mesh only: its buffers were sized at
        // [`P1::new`] and a captured graph bakes the launches' grids. A
        // different mesh is refused by name, never silently substituted.
        if m.n_cells != self.n_cells
            || m.n_internal_faces != self.n_internal_faces
            || m.n_boundary_faces != self.n_bf
        {
            return Err(Error::Config(format!(
                "P1::solve: the mesh handed in ({} cells, {} internal faces, \
                 {} boundary faces) is not the one P1::new was built on \
                 ({} cells, {} internal faces, {} boundary faces) - the \
                 step's buffers and its captured graph address one mesh only \
                 (SPEC-LIT §81)",
                m.n_cells,
                m.n_internal_faces,
                m.n_boundary_faces,
                self.n_cells,
                self.n_internal_faces,
                self.n_bf
            )));
        }
        let n = self.n_cells;
        let nl = n as Label;
        let nbl = self.n_bf as Label;
        let kappa = self.medium.kappa as Scalar;
        let sigma_s = self.medium.sigma_s as Scalar;
        let a1 = self.medium.a1 as Scalar;
        let sigma = SIGMA_SB * self.weight;

        // (135.7), then (135.14): Γ first, because the Marshak triple reads
        // its owner's cell value. Then (135.13)'s two face products. All
        // four launches are idempotent overwrites, so re-solving with a new
        // `t` cannot inherit anything from the last one.
        //
        // (159.7) of SPEC-LIT 159.4: a set cloud's absorption joins the
        // medium, so beta = kappa + a_p + sigma_s and the SAME per-cell
        // Gamma buffer the Marshak triple, the matrix and q_r all read is
        // written from it. Without a cloud the launch is exactly the one
        // that always ran.
        match &self.particles {
            Some((a_p, _e_p)) => unsafe {
                gpu.stream()
                    .launch_builder(&self.k.gamma_cells_particles)
                    .arg(&mut self.gamma_cells)
                    .arg(a_p)
                    .arg(&kappa)
                    .arg(&sigma_s)
                    .arg(&a1)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            },
            None => unsafe {
                gpu.stream()
                    .launch_builder(&self.k.gamma_cells)
                    .arg(&mut self.gamma_cells)
                    .arg(&kappa)
                    .arg(&sigma_s)
                    .arg(&a1)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            },
        }
        unsafe {
            gpu.stream()
                .launch_builder(&self.k.marshak_triple)
                .arg(&mut self.g.fr)
                .arg(&mut self.g.ref_value)
                .arg(&self.a_bf)
                .arg(&self.gamma_cells)
                .arg(&m.b_face_cells)
                .arg(&m.b_delta_coeffs)
                .arg(&self.tw_fixed)
                .arg(&self.fixed_flag)
                .arg(&t.bf)
                .arg(&sigma)
                .arg(&nbl)
                .launch(cfg_for(self.n_bf))?;
        }
        unsafe {
            gpu.stream()
                .launch_builder(&self.k.su_emission)
                .arg(&mut self.su)
                .arg(&t.f)
                .arg(&kappa)
                .arg(&sigma)
                .arg(&nl)
                .launch(cfg_for(n))?;
            if m.n_internal_faces > 0 {
                let nif = m.n_internal_faces as Label;
                gpu.stream()
                    .launch_builder(&self.k.gamma_faces)
                    .arg(&mut self.gamma_if)
                    .arg(&self.gamma_cells)
                    .arg(&m.weights)
                    .arg(&m.mag_sf)
                    .arg(&m.owner)
                    .arg(&m.neighbour)
                    .arg(&nif)
                    .launch(cfg_for(m.n_internal_faces))?;
            }
            if self.n_bf > 0 {
                let nbfl = self.n_bf as Label;
                gpu.stream()
                    .launch_builder(&self.k.gamma_boundary)
                    .arg(&mut self.gamma_bf)
                    .arg(&self.gamma_cells)
                    .arg(&m.b_face_cells)
                    .arg(&m.b_mag_sf)
                    .arg(&nbfl)
                    .launch(cfg_for(self.n_bf))?;
            }
        }

        // (135.12) on the crate's own operators. `sign = -1` in
        // `fvm_laplacian` puts the discretisation on its positive form -
        // `diag += Γ_f|S_f|Δ_f` per face, `upper = -Γ_f|S_f|Δ_f` - and the
        // same sign runs the Marshak faces' triple through §4's value
        // contributions, `diag += Γ_b|S_b|Δ_b·fr_b` and
        // `source += Γ_b|S_b|Δ_b·fr_b·G_w`, which is exactly the boundary
        // row of (135.12). `fvm_sp(+1)` adds the absorption `κ V_P` to the
        // diagonal and `fvm_su(+1)` the emission `4κσT^4·V_P` to the
        // source. The matrix stays symmetric throughout - the boundary
        // pass never writes an off-diagonal - which is what makes PCG
        // legal (SPEC-LIT §8.2).
        self.a.zero(gpu)?;
        fv::fvm_laplacian(gpu, &self.fvk, &mut self.a, m, &self.gamma_if, &self.gamma_bf, &self.g, -1.0)?;
        fv::fvm_sp(gpu, &self.fvk, &mut self.a, m, &self.kappa_cells, 1.0)?;
        // (159.7) of SPEC-LIT 159.4: a set cloud adds its absorption `a_p`
        // to the diagonal, `diag += a_p V_P`, and - below, after the gas's
        // emission - its grey emission `e_p` to the source,
        // `source += e_p V_P`, both per cell. The gas's own (137.2) energy
        // source is NOT touched: the gas term is `kappa` alone, and the
        // particles' side of the exchange is the parcels' own energy.
        if let Some((a_p, _e_p)) = &self.particles {
            fv::fvm_sp(gpu, &self.fvk, &mut self.a, m, a_p, 1.0)?;
        }
        fv::fvm_su(gpu, &self.fvk, &mut self.a, m, &self.su, 1.0)?;
        if let Some((_a_p, e_p)) = &self.particles {
            fv::fvm_su(gpu, &self.fvk, &mut self.a, m, e_p, 1.0)?;
        }
        ldu_ops::add_boundary_contributions(gpu, &self.lduk, &mut self.a, m)?;

        // §118.4-refined under `single` when the controls let it be. The
        // captured direct entry is taken only when capturing AND the
        // controls name PCG - the one combination `solver::solve` would
        // refuse inside a capture for its host-side symmetry check. Every
        // other combination lets `solver::solve` decide, so a P1 built with
        // PBiCGStab solves with PBiCGStab captured too.
        let perf = if gpu.is_capturing() && self.ctrl.solver == solver::LinearSolverKind::PCG {
            solver::solve_pcg(gpu, &self.solk, &mut self.g.f, &self.a, m, &mut self.ws, &self.ctrl)?
        } else {
            solver::solve(gpu, &self.solk, &mut self.g.f, &self.a, m, &mut self.ws, &self.ctrl)?
        };

        // The evaluated boundary values, so `fvc_grad_scalar` reads faces
        // that are not stale - and so a caller may write `G` out.
        field_ops::correct_boundary_conditions(gpu, &self.fldk, &mut self.g, m)?;

        // (135.16): the Green-Gauss gradient in place, then the kernel's
        // scaling to q_r = -Gamma grad G - by the SAME per-cell Gamma
        // buffer the matrix was assembled from, not a host scalar that
        // under `single` can sit an ulp away from it.
        fv::fvc_grad_scalar(gpu, &self.fvk, &mut self.qr, &self.g, m)?;
        unsafe {
            gpu.stream()
                .launch_builder(&self.k.qr_scale)
                .arg(&mut self.qr)
                .arg(&self.gamma_cells)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }

        // (135.15): every boundary face's wall flux out of the medium, 0 on
        // the faces that are not Marshak (their `a_b` is 0, and the kernel
        // multiplies by it rather than branching).
        if self.n_bf > 0 {
            let nbl = self.n_bf as Label;
            unsafe {
                gpu.stream()
                    .launch_builder(&self.k.wall_flux)
                    .arg(&mut self.wall_flux)
                    .arg(&self.a_bf)
                    .arg(&self.g.fr)
                    .arg(&self.g.ref_value)
                    .arg(&self.g.f)
                    .arg(&m.b_face_cells)
                    .arg(&nbl)
                    .launch(cfg_for(self.n_bf))?;
            }
        }
        Ok(perf)
    }

    /// (137.2) at `T* = t`'s cell values - `q_src = κ(G + 12σT*^4)`,
    /// `sp_src = -16κσT*^3` - registered on `sources` as its explicit power
    /// density and implicit sink. Call once per outer iteration, after
    /// `sources.clear`; registering twice doubles the source, exactly as
    /// the method says.
    ///
    /// Two launches plus the registry's own two accumulations, no host
    /// round-trip, so the registration captures with the rest of the step.
    ///
    /// The gas term is `kappa` alone: a set cloud's exchange with the gas
    /// is the PARCELS' own energy (SPEC-LIT 159.4), which the particle step
    /// carries - this registration never sees it.
    pub fn register_energy_source(
        &mut self,
        gpu: &Gpu,
        t: &GpuScalarField,
        sources: &mut EnergySources,
    ) -> Result<()> {
        let n = self.n_cells;
        let nl = n as Label;
        let kappa = self.medium.kappa as Scalar;
        let sigma = SIGMA_SB * self.weight;
        unsafe {
            gpu.stream()
                .launch_builder(&self.k.source_pair)
                .arg(&mut self.q_src)
                .arg(&mut self.sp_src)
                .arg(&self.g.f)
                .arg(&t.f)
                .arg(&kappa)
                .arg(&sigma)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        sources.register_explicit(gpu, &self.q_src)?;
        sources.register_implicit_sink(gpu, &self.sp_src)?;
        Ok(())
    }

    /// Hand the solve one cloud: `a_p` and `e_p`, `[n_cells]` each, the
    /// per-cell absorption and grey emission of (159.7) of SPEC-LIT 159.4
    /// - the output of [`crate::dust::particle::cloud_radiation`],
    /// uploaded. From the next [`P1::solve`] on, `a_p` joins the medium in
    /// (135.7) of SPEC-LIT 135.3 and the diagonal, and `e_p` joins the
    /// source; the (137.2) energy source the gas registers stays `kappa`
    /// alone - the particles' side of the exchange is the parcels' own.
    ///
    /// Refused, naming the argument and SPEC-LIT 159.4: a length other than
    /// the mesh's cell count, any value that is not finite or is negative,
    /// and any call at all when the solve is ONE grey gas of a WSGG sum
    /// (`weight != 1`): the particles' grey emission is not split over the
    /// gases.
    ///
    /// This UPLOADS, so it must never be called inside a CUDA-graph capture
    /// (SPEC-LIT 81): set the cloud, or swap its buffers, outside the
    /// capture.
    pub fn set_particles(&mut self, gpu: &Gpu, a_p: &[Scalar], e_p: &[Scalar]) -> Result<()> {
        if a_p.len() != self.n_cells || e_p.len() != self.n_cells {
            return Err(Error::Config(format!(
                "radiation::p1::P1::set_particles: a_p holds {} values and \
                 e_p {}, the mesh has {} cells - one absorption and one \
                 emission per cell (SPEC-LIT 159.4)",
                a_p.len(),
                e_p.len(),
                self.n_cells
            )));
        }
        for (name, v) in a_p
            .iter()
            .enumerate()
            .map(|(i, &x)| (format!("a_p[{i}]"), x))
            .chain(e_p.iter().enumerate().map(|(i, &x)| (format!("e_p[{i}]"), x)))
        {
            if !v.is_finite() || v < 0.0 {
                return Err(Error::Config(format!(
                    "radiation::p1::P1::set_particles: {name} = {v} is \
                     refused - a cloud's absorption and emission are finite \
                     and >= 0 (SPEC-LIT 159.4)"
                )));
            }
        }
        if self.weight != 1.0 {
            return Err(Error::Config(format!(
                "radiation::p1::P1::set_particles: the grey gas's weight {} \
                 is refused - a WSGG sum is a superposition of grey solves, \
                 and the particles' grey emission is not split over its \
                 gases (SPEC-LIT 159.4)",
                self.weight
            )));
        }
        match &mut self.particles {
            Some((a, e)) => {
                gpu.write(a, a_p)?;
                gpu.write(e, e_p)?;
            }
            slot => {
                *slot = Some((gpu.upload(a_p)?, gpu.upload(e_p)?));
            }
        }
        Ok(())
    }

    /// Drop the cloud: the next [`P1::solve`] is exactly the solve that
    /// runs without one.
    pub fn clear_particles(&mut self) {
        self.particles = None;
    }

    /// The cloud's `(a_p, e_p)` buffers, when one is set.
    pub fn particles(&self) -> Option<(&DevBuf<Scalar>, &DevBuf<Scalar>)> {
        self.particles.as_ref().map(|(a, e)| (a, e))
    }

    /// Make the solve ONE grey gas of a weighted sum (SPEC-LIT §136.4,
    /// §136.9): the absorption becomes `kappa` (scattering and `A_1` stay
    /// the medium's) and every emission term is weighted by `weight` -
    /// (136.10)'s `a_i κ_i I_b` in the medium and `a_i` on the Marshak walls'
    /// `4σT_w^4` and in (137.2)'s pair. `G` is zeroed, so the next
    /// [`P1::solve`] is the solve a fresh [`P1::new`] at this `kappa` would
    /// run. `kappa` is refused as [`P1::new`] refuses it (finite, positive);
    /// `weight` must be finite - a fitted WSGG weight may be negative inside
    /// its range and is used as published (§136.9). Launches only, so it
    /// captures with the step (SPEC-LIT §81).
    pub fn set_grey_gas(&mut self, gpu: &Gpu, kappa: f64, weight: f64) -> Result<()> {
        if !kappa.is_finite() || kappa <= 0.0 {
            return Err(Error::Config(format!(
                "radiation::p1::P1::set_grey_gas: kappa {kappa} is refused - a \
                 grey gas's κ must be positive and finite (SPEC-LIT §135.3, \
                 §136.4)"
            )));
        }
        if !weight.is_finite() {
            return Err(Error::Config(format!(
                "radiation::p1::P1::set_grey_gas: weight {weight} is refused - \
                 the grey gas's a_i must be finite (SPEC-LIT §136.4)"
            )));
        }
        self.medium.kappa = kappa;
        self.weight = weight as Scalar;
        field_ops::set_field(gpu, &self.fldk, &mut self.kappa_cells, kappa as Scalar, self.n_cells)?;
        gpu.fill_zero(&mut self.g.f)?;
        gpu.fill_zero(&mut self.g.bf)?;
        Ok(())
    }

    /// The medium the next [`P1::solve`] runs - [`P1::set_grey_gas`]'s
    /// `kappa` once it was called.
    pub fn medium(&self) -> &P1Medium {
        &self.medium
    }

    /// The incident radiation field `G`, internal and boundary values.
    pub fn g(&self) -> &GpuScalarField {
        &self.g
    }

    /// `[n_boundary_faces]` (135.15)'s `(q·n)_b`, W/m2, positive out of the
    /// medium; 0 on every face that is not Marshak.
    pub fn wall_flux(&self) -> &DevBuf<Scalar> {
        &self.wall_flux
    }

    /// `[n_cells]` (135.16)'s `q_r = -Γ ∇G`, W/m2.
    pub fn qr(&self) -> &DevBuf<Vec3> {
        &self.qr
    }
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blockgen::{self, BlockSpec, GradedAxis};

    /// The process's device, if the box has one: the convention of every
    /// GPU test in the crate.
    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    /// The 8-cell slab: x in [0, 1] m, one cell each in y and z over
    /// [0, 0.1], Marshak walls at xMin and xMax, the four other slots
    /// empty. T uniform 1000 K in every cell and on every face, walls at
    /// 0 K - the closed-form slab of SPEC-LIT §135.5 at tau = 1.
    fn slab(gpu: &Gpu) -> Result<(HostMesh, GpuMesh, GpuScalarField)> {
        let axis = |lo: f64, hi: f64, n: usize| GradedAxis {
            lo: lo as Scalar,
            hi: hi as Scalar,
            n,
            expansion: 1.0,
            two_sided: false,
        };
        let spec = BlockSpec {
            x: axis(0.0, 1.0, 8),
            y: axis(0.0, 0.1, 1),
            z: axis(0.0, 0.1, 1),
            patch_name: BlockSpec::default().patch_name,
            patch_type: ["wall", "wall", "empty", "empty", "empty", "empty"]
                .map(String::from),
            windows: Vec::new(),
            cyclic: Vec::new(),
        };
        let hm = blockgen::build_mesh(&spec)?;
        let m = GpuMesh::upload(gpu, &hm)?;
        let mut t = GpuScalarField::zeros(gpu, &m, "T")?;
        gpu.write(&mut t.f, &vec![1000.0 as Scalar; hm.n_cells])?;
        gpu.write(&mut t.bf, &vec![1000.0 as Scalar; hm.n_boundary_faces])?;
        Ok((hm, m, t))
    }

    fn medium() -> P1Medium {
        P1Medium { kappa: 1.0, sigma_s: 0.0, a1: 0.0 }
    }

    fn walls() -> Vec<MarshakPatch> {
        vec![
            MarshakPatch {
                patch: "xMin".to_string(),
                emissivity: 1.0,
                temperature: WallTemperature::Fixed(0.0),
            },
            MarshakPatch {
                patch: "xMax".to_string(),
                emissivity: 1.0,
                temperature: WallTemperature::Fixed(0.0),
            },
        ]
    }

    fn capture_controls() -> SolverControls {
        SolverControls {
            solver: solver::LinearSolverKind::PCG,
            fixed_iters: true,
            report_residuals: false,
            max_iter: 20,
            ..SolverControls::default()
        }
    }

    /// The registry's gate (`src/capture/registry.rs`, row
    /// `src/radiation/p1.rs`): one solve plus its energy-source
    /// registration captures into a CUDA graph and replays bitwise against
    /// the per-launch path, on the 8-cell slab. The state is `G`'s cells,
    /// the wall fluxes, the flux vector's three components, and the two
    /// registered source accumulators - what the step writes, and nothing
    /// less.
    #[test]
    fn p1_solve_replays_under_capture() {
        let Some(gpu) = gpu() else { return };

        let report = crate::capture::capture_replays_bitwise(
            &gpu,
            "the P1 solve and its energy source (SPEC-LIT 135.7)",
            || {
                let (_hm, m, t) = slab(&gpu)?;
                let p1 = P1::new(&gpu, &_hm, &m, &medium(), &walls(), capture_controls())?;
                let sources = EnergySources::new(&gpu, &m)?;
                Ok((p1, m, t, sources))
            },
            |(p1, m, t, sources): &mut (P1, GpuMesh, GpuScalarField, EnergySources)| {
                sources.clear(&gpu)?;
                p1.solve(&gpu, m, t)?;
                p1.register_energy_source(&gpu, t, sources)
            },
            |(p1, _m, _t, sources): &(P1, GpuMesh, GpuScalarField, EnergySources)| {
                let qr = gpu.download(p1.qr())?;
                let mut qx = Vec::with_capacity(qr.len());
                let mut qy = Vec::with_capacity(qr.len());
                let mut qz = Vec::with_capacity(qr.len());
                for v in &qr {
                    qx.push(v.x);
                    qy.push(v.y);
                    qz.push(v.z);
                }
                Ok(vec![
                    ("G cells", gpu.download(&p1.g().f)?),
                    ("wall flux", gpu.download(p1.wall_flux())?),
                    ("qr x", qx),
                    ("qr y", qy),
                    ("qr z", qz),
                    ("sources q", gpu.download(sources.q())?),
                    ("sources sp", gpu.download(sources.sp())?),
                ])
            },
        )
        .expect("SPEC-LIT 81.7: the P1 step must capture and replay bitwise");
        println!("  P1: {report}");
    }
}
