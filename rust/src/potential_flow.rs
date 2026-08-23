// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Educational use is free; research,
// publication and commercial use require a licence - simul@msimul.com
// See LICENSE at the repository root.

//! Potential flow: solve for a mass-conserving flux instead of inventing one.
//!
//! # Why this exists
//!
//! A driver that has no `phi` on disk has to get one from somewhere, and the
//! obvious move — [`crate::field_setup::compute_phi_from_u`], i.e.
//! `interpolate(U) & Sf` — is wrong in a way that is easy to miss. Nothing
//! constrains a hand-written cell-centred `U` to satisfy the *discrete*
//! continuity equation, so the flux it interpolates to does not either. On the
//! plume case, whose `0/U` prescribes a jet that tapers to zero inside the
//! domain, `max |sum_f phi|` per cell came out at 1.4e-2 m^3/s: mass entered at
//! the burner and never left. A transported temperature then has no mean
//! transport to carry heat out of the room, so the whole domain equilibrates to
//! the inlet value — a physically wrong answer that the `bounded` correction
//! (`- fvm::Sp(fvc::div(phi), psi)`) papers over rather than fixes.
//!
//! The fix is to stop inventing a velocity field and solve for one:
//!
//! ```text
//! laplacian(Phi) = 0
//!   inlet   : fixedGradient, dPhi/dn = -U_in   (n is OUTWARD, so this is inflow)
//!   outlet  : fixedValue Phi = 0
//!   walls   : zeroGradient                     (no flux through a wall)
//! ```
//!
//! This is the classical potential-flow initialisation (Ferziger & Peric
//! section 7.1): one Laplace solve, no momentum, no turbulence. The outlet
//! Dirichlet does two jobs: it lets the
//! outflow distribute itself instead of being prescribed, and it removes the
//! null space that an all-Neumann Laplace problem has.
//!
//! # The critical detail
//!
//! Do **not** compute a cell velocity and interpolate it to faces — that would
//! reintroduce exactly the error this module exists to remove. Build the flux
//! directly from the face gradient the laplacian assembled:
//!
//! ```text
//! phi_f  = deltaCoeffs[f]*(Phi[nei] - Phi[own])*magSf[f]     internal
//! phi_bf = internalCoeffs[bf]*Phi_c - boundaryCoeffs[bf]     boundary
//! ```
//!
//! Both are [`fv::sn_grad_flux`], which is written against the same
//! coefficients in the same multiplication order as [`fv::fvm_laplacian`].
//! With `source = 0`, the linear solver's residual in cell `c` *is*
//! `sum_f phi_f`, so the flux is discretely conservative to the solver
//! tolerance plus a few ulps of round-off — not to interpolation accuracy.
//! [`PotentialFlowResult::max_div_phi`] is the number that proves it, and it
//! lands near 1e-18 rather than 1e-2.
//!
//! # Deliberately orthogonal
//!
//! No non-orthogonal correction is applied. That correction enters the matrix
//! as an explicit source built from the *previous* iterate's `grad(Phi)`, so
//! the exactly-conservative flux would have to carry the same stale term; the
//! guarantee would then hold only at the deferred-correction fixed point
//! rather than unconditionally. On a mesh with real non-orthogonality this
//! costs some accuracy in `Phi` and costs nothing in conservation, which is
//! the right way round: `phi` is the object transport must be able to trust.
//! On the plume case, a uniform hex block, the correction is identically zero.
//!
//! # And then a velocity
//!
//! `grad(U)` still needs a cell-centred velocity for the turbulence production
//! term, so [`fv::fvc_reconstruct`] recovers one:
//!
//! ```text
//! U_c = inv(sum_f (Sf (x) Sf)/|Sf|) & (sum_f (Sf/|Sf|)*phi_f)
//! ```
//!
//! That direction is lossy and is meant to be. `phi` is what the transport
//! equations consume; `U` is what the model differentiates. Interpolating this
//! `U` back onto the faces would not return the flux it came from, which is
//! precisely why nothing here does.

use crate::device::Gpu;
use crate::error::{Error, Result};
use crate::field::{BcKind, GpuScalarField, GpuSurfaceScalarField, GpuVectorField};
use crate::field_ops::{self, FieldKernels};
use crate::fv::{self, FvKernels};
use crate::ldu::GpuLduMatrix;
use crate::ldu_ops::{self, LduKernels};
use crate::mesh::{GpuMesh, HostMesh, PatchInfo, PatchKind};
use crate::solver::{self, SolverKernels, SolverWorkspace};
use crate::{Label, Scalar, Vec3};

pub use crate::io::case::SolverControls;

// ==========================================================================
//  Specification and result
// ==========================================================================

/// Which patch the flow enters through, how fast, and which patch it leaves
/// through.
///
/// Exactly one inlet and exactly one outlet. Two openings would need a
/// pressure level each to decide how the outflow splits between them, and this
/// solve carries only the single Dirichlet reference the outlet supplies — so
/// rather than pick a split silently, the case geometry is expected to have
/// one opening. That is why `blockgen`'s plume case walls off everything but
/// `+x`.
#[derive(Debug, Clone, PartialEq)]
pub struct PotentialFlowSpec {
    pub inlet_patch: String,

    /// Speed of the flow ENTERING the domain, positive. It becomes
    /// `dPhi/dn = -inlet_normal_velocity` on the inlet, because `n` is the
    /// outward normal and this flux points inward.
    pub inlet_normal_velocity: Scalar,

    pub outlet_patch: String,
}

/// What the solve did, and the two numbers that show it worked.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PotentialFlowResult {
    pub iterations: usize,
    pub final_residual: Scalar,

    /// `max_c |sum_f phi_f|`, in m^3/s and undivided by the cell volume, so it
    /// is directly comparable with the fluxes themselves. Near the linear
    /// solver's tolerance is the pass mark; anything approaching the inlet
    /// flux means the flux was not built from the operator that was solved.
    pub max_div_phi: Scalar,

    /// `sum(phi_bf)` over the inlet, signed OUTWARD — so a real inlet reports
    /// a negative number.
    pub inlet_flux: Scalar,

    /// `sum(phi_bf)` over the outlet, signed outward and therefore positive.
    ///
    /// `inlet_flux + outlet_flux` is the global mass imbalance and is the
    /// headline claim of this module: it cancels to solver tolerance.
    pub outlet_flux: Scalar,
}

impl PotentialFlowResult {
    /// `inlet_flux + outlet_flux`: what did not balance.
    ///
    /// Both are signed outward, so a closed domain sums them to zero. Reported
    /// as a method rather than a field because it is derived, and a stored
    /// copy could drift from the two it is derived from.
    pub fn imbalance(&self) -> Scalar {
        self.inlet_flux + self.outlet_flux
    }
}

// ==========================================================================
//  The solve
// ==========================================================================

/// Solve `laplacian(Phi) = 0` and write the flux it implies into `phi`, then
/// reconstruct `u` from that flux.
///
/// `phi` is overwritten in full — internal and boundary faces alike.
///
/// `u`'s INTERNAL field is overwritten with the reconstruction; its boundary
/// conditions are left exactly as the caller set them up and are then
/// re-evaluated. That is deliberate: the walls' `noSlip` and the inlet's
/// prescribed velocity are physics the case file states and this solve has no
/// business overwriting — potential flow slips along every wall, and using its
/// wall value would delete the shear the turbulence model exists to compute.
///
/// `ctrl` is used as given except that residual reporting is forced on, since
/// [`PotentialFlowResult::final_residual`] is one of the things being asked
/// for. This is a setup-time solve, so the two eight-byte read-backs cost
/// nothing; nothing here is intended to run inside a time loop or a CUDA-graph
/// capture.
pub fn solve_potential_flow(
    gpu: &Gpu,
    hm: &HostMesh,
    m: &GpuMesh,
    phi: &mut GpuSurfaceScalarField,
    u: &mut GpuVectorField,
    spec: &PotentialFlowSpec,
    ctrl: &SolverControls,
) -> Result<PotentialFlowResult> {
    check_meshes(hm, m)?;
    let inlet = find_patch(hm, &spec.inlet_patch, "inlet")?;
    let outlet = find_patch(hm, &spec.outlet_patch, "outlet")?;

    if spec.inlet_patch == spec.outlet_patch {
        return Err(Error::Config(format!(
            "potential flow: the inlet and the outlet are both \"{}\"; the \
             outlet is the solve's only Dirichlet reference and cannot also \
             carry the prescribed inflow",
            spec.inlet_patch
        )));
    }

    if !spec.inlet_normal_velocity.is_finite() {
        return Err(Error::Config(format!(
            "potential flow: inlet_normal_velocity is {}, which cannot be a \
             boundary gradient",
            spec.inlet_normal_velocity
        )));
    }

    // An outlet that contributes no matrix coefficient leaves the problem
    // all-Neumann and therefore singular: `Phi + c` solves it for every `c`,
    // and PBiCGStab wanders off along that null space and returns a plausible
    // field with a meaningless level. Caught here, because the symptom - a
    // huge `Phi` and a flux that does not balance - points nowhere near the
    // cause. `empty` faces are skipped by `fvLapBoundary`, so an outlet on an
    // empty patch is a zero-size outlet as far as the matrix is concerned.
    if outlet.size == 0 || outlet.kind == PatchKind::Empty {
        return Err(Error::Config(format!(
            "potential flow: outlet \"{}\" has {} {} faces, so the solve has no \
             Dirichlet reference and the laplacian is singular",
            outlet.name,
            outlet.size,
            outlet.kind.as_str()
        )));
    }

    if m.n_cells == 0 {
        return Ok(PotentialFlowResult::default());
    }

    let fvk = FvKernels::new(gpu)?;
    let lduk = LduKernels::new(gpu)?;
    let fldk = FieldKernels::new(gpu)?;
    let solk = SolverKernels::new(gpu)?;

    // ---- Phi and its boundary conditions ---------------------------------
    let mut phi_pot = GpuScalarField::zeros(gpu, m, "Phi")?;
    apply_potential_bcs(gpu, &mut phi_pot, hm, inlet, outlet, spec)?;

    // ---- laplacian(Phi) == 0 ---------------------------------------------
    //
    // `gamma = 1`, so `gammaMagSf` is the mesh's own `magSf` and no scratch
    // face field is needed. Passing the mesh arrays straight through is also
    // what makes `sn_grad_flux` below provably the same operator: it is handed
    // the identical buffers.
    let mut a = GpuLduMatrix::new(gpu, m)?;
    a.zero(gpu)?;

    fv::fvm_laplacian(gpu, &fvk, &mut a, m, &m.mag_sf, &m.b_mag_sf, &phi_pot, 1.0)?;

    // No `relax` before this. Relaxation would leave a different matrix from
    // the one the flux is read off, and a linear problem has nothing to relax
    // towards anyway.
    ldu_ops::add_boundary_contributions(gpu, &lduk, &mut a, m)?;

    let mut ws = SolverWorkspace::for_mesh(gpu, m)?;
    let solve_ctrl = SolverControls { report_residuals: true, ..*ctrl };

    // SPEC-LIT 13.4: `solvers/Phi/solver`, honoured. The potential-flow
    // matrix is a pure laplacian and therefore symmetric, so `solver PCG;`
    // is both legal and the right answer here.
    let perf = solver::solve(
        gpu,
        &solk,
        &mut phi_pot.f,
        &a,
        m,
        &mut ws,
        &solve_ctrl,
    )?;

    // Not needed by `sn_grad_flux`, which rebuilds the boundary value from
    // `(fr, refValue, refGrad)` itself, but it leaves `Phi` in a state a
    // caller could write out or probe without finding stale faces.
    field_ops::correct_boundary_conditions(gpu, &fldk, &mut phi_pot, m)?;

    // ---- the flux, and the velocity it implies ---------------------------
    fv::sn_grad_flux(gpu, &fvk, phi, &phi_pot, &m.mag_sf, &m.b_mag_sf, m)?;
    fv::fvc_reconstruct(gpu, &fvk, &mut u.f, phi, m)?;
    field_ops::correct_boundary_conditions_vector(gpu, &fldk, u, m)?;

    // ---- what it came to -------------------------------------------------
    let bphi = gpu.download(&phi.bf)?;
    let iphi = gpu.download(&phi.f)?;

    Ok(PotentialFlowResult {
        iterations: perf.n_iterations,
        final_residual: perf.final_residual,
        max_div_phi: crate::field_setup::max_div_phi_host(&iphi, &bphi, hm),
        inlet_flux: patch_flux_host(&bphi, inlet),
        outlet_flux: patch_flux_host(&bphi, outlet),
    })
}

// ==========================================================================
//  Boundary conditions
// ==========================================================================

/// Stamp the three-way potential-flow condition onto `Phi`.
///
/// Everything that is not the inlet or the outlet keeps whatever
/// [`GpuScalarField::zeros`] gave it, which is `zeroGradient` on an ordinary
/// patch or a wall and the mesh's own verdict on `empty`, `cyclic` and
/// `symmetry`. All four are the right potential-flow condition already: no
/// flux through a wall or a symmetry plane, nothing at all through an empty
/// patch, and a couple that the matrix handles as a couple. Reading the seeded
/// kinds back rather than rebuilding them is what keeps that rule in one place.
fn apply_potential_bcs(
    gpu: &Gpu,
    psi: &mut GpuScalarField,
    hm: &HostMesh,
    inlet: &PatchInfo,
    outlet: &PatchInfo,
    spec: &PotentialFlowSpec,
) -> Result<()> {
    let n_bf = hm.n_boundary_faces;

    let mut kind = gpu.download(&psi.bc_kind)?;
    let mut fr = vec![0.0 as Scalar; n_bf];
    let mut ref_value = vec![0.0 as Scalar; n_bf];
    let mut ref_grad = vec![0.0 as Scalar; n_bf];

    // fixedGradient: fr = 0, refGrad = dPhi/dn. Outward normal, inward flow,
    // hence the minus.
    for bf in inlet.start..inlet.start + inlet.size {
        kind[bf] = BcKind::FixedGradient as Label;
        ref_grad[bf] = -spec.inlet_normal_velocity;
    }

    // fixedValue Phi = 0. The value is arbitrary - only gradients of Phi are
    // ever used - but it has to be *fixed* somewhere, or the matrix is
    // singular and the solver wanders along the null space.
    for bf in outlet.start..outlet.start + outlet.size {
        kind[bf] = BcKind::FixedValue as Label;
        fr[bf] = 1.0;
        ref_value[bf] = 0.0;
    }

    gpu.write(&mut psi.bc_kind, &kind)?;
    gpu.write(&mut psi.fr, &fr)?;
    gpu.write(&mut psi.ref_value, &ref_value)?;
    gpu.write(&mut psi.ref_grad, &ref_grad)?;

    // `Phi = 0` is the initial guess, and a deliberate one rather than a
    // leftover: the outlet fixes `Phi = 0`, so a uniform zero already
    // satisfies the only Dirichlet condition and the solve starts on the
    // right level instead of having to translate the whole field onto it.
    gpu.fill_zero(&mut psi.f)?;
    gpu.fill_zero(&mut psi.f0)?;
    gpu.fill_zero(&mut psi.bf)?;

    Ok(())
}

// ==========================================================================
//  Reporting helpers
// ==========================================================================

/// `sum(phi_bf)` over one patch, signed outward.
pub fn patch_flux(
    gpu: &Gpu,
    phi: &GpuSurfaceScalarField,
    hm: &HostMesh,
    patch: &str,
) -> Result<Scalar> {
    let p = find_patch(hm, patch, "patch")?;
    let bphi = gpu.download(&phi.bf)?;
    Ok(patch_flux_host(&bphi, p))
}

/// The host half of [`patch_flux`], so the sum can be tested without a device.
///
/// Summed in face order, which is the order the boundary arrays are stored in
/// and therefore the same order every run will produce.
pub fn patch_flux_host(bphi: &[Scalar], p: &PatchInfo) -> Scalar {
    let mut s: Scalar = 0.0;
    for bf in p.start..p.start + p.size {
        match bphi.get(bf) {
            Some(v) => s += *v,
            None => break,
        }
    }
    s
}

/// The mean speed at which flow ENTERS through a patch, from an evaluated
/// boundary velocity field: `-sum(U_b . Sf)/sum(|Sf|)`.
///
/// This is what [`PotentialFlowSpec::inlet_normal_velocity`] wants, and taking
/// it from the case's own `0/U` rather than from a constant in a driver is
/// what stops the two descriptions of the burner from drifting apart. Positive
/// means inflow.
///
/// `empty` faces are skipped, matching every surface integral in the crate.
pub fn mean_inflow_speed(ubf: &[Vec3], hm: &HostMesh, patch: &str) -> Result<Scalar> {
    let p = find_patch(hm, patch, "patch")?;

    if ubf.len() < hm.n_boundary_faces {
        return Err(Error::Config(format!(
            "mean_inflow_speed: boundary velocity has {} faces, mesh has {}",
            ubf.len(),
            hm.n_boundary_faces
        )));
    }

    let mut flux: Scalar = 0.0;
    let mut area: Scalar = 0.0;

    let faces = p.start..p.start + p.size;
    for (bf, u_b) in faces.clone().zip(&ubf[faces]) {
        if hm.b_kind[bf] == PatchKind::Empty as Label {
            continue;
        }
        flux += u_b.dot(hm.b_sf[bf]);
        area += hm.b_mag_sf[bf];
    }

    // `is_finite` first, so the test never has to reason about the NaN a
    // degenerate face can produce.
    if !area.is_finite() || area <= 0.0 {
        return Err(Error::Config(format!(
            "mean_inflow_speed: patch \"{patch}\" has no area to average over"
        )));
    }

    Ok(-flux / area)
}

// ==========================================================================
//  Validation
// ==========================================================================

fn find_patch<'a>(hm: &'a HostMesh, name: &str, role: &str) -> Result<&'a PatchInfo> {
    hm.patches.iter().find(|p| p.name == name).ok_or_else(|| {
        let have: Vec<&str> = hm.patches.iter().map(|p| p.name.as_str()).collect();
        Error::Config(format!(
            "potential flow: no {role} patch named \"{name}\"; the mesh has {}",
            have.join(", ")
        ))
    })
}

/// A `HostMesh` that is not the uploaded mesh would make every host-side sum
/// below address the wrong faces, and the failure would surface as a plausible
/// wrong flux rather than as an error.
fn check_meshes(hm: &HostMesh, m: &GpuMesh) -> Result<()> {
    if hm.n_cells != m.n_cells
        || hm.n_internal_faces != m.n_internal_faces
        || hm.n_boundary_faces != m.n_boundary_faces
    {
        return Err(Error::Config(format!(
            "potential flow: host mesh ({} cells, {} internal faces, {} boundary \
             faces) is not the uploaded mesh ({}, {}, {})",
            hm.n_cells,
            hm.n_internal_faces,
            hm.n_boundary_faces,
            m.n_cells,
            m.n_internal_faces,
            m.n_boundary_faces
        )));
    }
    Ok(())
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blockgen::{write_case, CaseKind};
    use crate::io::polymesh::{build_host_mesh, read_poly_mesh};
    use std::path::PathBuf;

    /// Every device test needs a card. Returning `None` makes the test pass
    /// vacuously on a machine without one, which is the convention the rest of
    /// the crate follows.
    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "ofgpu_potflow_{tag}_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    /// A generated case, read back through the real polyMesh reader so the
    /// test exercises the same addressing a run would.
    fn case_mesh(tag: &str, kind: CaseKind, n: (usize, usize, usize)) -> (PathBuf, HostMesh) {
        let dir = temp_dir(tag);
        write_case(&dir, kind, n.0, n.1, n.2).expect("write case");
        let hm = build_host_mesh(&read_poly_mesh(&dir).expect("read")).expect("host mesh");
        (dir, hm)
    }

    fn controls() -> SolverControls {
        SolverControls {
            tolerance: 1e-12,
            rel_tol: 0.0,
            max_iter: 5000,
            ..Default::default()
        }
    }

    fn spec(u_in: Scalar) -> PotentialFlowSpec {
        PotentialFlowSpec {
            inlet_patch: "inlet".to_string(),
            inlet_normal_velocity: u_in,
            outlet_patch: "outlet".to_string(),
        }
    }

    // ---- host-only --------------------------------------------------------

    #[test]
    fn a_patch_flux_sums_only_its_own_faces() {
        let p = PatchInfo {
            name: "outlet".to_string(),
            type_name: "patch".to_string(),
            kind: PatchKind::Generic,
            start: 2,
            size: 3,
            nbr_patch: None,
        };
        assert_eq!(patch_flux_host(&[9.0, 9.0, 1.0, 2.0, 4.0, 9.0], &p), 7.0);

        // A short array must not index past its end; the crate never truncates
        // silently anywhere it can help it, but a panic here would be worse.
        assert_eq!(patch_flux_host(&[9.0, 9.0, 1.0], &p), 1.0);
    }

    #[test]
    fn the_imbalance_is_the_signed_sum_of_the_two_fluxes() {
        let r = PotentialFlowResult {
            inlet_flux: -2.5,
            outlet_flux: 2.5,
            ..Default::default()
        };
        assert_eq!(r.imbalance(), 0.0);
    }

    /// `mean_inflow_speed` has to come out positive for flow that enters, or
    /// every driver reading it would prescribe the burner backwards.
    #[test]
    fn the_mean_inflow_speed_is_positive_for_flow_that_enters() {
        let (dir, hm) = case_mesh("inflow", CaseKind::Big, (4, 3, 3));

        // U = (1, 0, 0) on every boundary face. The inlet is at xMin, whose
        // outward normal is -x, so this enters.
        let ubf = vec![Vec3::new(1.0, 0.0, 0.0); hm.n_boundary_faces];

        let s_in = mean_inflow_speed(&ubf, &hm, "inlet").expect("inlet");
        let s_out = mean_inflow_speed(&ubf, &hm, "outlet").expect("outlet");
        assert!((s_in - 1.0).abs() < 1e-12, "inlet speed {s_in}");
        assert!((s_out + 1.0).abs() < 1e-12, "outlet speed {s_out}");

        assert!(mean_inflow_speed(&ubf, &hm, "nope").is_err());
        assert!(mean_inflow_speed(&[], &hm, "inlet").is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_or_duplicated_patch_is_refused() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };
        let (dir, hm) = case_mesh("refuse", CaseKind::Big, (4, 3, 3));
        let m = GpuMesh::upload(&g, &hm)?;

        let mut phi = GpuSurfaceScalarField::zeros(&g, &m, "phi")?;
        let mut u = GpuVectorField::zeros(&g, &m, "U")?;

        let bad_name = PotentialFlowSpec {
            inlet_patch: "burner".to_string(),
            ..spec(1.0)
        };
        let err = solve_potential_flow(&g, &hm, &m, &mut phi, &mut u, &bad_name, &controls())
            .expect_err("a mesh without a `burner` patch must not solve");
        assert!(format!("{err}").contains("burner"), "{err}");

        let same = PotentialFlowSpec {
            outlet_patch: "inlet".to_string(),
            ..spec(1.0)
        };
        assert!(
            solve_potential_flow(&g, &hm, &m, &mut phi, &mut u, &same, &controls()).is_err(),
            "one patch cannot be both the inflow and the pressure reference"
        );

        let nan = spec(Scalar::NAN);
        assert!(
            solve_potential_flow(&g, &hm, &m, &mut phi, &mut u, &nan, &controls()).is_err(),
            "a NaN inlet velocity must be refused, not assembled"
        );

        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }

    /// An outlet on an `empty` patch contributes no matrix coefficient, so the
    /// laplacian is all-Neumann and singular. The solve would still return a
    /// field, which is exactly why this has to be an error and not a warning.
    #[test]
    fn an_outlet_that_pins_nothing_is_refused() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        // The channel case is 2-D: `back` and `front` are `empty`.
        let (dir, hm) = case_mesh("singular", CaseKind::Channel, (6, 4, 1));
        let m = GpuMesh::upload(&g, &hm)?;

        let mut phi = GpuSurfaceScalarField::zeros(&g, &m, "phi")?;
        let mut u = GpuVectorField::zeros(&g, &m, "U")?;

        let s = PotentialFlowSpec {
            outlet_patch: "back".to_string(),
            ..spec(1.0)
        };
        let err = solve_potential_flow(&g, &hm, &m, &mut phi, &mut u, &s, &controls())
            .expect_err("an empty outlet leaves the matrix singular");
        assert!(format!("{err}").contains("singular"), "{err}");

        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }

    // ---- the analytic case ------------------------------------------------

    /// A box with the inlet on one end and the outlet on the other has the
    /// exact potential `Phi = x - L`, so the discrete answer is a uniform
    /// `U = (U_in, 0, 0)` and a flux of `U_in*A` through every x-face. Getting
    /// anything else back means the boundary conditions, the flux or the
    /// reconstruction is wrong, and this pins all three at once.
    #[test]
    fn a_duct_reproduces_uniform_flow_exactly() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        const U_IN: Scalar = 1.0;

        // CaseKind::Big is the unit cube: inlet at xMin, outlet at xMax, the
        // other four sides walls. Its inlet area is therefore exactly 1.
        let (dir, hm) = case_mesh("duct", CaseKind::Big, (7, 5, 4));
        let m = GpuMesh::upload(&g, &hm)?;

        let mut phi = GpuSurfaceScalarField::zeros(&g, &m, "phi")?;
        let mut u = GpuVectorField::zeros(&g, &m, "U")?;

        let r = solve_potential_flow(&g, &hm, &m, &mut phi, &mut u, &spec(U_IN), &controls())?;

        assert!(
            (r.inlet_flux + 1.0).abs() < 1e-10,
            "inlet flux {} is not -U_in*A",
            r.inlet_flux
        );
        assert!(
            (r.outlet_flux - 1.0).abs() < 1e-10,
            "outlet flux {} is not +U_in*A",
            r.outlet_flux
        );
        assert!(r.imbalance().abs() < 1e-12, "imbalance {}", r.imbalance());

        // Uniform flow reconstructs exactly, in all three components.
        for (c, uc) in g.download(&u.f)?.iter().enumerate() {
            assert!((uc.x - U_IN).abs() < 1e-10, "cell {c} Ux = {}", uc.x);
            assert!(uc.y.abs() < 1e-10, "cell {c} Uy = {}", uc.y);
            assert!(uc.z.abs() < 1e-10, "cell {c} Uz = {}", uc.z);
        }

        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }

    /// The headline claim. The plume geometry - one small burner in the floor,
    /// one opening at +x, walls everywhere else - is the case that produced
    /// `max |sum_f phi| = 1.4e-2` from an interpolated velocity. The solved
    /// flux has to be conservative to the solver's tolerance instead, and the
    /// two patch fluxes have to cancel.
    #[test]
    fn the_plume_flux_conserves_mass_where_an_interpolated_one_did_not() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        const U_IN: Scalar = 2.0;

        let (dir, hm) = case_mesh("plume", CaseKind::Plume, (14, 8, 6));
        let m = GpuMesh::upload(&g, &hm)?;

        let mut phi = GpuSurfaceScalarField::zeros(&g, &m, "phi")?;
        let mut u = GpuVectorField::zeros(&g, &m, "U")?;

        let r = solve_potential_flow(&g, &hm, &m, &mut phi, &mut u, &spec(U_IN), &controls())?;

        let inlet = find_patch(&hm, "inlet", "inlet")?;
        let area: Scalar = (inlet.start..inlet.start + inlet.size)
            .map(|bf| hm.b_mag_sf[bf])
            .sum();

        // The inlet flux is prescribed exactly by the boundary condition, so
        // this is a check on the BC plumbing rather than on the solve.
        assert!(
            (r.inlet_flux + U_IN * area).abs() < 1e-10 * U_IN * area,
            "inlet flux {} is not -{U_IN}*{area}",
            r.inlet_flux
        );

        // And this is the solve: everything that went in came out, through the
        // one opening, without the walls leaking.
        let scale = U_IN * area;
        assert!(
            r.imbalance().abs() < 1e-9 * scale,
            "imbalance {} against an inlet flux of {scale}",
            r.imbalance()
        );

        // The conservation claim itself. An interpolated velocity gave 1.4e-2
        // here; anything remotely near that means the flux stopped being the
        // one the laplacian assembled.
        assert!(
            r.max_div_phi < 1e-12 * scale,
            "max |sum_f phi| = {} against an inlet flux of {scale}",
            r.max_div_phi
        );

        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }

    /// No flux may cross a wall. This is what the whole geometry change in
    /// `blockgen` buys, and it is worth asserting separately from the global
    /// balance, which a pair of cancelling leaks would satisfy.
    #[test]
    fn nothing_leaks_through_a_wall() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        let (dir, hm) = case_mesh("walls", CaseKind::Plume, (12, 8, 5));
        let m = GpuMesh::upload(&g, &hm)?;

        let mut phi = GpuSurfaceScalarField::zeros(&g, &m, "phi")?;
        let mut u = GpuVectorField::zeros(&g, &m, "U")?;

        solve_potential_flow(&g, &hm, &m, &mut phi, &mut u, &spec(2.0), &controls())?;

        let bphi = g.download(&phi.bf)?;
        for p in hm.patches.iter().filter(|p| p.kind == PatchKind::Wall) {
            for (bf, q) in (p.start..).zip(&bphi[p.start..p.start + p.size]) {
                assert!(q.abs() < 1e-15, "{} face {bf} carries a flux of {q}", p.name);
            }
        }

        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }

    /// `fvc::reconstruct` must not fabricate a velocity in a direction the
    /// mesh does not resolve: on a 2-D case the empty patch contributes
    /// nothing, the summed tensor is singular, and a naive inverse returns
    /// NaN in every component rather than zero in one.
    #[test]
    fn a_two_dimensional_mesh_reconstructs_without_a_nan() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        // The channel case is 2-D: `back` and `front` are `empty`.
        let (dir, hm) = case_mesh("flat", CaseKind::Channel, (10, 6, 1));
        let m = GpuMesh::upload(&g, &hm)?;

        let mut phi = GpuSurfaceScalarField::zeros(&g, &m, "phi")?;
        let mut u = GpuVectorField::zeros(&g, &m, "U")?;

        let r = solve_potential_flow(&g, &hm, &m, &mut phi, &mut u, &spec(1.0), &controls())?;
        assert!(r.max_div_phi.is_finite(), "the 2-D solve produced a NaN");

        for (c, uc) in g.download(&u.f)?.iter().enumerate() {
            assert!(uc.x.is_finite() && uc.y.is_finite(), "cell {c} is {uc:?}");
            assert_eq!(uc.z, 0.0, "cell {c} grew a velocity in the empty direction");
        }

        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }
}
