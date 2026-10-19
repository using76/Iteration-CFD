// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Field operations: the launchers for `cuda/field.cu`.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` section 4 (the single mixed boundary representation,
//!     marked DESIGN there - it is ours), 2.4 (the boundary delta
//!     coefficient) and 6.1 (bounding k and epsilon, also DESIGN)
//!   Hirsch, *Numerical Computation of Internal and External Flows*, 2nd ed.
//!     (2007), on Robin/mixed boundary treatment
//!   Jasak (1996) section 3.2 for the symmetry-plane condition
//!   ofgpu `SPEC-LIT.md` section 13.3 (the second old time level and the
//!     order the rotation psi^{n-2} <- psi^{n-1} <- psi has to happen in)
//! No GPL-licensed source was consulted.
//!
//! # One boundary condition
//!
//! Every scalar condition in this solver is the triple `(fr, ref_value,
//! ref_grad)` and the one expression
//!
//! ```text
//! psi_b = fr*ref_value + (1 - fr)*(psi_P + ref_grad/Delta_b)
//! ```
//!
//! so [`correct_boundary_conditions`] has no per-type dispatch and a wall
//! function is just a kernel that rewrites the triple. `bc_kind` is consulted
//! for the three conditions the triple cannot express - `Calculated` (the
//! value belongs to a model), `Cyclic` (the value is in a cell the triple
//! cannot name) and, for vectors only, `Symmetry` (the condition is a
//! projection, not a scalar blend). `cuda/field.cu` carries the derivation and
//! the consequence, which is that the matrix sees a symmetry plane as plain
//! zero-gradient.
//!
//! # The elementwise helpers
//!
//! [`copy_field`] and its neighbours take a raw [`DevBuf`] and a length rather
//! than a field, because they are used on the solver's work vectors as much as
//! on fields. They are memory-bound to a fault: each touches its arrays once,
//! and there is deliberately no fused expression evaluator.

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::field::{BcKind, GpuScalarField, GpuSurfaceScalarField, GpuVectorField};
use crate::mesh::GpuMesh;
use crate::{Label, Scalar, Vec3};

/// Entry points of `cuda/field.cu`, resolved once.
pub struct FieldKernels {
    correct_bc_scalar: CudaFunction,
    correct_bc_vector: CudaFunction,
    inlet_outlet: CudaFunction,
    advective_speed: CudaFunction,
    advective_robin_scalar: CudaFunction,
    advective_robin_vector: CudaFunction,
    wedge_ref_grad: CudaFunction,
    pressure_inlet: CudaFunction,
    patch_mean_scalar: CudaFunction,
    patch_mean_vector: CudaFunction,
    extrapolated_sn_grad: CudaFunction,
    // SPEC-LIT 120.10: the recycled inflow's two refresh kernels.
    mapped: CudaFunction,
    turbulent_inlet: CudaFunction,
    // SPEC-LIT 120.12: the synthetic LES inflow's two per-face kernels and
    // the one-thread state step after them.
    digital_filter_inlet: CudaFunction,
    dfsem_inlet: CudaFunction,
    synthetic_step: CudaFunction,

    // SPEC-LIT 139.10 (PAR-14): the decomposed part's processor-face
    // fix-ups, launched right after the evaluation on a mesh that has
    // processor faces and never on any other (`n_proc_faces == 0`). The
    // scalar one is also the fix-up every later interpolated-to-face field
    // launches on a part (`Energy::update_k_eff`'s `rho_face`/`nut_face`).
    pub(crate) proc_face_scalar: CudaFunction,
    proc_face_vector: CudaFunction,

    copy: CudaFunction,
    copy_vector: CudaFunction,
    set: CudaFunction,
    set_vector: CudaFunction,
    bound: CudaFunction,
    clamp: CudaFunction,
    multiply: CudaFunction,
    add: CudaFunction,
    add_vector: CudaFunction,
    divide: CudaFunction,
    scale: CudaFunction,
}

impl FieldKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::FIELD)?;
        Ok(Self {
            correct_bc_scalar: k.func("fldCorrectBcScalar")?,
            correct_bc_vector: k.func("fldCorrectBcVector")?,
            inlet_outlet: k.func("fldInletOutletFraction")?,
            // SPEC-LIT 120.4: the advective outlet's speed and its Robin
            // rewrite, shared with CMP-17's wave-transmissive condition.
            advective_speed: k.func("fldAdvectiveSpeed")?,
            advective_robin_scalar: k.func("fldAdvectiveRobinScalar")?,
            advective_robin_vector: k.func("fldAdvectiveRobinVector")?,
            // SPEC-LIT 119.2: the wedge condition's refGrad rewrite, run
            // before the evaluation on a mesh that carries a wedge.
            wedge_ref_grad: k.func("fldWedgeRefGrad")?,
            // SPEC-LIT 120.5: the pressure-driven velocity inlets' refValue
            // rewrite, run after the flux switch and before the evaluation
            // wherever a caller opted in.
            pressure_inlet: k.func("fldPressureInletValue")?,
            // SPEC-LIT 120.6: the Outlets II conditions' triple rewrites -
            // the four patch-mean kinds on one 1-thread launch per rule,
            // the extrapolated gradient one thread per face of the patch.
            patch_mean_scalar: k.func("fldPatchMeanScalar")?,
            patch_mean_vector: k.func("fldPatchMeanVector")?,
            extrapolated_sn_grad: k.func("fldExtrapolatedSnGrad")?,
            mapped: k.func("fldMapped")?,
            turbulent_inlet: k.func("fldTurbulentInlet")?,
            digital_filter_inlet: k.func("fldDigitalFilterInlet")?,
            dfsem_inlet: k.func("fldDfsemInlet")?,
            synthetic_step: k.func("fldSyntheticStep")?,

            proc_face_scalar: k.func("fldProcFaceScalar")?,
            proc_face_vector: k.func("fldProcFaceVector")?,

            copy: k.func("fldCopy")?,
            copy_vector: k.func("fldCopyVector")?,
            set: k.func("fldSet")?,
            set_vector: k.func("fldSetVector")?,
            bound: k.func("fldBound")?,
            clamp: k.func("fldClamp")?,
            multiply: k.func("fldMultiply")?,
            add: k.func("fldAdd")?,
            add_vector: k.func("fldAddVector")?,
            divide: k.func("fldDivide")?,
            scale: k.func("fldScale")?,
        })
    }
}

/// A field and a mesh have to agree on their sizes, or the kernels index past
/// the end of one of them.
fn check_sizes(
    name: &str,
    n_cells: usize,
    n_bf: usize,
    m: &GpuMesh,
    what: &str,
) -> Result<()> {
    if n_cells != m.n_cells || n_bf != m.n_boundary_faces {
        return Err(Error::Field {
            field: name.to_string(),
            msg: format!(
                "{what}: the field has {n_cells} cells and {n_bf} boundary \
                 faces, the mesh has {} and {}",
                m.n_cells, m.n_boundary_faces
            ),
        });
    }
    Ok(())
}

// ==========================================================================
//  correctBoundaryConditions
// ==========================================================================

/// Re-evaluate every boundary face of a scalar field from its triple.
///
/// Called after anything that changes the internal field, and after any wall
/// function that rewrites the triple. `Calculated` faces are left as they are,
/// because their value is the model's output rather than this expression's.
pub fn correct_boundary_conditions(
    gpu: &Gpu,
    k: &FieldKernels,
    f: &mut GpuScalarField,
    m: &GpuMesh,
) -> Result<()> {
    check_sizes(
        &f.name,
        f.n_cells,
        f.n_boundary_faces,
        m,
        "correct_boundary_conditions",
    )?;

    let n = f.n_boundary_faces;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;

    let func = k.correct_bc_scalar.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&func)
            .arg(&mut f.bf)
            .arg(&f.f)
            .arg(&f.fr)
            .arg(&f.ref_value)
            .arg(&f.ref_grad)
            .arg(&f.bc_kind)
            .arg(&m.b_face_cells)
            .arg(&m.b_delta_coeffs)
            .arg(&m.b_nbr_cell)
            .arg(&m.b_weights)
            // SPEC-LIT (120.2b): the cyclic jump a coupled face sees.
            .arg(&f.jump)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }

    // SPEC-LIT 139.10 (PAR-14): on a decomposed part the processor faces'
    // values are the WHOLE mesh's internal interpolations, which the cyclic
    // branch above cannot form (it reads this side's `b_weights`). The
    // fix-up overwrites exactly the faces `b_proc` marks; every other mesh
    // launches nothing and moves no bit.
    if m.n_proc_faces > 0 {
        let func = k.proc_face_scalar.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&func)
                .arg(&mut f.bf)
                .arg(&f.f)
                .arg(&m.b_proc)
                .arg(&m.b_proc_w)
                .arg(&m.b_face_cells)
                .arg(&m.b_nbr_cell)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }
    Ok(())
}

/// Re-evaluate every boundary face of a vector field.
///
/// Componentwise the same expression, except on a symmetry plane, where the
/// value is the tangential part `U_P - n (n.U_P)` - the average of the cell
/// value and its mirror image - and on a `DirectionMixed` face (SPEC-LIT
/// 120.3), where the kernel evaluates (120.3a) from the face's `dm` block
/// into `bf` ONLY: the matrix sees (120.3h) - `fr = f_t`, `ref_value = r`,
/// `ref_grad = g` - so the tangential part enters it implicitly, and
/// `ref_value` is read here, never rewritten. On a mesh that carries a
/// wedge, `fldWedgeRefGrad` (SPEC-LIT 119.2) rewrites `ref_grad` on the
/// wedge faces from the current cell value BEFORE the evaluation, so the
/// section-4 expression returns (119.2c); a mesh without a wedge launches
/// nothing extra.
pub fn correct_boundary_conditions_vector(
    gpu: &Gpu,
    k: &FieldKernels,
    f: &mut GpuVectorField,
    m: &GpuMesh,
) -> Result<()> {
    check_sizes(
        &f.name,
        f.n_cells,
        f.n_boundary_faces,
        m,
        "correct_boundary_conditions_vector",
    )?;

    let n = f.n_boundary_faces;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;

    // SPEC-LIT 119.2: the wedge faces' ref_grad = Delta_b w, from the
    // CURRENT cell value, before the evaluation reads it. The five
    // condition scalars are formed in f64 from the f64 wedge of 119.1 and
    // cast to Scalar once (SPEC-LIT 118.2). Nothing extra on a mesh
    // without a wedge - it runs bit for bit as before.
    if let Some(w) = &m.wedge {
        let back = w.back as Label;
        let front = w.front as Label;
        let (ax, ay, az) = (
            w.axis.x as Scalar,
            w.axis.y as Scalar,
            w.axis.z as Scalar,
        );
        let sin_theta = w.theta.sin() as Scalar;
        let vers_theta = (2.0 * (0.5 * w.theta).sin() * (0.5 * w.theta).sin()) as Scalar;
        let func = k.wedge_ref_grad.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&func)
                .arg(&mut f.ref_grad)
                .arg(&f.f)
                .arg(&f.bc_kind)
                .arg(&m.b_patch)
                .arg(&m.b_face_cells)
                .arg(&m.b_delta_coeffs)
                .arg(&back)
                .arg(&front)
                .arg(&ax)
                .arg(&ay)
                .arg(&az)
                .arg(&sin_theta)
                .arg(&vers_theta)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }

    let func = k.correct_bc_vector.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&func)
            .arg(&mut f.bf)
            .arg(&f.f)
            .arg(&f.fr)
            .arg(&f.ref_value)
            .arg(&f.ref_grad)
            .arg(&f.dm)
            .arg(&f.bc_kind)
            .arg(&m.b_face_cells)
            .arg(&m.b_delta_coeffs)
            .arg(&m.b_sf)
            .arg(&m.b_nbr_cell)
            .arg(&m.b_weights)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }

    // SPEC-LIT 139.10 (PAR-14): the vector half of the processor-face
    // fix-up - per component fvFluxInternal's expression on the whole
    // mesh's two cells and weight. Nothing on a mesh without processor
    // faces.
    if m.n_proc_faces > 0 {
        let func = k.proc_face_vector.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&func)
                .arg(&mut f.bf)
                .arg(&f.f)
                .arg(&m.b_proc)
                .arg(&m.b_proc_w)
                .arg(&m.b_face_cells)
                .arg(&m.b_nbr_cell)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }
    Ok(())
}

/// Refresh the value fraction of every `InletOutlet` face from the flux.
///
/// `fr = 1` where `phi_b < 0` (the flux points into the domain, so the face
/// has to be told what it is bringing in), `fr = 0` where it points out. Run
/// it before [`correct_boundary_conditions`] and before assembly, once per
/// outer iteration; faces of every other kind are untouched, so it can sweep
/// the whole boundary in one launch.
pub fn update_inlet_outlet(
    gpu: &Gpu,
    k: &FieldKernels,
    fr: &mut DevBuf<Scalar>,
    bc_kind: &DevBuf<Label>,
    phi_b: &DevBuf<Scalar>,
    n_boundary_faces: usize,
) -> Result<()> {
    let n = n_boundary_faces;
    if fr.len() < n || bc_kind.len() < n || phi_b.len() < n {
        return Err(Error::Config(format!(
            "update_inlet_outlet: {n} boundary faces, but fr has {}, bc_kind \
             {} and phi {}",
            fr.len(),
            bc_kind.len(),
            phi_b.len()
        )));
    }
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;

    let func = k.inlet_outlet.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&func)
            .arg(&mut *fr)
            .arg(bc_kind)
            .arg(phi_b)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// [`update_inlet_outlet`] for a scalar field against a flux field.
pub fn update_inlet_outlet_scalar(
    gpu: &Gpu,
    k: &FieldKernels,
    f: &mut GpuScalarField,
    phi: &GpuSurfaceScalarField,
) -> Result<()> {
    let n = f.n_boundary_faces;
    update_inlet_outlet(gpu, k, &mut f.fr, &f.bc_kind, &phi.bf, n)
}

/// [`update_inlet_outlet`] for a vector field against a flux field.
pub fn update_inlet_outlet_vector(
    gpu: &Gpu,
    k: &FieldKernels,
    f: &mut GpuVectorField,
    phi: &GpuSurfaceScalarField,
) -> Result<()> {
    let n = f.n_boundary_faces;
    update_inlet_outlet(gpu, k, &mut f.fr, &f.bc_kind, &phi.bf, n)
}

/// Rewrite `ref_value` on the five pressure-driven velocity inlets of
/// SPEC-LIT 120.5 from the flux the last corrector left, (120.5e)-(120.5i) -
/// one launch over the whole boundary, no host sync (it runs inside
/// CUDA-graph capture, SPEC-LIT 81). Run it after
/// [`update_inlet_outlet_vector`], which switches the fr of kinds 44 and 45
/// on the same flux, and before [`correct_boundary_conditions_vector`],
/// once per outer iteration, wherever the caller set
/// `BcInputs::pressure_inlet`. The host twin of the kernel is
/// [`crate::field::pressure_inlet_values`].
pub fn update_pressure_inlet_velocity(
    gpu: &Gpu,
    k: &FieldKernels,
    u: &mut GpuVectorField,
    phi: &GpuSurfaceScalarField,
    m: &GpuMesh,
) -> Result<()> {
    check_sizes(
        &u.name,
        u.n_cells,
        u.n_boundary_faces,
        m,
        "update_pressure_inlet_velocity",
    )?;
    let n = u.n_boundary_faces;
    let checks = [
        (u.ref_value.len(), "ref_value"),
        (u.bc_kind.len(), "bc_kind"),
        (phi.bf.len(), "phi"),
        (m.b_patch.len(), "b_patch"),
        (m.b_sf.len(), "b_sf"),
        (m.b_mag_sf.len(), "b_mag_sf"),
        (m.b_face_cells.len(), "b_face_cells"),
    ];
    for (len, name) in checks {
        if len < n {
            return Err(Error::Config(format!(
                "update_pressure_inlet_velocity: {n} boundary faces, but \
                 {name} has {len}"
            )));
        }
    }
    // The kernel reads the CELL value behind each face out of the internal
    // field, so `f` covers the cells, not the boundary.
    if u.f.len() < m.n_cells {
        return Err(Error::Config(format!(
            "update_pressure_inlet_velocity: {} cells, but f has {}",
            m.n_cells,
            u.f.len()
        )));
    }
    if u.dm.len() < 12 * n {
        return Err(Error::Config(format!(
            "update_pressure_inlet_velocity: {n} boundary faces, but dm has {}",
            u.dm.len()
        )));
    }
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;

    let func = k.pressure_inlet.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&func)
            .arg(&mut u.ref_value)
            .arg(&u.bc_kind)
            .arg(&m.b_patch)
            .arg(&phi.bf)
            .arg(&m.b_sf)
            .arg(&m.b_mag_sf)
            .arg(&u.dm)
            .arg(&u.f)
            .arg(&m.b_face_cells)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// Rewrite the triple on the Outlets II patches of SPEC-LIT 120.6 from the
/// flux the last corrector left - (120.6a)-(120.6c) and (120.6f) on a
/// scalar field - one launch per rule held by `f.mean_rules`, no host sync
/// (it runs inside CUDA-graph capture, SPEC-LIT 81). Run it after the flux
/// switch and before [`correct_boundary_conditions`], once per outer
/// iteration, wherever the setup opted in with `BcInputs::patch_means`. A
/// field with no rule launches nothing, which keeps every existing case's
/// results bitwise. `grad` is the momentum predictor's `grad p`, which the
/// kind-48 rules read; passing `None` where a rule needs it is an error.
/// The host twins of the kernels are
/// [`crate::field::patch_mean::patch_mean_values_scalar`] and
/// [`crate::field::patch_mean::extrapolated_sn_grad`].
pub fn update_patch_means_scalar(
    gpu: &Gpu,
    k: &FieldKernels,
    f: &mut GpuScalarField,
    phi: &GpuSurfaceScalarField,
    grad: Option<&DevBuf<Vec3>>,
    m: &GpuMesh,
) -> Result<()> {
    check_sizes(
        &f.name,
        f.n_cells,
        f.n_boundary_faces,
        m,
        "update_patch_means_scalar",
    )?;
    if f.mean_rules.is_empty() {
        return Ok(());
    }
    let n = f.n_boundary_faces;
    let checks = [
        (f.fr.len(), "fr"),
        (f.ref_value.len(), "ref_value"),
        (f.ref_grad.len(), "ref_grad"),
        (f.bf.len(), "bf"),
        (f.bc_kind.len(), "bc_kind"),
        (phi.bf.len(), "phi"),
        (m.b_mag_sf.len(), "b_mag_sf"),
        (m.b_face_cells.len(), "b_face_cells"),
    ];
    for (len, name) in checks {
        if len < n {
            return Err(Error::Config(format!(
                "update_patch_means_scalar: {n} boundary faces, but {name} \
                 has {len}"
            )));
        }
    }
    // The kernels read the CELL value behind each face out of the internal
    // field, so `f` covers the cells, not the boundary.
    if f.f.len() < m.n_cells {
        return Err(Error::Config(format!(
            "update_patch_means_scalar: {} cells, but f has {}",
            m.n_cells,
            f.f.len()
        )));
    }
    for rule in &f.mean_rules {
        if rule.start + rule.size > n || rule.src_start + rule.src_size > n {
            return Err(Error::Config(format!(
                "update_patch_means_scalar: field {} carries a {:?} rule on \
                 faces {}..{} (source {}..{}) but the boundary has {n} faces",
                f.name,
                rule.kind,
                rule.start,
                rule.start + rule.size,
                rule.src_start,
                rule.src_start + rule.src_size
            )));
        }
        match rule.kind {
            BcKind::FixedFluxExtrapolatedPressure => {
                let Some(g) = grad else {
                    return Err(Error::Config(format!(
                        "update_patch_means_scalar: field {} carries a \
                         fixedFluxExtrapolatedPressure rule but no grad p \
                         was passed - (120.6f) reads the momentum \
                         predictor's gradient (SPEC-LIT 120.6)",
                        f.name
                    )));
                };
                let start = rule.start as Label;
                let size = rule.size as Label;
                let func = k.extrapolated_sn_grad.clone();
                unsafe {
                    gpu.stream()
                        .launch_builder(&func)
                        .arg(&mut f.fr)
                        .arg(&mut f.ref_value)
                        .arg(&mut f.ref_grad)
                        .arg(g)
                        .arg(&m.b_sf)
                        .arg(&m.b_mag_sf)
                        .arg(&m.b_face_cells)
                        .arg(&start)
                        .arg(&size)
                        .launch(cfg_for(rule.size))?;
                }
            }
            BcKind::FixedMean | BcKind::FixedMeanOutletInlet
            | BcKind::OutletMappedUniformInlet => {
                let kind = rule.kind as Label;
                let start = rule.start as Label;
                let size = rule.size as Label;
                let src_start = rule.src_start as Label;
                let src_size = rule.src_size as Label;
                let target = rule.target.x;
                let fraction = rule.fraction;
                let func = k.patch_mean_scalar.clone();
                unsafe {
                    gpu.stream()
                        .launch_builder(&func)
                        .arg(&mut f.fr)
                        .arg(&mut f.ref_value)
                        .arg(&mut f.ref_grad)
                        .arg(&f.f)
                        .arg(&f.bf)
                        .arg(&phi.bf)
                        .arg(&m.b_mag_sf)
                        .arg(&m.b_face_cells)
                        .arg(&kind)
                        .arg(&start)
                        .arg(&size)
                        .arg(&src_start)
                        .arg(&src_size)
                        .arg(&target)
                        .arg(&fraction)
                        .launch(cfg_for(1))?;
                }
            }
            // 50 is a vector condition; a scalar rule never carries it.
            _ => {}
        }
    }
    Ok(())
}

/// [`update_patch_means_scalar`] for a vector field - (120.6a)-(120.6c)
/// componentwise and (120.6e), kinds 46, 47, 49 and 50, the host twin
/// [`crate::field::patch_mean::patch_mean_values_vector`]. Kind 48 is
/// scalar-only and never reaches this side.
pub fn update_patch_means_vector(
    gpu: &Gpu,
    k: &FieldKernels,
    f: &mut GpuVectorField,
    phi: &GpuSurfaceScalarField,
    m: &GpuMesh,
) -> Result<()> {
    check_sizes(
        &f.name,
        f.n_cells,
        f.n_boundary_faces,
        m,
        "update_patch_means_vector",
    )?;
    if f.mean_rules.is_empty() {
        return Ok(());
    }
    let n = f.n_boundary_faces;
    let checks = [
        (f.fr.len(), "fr"),
        (f.ref_value.len(), "ref_value"),
        (f.ref_grad.len(), "ref_grad"),
        (f.bf.len(), "bf"),
        (f.bc_kind.len(), "bc_kind"),
        (phi.bf.len(), "phi"),
        (m.b_sf.len(), "b_sf"),
        (m.b_mag_sf.len(), "b_mag_sf"),
        (m.b_face_cells.len(), "b_face_cells"),
    ];
    for (len, name) in checks {
        if len < n {
            return Err(Error::Config(format!(
                "update_patch_means_vector: {n} boundary faces, but {name} \
                 has {len}"
            )));
        }
    }
    // The kernels read the CELL value behind each face out of the internal
    // field, so `f` covers the cells, not the boundary.
    if f.f.len() < m.n_cells {
        return Err(Error::Config(format!(
            "update_patch_means_vector: {} cells, but f has {}",
            m.n_cells,
            f.f.len()
        )));
    }
    for rule in &f.mean_rules {
        if rule.start + rule.size > n || rule.src_start + rule.src_size > n {
            return Err(Error::Config(format!(
                "update_patch_means_vector: field {} carries a {:?} rule on \
                 faces {}..{} (source {}..{}) but the boundary has {n} faces",
                f.name,
                rule.kind,
                rule.start,
                rule.start + rule.size,
                rule.src_start,
                rule.src_start + rule.src_size
            )));
        }
        match rule.kind {
            BcKind::FixedMean | BcKind::FixedMeanOutletInlet
            | BcKind::OutletMappedUniformInlet
            | BcKind::MatchedFlowRateOutletVelocity => {
                let kind = rule.kind as Label;
                let start = rule.start as Label;
                let size = rule.size as Label;
                let src_start = rule.src_start as Label;
                let src_size = rule.src_size as Label;
                let tx = rule.target.x;
                let ty = rule.target.y;
                let tz = rule.target.z;
                let fraction = rule.fraction;
                let func = k.patch_mean_vector.clone();
                unsafe {
                    gpu.stream()
                        .launch_builder(&func)
                        .arg(&mut f.fr)
                        .arg(&mut f.ref_value)
                        .arg(&mut f.ref_grad)
                        .arg(&f.f)
                        .arg(&f.bf)
                        .arg(&phi.bf)
                        .arg(&m.b_sf)
                        .arg(&m.b_mag_sf)
                        .arg(&m.b_face_cells)
                        .arg(&kind)
                        .arg(&start)
                        .arg(&size)
                        .arg(&src_start)
                        .arg(&src_size)
                        .arg(&tx)
                        .arg(&ty)
                        .arg(&tz)
                        .arg(&fraction)
                        .launch(cfg_for(1))?;
                }
            }
            // 48 is a scalar condition; a vector rule never carries it.
            _ => {}
        }
    }
    Ok(())
}

/// One `fldMapped` launch (SPEC-LIT 120.10): `ncomp` 1 for a scalar field,
/// 3 for a vector field, whose `ofvec3` buffers the kernel reads as three
/// `ofscalar` a face.
#[allow(clippy::too_many_arguments)]
fn launch_mapped<T: cudarc::driver::DeviceRepr>(
    gpu: &Gpu,
    k: &FieldKernels,
    name: &str,
    fr: &mut DevBuf<Scalar>,
    ref_value: &mut DevBuf<T>,
    ref_grad: &mut DevBuf<T>,
    psi: &DevBuf<T>,
    mp: &crate::field::mapped::GpuMapped,
    m: &GpuMesh,
    ncomp: Label,
) -> Result<()> {
    let r = &mp.rule;
    let n = m.n_boundary_faces;
    if r.start + r.size > n || mp.cells.len() < r.size || r.cells.len() != r.size {
        return Err(Error::Config(format!(
            "update_mapped: field {name} carries a mapped rule on faces {}..{} \
             with {} sampled cells, but the boundary has {n} faces (SPEC-LIT \
             120.10)",
            r.start,
            r.start + r.size,
            r.cells.len()
        )));
    }
    if r.size == 0 {
        return Ok(());
    }
    let start = r.start as Label;
    let size = r.size as Label;
    let set_average: Label = r.set_average.into();
    let (ax, ay, az) = (r.average.x, r.average.y, r.average.z);
    let func = k.mapped.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&func)
            .arg(fr)
            .arg(ref_value)
            .arg(ref_grad)
            .arg(psi)
            .arg(&mp.cells)
            .arg(&m.b_mag_sf)
            .arg(&start)
            .arg(&size)
            .arg(&ncomp)
            .arg(&set_average)
            .arg(&ax)
            .arg(&ay)
            .arg(&az)
            .launch(cfg_for(1))?;
    }
    Ok(())
}

/// Rewrite the `refValue` of every `mapped`/`mappedFixedValue` patch of a
/// scalar field from the cells its offset plane samples - (120.10a)-
/// (120.10c) of SPEC-LIT 120.10 - one launch per rule in `f.mapped`, no
/// host sync (it runs inside CUDA-graph capture, SPEC-LIT 81). Run it once
/// per outer iteration, before [`correct_boundary_conditions`], wherever
/// the setup opted in with `BcInputs::mapped`. A field with no rule
/// launches nothing, which keeps every existing case's results bitwise.
/// The host twin is [`crate::field::mapped::mapped_values`].
pub fn update_mapped_scalar(
    gpu: &Gpu,
    k: &FieldKernels,
    f: &mut GpuScalarField,
    m: &GpuMesh,
) -> Result<()> {
    if f.mapped.is_empty() {
        return Ok(());
    }
    check_sizes(&f.name, f.n_cells, f.n_boundary_faces, m, "update_mapped_scalar")?;
    for mp in &f.mapped {
        launch_mapped(
            gpu, k, &f.name, &mut f.fr, &mut f.ref_value, &mut f.ref_grad, &f.f, mp, m, 1,
        )?;
    }
    Ok(())
}

/// [`update_mapped_scalar`] for a vector field, componentwise (120.10c)
/// with the vector magnitudes of `a` and `m`.
pub fn update_mapped_vector(
    gpu: &Gpu,
    k: &FieldKernels,
    f: &mut GpuVectorField,
    m: &GpuMesh,
) -> Result<()> {
    if f.mapped.is_empty() {
        return Ok(());
    }
    check_sizes(&f.name, f.n_cells, f.n_boundary_faces, m, "update_mapped_vector")?;
    for mp in &f.mapped {
        launch_mapped(
            gpu, k, &f.name, &mut f.fr, &mut f.ref_value, &mut f.ref_grad, &f.f, mp, m, 3,
        )?;
    }
    Ok(())
}

/// One `fldTurbulentInlet` launch (SPEC-LIT 120.10).
fn launch_turbulent<T: cudarc::driver::DeviceRepr>(
    gpu: &Gpu,
    k: &FieldKernels,
    name: &str,
    n_bf: usize,
    ref_value: &mut DevBuf<T>,
    ti: &mut crate::field::mapped::GpuTurbulentInlet,
) -> Result<()> {
    let r = &ti.rule;
    if r.start + r.size > n_bf || r.reference.len() != r.size * r.ncomp {
        return Err(Error::Config(format!(
            "advance_turbulent_inlet: field {name} carries a turbulentInlet \
             rule on faces {}..{} with {} reference values, but the boundary \
             has {n_bf} faces (SPEC-LIT 120.10)",
            r.start,
            r.start + r.size,
            r.reference.len()
        )));
    }
    if r.size == 0 {
        return Ok(());
    }
    let start = r.start as Label;
    let size = r.size as Label;
    let ncomp = r.ncomp as Label;
    let (sx, sy, sz) = (r.scale.x, r.scale.y, r.scale.z);
    let alpha = r.alpha;
    let seed = r.seed;
    let func = k.turbulent_inlet.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&func)
            .arg(ref_value)
            .arg(&ti.reference)
            .arg(&mut ti.step)
            .arg(&start)
            .arg(&size)
            .arg(&ncomp)
            .arg(&sx)
            .arg(&sy)
            .arg(&sz)
            .arg(&alpha)
            .arg(&seed)
            .launch(cfg_for(1))?;
    }
    Ok(())
}

/// Draw the next step of every `turbulentInlet` patch of a scalar field -
/// (120.10g)-(120.10h) of SPEC-LIT 120.10 - one launch per rule in
/// `f.turbulent`, no host sync; the step counter lives on the device. Run
/// it once per TIME STEP, before the step's first outer iteration,
/// wherever the setup opted in with `BcInputs::turbulent_inlet`. A field
/// with no rule launches nothing. The host twin is
/// [`crate::field::mapped::turbulent_advance`].
pub fn advance_turbulent_inlet_scalar(
    gpu: &Gpu,
    k: &FieldKernels,
    f: &mut GpuScalarField,
) -> Result<()> {
    let n_bf = f.n_boundary_faces;
    for ti in &mut f.turbulent {
        launch_turbulent(gpu, k, &f.name, n_bf, &mut f.ref_value, ti)?;
    }
    Ok(())
}

/// [`advance_turbulent_inlet_scalar`] for a vector field, componentwise.
pub fn advance_turbulent_inlet_vector(
    gpu: &Gpu,
    k: &FieldKernels,
    f: &mut GpuVectorField,
) -> Result<()> {
    let n_bf = f.n_boundary_faces;
    for ti in &mut f.turbulent {
        launch_turbulent(gpu, k, &f.name, n_bf, &mut f.ref_value, ti)?;
    }
    Ok(())
}

/// Draw the next time step of every synthetic-inflow patch of a vector field,
/// SPEC-LIT 120.12's digital filter (120.12c)-(120.12e) or eddies
/// (120.12g)-(120.12k): one per-face launch and one state step per rule in
/// `f.synthetic`, no host sync; the state `[n, X]` lives on the device. Run
/// it once per TIME STEP with that step's `dt`, before the step's first
/// outer iteration, wherever the setup opted in with
/// `BcInputs::turbulent_inlet`. A field with no rule launches nothing. The
/// host twins are [`crate::field::synthetic::df_values`] and
/// [`crate::field::synthetic::dfsem_values`]. A digital filter whose time
/// filter at this `dt` would exceed its limits is refused by name.
pub fn advance_synthetic_inlet_vector(
    gpu: &Gpu,
    k: &FieldKernels,
    f: &mut GpuVectorField,
    dt: Scalar,
) -> Result<()> {
    use crate::field::synthetic::{self, SyntheticRule};
    let n_bf = f.n_boundary_faces;
    let dt64 = dt as f64;
    for sy in &mut f.synthetic {
        let (start, size) = (sy.rule.start(), sy.rule.size());
        if start + size > n_bf {
            return Err(Error::Config(format!(
                "advance_synthetic_inlet_vector: field {} carries a synthetic-inflow rule on \
                 faces {start}..{} but the boundary has {n_bf} faces (SPEC-LIT 120.12)",
                f.name,
                start + size
            )));
        }
        if size == 0 {
            continue;
        }
        let start_l = start as Label;
        let size_l = size as Label;
        match &sy.rule {
            SyntheticRule::DigitalFilter(r) => {
                let bt = synthetic::df_time_filters(r, dt64)?;
                let nt: [Label; 3] = [
                    (bt[0].len() / 2) as Label,
                    (bt[1].len() / 2) as Label,
                    (bt[2].len() / 2) as Label,
                ];
                let seed = r.seed;
                let func = k.digital_filter_inlet.clone();
                unsafe {
                    gpu.stream()
                        .launch_builder(&func)
                        .arg(&mut f.ref_value)
                        .arg(&sy.params)
                        .arg(&sy.nodes)
                        .arg(&sy.state)
                        .arg(&start_l)
                        .arg(&size_l)
                        .arg(&dt64)
                        .arg(&seed)
                        .arg(&nt[0])
                        .arg(&nt[1])
                        .arg(&nt[2])
                        .launch(cfg_for(size))?;
                }
            }
            SyntheticRule::Dfsem(r) => {
                let seed = r.seed;
                let func = k.dfsem_inlet.clone();
                unsafe {
                    gpu.stream()
                        .launch_builder(&func)
                        .arg(&mut f.ref_value)
                        .arg(&sy.params)
                        .arg(&sy.rel)
                        .arg(&sy.state)
                        .arg(&start_l)
                        .arg(&size_l)
                        .arg(&dt64)
                        .arg(&seed)
                        .launch(cfg_for(size))?;
                }
            }
        }
        let func = k.synthetic_step.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&func)
                .arg(&mut sy.state)
                .arg(&sy.params)
                .arg(&dt64)
                .launch(cfg_for(1))?;
        }
    }
    Ok(())
}

// ==========================================================================
//  The advective outlet (SPEC-LIT 120.4)
// ==========================================================================

/// Fill the wave speed and the relaxation rate of every face of `match_kind`
/// from the flux - (120.4b), `w = max(phi_b/|S_f|, 0)` (0 where `|S_f|` is
/// 0) and `K = w/lInf`.
///
/// `match_kind` is an input, not a constant: the generic condition passes
/// `BcKind::Advective`, and CMP-17 passes its own kind with its own `w` and
/// `K` of (126.6) to the same kernels. Every length must cover the whole
/// boundary, the same contract [`update_inlet_outlet`] checks.
#[allow(clippy::too_many_arguments)]
pub fn advective_speed(
    gpu: &Gpu,
    k: &FieldKernels,
    w: &mut DevBuf<Scalar>,
    k_relax: &mut DevBuf<Scalar>,
    bc_kind: &DevBuf<Label>,
    match_kind: BcKind,
    phi_b: &DevBuf<Scalar>,
    mag_sf: &DevBuf<Scalar>,
    inv_l_inf: &DevBuf<Scalar>,
    n_boundary_faces: usize,
) -> Result<()> {
    let n = n_boundary_faces;
    for (buf, name) in [
        (w.len(), "w"),
        (k_relax.len(), "k_relax"),
        (bc_kind.len(), "bc_kind"),
        (phi_b.len(), "phi_b"),
        (mag_sf.len(), "mag_sf"),
        (inv_l_inf.len(), "inv_l_inf"),
    ] {
        if buf < n {
            return Err(Error::Config(format!(
                "advective_speed: {n} boundary faces, but {name} has {buf}"
            )));
        }
    }
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let kind = match_kind as Label;

    let func = k.advective_speed.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&func)
            .arg(&mut *w)
            .arg(&mut *k_relax)
            .arg(bc_kind)
            .arg(&kind)
            .arg(phi_b)
            .arg(mag_sf)
            .arg(inv_l_inf)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// Rewrite the triple of every face of `match_kind` from (120.4d), in the
/// 120.4 operation order (`d = a_n + K`, `fr = d/(d + w·Delta_b)`,
/// `refValue = (K·phi_inf - a_0·b_o - a_00·b_oo)/d`, `ref_grad = 0`), with
/// `(a_n, a_0, a_00)` the run's own `DdtCoeffs` - the host twin is
/// [`crate::field::advective::advective_triple`]. The face value is then the
/// ordinary §4 evaluation: run
/// [`correct_boundary_conditions`] after this, never instead of it.
///
/// `b_old`/`b_old_old` are the face's two previous STEP values, rotated once
/// per time step by `field::advective`'s `store_old_time`.
#[allow(clippy::too_many_arguments)]
pub fn advective_robin_scalar(
    gpu: &Gpu,
    k: &FieldKernels,
    fr: &mut DevBuf<Scalar>,
    ref_value: &mut DevBuf<Scalar>,
    ref_grad: &mut DevBuf<Scalar>,
    bc_kind: &DevBuf<Label>,
    match_kind: BcKind,
    w: &DevBuf<Scalar>,
    k_relax: &DevBuf<Scalar>,
    phi_inf: &DevBuf<Scalar>,
    b_old: &DevBuf<Scalar>,
    b_old_old: &DevBuf<Scalar>,
    delta_coeffs: &DevBuf<Scalar>,
    c: crate::timescheme::DdtCoeffs,
    n_boundary_faces: usize,
) -> Result<()> {
    let n = n_boundary_faces;
    for (buf, name) in [
        (fr.len(), "fr"),
        (ref_value.len(), "ref_value"),
        (ref_grad.len(), "ref_grad"),
        (bc_kind.len(), "bc_kind"),
        (w.len(), "w"),
        (k_relax.len(), "k_relax"),
        (phi_inf.len(), "phi_inf"),
        (b_old.len(), "b_old"),
        (b_old_old.len(), "b_old_old"),
        (delta_coeffs.len(), "delta_coeffs"),
    ] {
        if buf < n {
            return Err(Error::Config(format!(
                "advective_robin_scalar: {n} boundary faces, but {name} has {buf}"
            )));
        }
    }
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let kind = match_kind as Label;

    let func = k.advective_robin_scalar.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&func)
            .arg(&mut *fr)
            .arg(&mut *ref_value)
            .arg(&mut *ref_grad)
            .arg(bc_kind)
            .arg(&kind)
            .arg(w)
            .arg(k_relax)
            .arg(phi_inf)
            .arg(b_old)
            .arg(b_old_old)
            .arg(delta_coeffs)
            .arg(&c.a_n)
            .arg(&c.a_0)
            .arg(&c.a_00)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// [`advective_robin_scalar`] for a vector field: one `fr` per face - a
/// vector field shares the fraction across its components (SPEC-LIT 120.4)
/// - and `ref_value`/`phi_inf`/`b_old`/`b_old_old` per component.
#[allow(clippy::too_many_arguments)]
pub fn advective_robin_vector(
    gpu: &Gpu,
    k: &FieldKernels,
    fr: &mut DevBuf<Scalar>,
    ref_value: &mut DevBuf<Vec3>,
    ref_grad: &mut DevBuf<Vec3>,
    bc_kind: &DevBuf<Label>,
    match_kind: BcKind,
    w: &DevBuf<Scalar>,
    k_relax: &DevBuf<Scalar>,
    phi_inf: &DevBuf<Vec3>,
    b_old: &DevBuf<Vec3>,
    b_old_old: &DevBuf<Vec3>,
    delta_coeffs: &DevBuf<Scalar>,
    c: crate::timescheme::DdtCoeffs,
    n_boundary_faces: usize,
) -> Result<()> {
    let n = n_boundary_faces;
    for (buf, name) in [
        (fr.len(), "fr"),
        (ref_value.len(), "ref_value"),
        (ref_grad.len(), "ref_grad"),
        (bc_kind.len(), "bc_kind"),
        (w.len(), "w"),
        (k_relax.len(), "k_relax"),
        (phi_inf.len(), "phi_inf"),
        (b_old.len(), "b_old"),
        (b_old_old.len(), "b_old_old"),
        (delta_coeffs.len(), "delta_coeffs"),
    ] {
        if buf < n {
            return Err(Error::Config(format!(
                "advective_robin_vector: {n} boundary faces, but {name} has {buf}"
            )));
        }
    }
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let kind = match_kind as Label;

    let func = k.advective_robin_vector.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&func)
            .arg(&mut *fr)
            .arg(&mut *ref_value)
            .arg(&mut *ref_grad)
            .arg(bc_kind)
            .arg(&kind)
            .arg(w)
            .arg(k_relax)
            .arg(phi_inf)
            .arg(b_old)
            .arg(b_old_old)
            .arg(delta_coeffs)
            .arg(&c.a_n)
            .arg(&c.a_0)
            .arg(&c.a_00)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

// ==========================================================================
//  Time levels and bounding
// ==========================================================================

/// `f0 = f`: remember this sub-step's starting value.
///
/// This is the ONE-level refresh, and it is what an outer corrector wants: a
/// SIMPLE iteration differences against the start of the sub-step it is in.
/// It deliberately leaves `f00` alone, because `psi^{n-2}` is a property of the
/// TIME step and not of the corrector - see [`advance_time_levels`], which is
/// the once-per-time-step rotation BDF2 needs.
pub fn store_old_time(gpu: &Gpu, k: &FieldKernels, f: &mut GpuScalarField) -> Result<()> {
    let n = f.n_cells;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;

    let func = k.copy.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&func)
            .arg(&mut f.f0)
            .arg(&f.f)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `f0 = f` for a vector field. See [`store_old_time`].
pub fn store_old_time_vector(gpu: &Gpu, k: &FieldKernels, f: &mut GpuVectorField) -> Result<()> {
    let n = f.n_cells;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;

    let func = k.copy_vector.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&func)
            .arg(&mut f.f0)
            .arg(&f.f)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// Rotate the time levels: `psi^{n-2} <- psi^{n-1} <- psi`, IN THAT ORDER.
///
/// **Once per time step, at the top - not once per outer corrector.** Getting
/// that wrong makes BDF2 quietly first order again: a second rotation inside
/// the same step collapses `psi^{n-2}` onto `psi^{n-1}`, and the BDF2
/// coefficients `(3/2, -2, 1/2)` evaluated on two equal old levels sum to the
/// Euler ones. Nothing about the result looks wrong; it is simply one order
/// worse than the log claims.
///
/// The ORDER inside the rotation matters too, and is the reason this is one
/// function and not two calls: done the other way round, `f00` would receive
/// the value `f0` is about to be given, i.e. `psi^n` twice.
///
/// SPEC-LIT 13.3.
pub fn advance_time_levels(gpu: &Gpu, k: &FieldKernels, f: &mut GpuScalarField) -> Result<()> {
    let n = f.n_cells;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;

    let func = k.copy.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&func)
            .arg(&mut f.f00)
            .arg(&f.f0)
            .arg(&nl)
            .launch(cfg_for(n))?;

        gpu.stream()
            .launch_builder(&func)
            .arg(&mut f.f0)
            .arg(&f.f)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// The same rotation for a vector field. See [`advance_time_levels`].
pub fn advance_time_levels_vector(
    gpu: &Gpu,
    k: &FieldKernels,
    f: &mut GpuVectorField,
) -> Result<()> {
    let n = f.n_cells;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;

    let func = k.copy_vector.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&func)
            .arg(&mut f.f00)
            .arg(&f.f0)
            .arg(&nl)
            .launch(cfg_for(n))?;

        gpu.stream()
            .launch_builder(&func)
            .arg(&mut f.f0)
            .arg(&f.f)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// Seed BOTH old levels from the current field.
///
/// The start of a run. Leaving `f00` at whatever the allocation held would make
/// the first BDF2 step differentiate against a fictitious `psi^{n-2}` of zero.
/// The first step degrades to Euler anyway (`SPEC-LIT` §3.3), which is what
/// makes this belt-and-braces rather than load-bearing - but a driver
/// restarting a BDF2 run has to be able to say "both levels are this".
pub fn seed_old_time(gpu: &Gpu, k: &FieldKernels, f: &mut GpuScalarField) -> Result<()> {
    advance_time_levels(gpu, k, f)?;
    advance_time_levels(gpu, k, f)
}

/// See [`seed_old_time`].
pub fn seed_old_time_vector(gpu: &Gpu, k: &FieldKernels, f: &mut GpuVectorField) -> Result<()> {
    advance_time_levels_vector(gpu, k, f)?;
    advance_time_levels_vector(gpu, k, f)
}

/// `f = max(f, lo)` on the internal field AND on the evaluated boundary
/// values.
///
/// SPEC-LIT 6.1 requires `k` and `epsilon` to stay positive and marks the
/// choice of limiter as ours. The boundary values are bounded too, which is a
/// second decision: most of them are about to be regenerated by
/// [`correct_boundary_conditions`] and would not need it, but the ones written
/// by a model (`Calculated`, which that function deliberately does not touch)
/// are exactly the ones a negative value would leak in through.
pub fn bound(gpu: &Gpu, k: &FieldKernels, f: &mut GpuScalarField, lo: Scalar) -> Result<()> {
    bound_field(gpu, k, &mut f.f, lo, f.n_cells)?;
    let n_bf = f.n_boundary_faces;
    bound_field(gpu, k, &mut f.bf, lo, n_bf)
}

// ==========================================================================
//  Elementwise algebra on raw buffers
// ==========================================================================

/// Two buffers both have to be at least `n` long, or a kernel walks off the
/// end of one of them.
fn check_len(what: &str, len: usize, n: usize, which: &str) -> Result<()> {
    if len < n {
        return Err(Error::Config(format!(
            "{what}: {which} holds {len} elements, {n} were asked for"
        )));
    }
    Ok(())
}

/// `dst = src`.
pub fn copy_field(
    gpu: &Gpu,
    k: &FieldKernels,
    dst: &mut DevBuf<Scalar>,
    src: &DevBuf<Scalar>,
    n: usize,
) -> Result<()> {
    check_len("copy_field", dst.len(), n, "dst")?;
    check_len("copy_field", src.len(), n, "src")?;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;

    let func = k.copy.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&func)
            .arg(&mut *dst)
            .arg(src)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `dst = src` for a vector buffer.
pub fn copy_field_vector(
    gpu: &Gpu,
    k: &FieldKernels,
    dst: &mut DevBuf<Vec3>,
    src: &DevBuf<Vec3>,
    n: usize,
) -> Result<()> {
    check_len("copy_field_vector", dst.len(), n, "dst")?;
    check_len("copy_field_vector", src.len(), n, "src")?;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;

    let func = k.copy_vector.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&func)
            .arg(&mut *dst)
            .arg(src)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `dst = value`.
pub fn set_field(
    gpu: &Gpu,
    k: &FieldKernels,
    dst: &mut DevBuf<Scalar>,
    value: Scalar,
    n: usize,
) -> Result<()> {
    check_len("set_field", dst.len(), n, "dst")?;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;

    let func = k.set.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&func)
            .arg(&mut *dst)
            .arg(&value)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `dst = value` for a vector buffer.
pub fn set_field_vector(
    gpu: &Gpu,
    k: &FieldKernels,
    dst: &mut DevBuf<Vec3>,
    value: Vec3,
    n: usize,
) -> Result<()> {
    check_len("set_field_vector", dst.len(), n, "dst")?;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let (x, y, z) = (value.x, value.y, value.z);

    let func = k.set_vector.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&func)
            .arg(&mut *dst)
            .arg(&x)
            .arg(&y)
            .arg(&z)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `x = max(x, lo)`.
pub fn bound_field(
    gpu: &Gpu,
    k: &FieldKernels,
    x: &mut DevBuf<Scalar>,
    lo: Scalar,
    n: usize,
) -> Result<()> {
    check_len("bound_field", x.len(), n, "x")?;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;

    let func = k.bound.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&func)
            .arg(&mut *x)
            .arg(&lo)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `x = min(max(x, lo), hi)`.
pub fn clamp_field(
    gpu: &Gpu,
    k: &FieldKernels,
    x: &mut DevBuf<Scalar>,
    lo: Scalar,
    hi: Scalar,
    n: usize,
) -> Result<()> {
    check_len("clamp_field", x.len(), n, "x")?;
    if lo > hi {
        return Err(Error::Config(format!(
            "clamp_field: the lower bound {lo} is above the upper bound {hi}"
        )));
    }
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;

    let func = k.clamp.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&func)
            .arg(&mut *x)
            .arg(&lo)
            .arg(&hi)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `dst *= src`, elementwise.
pub fn multiply_field(
    gpu: &Gpu,
    k: &FieldKernels,
    dst: &mut DevBuf<Scalar>,
    src: &DevBuf<Scalar>,
    n: usize,
) -> Result<()> {
    check_len("multiply_field", dst.len(), n, "dst")?;
    check_len("multiply_field", src.len(), n, "src")?;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;

    let func = k.multiply.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&func)
            .arg(&mut *dst)
            .arg(src)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `dst += src`, elementwise.
///
/// The partner of [`multiply_field`], and the second half of SPEC-LIT
/// (S59.3)'s blend `dst = dst*mask + other`. Written as two elementwise
/// kernels rather than one fused one with a branch on purpose: a branch would
/// be a second code path, and SPEC-LIT §59.5 has to prove there is only one.
/// On the side the mask keeps, the pair is `x*1.0 + 0.0`, which is `x` in
/// every bit.
pub fn add_field(
    gpu: &Gpu,
    k: &FieldKernels,
    dst: &mut DevBuf<Scalar>,
    src: &DevBuf<Scalar>,
    n: usize,
) -> Result<()> {
    check_len("add_field", dst.len(), n, "dst")?;
    check_len("add_field", src.len(), n, "src")?;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;

    let func = k.add.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&func)
            .arg(&mut *dst)
            .arg(src)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `dst += src`, elementwise, on a vector field.
///
/// The vector twin of [`add_field`]. What SPEC-LIT S18's whole-field source
/// registries accumulate into: several producers, one array, and an assembly
/// that reads it once and asks nothing about where it came from.
pub fn add_field_vector(
    gpu: &Gpu,
    k: &FieldKernels,
    dst: &mut DevBuf<Vec3>,
    src: &DevBuf<Vec3>,
    n: usize,
) -> Result<()> {
    check_len("add_field_vector", dst.len(), n, "dst")?;
    check_len("add_field_vector", src.len(), n, "src")?;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;

    let func = k.add_vector.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&func)
            .arg(&mut *dst)
            .arg(src)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `dst /= src`, elementwise.
///
/// No epsilon in the denominator: a zero divisor means the caller handed over
/// a field it should have bounded first, and regularising it here would hide
/// the bug rather than fix it.
pub fn divide_field(
    gpu: &Gpu,
    k: &FieldKernels,
    dst: &mut DevBuf<Scalar>,
    src: &DevBuf<Scalar>,
    n: usize,
) -> Result<()> {
    check_len("divide_field", dst.len(), n, "dst")?;
    check_len("divide_field", src.len(), n, "src")?;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;

    let func = k.divide.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&func)
            .arg(&mut *dst)
            .arg(src)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `dst *= s`, one scalar for the whole buffer.
pub fn scale_field(
    gpu: &Gpu,
    k: &FieldKernels,
    dst: &mut DevBuf<Scalar>,
    s: Scalar,
    n: usize,
) -> Result<()> {
    check_len("scale_field", dst.len(), n, "dst")?;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;

    let func = k.scale.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&func)
            .arg(&mut *dst)
            .arg(&s)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    use crate::field::BcKind;
    use crate::ldu_ops::tests::chain_mesh;
    use crate::mesh::HostMesh;

    /// The device, a mesh on it, and the kernels - or `None` when there is no
    /// device. Only `Gpu::new` may fail quietly; a kernel that will not load
    /// is a real failure.
    fn ctx(hm: &HostMesh) -> Option<(Gpu, GpuMesh, FieldKernels)> {
        let gpu = Gpu::new(0).ok()?;
        let m = GpuMesh::upload(&gpu, hm).expect("upload the mesh");
        let k = FieldKernels::new(&gpu).expect("load cuda/field.cu");
        Some((gpu, m, k))
    }

    /// The four cell values and the four boundary delta coefficients of
    /// `chain_mesh`, which the closed forms below are written against.
    const PSI: [Scalar; 4] = [1.5, -0.25, 3.0, 0.75];
    const DELTA: [Scalar; 4] = [8.0, 4.0, 2.0, 16.0];
    /// `chain_mesh` puts bf0 and bf2 on cell 0, bf1 and bf3 on cell 3.
    const FACE_CELL: [usize; 4] = [0, 3, 0, 3];

    fn max_diff(a: &[Scalar], b: &[Scalar]) -> Scalar {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .fold(0.0 as Scalar, |m, (x, y)| m.max((x - y).abs()))
    }

    // ----------------------------------------------------------------------
    //  The single mixed expression
    // ----------------------------------------------------------------------

    /// The enum in `src/field.rs` and the `#define`s in `cuda/field.cu` are
    /// two spellings of the same numbers. Nothing but this test stops them
    /// drifting apart, and the failure mode if they do - a cyclic face
    /// silently evaluated as a Dirichlet one - is invisible in the field
    /// files.
    #[test]
    fn bc_kind_values_match_the_device() {
        assert_eq!(BcKind::Calculated as i32, 4);
        assert_eq!(BcKind::Symmetry as i32, 6);
        assert_eq!(BcKind::Cyclic as i32, 7);
        assert_eq!(BcKind::InletOutlet as i32, 8);

        // The FLUX-SWITCHED block of `cuda/field.cu`. Every kind in it is
        // Dirichlet on inflow and zero-gradient on outflow, and the kernel
        // tests the RANGE - so a kind slipped in above 12 would be switched
        // by a flux it has nothing to do with, and one added inside the
        // range without being flux-switched would be switched when it should
        // not be.
        assert_eq!(crate::field::FLUX_SWITCHED_FIRST, 8);
        assert_eq!(crate::field::FLUX_SWITCHED_LAST, 12);
        assert_eq!(BcKind::PressureInletOutletVelocity as i32, 12);

        // `OFGPU_BC_NUT_LOW_RE` in cuda/turbulence.cu, where nu_t is pinned
        // to zero at the wall (SPEC-LIT 15.2).
        assert_eq!(BcKind::NutLowReWallFunction as i32, 22);

        // `OFGPU_BC_DIRECTION_MIXED` in cuda/field.cu: the tensorial mixed
        // condition of SPEC-LIT 120.3, the branch in fldCorrectBcVector that
        // evaluates (120.3a) into the face value while the matrix sees
        // (120.3h) - the tangential part implicitly.
        assert_eq!(BcKind::DirectionMixed as i32, 37);

        // `OFGPU_BC_OUTLET_INLET` and `OFGPU_BC_ADVECTIVE` in cuda/field.cu
        // (SPEC-LIT 120.4): the reverse flux switch, OUTSIDE the range above
        // because its Dirichlet half is the OUTflow half, and the Robin
        // outlet whose triple `fldAdvectiveRobin*` rewrites from (120.4d).
        assert_eq!(BcKind::OutletInlet as i32, 38);
        assert_eq!(BcKind::Advective as i32, 39);

        // `OFGPU_BC_WEDGE` in cuda/field.cu (SPEC-LIT 119.2): the wedge
        // condition's kind, whose ref_grad the fldWedgeRefGrad kernel
        // rewrites before the evaluation and which adds no branch to
        // fldCorrectBcScalar/fldCorrectBcVector.
        assert_eq!(BcKind::Wedge as i32, 40);

        // `OFGPU_BC_PRESSURE_INLET_FIRST`/`_LAST` in cuda/field.cu AND in
        // cuda/momentum.cu (SPEC-LIT 120.5): the five pressure-driven
        // velocity inlets, whose ref_value the fldPressureInletValue kernel
        // rewrites from the flux and whose flux the momFluxIsPrescribed in
        // momentum.cu leaves to the pressure equation whatever their fr -
        // both macros must keep naming the same five kinds.
        assert_eq!(crate::field::PRESSURE_INLET_FIRST, 41);
        assert_eq!(crate::field::PRESSURE_INLET_LAST, 45);
        assert_eq!(BcKind::PressureInletVelocity as i32, 41);
        assert_eq!(BcKind::PressureInletUniformVelocity as i32, 42);
        assert_eq!(BcKind::PressureDirectedInletVelocity as i32, 43);
        assert_eq!(BcKind::PressureDirectedInletOutletVelocity as i32, 44);
        assert_eq!(BcKind::PressureInletOutletParSlipVelocity as i32, 45);

        // `OFGPU_BC_FIXED_MEAN` .. `OFGPU_BC_MATCHED_FLOW_RATE_OUTLET` in
        // cuda/field.cu (SPEC-LIT 120.6): the five Outlets II kinds, whose
        // triples the fldPatchMeanScalar, fldPatchMeanVector and
        // fldExtrapolatedSnGrad kernels rewrite once per outer iteration
        // wherever the caller opted in (BcInputs::patch_means).
        assert_eq!(BcKind::FixedMean as i32, 46);
        assert_eq!(BcKind::FixedMeanOutletInlet as i32, 47);
        assert_eq!(BcKind::FixedFluxExtrapolatedPressure as i32, 48);
        assert_eq!(BcKind::OutletMappedUniformInlet as i32, 49);
        assert_eq!(BcKind::MatchedFlowRateOutletVelocity as i32, 50);
    }

    /// Every branch of SPEC-LIT section 4 against the closed form it
    /// specialises to: `fr = 1` Dirichlet, `fr = 0 g = 0` zero-gradient,
    /// `fr = 0 g != 0` Neumann, `0 < fr < 1` Robin.
    #[test]
    fn every_scalar_bc_branch_matches_the_closed_form() {
        let hm = chain_mesh();
        let Some((gpu, m, k)) = ctx(&hm) else { return };

        let mut f = GpuScalarField::zeros(&gpu, &m, "psi").expect("field");
        gpu.write(&mut f.f, &PSI).expect("psi");

        // One face per branch. bf2 and bf3 carry a cyclic neighbour in the
        // mesh, and are given non-cyclic kinds here on purpose: the kind, not
        // the topology, is what selects the branch.
        let kind = [
            BcKind::FixedValue as Label,
            BcKind::ZeroGradient as Label,
            BcKind::FixedGradient as Label,
            BcKind::Mixed as Label,
        ];
        let fr: [Scalar; 4] = [1.0, 0.0, 0.0, 0.4];
        let ref_value: [Scalar; 4] = [2.75, 0.0, 0.0, -1.25];
        let ref_grad: [Scalar; 4] = [0.0, 0.0, 5.0, -3.0];

        gpu.write(&mut f.bc_kind, &kind).expect("kind");
        gpu.write(&mut f.fr, &fr).expect("fr");
        gpu.write(&mut f.ref_value, &ref_value).expect("refValue");
        gpu.write(&mut f.ref_grad, &ref_grad).expect("refGrad");

        correct_boundary_conditions(&gpu, &k, &mut f, &m).expect("correctBcs");
        gpu.sync().expect("sync");

        let want: Vec<Scalar> = (0..4)
            .map(|i| {
                let p = PSI[FACE_CELL[i]];
                fr[i] * ref_value[i] + (1.0 - fr[i]) * (p + ref_grad[i] / DELTA[i])
            })
            .collect();

        let got = gpu.download(&f.bf).expect("bf");
        assert!(max_diff(&got, &want) < 1e-14, "{got:?} vs {want:?}");

        // And each specialisation says what section 4 says it says.
        assert!((got[0] - 2.75).abs() < 1e-14, "Dirichlet: {}", got[0]);
        assert!((got[1] - PSI[3]).abs() < 1e-14, "zero-gradient: {}", got[1]);
        assert!(
            (got[2] - (PSI[0] + 5.0 / 2.0)).abs() < 1e-14,
            "Neumann: {}",
            got[2]
        );
        // Robin: the boundary value sits between the reference value and the
        // extrapolated internal one.
        let extrapolated = PSI[3] + -3.0 / 16.0;
        assert!(
            (got[3] - (0.4 * -1.25 + 0.6 * extrapolated)).abs() < 1e-14,
            "Robin: {}",
            got[3]
        );
    }

    #[test]
    fn a_cyclic_face_interpolates_across_the_couple() {
        let hm = chain_mesh();
        let Some((gpu, m, k)) = ctx(&hm) else { return };

        let mut f = GpuScalarField::zeros(&gpu, &m, "psi").expect("field");
        gpu.write(&mut f.f, &PSI).expect("psi");

        // A triple that would give a completely different answer if the
        // cyclic branch were not taken.
        let kind = [
            BcKind::ZeroGradient as Label,
            BcKind::ZeroGradient as Label,
            BcKind::Cyclic as Label,
            BcKind::Cyclic as Label,
        ];
        gpu.write(&mut f.bc_kind, &kind).expect("kind");
        gpu.write(&mut f.fr, &[0.0, 0.0, 1.0, 1.0]).expect("fr");
        gpu.write(&mut f.ref_value, &[0.0, 0.0, 99.0, 99.0])
            .expect("refValue");

        correct_boundary_conditions(&gpu, &k, &mut f, &m).expect("correctBcs");
        gpu.sync().expect("sync");

        let got = gpu.download(&f.bf).expect("bf");

        // bf2: cell 0, weight 0.35, neighbour cell 3.
        // bf3: cell 3, weight 0.65, neighbour cell 0.
        let want2 = 0.35 * PSI[0] + 0.65 * PSI[3];
        let want3 = 0.65 * PSI[3] + 0.35 * PSI[0];
        assert!((got[2] - want2).abs() < 1e-14, "{} vs {want2}", got[2]);
        assert!(
            ((got[3] - want3).abs() as f64) < crate::precision::roundoff_tol(1e-14),
            "{} vs {want3}",
            got[3]
        );
    }

    /// SPEC-LIT §31.1's advection test, at the mechanism `correctBcs` actually
    /// exercises: a UNIFORM field, on a REAL periodic-channel mesh built by
    /// `blockgen::build_mesh` (the new case-file path, not a hand-rolled
    /// `HostMesh`), comes back unchanged once every boundary face - the
    /// cyclic couple included - is re-evaluated from it. A wrong `b_nbr_cell`
    /// or a wrong cyclic weight would show up here as a boundary value that
    /// is no longer the uniform one.
    #[test]
    fn a_uniform_field_survives_a_generated_cyclic_pair() {
        let mut b = crate::blockgen::BlockSpec::default();
        b.x.hi = 2.0;
        b.x.n = 6;
        b.y.hi = 1.0;
        b.y.n = 4;
        b.z.hi = 0.5;
        b.z.n = 3;
        b.set_cyclic_axis(0).expect("axis 0 is x");
        let hm = crate::blockgen::build_mesh(&b).expect("build_mesh");

        let Some((gpu, m, k)) = ctx(&hm) else { return };

        const T0: Scalar = 42.5;
        let mut f = GpuScalarField::zeros(&gpu, &m, "T").expect("field");
        gpu.write(&mut f.f, &vec![T0; hm.n_cells]).expect("T");

        correct_boundary_conditions(&gpu, &k, &mut f, &m).expect("correctBcs");
        gpu.sync().expect("sync");

        let got = gpu.download(&f.bf).expect("bf");
        for (i, &v) in got.iter().enumerate() {
            assert!(
                (v - T0).abs() < 1e-10,
                "boundary face {i}: {v} != uniform {T0} - a periodic pair must not \
                 perturb a field that had nothing to advect"
            );
        }
    }

    /// SPEC-LIT §34.2's own 34.3-table check: the same test as
    /// [`a_uniform_field_survives_a_generated_cyclic_pair`], on a mesh with
    /// TWO cyclic pairs (a plane channel periodic in x and y) - a uniform
    /// field must return to itself through both couples at once, not just
    /// whichever one an implementation happened to wire up first.
    #[test]
    fn a_uniform_field_survives_two_generated_cyclic_pairs() {
        let mut b = crate::blockgen::BlockSpec::default();
        b.x.hi = 2.0;
        b.x.n = 6;
        b.y.hi = 1.0;
        b.y.n = 4;
        b.z.hi = 0.5;
        b.z.n = 3;
        b.set_cyclic_axis(0).expect("axis 0 is x");
        b.set_cyclic_axis(1).expect("axis 1 is y");
        let hm = crate::blockgen::build_mesh(&b).expect("build_mesh");

        let Some((gpu, m, k)) = ctx(&hm) else { return };

        const T0: Scalar = -7.25;
        let mut f = GpuScalarField::zeros(&gpu, &m, "T").expect("field");
        gpu.write(&mut f.f, &vec![T0; hm.n_cells]).expect("T");

        correct_boundary_conditions(&gpu, &k, &mut f, &m).expect("correctBcs");
        gpu.sync().expect("sync");

        let got = gpu.download(&f.bf).expect("bf");
        for (i, &v) in got.iter().enumerate() {
            assert!(
                (v - T0).abs() < 1e-10,
                "boundary face {i}: {v} != uniform {T0} - two periodic pairs together \
                 must not perturb a field that had nothing to advect"
            );
        }
    }

    #[test]
    fn a_calculated_face_is_left_alone() {
        let hm = chain_mesh();
        let Some((gpu, m, k)) = ctx(&hm) else { return };

        let mut f = GpuScalarField::zeros(&gpu, &m, "nut").expect("field");
        gpu.write(&mut f.f, &PSI).expect("psi");

        let written: [Scalar; 4] = [11.0, 12.0, 13.0, 14.0];
        gpu.write(&mut f.bf, &written).expect("bf");

        let kind = [BcKind::Calculated as Label; 4];
        gpu.write(&mut f.bc_kind, &kind).expect("kind");
        // A triple that WOULD overwrite it, if the kind were not honoured.
        gpu.write(&mut f.fr, &[1.0 as Scalar; 4]).expect("fr");
        gpu.write(&mut f.ref_value, &[0.0 as Scalar; 4])
            .expect("refValue");

        correct_boundary_conditions(&gpu, &k, &mut f, &m).expect("correctBcs");
        gpu.sync().expect("sync");

        let got = gpu.download(&f.bf).expect("bf");
        assert!(max_diff(&got, &written) < 1e-15, "{got:?}");
    }

    /// A degenerate boundary face has `Delta_b = 0`, and `fr = 1` would then
    /// multiply an infinity by zero. It has to come out Dirichlet, not NaN.
    #[test]
    fn a_zero_delta_coefficient_does_not_produce_a_nan() {
        let mut hm = chain_mesh();
        hm.b_delta_coeffs[0] = 0.0;
        let Some((gpu, m, k)) = ctx(&hm) else { return };

        let mut f = GpuScalarField::zeros(&gpu, &m, "psi").expect("field");
        gpu.write(&mut f.f, &PSI).expect("psi");
        gpu.write(&mut f.bc_kind, &[BcKind::FixedValue as Label; 4])
            .expect("kind");
        gpu.write(&mut f.fr, &[1.0 as Scalar; 4]).expect("fr");
        gpu.write(&mut f.ref_value, &[2.0 as Scalar; 4])
            .expect("refValue");
        gpu.write(&mut f.ref_grad, &[7.0 as Scalar; 4])
            .expect("refGrad");

        correct_boundary_conditions(&gpu, &k, &mut f, &m).expect("correctBcs");
        gpu.sync().expect("sync");

        let got = gpu.download(&f.bf).expect("bf");
        assert!(got.iter().all(|v| v.is_finite()), "{got:?}");
        assert!((got[0] - 2.0).abs() < 1e-15);
    }

    // ----------------------------------------------------------------------
    //  Vectors
    // ----------------------------------------------------------------------

    #[test]
    // f32 (F32-17): the exact-zero normal residual is ~1e-8, not R-eligible
    #[cfg_attr(feature = "single", ignore = "fails at f32: SPEC-LIT 112.3")]
    fn vector_symmetry_removes_the_normal_component_and_keeps_the_rest() {
        let mut hm = chain_mesh();
        // A slanted face, so the projection is a real one rather than the
        // deletion of a single component. (0.36, 0.48, 0.8) is a unit vector.
        let n = Vec3::new(0.36, 0.48, 0.8);
        hm.b_sf[1] = n * hm.b_mag_sf[1];

        let Some((gpu, m, k)) = ctx(&hm) else { return };

        let u: Vec<Vec3> = vec![
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-1.0, 0.5, 0.25),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.5, -1.5, 0.75),
        ];

        let mut f = GpuVectorField::zeros(&gpu, &m, "U").expect("field");
        gpu.write(&mut f.f, &u).expect("U");
        gpu.write(&mut f.bc_kind, &[BcKind::Symmetry as Label; 4])
            .expect("kind");

        correct_boundary_conditions_vector(&gpu, &k, &mut f, &m).expect("correctBcs");
        gpu.sync().expect("sync");

        let got = gpu.download(&f.bf).expect("bf");

        // bf1 sits on cell 3.
        let up = u[3];
        let want = up - n * n.dot(up);
        assert!((got[1] - want).mag() < 1e-14, "{:?} vs {want:?}", got[1]);
        assert!(n.dot(got[1]).abs() < 1e-14, "normal component survived");

        // The tangential part is untouched: subtracting the projection twice
        // is the same as subtracting it once.
        let twice = got[1] - n * n.dot(got[1]);
        assert!((twice - got[1]).mag() < 1e-14);
    }

    #[test]
    fn vector_boundaries_use_the_same_expression_per_component() {
        let hm = chain_mesh();
        let Some((gpu, m, k)) = ctx(&hm) else { return };

        let u: Vec<Vec3> = vec![
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-1.0, 0.5, 0.25),
            Vec3::new(0.0, -2.0, 1.0),
            Vec3::new(2.5, -1.5, 0.75),
        ];

        let mut f = GpuVectorField::zeros(&gpu, &m, "U").expect("field");
        gpu.write(&mut f.f, &u).expect("U");

        let kind = [
            BcKind::FixedValue as Label,
            BcKind::FixedGradient as Label,
            BcKind::Mixed as Label,
            BcKind::ZeroGradient as Label,
        ];
        let fr: [Scalar; 4] = [1.0, 0.0, 0.25, 0.0];
        let ref_value = vec![
            Vec3::new(9.0, 8.0, 7.0),
            Vec3::ZERO,
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::ZERO,
        ];
        let ref_grad = vec![
            Vec3::ZERO,
            Vec3::new(1.0, -2.0, 3.0),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::ZERO,
        ];

        gpu.write(&mut f.bc_kind, &kind).expect("kind");
        gpu.write(&mut f.fr, &fr).expect("fr");
        gpu.write(&mut f.ref_value, &ref_value).expect("refValue");
        gpu.write(&mut f.ref_grad, &ref_grad).expect("refGrad");

        correct_boundary_conditions_vector(&gpu, &k, &mut f, &m).expect("correctBcs");
        gpu.sync().expect("sync");

        let got = gpu.download(&f.bf).expect("bf");

        for i in 0..4 {
            let p = u[FACE_CELL[i]];
            let g = ref_grad[i] / DELTA[i];
            let want = ref_value[i] * fr[i] + (p + g) * (1.0 - fr[i]);
            assert!((got[i] - want).mag() < 1e-14, "face {i}: {:?} vs {want:?}", got[i]);
        }
    }

    // ----------------------------------------------------------------------
    //  inletOutlet
    // ----------------------------------------------------------------------

    #[test]
    fn inlet_outlet_switches_on_the_sign_of_the_flux() {
        let hm = chain_mesh();
        let Some((gpu, m, k)) = ctx(&hm) else { return };

        let mut f = GpuScalarField::zeros(&gpu, &m, "T").expect("field");
        let kind = [
            BcKind::InletOutlet as Label,
            BcKind::InletOutlet as Label,
            BcKind::FixedValue as Label,
            BcKind::InletOutlet as Label,
        ];
        gpu.write(&mut f.bc_kind, &kind).expect("kind");
        gpu.write(&mut f.fr, &[0.5 as Scalar; 4]).expect("fr");

        let mut phi = GpuSurfaceScalarField::zeros(&gpu, &m, "phi").expect("phi");
        // Sf points out of the domain: negative is inflow.
        gpu.write(&mut phi.bf, &[-1.0 as Scalar, 2.0, -3.0, 0.0])
            .expect("phiBf");

        update_inlet_outlet_scalar(&gpu, &k, &mut f, &phi).expect("inletOutlet");
        gpu.sync().expect("sync");

        let got = gpu.download(&f.fr).expect("fr");
        assert_eq!(got[0], 1.0, "inflow should be Dirichlet");
        assert_eq!(got[1], 0.0, "outflow should be zero-gradient");
        assert_eq!(got[2], 0.5, "a face of another kind must not be touched");
        assert_eq!(got[3], 0.0, "zero flux is not inflow");
    }

    // ----------------------------------------------------------------------
    //  Time levels and elementwise algebra
    // ----------------------------------------------------------------------

    #[test]
    fn store_old_time_copies_the_internal_field() {
        let hm = chain_mesh();
        let Some((gpu, m, k)) = ctx(&hm) else { return };

        let mut f = GpuScalarField::zeros(&gpu, &m, "psi").expect("field");
        gpu.write(&mut f.f, &PSI).expect("psi");
        store_old_time(&gpu, &k, &mut f).expect("storeOldTime");

        // Moving on a step must not drag the stored level with it.
        gpu.write(&mut f.f, &[0.0 as Scalar; 4]).expect("psi");
        gpu.sync().expect("sync");

        assert!(max_diff(&gpu.download(&f.f0).expect("f0"), &PSI) < 1e-15);

        let mut v = GpuVectorField::zeros(&gpu, &m, "U").expect("field");
        let u = vec![
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(4.0, 5.0, 6.0),
            Vec3::new(7.0, 8.0, 9.0),
            Vec3::new(-1.0, -2.0, -3.0),
        ];
        gpu.write(&mut v.f, &u).expect("U");
        store_old_time_vector(&gpu, &k, &mut v).expect("storeOldTime");
        gpu.sync().expect("sync");

        let got = gpu.download(&v.f0).expect("U0");
        for i in 0..4 {
            assert!((got[i] - u[i]).mag() < 1e-15);
        }
    }

    #[test]
    fn the_elementwise_helpers_do_what_they_say() {
        let hm = chain_mesh();
        let Some((gpu, _m, k)) = ctx(&hm) else { return };

        let n = 5usize;
        let src = gpu.upload(&[1.0 as Scalar, 2.0, 4.0, -8.0, 0.5]).expect("src");
        let mut dst: DevBuf<Scalar> = gpu.zeros(n).expect("dst");

        copy_field(&gpu, &k, &mut dst, &src, n).expect("copy");
        gpu.sync().expect("sync");
        assert_eq!(gpu.download(&dst).expect("dst"), vec![1.0, 2.0, 4.0, -8.0, 0.5]);

        multiply_field(&gpu, &k, &mut dst, &src, n).expect("multiply");
        gpu.sync().expect("sync");
        assert_eq!(
            gpu.download(&dst).expect("dst"),
            vec![1.0, 4.0, 16.0, 64.0, 0.25]
        );

        divide_field(&gpu, &k, &mut dst, &src, n).expect("divide");
        gpu.sync().expect("sync");
        assert_eq!(
            gpu.download(&dst).expect("dst"),
            vec![1.0, 2.0, 4.0, -8.0, 0.5]
        );

        scale_field(&gpu, &k, &mut dst, 2.0, n).expect("scale");
        gpu.sync().expect("sync");
        assert_eq!(
            gpu.download(&dst).expect("dst"),
            vec![2.0, 4.0, 8.0, -16.0, 1.0]
        );

        bound_field(&gpu, &k, &mut dst, 0.0, n).expect("bound");
        gpu.sync().expect("sync");
        assert_eq!(gpu.download(&dst).expect("dst"), vec![2.0, 4.0, 8.0, 0.0, 1.0]);

        clamp_field(&gpu, &k, &mut dst, 1.5, 5.0, n).expect("clamp");
        gpu.sync().expect("sync");
        assert_eq!(gpu.download(&dst).expect("dst"), vec![2.0, 4.0, 5.0, 1.5, 1.5]);

        set_field(&gpu, &k, &mut dst, -7.0, n).expect("set");
        gpu.sync().expect("sync");
        assert_eq!(gpu.download(&dst).expect("dst"), vec![-7.0; 5]);

        // Vectors.
        let mut v: DevBuf<Vec3> = gpu.zeros(n).expect("v");
        set_field_vector(&gpu, &k, &mut v, Vec3::new(1.0, -2.0, 3.0), n).expect("setVec");
        gpu.sync().expect("sync");
        assert!(gpu
            .download(&v)
            .expect("v")
            .iter()
            .all(|x| *x == Vec3::new(1.0, -2.0, 3.0)));

        let mut w: DevBuf<Vec3> = gpu.zeros(n).expect("w");
        copy_field_vector(&gpu, &k, &mut w, &v, n).expect("copyVec");
        gpu.sync().expect("sync");
        assert_eq!(gpu.download(&w).expect("w"), gpu.download(&v).expect("v"));
    }

    /// `bound` reaches the boundary values too - the ones a model wrote and
    /// `correct_boundary_conditions` will not regenerate.
    #[test]
    fn bound_covers_the_boundary_values() {
        let hm = chain_mesh();
        let Some((gpu, m, k)) = ctx(&hm) else { return };

        let mut f = GpuScalarField::zeros(&gpu, &m, "k").expect("field");
        gpu.write(&mut f.f, &[-1.0 as Scalar, 0.5, -0.25, 2.0])
            .expect("k");
        gpu.write(&mut f.bf, &[-3.0 as Scalar, 1.0, -0.5, 4.0])
            .expect("kBf");

        bound(&gpu, &k, &mut f, 1e-9).expect("bound");
        gpu.sync().expect("sync");

        let internal = gpu.download(&f.f).expect("k");
        let boundary = gpu.download(&f.bf).expect("kBf");
        assert!(internal.iter().all(|v| *v >= 1e-9), "{internal:?}");
        assert!(boundary.iter().all(|v| *v >= 1e-9), "{boundary:?}");
        // What was already above the floor is untouched.
        assert_eq!(internal[1], 0.5);
        assert_eq!(boundary[3], 4.0);
    }

    /// A grid dimension of zero is an invalid launch configuration, not a
    /// no-op. A 2-D case with no boundary faces on some field, or a work
    /// vector of length zero, would otherwise take the whole run down with
    /// CUDA_ERROR_INVALID_VALUE.
    #[test]
    fn an_empty_field_launches_nothing() {
        let Some(gpu) = Gpu::new(0).ok() else { return };
        let k = FieldKernels::new(&gpu).expect("load cuda/field.cu");

        let hm = HostMesh {
            n_cells: 0,
            n_internal_faces: 0,
            n_boundary_faces: 0,
            cf_offset: vec![0],
            bcf_offset: vec![0],
            ..Default::default()
        };
        let m = GpuMesh::upload(&gpu, &hm).expect("upload");

        let mut s = GpuScalarField::zeros(&gpu, &m, "psi").expect("field");
        let mut v = GpuVectorField::zeros(&gpu, &m, "U").expect("field");
        let phi = GpuSurfaceScalarField::zeros(&gpu, &m, "phi").expect("phi");

        correct_boundary_conditions(&gpu, &k, &mut s, &m).expect("correctBcs");
        correct_boundary_conditions_vector(&gpu, &k, &mut v, &m).expect("correctBcs");
        update_inlet_outlet_scalar(&gpu, &k, &mut s, &phi).expect("inletOutlet");
        store_old_time(&gpu, &k, &mut s).expect("storeOldTime");
        store_old_time_vector(&gpu, &k, &mut v).expect("storeOldTime");
        bound(&gpu, &k, &mut s, 0.0).expect("bound");

        let empty: DevBuf<Scalar> = gpu.zeros(0).expect("alloc");
        let mut out: DevBuf<Scalar> = gpu.zeros(0).expect("alloc");
        copy_field(&gpu, &k, &mut out, &empty, 0).expect("copy");
        multiply_field(&gpu, &k, &mut out, &empty, 0).expect("multiply");
        divide_field(&gpu, &k, &mut out, &empty, 0).expect("divide");
        set_field(&gpu, &k, &mut out, 1.0, 0).expect("set");
        scale_field(&gpu, &k, &mut out, 1.0, 0).expect("scale");
        bound_field(&gpu, &k, &mut out, 0.0, 0).expect("bound");
        clamp_field(&gpu, &k, &mut out, 0.0, 1.0, 0).expect("clamp");
        gpu.sync().expect("sync");
    }

    #[test]
    fn the_helpers_refuse_a_buffer_that_is_too_short() {
        let hm = chain_mesh();
        let Some((gpu, _m, k)) = ctx(&hm) else { return };

        let src: DevBuf<Scalar> = gpu.zeros(3).expect("src");
        let mut dst: DevBuf<Scalar> = gpu.zeros(2).expect("dst");
        assert!(copy_field(&gpu, &k, &mut dst, &src, 3).is_err());
        assert!(clamp_field(&gpu, &k, &mut dst, 2.0, 1.0, 2).is_err());
    }
}
