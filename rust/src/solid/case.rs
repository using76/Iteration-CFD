// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The thermo-elastic CASE reaching SPEC-LIT §95's solid - the bridge from a
//! lowered `mechanics` block to the outer loop, from its result to the
//! per-region VTU, and the §13.4.2 banner. SPEC-LIT §96.
//!
//! Provenance: ORIGINAL - a bridge, not numerics. Every equation it reaches is
//! §95's, implemented in `crate::solid`, and the coupling parameter it prints
//! is `crate::solid::two_way_coupling_delta`'s (Boley & Weiner, *Theory of
//! Thermal Stresses*, Wiley (1960) ch. 1-2). No GPL-licensed source was consulted.

use std::path::{Path, PathBuf};

use crate::cht::ChtSolution;
use crate::error::{Error, IoContext, Result};
use crate::io::case_cht::{LoweredChtCase, LoweredMechanicalBc};
use crate::io::output_types::OutputField;
use crate::io::pointfield::PointInterpolator;
use crate::io::vtu::write_vtu_points;
use crate::mesh::{GpuMesh, HostMesh};
use crate::solid::bc::CompBc;
use crate::solid::displacement::Displacement;
use crate::solid::materials::MaterialMap;
use crate::solid::outer::{self, OuterControls, OuterReport, Relaxation};
use crate::solid::stress::{point_displacement, StressFields};
use crate::solver::SolverPerformance;
use crate::{Gpu, Label, Scalar, Tensor, Vec3};

// ==========================================================================
//  The thermal verdict - SPEC-LIT 96.2 refuses to go on otherwise
// ==========================================================================

/// SPEC-LIT 96.2: the thermal solve every mechanical region reads its `T`
/// from has to have CONVERGED - every region's own 8.4 residual inside
/// `numerics.tolerance`, S1's verdict read, never re-derived.
pub fn thermal_converged(low: &LoweredChtCase, sol: &ChtSolution) -> Result<()> {
    let names: Vec<&str> = sol.mesh.regions.iter().map(|r| r.name.as_str()).collect();
    verdict(&names, &sol.region_residuals, sol.converged, low.solver.tolerance)
}

/// The host-testable half: `names` in region order, one residual per region,
/// `all` the solve's own `converged`, `tol` the case's `numerics.tolerance`.
pub(crate) fn verdict(
    names: &[&str],
    residuals: &[SolverPerformance],
    all: bool,
    tol: Scalar,
) -> Result<()> {
    if residuals.is_empty() {
        return Err(Error::Config(format!(
            "stress mode: the per-region residuals were not measured, so no \
             region's convergence is observable - run the thermal solve with \
             report_residuals on (SPEC-LIT 96.2 refuses to guess)"
        )));
    }
    if all {
        return Ok(());
    }
    let (i, rp) = residuals
        .iter()
        .enumerate()
        .find(|(_, rp)| !rp.converged)
        .expect("all=false implies some region did not converge");
    Err(Error::Config(format!(
        "stress mode: region '{}' did not meet the thermal solve's criterion - \
         final residual {:.3e} against tolerance {:e}; raise numerics.maxIter or \
         loosen numerics.tolerance (SPEC-LIT 96.2 solves displacement on a \
         converged thermal solve only)",
        names.get(i).unwrap_or(&"?"),
        rp.final_residual,
        tol
    )))
}

// ==========================================================================
//  The per-patch statement - SPEC-LIT 96.2's lowering table
// ==========================================================================

/// One row per patch of the region's own mesh, in patch order - the table
/// [`Displacement::with_materials`] and [`crate::solid::bc::check_patches`]
/// read. An `empty` patch is never named (SPEC-LIT 96.3 row 12) and is
/// filled with three `Traction(0.0)`, which no kernel reads on an empty
/// face. A named patch the mesh does not have is refused, by name.
pub(crate) fn per_patch_of(
    m: &HostMesh,
    bcs: &[(String, LoweredMechanicalBc)],
) -> Result<Vec<[CompBc; 3]>> {
    let mut out = vec![[CompBc::Traction(0.0); 3]; m.patches.len()];
    for (name, bc) in bcs {
        let Some(p) = m.patches.iter().position(|p| &p.name == name) else {
            return Err(Error::Config(format!(
                "regions/mechanics/patches: patch '{name}' is not a patch of \
                 this region's mesh - name every non-empty patch exactly once"
            )));
        };
        out[p] = match bc {
            LoweredMechanicalBc::Fixed(v) => {
                let mut row = [CompBc::Traction(0.0); 3];
                for (i, v) in v.iter().enumerate() {
                    if let Some(v) = v {
                        row[i] = CompBc::Fixed(*v);
                    }
                }
                row
            }
            LoweredMechanicalBc::Traction(t) => {
                [CompBc::Traction(t.x), CompBc::Traction(t.y), CompBc::Traction(t.z)]
            }
            LoweredMechanicalBc::Symmetry { axis } => {
                let mut row = [CompBc::Traction(0.0); 3];
                row[*axis] = CompBc::Fixed(0.0);
                row
            }
            LoweredMechanicalBc::Free => [CompBc::Traction(0.0); 3],
        };
    }
    Ok(out)
}

// ==========================================================================
//  The run - SPEC-LIT 96.2
// ==========================================================================

/// What one mechanical region's run produced - what `summary_lines` prints
/// and `write_region_vtu` writes.
#[derive(Debug)]
pub struct RegionStress {
    /// The region's index.
    pub region: usize,
    /// The region's name.
    pub name: String,
    /// The outer loop's own report.
    pub report: OuterReport,
    /// [`MaterialMap::describe`] - one line per zone, then the bond faces.
    pub describe: String,
    /// `[n_cells]` cell displacement.
    pub u: Vec<Vec3>,
    /// `[n_points]` displacement on the mesh's own points.
    pub u_points: Vec<Vec3>,
    /// `[n_cells]` Cauchy stress.
    pub sigma: Vec<Tensor>,
    /// `[n_cells]` von Mises.
    pub von_mises: Vec<Scalar>,
    /// `[n_cells]` principal stresses, largest first.
    pub principal: Vec<Vec3>,
    /// `[n_cells]` `|u|`.
    pub mag_u: Vec<Scalar>,
    /// `max|u|` over the region.
    pub max_u: Scalar,
    /// `(value, cell, centroid)`, the peak von Mises.
    pub peak_von_mises: (Scalar, usize, Vec3),
}

/// Solve SPEC-LIT 95's displacement on every solid region carrying a
/// `mechanics` block, in region order, each on ITS OWN `HostMesh`
/// (docs/10's decision B2 - never the concatenated mesh), reading its `T`
/// and `bT` as that region's slice of the conjugate solution.
pub fn run_stress(gpu: &Gpu, low: &LoweredChtCase, sol: &ChtSolution) -> Result<Vec<RegionStress>> {
    let mut out = Vec::new();
    for r in 0..low.meshes.len() {
        let Some(m) = &low.mechanics[r] else { continue };
        let host = &low.meshes[r];
        // The region's own slices of the conjugate field: concatenation
        // preserves each region's cells and its boundary faces, in order.
        let reg = &sol.mesh.regions[r];
        let t_c: Vec<Scalar> = sol.t[reg.cells()].to_vec();
        let (bo, nb) = (reg.boundary_face_offset, reg.n_boundary_faces);
        let bt: Vec<Scalar> = sol.bt[bo..bo + nb].to_vec();

        let gm = GpuMesh::upload(gpu, host)?;
        let entries: Vec<(&str, crate::solid::Material, Option<Scalar>, Vec<Label>)> = m
            .zones
            .iter()
            .enumerate()
            .map(|(zi, z)| {
                let cells: Vec<Label> = (0..m.zone_of_cell.len())
                    .filter(|&c| m.zone_of_cell[c] == zi)
                    .map(|c| c as Label)
                    .collect();
                (z.name.as_str(), z.material, Some(z.t_ref), cells)
            })
            .collect();
        let map = MaterialMap::from_cell_lists(&entries, host.n_cells, m.bond)?;
        let per_patch = per_patch_of(host, &m.patch_bcs)?;
        // One `numerics` block serves the conduction matrix and the three
        // displacement component solves (SPEC-LIT 96.2).
        let mut d = Displacement::with_materials(gpu, &gm, host, &map, &per_patch, low.solver)?;
        // The fallback is never read: every entry above carries `Some(t_ref)`,
        // and `set_temperature` writes `map.t_ref_per_cell` over it.
        d.set_temperature(gpu, &t_c, &bt, m.zones[0].t_ref)?;
        let ctrl = OuterControls {
            relaxation: Relaxation::Anderson(outer::ANDERSON_DEPTH),
            decades: -m.solver.tolerance.log10(),
            max_outer: m.solver.max_outer,
            boundary_passes: 3,
        };
        let report = outer::solve(gpu, &mut d, &ctrl)?;
        if !report.converged {
            return Err(Error::Config(format!(
                "regions/{}/mechanics/solver/maxOuter: the outer loop stopped \
                 after {} iterations without converging - raise maxOuter or \
                 loosen mechanics/solver/tolerance (SPEC-LIT 96.3 row 21)",
                low.region_names[r], report.iterations
            )));
        }
        // `outer::solve`'s last act is `correct_boundary`, so `d.grad`
        // already belongs to the accepted `u`; `compute_with` recomputes
        // nothing (SPEC-LIT 95.8's read-out on the map's own per-cell arrays).
        let mut sf = StressFields::new(gpu, host.n_cells)?;
        sf.compute_with(gpu, &d.cells, &d.grad, &d.u.f, &d.t)?;
        let hs = sf.download(gpu)?;
        let u: Vec<Vec3> = gpu.download(&d.u.f)?;
        let u_points = point_displacement(&low.raw[r], host, &u)?;
        let max_u = u.iter().map(|a| a.mag()).fold(0.0 as Scalar, Scalar::max);
        let peak = (0..u.len())
            .max_by(|&a, &b| hs.von_mises[a].total_cmp(&hs.von_mises[b]))
            .unwrap_or(0);
        let peak_von_mises = (hs.von_mises[peak], peak, host.c[peak]);
        let describe = d.map.describe(d.bonds.n_bond());
        out.push(RegionStress {
            region: r,
            name: low.region_names[r].clone(),
            report,
            describe,
            u,
            u_points,
            sigma: hs.sigma,
            von_mises: hs.von_mises,
            principal: hs.principal,
            mag_u: hs.mag_u,
            max_u,
            peak_von_mises,
        });
    }
    Ok(out)
}

// ==========================================================================
//  The 13.4.2 banner, and the summary after the run
// ==========================================================================

/// The start-up block, BEFORE the run: per mechanical region and zone, the
/// constants, the Lamé conversion, the predicted contraction, the measured
/// `nu` edge the case passed, and the two-way coupling parameter the
/// one-way energy equation drops (docs/09's D.5 item 9 - printed, not
/// hidden, because a number is the only honest way to say what was
/// neglected). Host only; nothing here calls `println!`.
pub fn banner_lines(low: &LoweredChtCase) -> Vec<String> {
    let mut out = Vec::new();
    for (r, m) in low.mechanics.iter().enumerate() {
        let Some(m) = m else {
            out.push(format!(
                "  region {r} '{}': no mechanics - thermal only",
                low.region_names[r]
            ));
            continue;
        };
        // The region's THERMAL material carries rho and c - the coupling
        // parameter is a property of the coupled problem, not of the
        // elastic constants alone.
        let (rho, c) = (low.materials[r].rho, low.materials[r].c);
        for z in &m.zones {
            let mat = z.material;
            out.push(format!(
                "  region {r} '{}' zone '{}': E = {:e} Pa, nu = {:.2}, alpha = {:e} /K, \
                 TRef = {:.2} K | mu = {:e}, lambda = {:e}, 2mu+lambda = {:e} | \
                 predicted contraction (mu+lambda)/(2mu+lambda) = {:.4} \
                 (nu = {:.2} <= {:.2} accepted)",
                low.region_names[r],
                z.name,
                mat.e,
                mat.nu,
                mat.alpha,
                z.t_ref,
                mat.mu(),
                mat.lambda(),
                mat.implicit_gamma(),
                mat.predicted_contraction(),
                mat.nu,
                crate::solid::NU_MAX,
            ));
            let delta = crate::solid::two_way_coupling_delta(&mat, rho, c, z.t_ref);
            out.push(format!(
                "    two-way coupling delta = {delta:.3e} - the size of the \
                 -(3lambda+2mu) alpha T0 d(tr eps)/dt term the one-way energy \
                 equation omits (Boley & Weiner ch. 1-2)"
            ));
        }
    }
    out
}

/// The after-the-run block: what each mechanical region cost and what it
/// produced.
pub fn summary_lines(low: &LoweredChtCase, stress: &[RegionStress]) -> Vec<String> {
    let mut out = Vec::new();
    for s in stress {
        out.push(format!(
            "  region {} '{}': outer iterations {}, converged {}, \
             observed contraction {:.4} vs predicted {:.4}, max|u| = {:.4e} m, \
             motion_ratio (max|u|/h_min) = {:.3e}",
            s.region,
            s.name,
            s.report.iterations,
            s.report.converged,
            s.report.observed_contraction,
            s.report.predicted_contraction,
            s.max_u,
            s.report.motion_ratio,
        ));
        out.push(format!(
            "    peak von Mises {:.4e} Pa at cell {} (centroid {:.4e}, {:.4e}, {:.4e}) m",
            s.peak_von_mises.0,
            s.peak_von_mises.1,
            s.peak_von_mises.2.x,
            s.peak_von_mises.2.y,
            s.peak_von_mises.2.z,
        ));
        out.push(s.describe.clone());
    }
    let _ = low;
    out
}

// ==========================================================================
//  The output - SPEC-LIT 96.2's write, one real-point VTU per region
// ==========================================================================

/// One `.vtu` per region - NOT through `OutputPipeline`, which carries one
/// `HostMesh` and cell fields only (SPEC-LIT 44.1's `exact` promise kept
/// with the real points, which is what a deformed shape wants). Cell
/// fields `T, u, sigma, vonMises, sigmaPrincipal, magU`, point fields
/// `T, u`, in that order and with those names. A region without
/// `mechanics` - and every region of a thermal-mode run - writes `T` only.
pub fn write_region_vtu(
    dir: &Path,
    low: &LoweredChtCase,
    sol: &ChtSolution,
    stress: &[RegionStress],
) -> Result<Vec<PathBuf>> {
    std::fs::create_dir_all(dir).path(dir)?;
    let mut written = Vec::new();
    for r in 0..low.meshes.len() {
        let host = &low.meshes[r];
        let reg = &sol.mesh.regions[r];
        let t_c: Vec<Scalar> = sol.t[reg.cells()].to_vec();
        let interp = PointInterpolator::new(&low.raw[r], host)?;
        let t_points = interp.cell_to_point(&t_c)?;
        let (cell, point) = match stress.iter().find(|s| s.region == r) {
            Some(s) => {
                let cell = vec![
                    OutputField::scalar("T", &t_c),
                    OutputField::vector("u", &s.u),
                    OutputField::tensor("sigma", &s.sigma),
                    OutputField::scalar("vonMises", &s.von_mises),
                    OutputField::vector("sigmaPrincipal", &s.principal),
                    OutputField::scalar("magU", &s.mag_u),
                ];
                let point = vec![
                    OutputField::scalar("T", &t_points),
                    OutputField::vector("u", &s.u_points),
                ];
                (cell, point)
            }
            None => (
                vec![OutputField::scalar("T", &t_c)],
                vec![OutputField::scalar("T", &t_points)],
            ),
        };
        let path = dir.join(format!("{}.vtu", low.region_names[r]));
        write_vtu_points(&path, &low.raw[r], &cell, &point, Some(0.0))?;
        written.push(path);
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::PatchKind;
    use crate::mesh::topology::tests::box_mesh;

    fn met(v: Scalar) -> SolverPerformance {
        SolverPerformance {
            initial_residual: 1.0,
            final_residual: v,
            n_iterations: 10,
            converged: true,
        }
    }

    fn missed(v: Scalar) -> SolverPerformance {
        SolverPerformance {
            initial_residual: 1.0,
            final_residual: v,
            n_iterations: 10,
            converged: false,
        }
    }

    /// SPEC-LIT 96.2: the verdict names the FIRST region that missed, its
    /// residual and the tolerance, and says what to do; `Ok` when every
    /// region met the criterion; and residuals that were never measured are
    /// a refusal, not a pass.
    #[test]
    fn the_verdict_names_the_first_region_that_missed() {
        let e = verdict(&["a", "b"], &[met(1e-13), missed(2.5e-9)], false, 1e-12)
            .expect_err("must refuse");
        let m = e.to_string();
        assert!(m.contains("'b'"), "{m}");
        assert!(m.contains("2.500e-9"), "{m}");
        assert!(m.contains("1e-12"), "{m}");
        assert!(m.contains("numerics.maxIter"), "{m}");
        assert!(
            verdict(&["a", "b"], &[met(1e-13), met(1e-13)], true, 1e-12).is_ok(),
            "every region met: the verdict passes"
        );
        let e = verdict(&["a", "b"], &[], false, 1e-12).expect_err("must refuse");
        assert!(e.to_string().contains("report_residuals"));
    }

    /// The lowering table of SPEC-LIT 96.2, on a mesh whose `z` patches are
    /// `empty` (`nz = 1`): every spelling lands on its row, an unnamed
    /// `empty` patch is filled with three `Traction(0.0)`, and a name the
    /// mesh does not have is refused, by name.
    #[test]
    fn the_bc_table_lowers_every_spelling() {
        let (mut m, pts, faces) = box_mesh([2, 2, 1], Vec3::new(1.0, 1.0, 1.0));
        m.compute_geometry(&pts, &faces).expect("geometry");
        m.build_cell_face_maps();

        let bcs = vec![
            (m.patches[0].name.clone(), LoweredMechanicalBc::Fixed([Some(1.0), None, Some(2.0)])),
            (m.patches[1].name.clone(), LoweredMechanicalBc::Traction(Vec3::new(3.0, 4.0, 5.0))),
            (m.patches[2].name.clone(), LoweredMechanicalBc::Symmetry { axis: 1 }),
            (m.patches[3].name.clone(), LoweredMechanicalBc::Free),
        ];
        let table = per_patch_of(&m, &bcs).expect("table");
        assert_eq!(table.len(), m.patches.len());
        assert_eq!(table[0], [CompBc::Fixed(1.0), CompBc::Traction(0.0), CompBc::Fixed(2.0)]);
        assert_eq!(
            table[1],
            [CompBc::Traction(3.0), CompBc::Traction(4.0), CompBc::Traction(5.0)]
        );
        assert_eq!(table[2], [CompBc::Traction(0.0), CompBc::Fixed(0.0), CompBc::Traction(0.0)]);
        assert_eq!(table[3], [CompBc::Traction(0.0); 3]);
        for (i, p) in m.patches.iter().enumerate() {
            if p.kind == PatchKind::Empty {
                assert_eq!(table[i], [CompBc::Traction(0.0); 3], "patch {}", p.name);
            }
        }
        let e = per_patch_of(&m, &[("nope".to_string(), LoweredMechanicalBc::Free)])
            .expect_err("must refuse");
        assert!(e.to_string().contains("nope"));
    }
}
