// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
//
// Provenance: ORIGINAL - the tests for SPEC-LIT §59 and §60. Two published
// numbers appear here and both are cited at the point they are used: de Vahl
// Davis (1983)'s four benchmark Nusselt numbers, quoted from Qi et al.,
// Nanoscale Research Letters 8 (2013) 56 Table 3 (open access), and Belazizia
// et al. (2012)'s conjugate table, read from the open-access PDF. Everything
// else is a closed form derived in SPEC-LIT §59/§60 or an identity the code
// checks against itself.
// No GPL-licensed source was consulted.

use super::*;

use crate::blockgen::{self, BlockSpec, GradedAxis};
use crate::io::case::{LinearSolverKind, Preconditioner};
use crate::io::case_cht::LoweredBc;

fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// One axis-aligned block, `n` cells from `lo` to `hi`, patches named and
/// typed by the caller.
pub(crate) fn block(
    n: [usize; 3],
    lo: Vec3,
    hi: Vec3,
    names: [&str; 6],
    types: [&str; 6],
) -> HostMesh {
    let axis = |i: usize| GradedAxis {
        lo: [lo.x, lo.y, lo.z][i],
        hi: [hi.x, hi.y, hi.z][i],
        n: n[i],
        expansion: 1.0,
        two_sided: false,
    };
    blockgen::build_mesh(&BlockSpec {
        x: axis(0),
        y: axis(1),
        z: axis(2),
        patch_name: std::array::from_fn(|i| names[i].to_string()),
        patch_type: std::array::from_fn(|i| types[i].to_string()),
        windows: Vec::new(),
        cyclic: Vec::new(),
    })
    .expect("blockgen")
}

/// A 2-D fluid cavity: four walls, `empty` front and back.
pub(crate) fn cavity_block(n: usize, x0: Scalar, x1: Scalar, dz: Scalar) -> HostMesh {
    block(
        [((x1 - x0) / (1.0 / n as Scalar)).round() as usize, n, 1],
        Vec3::new(x0, 0.0, 0.0),
        Vec3::new(x1, 1.0, dz),
        ["left", "right", "bottom", "top", "front", "back"],
        ["wall", "wall", "wall", "wall", "empty", "empty"],
    )
}

pub(crate) const T_REF: Scalar = 300.0;
pub(crate) const D_T: Scalar = 0.1;

/// The gravity that makes a unit-width, unit-height cavity of this fluid run
/// at `Ra`.
///
/// `Ra = g beta dT L^3/(nu alpha)` with `beta = 1/TRef` - the exact
/// linearisation of SPEC-LIT §9's `g(TRef/T - 1)`, whose remaining error is
/// `O(dT/TRef)` and is `3.3e-4` here (SPEC-LIT §59.9).
pub(crate) fn g_for_ra(ra: Scalar, f: &FluidMaterial) -> Scalar {
    ra * f.nu() * f.alpha() * T_REF / (D_T * 1.0)
}

pub(crate) fn air(kappa: Scalar) -> FluidMaterial {
    // rho = cp = 1 makes alpha = kappa exactly; mu = 0.71 kappa makes
    // Pr = mu cp/kappa = 0.71, air's.
    FluidMaterial {
        name: "air".to_string(),
        rho: 1.0,
        cp: 1.0,
        kappa,
        mu: 0.71 * kappa,
    }
}

pub(crate) fn flow_controls(
    iterations: usize,
    residual: Scalar,
    relax_u: Scalar,
    relax_p: Scalar,
) -> FlowControls {
    FlowControls {
        iterations,
        residual,
        relax_u,
        relax_p,
        relax_t: 0.7,
        div_u: DivScheme::Central,
        div_t: DivEntry { scheme: DivScheme::Central, bounded: true },
        u_solver: SolverControls {
            solver: LinearSolverKind::PBiCGStab,
            precon: Preconditioner::Dilu,
            tolerance: 1e-14,
            rel_tol: 0.01,
            max_iter: 200,
            check_interval: 10,
            ..SolverControls::default()
        },
        p_solver: SolverControls {
            solver: LinearSolverKind::PCG,
            precon: Preconditioner::Dic,
            tolerance: 1e-14,
            rel_tol: 0.001,
            max_iter: 800,
            check_interval: 10,
            ..SolverControls::default()
        },
        n_non_orth_correctors: 0,
        simplec: false,
    }
}

pub(crate) fn t_solver() -> SolverControls {
    SolverControls {
        solver: LinearSolverKind::PBiCGStab,
        precon: Preconditioner::Dilu,
        tolerance: 1e-16,
        rel_tol: 0.0,
        max_iter: 500,
        check_interval: 10,
        ..SolverControls::default()
    }
}

// ==========================================================================
//  Gate 59-A - de Vahl Davis (1983), the fluid-only anchor
// ==========================================================================

/// The differentially heated square cavity, fluid only: no solid region, no
/// interface, so what it measures is `Energy` + `Simple` and nothing else.
///
/// Returns `(Nu_cold, Nu_hot, iterations, converged)`. The two Nusselt numbers
/// are computed from the two walls independently and must agree - a global
/// energy balance the case never told the solver about.
pub(crate) fn de_vahl_davis(
    gpu: &Gpu,
    ra: Scalar,
    n: usize,
    iterations: usize,
    residual: Scalar,
) -> Result<(Scalar, Scalar, usize, bool)> {
    let dz: Scalar = 1.0 / n as Scalar;
    let m = cavity_block(n, 0.0, 1.0, dz);
    let f = air(1.0);
    let meshes = [m];

    let case = FlowCase {
        name: "deVahlDavis".to_string(),
        regions: vec![FlowRegion {
            name: "air".to_string(),
            kind: RegionKind::Fluid,
            solid: None,
            fluid: Some(f.clone()),
            source: 0.0,
        }],
        meshes: &meshes,
        interfaces: Vec::new(),
        patch_bcs: vec![
            (0, "left".to_string(), LoweredBc::FixedValue(T_REF + 0.5 * D_T)),
            (0, "right".to_string(), LoweredBc::FixedValue(T_REF - 0.5 * D_T)),
            (0, "bottom".to_string(), LoweredBc::ZeroGradient),
            (0, "top".to_string(), LoweredBc::ZeroGradient),
        ],
        buoyancy: Some(Buoyancy {
            g: Vec3::new(0.0, -g_for_ra(ra, &f), 0.0),
            t_ref: T_REF,
        }),
        openings: None,
        initial_t: T_REF,
        flow: flow_controls(iterations, residual, 0.7, 0.3),
        t_solver: t_solver(),
        n_non_orthogonal_correctors: 0,
        tolerances: PairingTolerances::default(),
        p0: 101_325.0,
    };

    let sol = run_flow_case(gpu, &case)?;
    // Nu = Q L/(k dT H d_z), L = H = 1.
    let scale = f.kappa * D_T * dz;
    let nu_cold = -sol.patch_heat_flow(0, "right")? / scale;
    let nu_hot = sol.patch_heat_flow(0, "left")? / scale;
    Ok((nu_cold, nu_hot, sol.iterations, sol.converged))
}

#[test]
fn gate_59a_de_vahl_davis_square_cavity() -> Result<()> {
    let Some(gpu) = gpu() else { return Ok(()) };

    // de Vahl Davis (1983), quoted from Qi et al., Nanoscale Research Letters
    // 8 (2013) 56, Table 3, which lists them beside two other codes'.
    for &(ra, n, expected, tol) in &[
        (1.0e3 as Scalar, 40usize, 1.118 as Scalar, 0.02 as Scalar),
        (1.0e4, 60, 2.243, 0.02),
        (1.0e5, 80, 4.519, 0.03),
    ] {
        let (nu_c, nu_h, its, conv) = de_vahl_davis(&gpu, ra, n, 6000, 1e-7)?;
        let rel = (nu_c - expected).abs() / expected;
        let balance = (nu_c - nu_h).abs() / nu_c.abs().max(1e-30);
        println!(
            "Ra {ra:.0e} n {n}: Nu_cold {nu_c:.4} Nu_hot {nu_h:.4} \
             (published {expected}, {:+.2}%), balance {balance:.2e}, \
             {its} iterations, converged {conv}",
            100.0 * (nu_c / expected - 1.0)
        );
        assert!(
            balance < 5e-3,
            "the two walls disagree by {balance:.3e}: at steady state the heat \
             in at the hot wall IS the heat out at the cold one, and a gap says \
             the run is not converged"
        );
        assert!(
            rel < tol,
            "Ra = {ra:.0e}: Nu = {nu_c:.4} against de Vahl Davis (1983)'s \
             {expected} - {:+.2}%, tolerance {:.1}%",
            100.0 * (nu_c / expected - 1.0),
            100.0 * tol
        );
    }
    Ok(())
}

// ==========================================================================
//  Gate 5 - the Kaminski & Prakash (1986) configuration
// ==========================================================================

/// Conjugate natural convection in a square enclosure with one conducting
/// vertical wall - SPEC-LIT §60.5.
///
/// The dimensionless form is Belazizia *et al.*'s and §60.5 records why:
/// **the total width, solid plus fluid, is the unit length**, so the solid
/// occupies `X in [0, D]` and the fluid `X in [D, 1]` of a `1 x 1` domain.
/// The outer face of the solid is held at `T_h`, the far face of the fluid at
/// `T_c`, and both horizontal boundaries are adiabatic across solid and fluid
/// alike.
///
/// Returns `(Nu_cold, Nu_hot, Nu_interface, iterations, converged)`. All three
/// are the same number at steady state, and their spread is this gate's own
/// error bar.
#[allow(clippy::too_many_lines)]
pub(crate) fn kaminski_prakash(
    gpu: &Gpu,
    ra: Scalar,
    kr: Scalar,
    n: usize,
    iterations: usize,
    residual: Scalar,
) -> Result<(Scalar, Scalar, Scalar, usize, bool)> {
    const D: Scalar = 0.2;

    assert!(n % 5 == 0, "the wall is 0.2 of the width; n must be a multiple of 5");
    let dz: Scalar = 1.0 / n as Scalar;
    let f = air(1.0);

    // Square cells on both sides: the solid gets D*n columns and the fluid
    // (1-D)*n.
    let n_solid = (D * n as Scalar).round() as usize;
    let fluid_mesh = block(
        [n - n_solid, n, 1],
        Vec3::new(D, 0.0, 0.0),
        Vec3::new(1.0, 1.0, dz),
        ["airToWall", "cold", "airBottom", "airTop", "airFront", "airBack"],
        ["wall", "wall", "wall", "wall", "empty", "empty"],
    );
    let solid_mesh = block(
        [n_solid, n, 1],
        Vec3::ZERO,
        Vec3::new(D, 1.0, dz),
        ["hot", "wallToAir", "wallBottom", "wallTop", "wallFront", "wallBack"],
        ["patch", "patch", "patch", "patch", "empty", "empty"],
    );
    let meshes = [fluid_mesh, solid_mesh];

    let case = FlowCase {
        name: "kaminskiPrakash".to_string(),
        regions: vec![
            FlowRegion {
                name: "air".to_string(),
                kind: RegionKind::Fluid,
                solid: None,
                fluid: Some(f.clone()),
                source: 0.0,
            },
            FlowRegion {
                name: "wall".to_string(),
                kind: RegionKind::Solid,
                solid: Some(SolidMaterial::isotropic("wall", 1.0, 1.0, kr * f.kappa)),
                fluid: None,
                source: 0.0,
            },
        ],
        meshes: &meshes,
        interfaces: vec![InterfaceRequest::new(0, "airToWall", 1, "wallToAir", 0.0)],
        patch_bcs: vec![
            (0, "cold".to_string(), LoweredBc::FixedValue(T_REF - 0.5 * D_T)),
            (0, "airBottom".to_string(), LoweredBc::ZeroGradient),
            (0, "airTop".to_string(), LoweredBc::ZeroGradient),
            (1, "hot".to_string(), LoweredBc::FixedValue(T_REF + 0.5 * D_T)),
            (1, "wallBottom".to_string(), LoweredBc::ZeroGradient),
            (1, "wallTop".to_string(), LoweredBc::ZeroGradient),
        ],
        buoyancy: Some(Buoyancy {
            g: Vec3::new(0.0, -g_for_ra(ra, &f), 0.0),
            t_ref: T_REF,
        }),
        openings: None,
        initial_t: T_REF,
        flow: flow_controls(iterations, residual, 0.7, 0.3),
        t_solver: t_solver(),
        n_non_orthogonal_correctors: 0,
        tolerances: PairingTolerances::default(),
        p0: 101_325.0,
    };

    let sol = run_flow_case(gpu, &case)?;

    // SPEC-LIT (S60.1): Nu = Q L/(k_f dT H d_z), L = H = 1.
    let scale = f.kappa * D_T * dz;
    let nu_cold = -sol.patch_heat_flow(0, "cold")? / scale;
    let nu_hot = sol.patch_heat_flow(1, "hot")? / scale;
    let nu_iface = sol
        .interface_flows()
        .first()
        .map(|(_, into_a, _)| *into_a / scale)
        .unwrap_or(0.0);

    // Gate 4 is always on: a mis-paired face or a sign error shows here
    // before it shows in any Nusselt number.
    assert!(
        sol.interface.imbalance() < 1e-12,
        "SPEC-LIT 47.12 Gate 4: interface imbalance {:.3e}",
        sol.interface.imbalance()
    );
    Ok((nu_cold, nu_hot, nu_iface, sol.iterations, sol.converged))
}

/// **Gate 59-B, exact and analytic.** At `Ra -> 0` the §60.5 configuration is
/// a two-material 1-D slab, so `Nu = 1/(D/Kr + (1 - D))` - SPEC-LIT (S59.7).
/// No published data at all, and it is the gate that says the INTERFACE is
/// right independently of any flow.
#[test]
fn gate_59b_the_conduction_limit_is_the_series_resistance() -> Result<()> {
    let Some(gpu) = gpu() else { return Ok(()) };
    const D: Scalar = 0.2;

    for &kr in &[0.1 as Scalar, 1.0, 10.0] {
        // Ra = 1 is not zero, but the fluid Rayleigh number it implies is
        // O(1) and the convective transport is below the tolerance below by
        // orders of magnitude. Using an exactly zero `g` would be refused
        // (SPEC-LIT §60.3) and would also not exercise the buoyancy path.
        let (nu_c, nu_h, nu_i, its, conv) = kaminski_prakash(&gpu, 1.0, kr, 10, 2500, 0.0)?;
        let exact = 1.0 / (D / kr + (1.0 - D));
        println!(
            "Kr {kr}: Nu cold {nu_c:.9} hot {nu_h:.9} interface {nu_i:.9} \
             against the exact {exact:.9} ({:+.2e} relative), {its} iterations, \
             converged {conv}",
            nu_c / exact - 1.0
        );
        assert!(
            (nu_c / exact - 1.0).abs() < 1e-6,
            "Kr = {kr}: Nu = {nu_c} against the exact series resistance {exact}"
        );
        assert!(
            (nu_h / exact - 1.0).abs() < 1e-6 && (nu_i / exact - 1.0).abs() < 1e-6,
            "Kr = {kr}: the three heat flows disagree - cold {nu_c}, hot {nu_h}, \
             interface {nu_i}"
        );
    }
    Ok(())
}

/// **SPEC-LIT §60.5, Gate 5 - the whole sweep.**
///
/// Ignored by default and run by hand: it is eighteen steady conjugate
/// natural-convection solves, which is a live multi-minute GPU run, and this
/// repository's standing rule is that those belong in a driver invocation a
/// human chooses to make rather than in `cargo test` (the same rule that keeps
/// SPEC-LIT §33.3's channel out of the fast suite). What IS in the fast suite
/// is Gate 59-B, which is exact and needs no published number at all, and the
/// two cheapest points of this table, which `ofgpu-validate` runs live.
///
/// The reference values are Belazizia *et al.* (2012) Fig. 6's captions, read
/// from the open-access PDF. SPEC-LIT §60.5 records that they are a SECONDARY
/// source, that Kaminski & Prakash's own table could not be obtained, and
/// exactly what was searched.
#[test]
#[ignore = "eighteen steady conjugate solves - a live multi-minute GPU run"]
fn gate_5_kaminski_prakash_sweep() -> Result<()> {
    let Some(gpu) = gpu() else { return Ok(()) };

    // (Ra, Kr, Belazizia et al. 2012 Fig. 6)
    //
    // The Ra = 500 column of that figure is DELIBERATELY not swept, and the
    // reason belongs on the record rather than in a commit message: it reads
    // 0.382 / 1.03 / 1.24 at Kr = 0.1 / 1 / 10, which is 3-7 % ABOVE the
    // analytic conduction limit 1/(D/Kr + 1 - D) = 0.35714 / 1.0 / 1.21951.
    // At Ra = 500 the fluid-layer Rayleigh number is O(100) and convection
    // cannot add 7 %; those three numbers are therefore not a 3 %-quality
    // reference. What the conduction limit IS gated against is Gate 59-B,
    // which reproduces it exactly and needs no published number at all.
    const PUBLISHED: &[(Scalar, Scalar, Scalar)] = &[
        (1.0e4, 0.1, 0.41),
        (1.0e4, 1.0, 1.57),
        (1.0e4, 10.0, 2.28),
        (1.0e5, 0.1, 0.461),
        (1.0e5, 1.0, 2.35),
        (1.0e5, 10.0, 4.25),
    ];

    println!(
        "\n  Ra        Kr     n    Nu(cold)    Nu(hot)     Nu(iface)   spread    \
         published  diff      d(mesh)  its    conv"
    );
    for &(ra, kr, pubv) in PUBLISHED {
        let mut prev: Option<Scalar> = None;
        for &n in &[40usize, 60, 80] {
            let (nu_c, nu_h, nu_i, its, conv) =
                kaminski_prakash(&gpu, ra, kr, n, 15_000, 1e-7)?;
            let dmesh = prev.map_or(Scalar::NAN, |p| (nu_c / p - 1.0).abs());
            // The three heat flows are the SAME number at steady state, and
            // their spread is this gate's own convergence measure - the
            // residual criterion is noise-dominated wherever the flow is weak
            // (SPEC-LIT 60.5).
            let spread = (nu_c.max(nu_h).max(nu_i) - nu_c.min(nu_h).min(nu_i)) / nu_c.abs();
            println!(
                "  {ra:<9.0e} {kr:<6} {n:<4} {nu_c:<11.5} {nu_h:<11.5} {nu_i:<11.5} \
                 {spread:<9.2e} {pubv:<10} {:+7.2}%  {:>7.2}%  {its:<6} {conv}",
                100.0 * (nu_c / pubv - 1.0),
                100.0 * dmesh
            );
            prev = Some(nu_c);
        }
    }
    Ok(())
}

// ==========================================================================
//  SPEC-LIT §59 - the retarget itself
//
//  These build an `Energy` by hand rather than going through
//  `run_flow_case`, because what they measure is the ASSEMBLY: matrix arrays,
//  bit for bit. A driver in between would only make the comparison harder to
//  read.
// ==========================================================================

use crate::cht::{
    Conduction, InterfaceRequest, PairingTolerances, RegionInput, ThermalMesh,
};
use crate::energy::{DomainKind, Energy, EnergyControls, GasProperties, GasState};
use crate::field::BcKind;
use crate::ldu::GpuLduMatrix;
use crate::mesh::GpuMesh;
use crate::timescheme::DdtScheme;
use crate::Label;

/// A `1 x 1` two-region rig whose FIRST region is fluid - the shape §47.4
/// requires and `run_flow_case` builds, reduced to the smallest mesh that
/// still has an interface several faces wide.
pub(crate) fn conjugate_rig(n: usize, kappa_s: Scalar) -> Result<(Vec<HostMesh>, ThermalMesh)> {
    let dz = 1.0 / n as Scalar;
    let n_solid = (0.2 * n as Scalar).round() as usize;
    let meshes = vec![
        block(
            [n - n_solid, n, 1],
            Vec3::new(0.2, 0.0, 0.0),
            Vec3::new(1.0, 1.0, dz),
            ["airToWall", "cold", "airBottom", "airTop", "airFront", "airBack"],
            ["wall", "wall", "wall", "wall", "empty", "empty"],
        ),
        block(
            [n_solid, n, 1],
            Vec3::ZERO,
            Vec3::new(0.2, 1.0, dz),
            ["hot", "wallToAir", "wallBottom", "wallTop", "wallFront", "wallBack"],
            ["patch", "patch", "patch", "patch", "empty", "empty"],
        ),
    ];
    let _ = kappa_s;
    // The borrow has to outlive the ThermalMesh, so the meshes are returned
    // beside it and the RegionInputs are built here from a local slice.
    let tm = {
        let inputs = [
            RegionInput { name: "air".into(), kind: RegionKind::Fluid, mesh: &meshes[0] },
            RegionInput { name: "wall".into(), kind: RegionKind::Solid, mesh: &meshes[1] },
        ];
        ThermalMesh::build(
            &inputs,
            &[InterfaceRequest::new(0, "airToWall", 1, "wallToAir", 0.0)],
            PairingTolerances::default(),
        )?
    };
    Ok((meshes, tm))
}

fn conduction_for(tm: &ThermalMesh, kappa_f: Scalar, kappa_s: Scalar) -> Result<Conduction> {
    Conduction::uniform_per_region(
        tm,
        &[
            SolidMaterial::isotropic("air", 1.0, 1.0, kappa_f),
            SolidMaterial::isotropic("wall", 2000.0, 800.0, kappa_s),
        ],
    )
}

/// The controls every §59 test runs with: steady, central convection, one
/// pass, a tight linear solve.
fn energy_controls() -> EnergyControls {
    EnergyControls {
        t_solver: t_solver(),
        t_relax: 0.7,
        div_scheme: DivEntry { scheme: DivScheme::Central, bounded: true },
        grad_scheme: GradScheme::GAUSS,
        sn_grad: SnGradScheme::Uncorrected,
        n_non_orth_correctors: 0,
        ddt: DdtScheme::SteadyState,
        steady: true,
        delta_t: 1.0,
    }
}

fn air_props() -> GasProperties {
    let d = GasProperties::default();
    GasProperties { cp: 1.0, k: 1.0, pr: 0.71, w: d.r_universal * 300.0 / 101_325.0, ..d }
}

/// `fixedValue` on one patch range.
fn fix(gpu: &Gpu, t: &mut GpuScalarField, faces: std::ops::Range<usize>, v: Scalar) -> Result<()> {
    let mut kind = gpu.download(&t.bc_kind)?;
    let mut fr = gpu.download(&t.fr)?;
    let mut rv = gpu.download(&t.ref_value)?;
    for bf in faces {
        if kind[bf] == BcKind::Empty as Label {
            continue;
        }
        kind[bf] = BcKind::FixedValue as Label;
        fr[bf] = 1.0;
        rv[bf] = v;
    }
    gpu.write(&mut t.bc_kind, &kind)?;
    gpu.write(&mut t.fr, &fr)?;
    gpu.write(&mut t.ref_value, &rv)
}

/// Every array of an assembled matrix, downloaded - the six §59.5 compares.
fn matrix_arrays(gpu: &Gpu, a: &GpuLduMatrix) -> Result<[Vec<Scalar>; 6]> {
    Ok([
        gpu.download(&a.diag)?,
        gpu.download(&a.upper)?,
        gpu.download(&a.lower)?,
        gpu.download(&a.source)?,
        gpu.download(&a.internal_coeffs)?,
        gpu.download(&a.boundary_coeffs)?,
    ])
}

const ARRAY_NAMES: [&str; 6] = [
    "diag",
    "upper",
    "lower",
    "source",
    "internalCoeffs",
    "boundaryCoeffs",
];

/// **SPEC-LIT §59.5's second claim, and the one that replaces §47.11's
/// "`energy.rs` is not modified at all".**
///
/// A whole run - mesh, boundary conditions, a source, several outer
/// iterations - solved twice in one process: once by an `Energy` that was
/// never told about a conjugate mesh, and once by an `Energy` handed a
/// thermal mesh with ONE region, that region FLUID, and no interfaces. The
/// second is the retargeted code path running every blend of (S59.3) against
/// an all-ones mask, and it must produce the same bits: the temperature
/// field, its boundary values, and all six matrix arrays.
///
/// A test that compared one coefficient would pass while the run drifted.
#[test]
fn a_one_region_fluid_retarget_is_bitwise_the_plain_energy() -> Result<()> {
    let Some(gpu) = gpu() else { return Ok(()) };

    const N: usize = 12;
    let hm = cavity_block(N, 0.0, 1.0, 1.0 / N as Scalar);
    let gm = GpuMesh::upload(&gpu, &hm)?;

    // The retargeted twin's thermal mesh: one region, fluid, no interface.
    let inputs = [RegionInput { name: "air".into(), kind: RegionKind::Fluid, mesh: &hm }];
    let tm = ThermalMesh::build(&inputs, &[], PairingTolerances::default())?;
    let cond = Conduction::uniform_per_region(&tm, &[SolidMaterial::isotropic("air", 1.0, 1.0, 1.0)])?;
    // The concatenation of one region is the region: if that ever stopped
    // being true the comparison below would be measuring two meshes.
    assert_eq!(tm.host.n_cells, hm.n_cells);
    assert_eq!(tm.host.v, hm.v);
    assert_eq!(tm.host.mag_sf, hm.mag_sf);
    assert_eq!(tm.host.b_delta_coeffs, hm.b_delta_coeffs);

    // Both `Energy` objects borrow the SAME device mesh, so not even the
    // upload can differ.
    let run = |attach: bool| -> Result<(Vec<Scalar>, Vec<Scalar>, [Vec<Scalar>; 6])> {
        let props = air_props();
        let mut gas = GasState::new(&gpu, &gm, props, DomainKind::Open, 101_325.0)?;
        let mut e = Energy::new(&gpu, &gm, energy_controls(), props)?;
        if attach {
            e.attach_conjugate(&gpu, &tm, &cond)?;
        }

        let t0 = vec![300.0 as Scalar; hm.n_cells];
        gpu.write(&mut e.field_mut().f, &t0)?;
        fix(&gpu, e.field_mut(), tm.patch_range(0, "left")?, 305.0)?;
        fix(&gpu, e.field_mut(), tm.patch_range(0, "right")?, 295.0)?;
        e.initialise(&gpu)?;

        // A volumetric source, so §18's registry is in the comparison too.
        let q: Vec<Scalar> = (0..hm.n_cells).map(|c| 10.0 + (c % 7) as Scalar).collect();
        let dq = gpu.upload(&q)?;
        e.sources_mut().register_explicit(&gpu, &dq)?;

        // A flux that is nothing like solenoidal, so the convection operator
        // and the bounded correction both do real work.
        let phi_f: Vec<Scalar> = (0..hm.n_internal_faces)
            .map(|f| 1e-3 * (((f * 37) % 23) as Scalar - 11.0))
            .collect();
        let mut phi = GpuSurfaceScalarField::zeros(&gpu, &gm, "phi")?;
        gpu.write(&mut phi.f, &phi_f)?;
        let nut = GpuScalarField::zeros(&gpu, &gm, "nut")?;
        let tke: DevBuf<Scalar> = gpu.zeros(hm.n_cells)?;

        for _ in 0..5 {
            gas.update_density(&gpu, e.field())?;
            e.correct(&gpu, &phi, &nut, &tke, 0.71, &gas)?;
        }
        Ok((
            gpu.download(&e.field().f)?,
            gpu.download(&e.field().bf)?,
            matrix_arrays(&gpu, e.matrix())?,
        ))
    };

    let (t_plain, bt_plain, m_plain) = run(false)?;
    let (t_cht, bt_cht, m_cht) = run(true)?;

    assert!(
        t_plain == t_cht,
        "SPEC-LIT 59.5: the retargeted one-region run moved T. First difference \
         at cell {:?}",
        t_plain.iter().zip(&t_cht).position(|(a, b)| a != b)
    );
    assert!(bt_plain == bt_cht, "SPEC-LIT 59.5: the retargeted run moved T's boundary values");
    for (i, (a, b)) in m_plain.iter().zip(&m_cht).enumerate() {
        assert!(
            a == b,
            "SPEC-LIT 59.5: the retargeted run moved the matrix's `{}` - first \
             difference at index {:?}",
            ARRAY_NAMES[i],
            a.iter().zip(b).position(|(x, y)| x != y)
        );
    }
    Ok(())
}

/// **SPEC-LIT §59.2, and the claim that matters most.**
///
/// The convective term vanishes in the solid in EVERY BIT, not to round-off.
/// The two runs differ only in the flux handed to [`Energy::correct`] - one
/// exactly zero, one a large arbitrary field on **every** face, solid and
/// interface faces included - and every solid row of the assembled matrix
/// must be identical.
///
/// Handing the second run a flux on the solid faces too is deliberate: it
/// tests (S59.3)'s mask rather than the driver's care. A driver bug cannot
/// leak convection into a solid, because `Energy` masks `phi_conv` itself.
#[test]
fn the_convective_term_vanishes_in_the_solid_in_every_bit() -> Result<()> {
    let Some(gpu) = gpu() else { return Ok(()) };

    const N: usize = 15;
    let (_meshes, tm) = conjugate_rig(N, 5.0)?;
    let cond = conduction_for(&tm, 1.0, 5.0)?;
    let gm = GpuMesh::upload(&gpu, &tm.host)?;

    let run = |scale: Scalar| -> Result<[Vec<Scalar>; 6]> {
        let props = air_props();
        let mut gas = GasState::new(&gpu, &gm, props, DomainKind::Open, 101_325.0)?;
        let mut e = Energy::new(&gpu, &gm, energy_controls(), props)?;
        e.attach_conjugate(&gpu, &tm, &cond)?;

        let t0: Vec<Scalar> = (0..tm.host.n_cells)
            .map(|c| 300.0 + ((c * 13) % 17) as Scalar * 0.01)
            .collect();
        gpu.write(&mut e.field_mut().f, &t0)?;
        fix(&gpu, e.field_mut(), tm.patch_range(1, "hot")?, 305.0)?;
        fix(&gpu, e.field_mut(), tm.patch_range(0, "cold")?, 295.0)?;
        crate::cht::mark_coupled_faces(&gpu, e.field_mut(), &tm)?;
        e.initialise(&gpu)?;

        // EVERY face, solid ones included.
        let phi_f: Vec<Scalar> = (0..tm.host.n_internal_faces)
            .map(|f| scale * (((f * 29) % 19) as Scalar - 9.0))
            .collect();
        let phi_b: Vec<Scalar> = (0..tm.host.n_boundary_faces)
            .map(|b| scale * (((b * 11) % 7) as Scalar - 3.0))
            .collect();
        let mut phi = GpuSurfaceScalarField::zeros(&gpu, &gm, "phi")?;
        gpu.write(&mut phi.f, &phi_f)?;
        gpu.write(&mut phi.bf, &phi_b)?;
        let nut = GpuScalarField::zeros(&gpu, &gm, "nut")?;
        let tke: DevBuf<Scalar> = gpu.zeros(tm.host.n_cells)?;

        gas.update_density(&gpu, e.field())?;
        e.correct(&gpu, &phi, &nut, &tke, 0.71, &gas)?;
        matrix_arrays(&gpu, e.matrix())
    };

    let still = run(0.0)?;
    let moving = run(1.0)?;

    // Which rows and faces belong to the solid.
    let solid = &tm.regions[1];
    let h = &tm.host;
    let mut n_checked = 0usize;
    for c in solid.cells() {
        assert!(
            still[0][c] == moving[0][c] && still[3][c] == moving[3][c],
            "SPEC-LIT 59.2: solid cell {c}'s diag/source moved when the flux \
             did - {} / {} against {} / {}",
            still[0][c],
            still[3][c],
            moving[0][c],
            moving[3][c]
        );
        n_checked += 1;
    }
    for f in solid.internal_face_offset..solid.internal_face_offset + solid.n_internal_faces {
        assert!(
            still[1][f] == moving[1][f] && still[2][f] == moving[2][f],
            "SPEC-LIT 59.2: solid internal face {f}'s off-diagonals moved when \
             the flux did"
        );
        n_checked += 1;
    }
    for bf in solid.boundary_face_offset..solid.boundary_face_offset + solid.n_boundary_faces {
        assert!(
            still[4][bf] == moving[4][bf] && still[5][bf] == moving[5][bf],
            "SPEC-LIT 59.2: solid boundary face {bf}'s coefficients moved when \
             the flux did"
        );
        n_checked += 1;
    }
    assert!(n_checked > 100, "only {n_checked} solid entries were checked");

    // And the FLUID side of every interface face too: no mass crosses a
    // fluid/solid interface, so its coefficients cannot move with the flux
    // either. This is SPEC-LIT §59.2 point 3 - the second boundary mask -
    // and it is checked here rather than argued, because the driver of §59.4
    // happens to hand over a zero `phi` there and would hide the hole.
    for p in &tm.pairs {
        for bf in [p.bf_a as usize, p.bf_b as usize] {
            assert!(
                still[4][bf] == moving[4][bf] && still[5][bf] == moving[5][bf],
                "SPEC-LIT 59.2 point 3: interface face {bf}'s coefficients moved                  when the flux did - a flux on an interface face would convect                  heat through a wall"
            );
        }
    }

    // And the FLUID rows really did move, or the test above would pass on a
    // solver that ignored the flux everywhere.
    let moved = (0..tm.regions[0].n_cells).any(|c| still[0][c] != moving[0][c]);
    assert!(moved, "the flux changed nothing anywhere - the test proves nothing");

    // SPEC-LIT §59.7: the solid block IS §46's conduction matrix. `fvLapFaces`
    // writes `coef = sign gammaMagSf deltaCoeffs` into both off-diagonals and
    // nothing else touches them on a steady solid face - not `relax`, which
    // moves only `diag` and `source`, and not the boundary fold. So the
    // off-diagonal must be exactly `-Dhat_f`, the conductance §46.2/§46.3
    // computed on the host, and the agreement is round-off because the two
    // paths multiply `Dhat/Delta` by `Delta` in different orders.
    let mut worst: Scalar = 0.0;
    for f in solid.internal_face_offset..solid.internal_face_offset + solid.n_internal_faces {
        let want = -cond.gamma_mag_sf[f] * h.delta_coeffs[f];
        worst = worst.max((moving[1][f] - want).abs() / want.abs().max(1e-300));
    }
    // Measured 0.000e0 - BITWISE, not round-off, because `fvLapFaces` forms
    // `gammaMagSf[f]*deltaCoeffs[f]` from the same two f64s in the same order
    // this check does. The threshold stays at round-off rather than at zero,
    // because a future non-orthogonal correction on a solid face would move it
    // legitimately and should not fail here.
    println!("S59.7: worst relative departure of a solid off-diagonal from -Dhat_f: {worst:.3e}");
    assert!(
        worst < 1e-13,
        "SPEC-LIT 59.7: a solid internal face's off-diagonal is off S46's own          Dhat_f by a relative {worst}"
    );
    Ok(())
}

/// SPEC-LIT §59.7: `dp0/dt` is exactly zero in the solid. A solid has no
/// thermodynamic pressure, and §25.2's term reaching one would be heat
/// appearing out of the gas law.
#[test]
fn the_thermodynamic_pressure_term_does_not_reach_the_solid() -> Result<()> {
    let Some(gpu) = gpu() else { return Ok(()) };

    const N: usize = 10;
    let (_meshes, tm) = conjugate_rig(N, 5.0)?;
    let cond = conduction_for(&tm, 1.0, 5.0)?;
    let gm = GpuMesh::upload(&gpu, &tm.host)?;

    let run = |dp0dt: Scalar| -> Result<Vec<Scalar>> {
        let props = air_props();
        let mut gas = GasState::new(&gpu, &gm, props, DomainKind::Sealed, 101_325.0)?;
        gas.set_dp0dt(dp0dt);
        let mut e = Energy::new(&gpu, &gm, energy_controls(), props)?;
        e.attach_conjugate(&gpu, &tm, &cond)?;

        let t0 = vec![300.0 as Scalar; tm.host.n_cells];
        gpu.write(&mut e.field_mut().f, &t0)?;
        fix(&gpu, e.field_mut(), tm.patch_range(1, "hot")?, 305.0)?;
        fix(&gpu, e.field_mut(), tm.patch_range(0, "cold")?, 295.0)?;
        crate::cht::mark_coupled_faces(&gpu, e.field_mut(), &tm)?;
        e.initialise(&gpu)?;

        let phi = GpuSurfaceScalarField::zeros(&gpu, &gm, "phi")?;
        let nut = GpuScalarField::zeros(&gpu, &gm, "nut")?;
        let tke: DevBuf<Scalar> = gpu.zeros(tm.host.n_cells)?;
        gas.update_density(&gpu, e.field())?;
        e.correct(&gpu, &phi, &nut, &tke, 0.71, &gas)?;
        gpu.download(&e.matrix().source)
    };

    let none = run(0.0)?;
    let some = run(1.0e5)?;

    for c in tm.regions[1].cells() {
        assert!(
            none[c] == some[c],
            "SPEC-LIT 59.7: solid cell {c}'s source moved with dp0/dt - {} \
             against {}",
            none[c],
            some[c]
        );
    }
    let moved = tm.regions[0].cells().any(|c| none[c] != some[c]);
    assert!(moved, "dp0/dt changed nothing in the FLUID either - the test proves nothing");
    Ok(())
}

/// SPEC-LIT §59.7: the solid's `ddt` weight is `rho_s c_s`, not `rho(T) cp`.
///
/// Two rigs differing only in the solid's `rho c` must move every solid
/// diagonal by exactly `(d(rho c)) V/dt`, and must not move a fluid one at
/// all. That is both the "the solid carries its own heat capacity" check and
/// the §13.4.1 pair test on `rhoSolid`/`cSolid` under a FLUID case.
#[test]
fn the_solid_ddt_weight_is_the_solids_own_rho_c() -> Result<()> {
    let Some(gpu) = gpu() else { return Ok(()) };

    const N: usize = 10;
    const DT: Scalar = 0.25;
    let (_meshes, tm) = conjugate_rig(N, 5.0)?;
    let gm = GpuMesh::upload(&gpu, &tm.host)?;

    let run = |rho_c: Scalar| -> Result<Vec<Scalar>> {
        let cond = Conduction::uniform_per_region(
            &tm,
            &[
                SolidMaterial::isotropic("air", 1.0, 1.0, 1.0),
                SolidMaterial::isotropic("wall", rho_c, 1.0, 5.0),
            ],
        )?;
        let props = air_props();
        let mut gas = GasState::new(&gpu, &gm, props, DomainKind::Open, 101_325.0)?;
        let ctrl = EnergyControls {
            ddt: DdtScheme::Euler,
            steady: false,
            delta_t: DT,
            t_relax: 1.0,
            ..energy_controls()
        };
        let mut e = Energy::new(&gpu, &gm, ctrl, props)?;
        e.attach_conjugate(&gpu, &tm, &cond)?;

        let t0 = vec![300.0 as Scalar; tm.host.n_cells];
        gpu.write(&mut e.field_mut().f, &t0)?;
        fix(&gpu, e.field_mut(), tm.patch_range(1, "hot")?, 305.0)?;
        fix(&gpu, e.field_mut(), tm.patch_range(0, "cold")?, 295.0)?;
        crate::cht::mark_coupled_faces(&gpu, e.field_mut(), &tm)?;
        e.initialise(&gpu)?;

        let phi = GpuSurfaceScalarField::zeros(&gpu, &gm, "phi")?;
        let nut = GpuScalarField::zeros(&gpu, &gm, "nut")?;
        let tke: DevBuf<Scalar> = gpu.zeros(tm.host.n_cells)?;
        gas.update_density(&gpu, e.field())?;
        e.correct(&gpu, &phi, &nut, &tke, 0.71, &gas)?;
        gpu.download(&e.matrix().diag)
    };

    let a = run(1000.0)?;
    let b = run(3000.0)?;
    let h = &tm.host;

    let mut worst: Scalar = 0.0;
    for c in tm.regions[1].cells() {
        // rho_s c_s = rho * c, and only `rho` moved: d(rho c) = 2000 * 1.0.
        let want = 2000.0 * h.v[c] / DT;
        let got = b[c] - a[c];
        worst = worst.max((got - want).abs() / want);
    }
    assert!(
        worst < 1e-13,
        "SPEC-LIT 59.7: the solid ddt diagonal moved by a relative {worst} away \
         from (d(rho c)) V/dt. If it moved by ZERO the case said rhoSolid and \
         the solver ignored it (SPEC-LIT 13.4.1); if it moved by something else \
         the weight is not the solid's own"
    );
    for c in tm.regions[0].cells() {
        assert!(
            a[c] == b[c],
            "SPEC-LIT 59.1: fluid cell {c}'s diagonal moved with the SOLID's \
             rho c - the blend is leaking across the mask"
        );
    }
    Ok(())
}

/// SPEC-LIT §59.7: the two coupled coefficients across a FLUID/solid
/// interface are equal in every bit, exactly as §47.11 requires across a
/// solid/solid one. One kernel writes both sides from one `h_G` and one
/// `|Sf|`, and `fvLapBoundary`'s interface branch takes the coefficient
/// directly (S47.9).
#[test]
fn the_coupled_interface_coefficients_are_bitwise_equal_with_a_fluid_on_one_side() -> Result<()> {
    let Some(gpu) = gpu() else { return Ok(()) };

    const N: usize = 15;
    let (_meshes, tm) = conjugate_rig(N, 100.0)?;
    let cond = conduction_for(&tm, 1.0, 100.0)?;
    let gm = GpuMesh::upload(&gpu, &tm.host)?;

    let props = air_props();
    let mut gas = GasState::new(&gpu, &gm, props, DomainKind::Open, 101_325.0)?;
    let mut e = Energy::new(&gpu, &gm, energy_controls(), props)?;
    e.attach_conjugate(&gpu, &tm, &cond)?;

    let t0: Vec<Scalar> = (0..tm.host.n_cells)
        .map(|c| 300.0 + ((c * 7) % 23) as Scalar * 0.1)
        .collect();
    gpu.write(&mut e.field_mut().f, &t0)?;
    fix(&gpu, e.field_mut(), tm.patch_range(1, "hot")?, 305.0)?;
    fix(&gpu, e.field_mut(), tm.patch_range(0, "cold")?, 295.0)?;
    crate::cht::mark_coupled_faces(&gpu, e.field_mut(), &tm)?;
    e.initialise(&gpu)?;

    let phi = GpuSurfaceScalarField::zeros(&gpu, &gm, "phi")?;
    let nut = GpuScalarField::zeros(&gpu, &gm, "nut")?;
    let tke: DevBuf<Scalar> = gpu.zeros(tm.host.n_cells)?;

    // SPEC-LIT §47.12 Gate 4 at the FIRST, unconverged iterate - the half a
    // partitioned scheme cannot satisfy. `prepare_coefficients` is what writes
    // the triple, and `correct` runs it before it assembles anything.
    gas.update_density(&gpu, e.field())?;
    e.correct(&gpu, &phi, &nut, &tke, 0.71, &gas)?;

    let bc = gpu.download(&e.matrix().boundary_coeffs)?;
    let ic = gpu.download(&e.matrix().internal_coeffs)?;
    assert!(!tm.pairs.is_empty());
    for p in &tm.pairs {
        let (a, b) = (p.bf_a as usize, p.bf_b as usize);
        assert!(
            bc[a] == bc[b],
            "SPEC-LIT 59.7/47.11: the two coupled boundary coefficients of pair \
             ({a}, {b}) differ - {} against {}. They must be the same NUMBER, \
             not the same number to round-off",
            bc[a],
            bc[b]
        );
        assert!(ic[a] == ic[b], "the two internalCoeffs of pair ({a}, {b}) differ");
        assert!(bc[a] != 0.0, "the interface coefficient is zero - nothing was coupled");
    }

    // And the interface flux balances, on this same unconverged field.
    let flux = e.interface_flux(&gpu)?;
    assert!(
        flux.imbalance() < 1e-12,
        "SPEC-LIT 47.12 Gate 4 with a fluid on side A: imbalance {:.3e}",
        flux.imbalance()
    );
    Ok(())
}

/// SPEC-LIT §59.6: a face carries ONE condition, and the two ways to give it
/// two are refused by name.
#[test]
fn a_face_cannot_be_both_an_interface_and_a_wall_condition() -> Result<()> {
    let Some(gpu) = gpu() else { return Ok(()) };

    const N: usize = 10;
    let (_meshes, tm) = conjugate_rig(N, 5.0)?;
    let cond = conduction_for(&tm, 1.0, 5.0)?;
    let gm = GpuMesh::upload(&gpu, &tm.host)?;
    let props = air_props();

    let mut faces = vec![false; tm.host.n_boundary_faces];
    for bf in tm.patch_range(0, "airToWall")? {
        faces[bf] = true;
    }

    let mut e = Energy::new(&gpu, &gm, energy_controls(), props)?;
    e.attach_conjugate(&gpu, &tm, &cond)?;
    let msg = match e.set_fixed_flux_walls(&gpu, &faces) {
        Err(err) => err.to_string(),
        Ok(()) => panic!("a fixed-flux wall on an interface face must be refused"),
    };
    assert!(msg.contains("47.6"), "{msg}");
    assert!(msg.contains("fixedFluxTemperature"), "{msg}");

    let mut e = Energy::new(&gpu, &gm, energy_controls(), props)?;
    e.attach_conjugate(&gpu, &tm, &cond)?;
    let msg = match e.set_thermal_wall(&gpu, crate::wallfunctions::WallFunctionCoeffs::default(), &faces) {
        Err(err) => err.to_string(),
        Ok(()) => panic!("a thermal wall function on an interface face must be refused"),
    };
    assert!(msg.contains("47.6"), "{msg}");
    assert!(msg.contains("wall function"), "{msg}");

    // And the order is refused too, so the check above can always run.
    let mut e = Energy::new(&gpu, &gm, energy_controls(), props)?;
    e.set_fixed_flux_walls(&gpu, &vec![false; tm.host.n_boundary_faces])?;
    let msg = match e.attach_conjugate(&gpu, &tm, &cond) {
        Err(err) => err.to_string(),
        Ok(()) => panic!("attaching after set_fixed_flux_walls must be refused"),
    };
    assert!(msg.contains("Attach, then set the walls"), "{msg}");
    Ok(())
}

/// SPEC-LIT §59.6: `attach_conjugate` refuses a mesh that is not the one this
/// `Energy` was built on, and a thermal mesh whose region 0 is a solid.
#[test]
fn attach_conjugate_refuses_the_wrong_mesh_and_a_solid_first_region() -> Result<()> {
    let Some(gpu) = gpu() else { return Ok(()) };

    const N: usize = 10;
    let (meshes, tm) = conjugate_rig(N, 5.0)?;
    let cond = conduction_for(&tm, 1.0, 5.0)?;
    let props = air_props();

    // Built on the FLUID mesh, handed the concatenated one.
    let fluid_gm = GpuMesh::upload(&gpu, &meshes[0])?;
    let mut e = Energy::new(&gpu, &fluid_gm, energy_controls(), props)?;
    let msg = match e.attach_conjugate(&gpu, &tm, &cond) {
        Err(err) => err.to_string(),
        Ok(()) => panic!("a size mismatch must be refused"),
    };
    assert!(msg.contains("CONCATENATED"), "{msg}");

    // A thermal mesh with no fluid at all.
    let solid_only = {
        let inputs = [
            RegionInput { name: "a".into(), kind: RegionKind::Solid, mesh: &meshes[1] },
        ];
        ThermalMesh::build(&inputs, &[], PairingTolerances::default())?
    };
    let sc = Conduction::uniform_per_region(
        &solid_only,
        &[SolidMaterial::isotropic("a", 2000.0, 800.0, 5.0)],
    )?;
    let sgm = GpuMesh::upload(&gpu, &solid_only.host)?;
    let mut e = Energy::new(&gpu, &sgm, energy_controls(), props)?;
    let msg = match e.attach_conjugate(&gpu, &solid_only, &sc) {
        Err(err) => err.to_string(),
        Ok(()) => panic!("a solid-only thermal mesh must be refused"),
    };
    assert!(msg.contains("ConjugateHeat"), "{msg}");
    Ok(())
}
