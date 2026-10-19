// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! `ofgpu-validate`'s gates in files of their own (docs/17 section 5.3 rule 2,
//! VAL decision D10): one file per gate, each reached from `validate.rs`
//! through one `c.section` / `enter_gate` call appended to its dispatch block.
//! Each file is a child of the binary's root, so it uses the root's `Checks`,
//! `GateReport` and answer-key reader as they are, and adds no second registry.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §69 (the gate registry), §94 (the mesh study) and
//!     the section each gate file names
//! No GPL-licensed source was consulted.

/// SPEC-LIT §145.2, Gate 145-A: the dam break against Martin & Moyce (1952).
pub(crate) mod martin_moyce;

/// SPEC-LIT §146.2, Gate 78-D's measured leg: the two splash criteria of §78.4
/// against Riboux & Gordillo's (2014) threshold speeds.
pub(crate) mod splash;

/// SPEC-LIT §144.2, Gate 132-L: the laminar flat plate against Blasius.
pub(crate) mod blasius_plate;

/// SPEC-LIT §144.3: the T3A plate on the flat-plate harness, leg 1 in 2-D
/// (recorded runs, replayed).
pub(crate) mod t3a;

/// SPEC-LIT §144.5, Gate 132-Z: the TMR zero-pressure-gradient plate with SA
/// and SST against Karman-Schoenherr and Coles (recorded runs, replayed).
pub(crate) mod zpg;

/// SPEC-LIT §143.5, Gate 131-P: TG0's turbulent pipe with kOmegaSST on the
/// true wedge against its four bands (recorded ofgpu-lowmach runs, replayed).
pub(crate) mod pipe;

/// SPEC-LIT §116.17, Gate 116-P: Hagen-Poiseuille on tetrahedra and prisms,
/// the L1 pipe at Re 100 (recorded ofgpu-lowmach run, replayed).
pub(crate) mod pipe_tet;

/// SPEC-LIT §110.2/§110.5, Gate 110-A: the plane channel's recorded runs at
/// three `Re_tau` (data replayed by `validate.rs`'s channel gate).
pub(crate) mod channel_dns;

/// SPEC-LIT §110.4/§110.5, Gate 110-C: the hot-gas plume's recorded runs at
/// three levels (data replayed by `validate.rs`'s plume gate).
pub(crate) mod plume;

/// SPEC-LIT §110.3/§110.5, Gate 110-B: the backward-facing step's recorded
/// runs at three `dx/H` levels and three single builds (data replayed by
/// `validate.rs`'s step gate).
pub(crate) mod backstep;

/// SPEC-LIT §127.6, Gate CMP-SOD: Sod's shock tube on three meshes against the
/// exact Riemann solution (CMP-19).
pub(crate) mod sod;

/// SPEC-LIT §127.6, Gate CMP-AC: a small acoustic pulse's speed against
/// (122.4)'s `c_0` (CMP-20).
pub(crate) mod acoustic;

/// SPEC-LIT §127.6, Gate CMP-LM: the lid-driven cavity at lid Mach 0.05
/// against Ghia, Ghia & Shin (1982) with Gate 94-D's tolerance (CMP-20).
pub(crate) mod cavity_lm;

/// SPEC-LIT §127.6, Gate CMP-COU: compressible Couette with viscous
/// heating against (127.12) (CMP-22).
pub(crate) mod couette;

/// SPEC-LIT §127.6, Gate CMP-NOZ: the quasi-1-D nozzle, subsonic,
/// supersonic started and shocked, against (127.23)-(127.25) (CMP-21).
pub(crate) mod nozzle;

/// SPEC-LIT §127.6, Gate CMP-OBL: the oblique shock on a 2-D compression
/// ramp at M 2, 10 degrees, against (127.4)'s theta-beta-M (CMP-23).
pub(crate) mod oblique;

/// SPEC-LIT §127.6, Gate CMP-RING: Ringleb's transonic smooth flow on
/// three meshes, h, h/2, h/4, the observed order of the L2 entropy error
/// against the isentropic key (CMP-24).
pub(crate) mod ringleb;

/// SPEC-LIT §127.6, Gate CMP-RAE: RAE 2822 Case 6 with Spalart-Allmaras
/// against AGARD AR-138's measured Cp (NPARC's corrected conditions), a
/// live coarse smoke and the three recorded C-meshes, replayed (CMP-27).
pub(crate) mod rae2822;

/// SPEC-LIT §144.7, Gates 88-T and 90-T leg 3: T3B and T3A- (and T3A
/// refitted) on §144.3's harness with a per-case inflow omega fitted to
/// ERCOFTAC's measured free-stream Tu(x), onset ordering and spread held
/// against Roach & Brierley (docs/17 VAL-35).
pub(crate) mod t3_fitted;
/// SPEC-LIT §157.5, Gate DST-LEEDS: the Leeds fan-stirred cornflour bomb,
/// four live runs of `ofgpu-lowmach`, against Bradley, Chen & Swithenbank
/// (1989) via Huang et al. (2020) Fig. 9.
pub(crate) mod dst_leeds;

/// SPEC-LIT §157.6, Gate DST-VES: the closed vessels - the two thin-flame
/// spheres against the host (157.4) integration, the constant-rho_u
/// negative control, the cube-root law, and the REMBE closed stage
/// against Huang, Bloching & Lipatnikov (2022) Fig. 10.
pub(crate) mod dst_vessel;

/// SPEC-LIT §133.12, Gate CHR-13: the 1-D freely propagating CH4-air flame,
/// BFER phi 1.0, S_L by the 1500 K isotherm on three meshes, replayed from
/// the recorded runs (CHR-13).
pub(crate) mod flame1d;

/// SPEC-LIT §134.8, Gate CHR-16: Sandia Flame D, EDC + DRM19, k-epsilon
/// C1 1.6, REDUCED on the true wedge, replayed from the recorded run (CHR-16).
pub(crate) mod flame_d;

/// SPEC-LIT §110.4 (leg 2 addendum)/§110.5, Gate 110-C leg 2: the radiating
/// plume, P1 grey kappa 1.0 1/m, walls at T_inf, on
/// cases/plumeMcCaffreyRad.jsonc at three levels, replayed through the same
/// P1-P4 bands as leg 1 (docs/17 VAL-54).
pub(crate) mod plume_rad;

/// The CHEMRAD sections' shared fixture (docs/17 CHR-27): the TM-4513
/// ThermoSet read from its tracked key, the ten-species mechanism over it,
/// the two mole recipes, and the radiation and round-off helpers the
/// chr_* gate files port their tests through. No gate of its own.
pub(crate) mod chr_fixture;

/// SPEC-LIT §128.7, Gate CHR-THERMO: the TM-4513 records' NASA-7
/// thermochemistry - cp against TP-2002 Table B1, H against the heats of
/// formation, the T_mid fit constraint, the range refusals, the molar
/// masses (docs/17 CHR-27).
pub(crate) mod chr_thermo;

/// SPEC-LIT §131.4, Gate CHR-EQ: gas-phase equilibrium by Gibbs minimisation
/// on the TM-4513 fixture - element conservation, the 40-digit oracle,
/// detailed balance, the adiabatic flame temperature, the absent-element
/// exact zeros (docs/17 CHR-27).
pub(crate) mod chr_equilibrium;

/// SPEC-LIT §132.6, Gate CHR-ODE: RODAS3 and ROS3 - the order conditions,
/// Robertson and HIRES against the IVP test-set keys, the observed order -
/// and the constant-pressure reactor on equilibrium and its ignition delay
/// (docs/17 CHR-27).
pub(crate) mod chr_ode;

/// SPEC-LIT §134.5, Gate CHR-EDM: EDM's closed-form box and the two-step
/// element balance, PaSR's two limits, and the device kernels
/// `chemEdmSource` and `chemPasrScale` against their f64 host twins. The
/// library and kernel gate of CHR-14, run live, not a driver run
/// (docs/17 CHR-27).
pub(crate) mod chr_edm;

/// SPEC-LIT §135.5, §136.7, Gate CHR-SLAB: the isothermal grey slab - P1
/// against its closed form (135.11) on §135.5's seven cases, the S8 fvDOM
/// slab against 1 - 2 E_3 with tau 0.1's recorded miss, and the step
/// recurrence (136.12) (docs/17 CHR-27).
pub(crate) mod chr_slab;

/// SPEC-LIT §136.8, Gate CHR-CUBE: fvDOM with scattering on the slab
/// against (136.5) solved in the gate, the isothermal cube at 24^3 against
/// (136.6) with the kL 0.1 misses, and Jamaluddin & Smith (1988) named and
/// NOT RUN (docs/17 CHR-27).
pub(crate) mod chr_cube;

/// SPEC-LIT §136.9, Gate CHR-WSGG: the weighted sum of grey gases on the
/// synthetic four-gas set - the per-gas fvDOM and P1 sums and the grey
/// mean; Smith 1982 and Bordbar 2014 named and NOT RUN (docs/17 CHR-27).
pub(crate) mod chr_wsgg;

/// SPEC-LIT §159.11, Gate DST-B-LAM: the Phase B cloud's 1-D laminar flame
/// against the gas flame of its volatile.
pub(crate) mod dst_b_laminar;

/// SPEC-LIT §159.12, Gate DST-B-VES: the Phase B cloud in a closed 20 L
/// sphere, replayed from its records.
pub(crate) mod dst_b_vessel;
