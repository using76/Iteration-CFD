// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The LIBRARY one dust step is made of (docs/17 DST-06 part 2, SPEC-LIT
//! §157.4): [`DustSetup`], every constant of the model formed once from the
//! lowered case, and [`DustSource`], the c~ equation, its closures and its
//! thermo resident on the device. One call,
//!
//! ```text
//! before_flow: (133.18) cp, the closures, the ignition bracket, Gamma,
//!              the face fluxes, one solve of (155.3), the heat release
//!              (152.12), the enthalpy diffusion (133.17) and the molar
//!              change (133.24)
//! ```
//!
//! leaves the fields the low-Mach driver's flow solve reads: the c~ of
//! (155.3), the heat release and the (133.24) molar source on the
//! [`crate::energy::GasState`], and the mixture `cp` on
//! [`crate::energy::Energy`]. The gas is the two pseudo-species of §152.4
//! (Y_B = c~, the inert U), so the mixture density is (133.20) and Le = 1
//! makes Gamma = kappa + D_t the SAME scalar both c~ and T diffuse with.

//! The step's order is SPEC-LIT §157.4's: cp field, closures, ignition,
//! Gamma, fluxes, the c~ solve, wdot_c = S_U - S_P' c~, the heat release,
//! the enthalpy diffusion, the molar change. The three thermo kernels of
//! `cuda/dust.cu` are launched through [`crate::dust::device::SourceKernels`];
//! the closures and the ignition through the DST-03/DST-04 ones. Nothing
//! here reads a case file: the driver (DST-06 part 3) lowers, builds the
//! [`crate::energy::GasState`] on [`DustSource::mixture`], and drives.
//!
//! DST-10 adds [`CompressibleDust`], the same model inside the compressible
//! solver (SPEC-LIT §158.6): (155.3) on the solver's mass flux, the mixture
//! gas law of the constant-c_p pseudo-species, the local rho_u(p), and the
//! energy equation (158.4) in energy units on `rho cp(c~)`, with
//! `h_s = cp(c~) T` and q = Δh_c wdot_c.
//!
//! DST-20's option (SPEC-LIT §160.5): a case stating `flameAge: local` runs
//! the closures on the transported flame age a* of (160.4) instead of the
//! papers' global clock.
//!
//! Written from:
//!   SPEC-LIT §158.4, §158.6 - the compressible coupling (DST-10)
//!   SPEC-LIT §157.4 - the step and its order
//!   SPEC-LIT (152.12), §152.5 - the heat release and the states
//!   SPEC-LIT §153.2 - the reaction time (153.2) with Psi at n = 1
//!   SPEC-LIT §154, §155 - the closures, the ignition and (155.3)
//!   SPEC-LIT (133.17), (133.18), (133.21), (133.24) - the enthalpy
//!     diffusion, the mixture cp, the molar sum and S_B
//!   SPEC-LIT (155.2), (155.9), (155.10) - the net rate, the flame-Courant
//!     number and rho_u(p)
//!   C. Huang, A. N. Lipatnikov & K. Nessvi, J. Loss Prev. Process Ind. 67
//!     (2020) 104237, DOI 10.1016/j.jlp.2020.104237
//!   C. Huang, M. Bloching & A. N. Lipatnikov, J. Loss Prev. Process Ind. 75
//!     (2022) 104707, DOI 10.1016/j.jlp.2021.104707
//! The specification was informed by a review of FSCDustFoam (Huang &
//! Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

use crate::chem::thermo::R_GAS;
use crate::compressible::energy::RhoEnergy;
use crate::compressible::thermo::{DustHConst, MixKernels};
use crate::compressible::CompressibleState;
use crate::device::{DevBuf, Gpu};
use crate::dust::age::{C_A, FlameAge};
use crate::dust::device::{
    self as dev, FscFields, FscInputsDev, ProgressKernels, SourceKernels,
};
use crate::dust::fsc::{development_time, FscConstants, FscInputs, THETA};
use crate::dust::ignition::IgnitionSource;
use crate::dust::progress::{
    unburned_density, unburned_temperature, ProgressCoupling, ProgressEquation, ProgressSolve,
};
use crate::dust::psi::{reaction_time, AdiabaticCurve, PsiOptions, PsiProblem};
use crate::dust::thermo::EquivalentGas;
use crate::dust::vessel0d::{BurningLaw, UnburnedDensity};
use crate::error::{Error, Result};
use crate::field::{BcKind, GpuScalarField, GpuSurfaceScalarField, GpuVectorField};
use crate::field_ops;
use crate::field_setup::setup_scalar_field;
use crate::fv::{self, FvKernels};
use crate::io::case::TurbulenceControls;
use crate::io::case_dust::LoweredDust;
use crate::io::fields::RawScalarField;
use crate::mesh::{GpuMesh, HostMesh};
use crate::models::coupled::CoupledTurbulence;
use crate::models::k_omega_sst::SstDensity;
use crate::turbulence::FlowState;
use crate::{Acc, Scalar, Vec3};

/// Everything of the dust model that is formed ONCE, on the host in f64,
/// from the lowered blocks at the initial pressure `p_init` and the energy
/// equation's molecular conductivity `k_mol` (W/(m K)) - SPEC-LIT §157.4's
/// stated inputs and the derived state of (152.10), (153.2), (155.10).
#[derive(Clone)]
pub struct DustSetup {
    /// The equivalent gas at the lowered thermo (§152.4, §152.9).
    pub gas: EquivalentGas,
    /// The closures' constants (§154.7).
    pub fsc: FscConstants,
    /// The ignition source of (155.4); `None` when the case named none.
    pub ignition: Option<IgnitionSource>,
    /// Gate DST-VES's prescribed `U_t` law (SPEC-LIT §157.6); `None` runs
    /// the §154 closures.
    pub prescribed_ut: Option<BurningLaw>,
    /// Which density the propagation reads (SPEC-LIT §157.6): the
    /// compressed (155.10) gas, or the initial one for the negative
    /// control.
    pub unburned_density: UnburnedDensity,
    /// The ignition's t0 (§151.5): the origin of the flame development time.
    /// 0.0 without an ignition, so t_fd is the clock itself.
    pub t_fd_origin: f64,
    /// The clock of the closures (SPEC-LIT §160.5): the papers' global one
    /// (`Global`), or the transported local flame age (`Local`), which
    /// switches the age on in the constructors.
    pub flame_age: FlameAge,
    /// K, the unburned temperature (the case's `initial.T`).
    pub t_u: f64,
    /// Pa, the initial thermodynamic pressure.
    pub p_init: f64,
    /// kg/m3, rho_u at `p_init` (152.9) - the (155.10) datum.
    pub rho_u0: f64,
    /// gamma_u = cp_U(T_u)/(cp_U(T_u) - R0/W_U).
    pub gamma_u: f64,
    /// K, the isobaric burned temperature (152.10).
    pub t_b: f64,
    /// rho_u/rho_b of (152.10).
    pub sigma: f64,
    /// kg/m3, the burned density at `p_init`, T_b.
    pub rho_b0: f64,
    /// W/(m K), the molecular conductivity the energy equation runs with.
    pub k_mol: f64,
    /// m2/s, kappa_u = k_mol/(rho_u0 cp_u0).
    pub kappa_u: f64,
    /// m2/s, kappa_b = k_mol/(rho_b0 cp_B(T_b)).
    pub kappa_b: f64,
    /// m/s, the laminar burning velocity (152.13).
    pub s_l: f64,
    /// The eigen-solution's Psi (§153.2, n = 1).
    pub psi: f64,
    /// s, the reaction time (153.2).
    pub t_r: f64,
    /// The flame-Courant bound of (155.9).
    pub max_flame_courant: f64,
}

impl DustSetup {
    /// From the lowered blocks at the initial `p_init` and the energy
    /// equation's `k_mol` (SPEC-LIT §157.4): the gas of §152.4/§152.9, the
    /// state of (152.9)-(152.10), gamma_u, the two molecular heat
    /// diffusivities, Psi at n = 1 (rho kappa constant, §153.2), the
    /// reaction time (153.2) and the closures' constants - each ignition
    /// validated. Refuses, by name, anything the pieces refuse.
    pub fn new(d: &LoweredDust, p_init: f64, k_mol: f64) -> Result<Self> {
        const W: &str = "dust::source::DustSetup::new";
        let bad = |name: &str, v: f64| {
            Error::Refused(format!("{W}: {name} = {v} is not a finite positive value"))
        };
        for (name, v) in [("p_init", p_init), ("k_mol", k_mol)] {
            if !(v.is_finite() && v > 0.0) {
                return Err(bad(name, v));
            }
        }
        let gas = d.gas()?;
        let cp_u0 = gas.cp_u(d.t_u)?;
        let rho_u0 = gas.rho(p_init, d.t_u, 0.0)?;
        let gamma_u = cp_u0 / (cp_u0 - R_GAS / gas.w_u());
        let b = gas.isobaric_burned(d.t_u)?;
        let (t_b, sigma) = (b.t_b, b.sigma);
        let rho_b0 = gas.rho(p_init, t_b, 1.0)?;
        let kappa_u = k_mol / (rho_u0 * cp_u0);
        let kappa_b = k_mol / (rho_b0 * gas.cp_b(t_b)?);
        let s_l = d.s_l;
        let curve = AdiabaticCurve::new(&gas, d.t_u)?;
        let problem = PsiProblem { curve: &curve, theta_u: THETA / d.t_u, n: 1.0 };
        let psi = problem.eigen(&PsiOptions::default())?.psi;
        let t_r = reaction_time(psi, kappa_u, s_l)?;
        let fsc = FscConstants::new(&FscInputs {
            model: d.model,
            a: d.a,
            s_l,
            kappa_u,
            pr_t: d.pr_t,
            t_r,
            t_b,
            kappa_b,
            t_flacc: d.t_flacc,
        })?;
        let ignition = match &d.ignition {
            None => None,
            Some(src) => {
                src.validate()?;
                Some(*src)
            }
        };
        let t_fd_origin = d.ignition.as_ref().map(|src| src.t0).unwrap_or(0.0);
        Ok(Self {
            gas,
            fsc,
            ignition,
            prescribed_ut: d.prescribed_ut,
            unburned_density: d.unburned_density,
            t_fd_origin,
            flame_age: d.flame_age,
            t_u: d.t_u,
            p_init,
            rho_u0,
            gamma_u,
            t_b,
            sigma,
            rho_b0,
            k_mol,
            kappa_u,
            kappa_b,
            s_l,
            psi,
            t_r,
            max_flame_courant: d.max_flame_courant,
        })
    }

    /// (155.10): the CURRENT unburned density `rho_u0 (p/p0)^{1/gamma_u}`
    /// at the run's `p0`.
    pub fn rho_u(&self, p: f64) -> f64 {
        unburned_density(self.rho_u0, p, self.p_init, self.gamma_u)
    }

    /// The density the flame PROPAGATION reads at `p` (SPEC-LIT §157.6):
    /// the (155.10) compressed gas of [`Self::rho_u`], or `rho_u0` held at
    /// the initial state for `unburnedDensity "initial"` - INSIGHTS (g)'s
    /// negative control.
    pub fn propagation_density(&self, p: f64) -> f64 {
        match self.unburned_density {
            UnburnedDensity::Compressed => self.rho_u(p),
            UnburnedDensity::Initial => self.rho_u0,
        }
    }

    /// (155.2)'s burned density: `rho_b(p) = p W_B/(R0 T_b(p))` with
    /// `T_b(p)` the isobaric burned state of `T_u(p) = T_u
    /// (p/p0)^{(gamma_u-1)/gamma_u}`.
    pub fn rho_b(&self, p: f64) -> Result<f64> {
        let t_up = unburned_temperature(self.t_u, p, self.p_init, self.gamma_u);
        let t_bp = self.gas.isobaric_burned(t_up)?.t_b;
        Ok(p * self.gas.w_b() / (R_GAS * t_bp))
    }
}

/// The frozen turbulence one step's closures read (§154.4's k, omega and
/// the SST's nu_t), all `n_cells` long.
pub struct DustTurbulence<'a> {
    pub k: &'a DevBuf<Scalar>,
    pub omega: &'a DevBuf<Scalar>,
    pub nu_t: &'a DevBuf<Scalar>,
}

/// One dust step's machinery, resident on the device (SPEC-LIT §157.4). The
/// `'m` is the mesh's borrow, as every equation here holds.
pub struct DustSource<'m> {
    setup: DustSetup,
    progress: ProgressEquation<'m>,
    sk: SourceKernels,
    fk: crate::dust::device::FscKernels,
    pk: ProgressKernels,
    fvk: FvKernels,
    fldk: crate::field_ops::FieldKernels,
    row_u: DevBuf<Acc>,
    row_b: DevBuf<Acc>,
    cp_cells: DevBuf<Scalar>,
    cp_faces: DevBuf<Scalar>,
    d_t: DevBuf<Scalar>,
    u_t: DevBuf<Scalar>,
    u_t_isp: DevBuf<Scalar>,
    tau_l: DevBuf<Scalar>,
    k_rate: DevBuf<Scalar>,
    w_ign: DevBuf<Scalar>,
    wdot: DevBuf<Scalar>,
    n_zero: DevBuf<u64>,
    /// The three frozen-turbulence buffers a laminar run's closures read:
    /// k = omega = nu_t = 0 leaves the turbulent closure zero (decision D5).
    zero: DevBuf<Scalar>,
    /// The prediction's T input: u_t_isp reads no T, and a cell at T = 0 is
    /// zero-rule (no K), so the one launch below is exact without a field.
    t_pred: DevBuf<Scalar>,
    tmp: DevBuf<Scalar>,
    diff: GpuScalarField,
    rho_face: GpuSurfaceScalarField,
    phi_mass: GpuSurfaceScalarField,
    gamma_face: GpuSurfaceScalarField,
    flux: GpuSurfaceScalarField,
    net: DevBuf<Scalar>,
    q: DevBuf<Scalar>,
    qhd: DevBuf<Scalar>,
    grad_c: DevBuf<Vec3>,
    grad_t: DevBuf<Vec3>,
    co: DevBuf<Scalar>,
    cbar: DevBuf<Scalar>,
    p_local: Option<Vec3>,
    raw_c: RawScalarField,
    n: usize,
    nf: usize,
    nb: usize,
}

impl<'m> DustSource<'m> {
    /// The step's machinery on one mesh: the c~ equation with the case's
    /// `raw_c` seeded into it (then [`ProgressEquation::initialise`]), the
    /// two NASA-7 rows of the pseudo-species, and the buffers §157.4's
    /// order runs through. `diff`'s boundary faces are all zeroGradient, so
    /// the Gamma a boundary Laplacian reads is the owner cell's.
    pub fn new(
        gpu: &Gpu,
        hm: &HostMesh,
        m: &'m GpuMesh,
        setup: DustSetup,
        raw_c: &RawScalarField,
        ctrl: TurbulenceControls,
    ) -> Result<Self> {
        let (n, nf, nb) = (m.n_cells, m.n_internal_faces, m.n_boundary_faces);
        let cells = n.max(1);
        let mut progress = ProgressEquation::new(gpu, hm, m, ctrl)?;
        setup_scalar_field(gpu, progress.field_mut(), raw_c, hm)?;
        progress.initialise(gpu)?;
        // The local flame-age clock (SPEC-LIT §160.5): the case's call.
        if setup.flame_age == FlameAge::Local {
            progress.enable_age(gpu, setup.t_fd_origin, C_A)?;
        }
        let mut diff = GpuScalarField::zeros(gpu, m, "dustGamma")?;
        gpu.write(&mut diff.bc_kind, &vec![BcKind::ZeroGradient as crate::Label; nb])?;
        let p_local = setup.ignition.map(|src| src.local_point(hm));
        Ok(Self {
            sk: SourceKernels::new(gpu)?,
            fk: crate::dust::device::FscKernels::new(gpu)?,
            pk: ProgressKernels::new(gpu)?,
            fvk: FvKernels::new(gpu)?,
            fldk: field_ops::FieldKernels::new(gpu)?,
            row_u: gpu.upload(&dev::thermo_row(&setup.gas.unburned))?,
            row_b: gpu.upload(&dev::thermo_row(&setup.gas.burned))?,
            cp_cells: gpu.zeros(cells)?,
            cp_faces: gpu.zeros(nb.max(1))?,
            d_t: gpu.zeros(cells)?,
            u_t: gpu.zeros(cells)?,
            u_t_isp: gpu.zeros(cells)?,
            tau_l: gpu.zeros(cells)?,
            k_rate: gpu.zeros(cells)?,
            w_ign: gpu.zeros(cells)?,
            wdot: gpu.zeros(cells)?,
            n_zero: gpu.zeros(1)?,
            zero: gpu.zeros(cells)?,
            t_pred: gpu.zeros(cells)?,
            tmp: gpu.zeros(cells)?,
            diff,
            rho_face: GpuSurfaceScalarField::zeros(gpu, m, "dustRhoF")?,
            phi_mass: GpuSurfaceScalarField::zeros(gpu, m, "dustPhiM")?,
            gamma_face: GpuSurfaceScalarField::zeros(gpu, m, "dustGamF")?,
            flux: GpuSurfaceScalarField::zeros(gpu, m, "dustSnGrad")?,
            net: gpu.zeros(cells)?,
            q: gpu.zeros(cells)?,
            qhd: gpu.zeros(cells)?,
            grad_c: gpu.zeros(m.n_local())?,
            grad_t: gpu.zeros(m.n_local())?,
            co: gpu.zeros(cells)?,
            cbar: gpu.zeros(cells)?,
            progress,
            setup,
            p_local,
            raw_c: raw_c.clone(),
            n,
            nf,
            nb,
        })
    }

    /// The constants the step runs with.
    pub fn setup(&self) -> &DustSetup {
        &self.setup
    }

    /// This field's own `div(phi,c)` entry (§11.7), the driver's
    /// `c_controls` forwarding the case's named scheme.
    pub fn set_convection(&mut self, conv: crate::io::DivEntry) {
        self.progress.set_convection(conv);
    }

    /// The face density `form_mass_flux` last interpolated - the `rho_face`
    /// a rho-weighted SST correction reads (§156.1).
    pub fn rho_face(&self) -> &GpuSurfaceScalarField {
        &self.rho_face
    }

    /// The MASS flux `form_mass_flux` last formed - the `FlowState::phi` a
    /// rho-weighted SST correction runs on (§156.1's §86.3 contract).
    pub fn phi_mass(&self) -> &GpuSurfaceScalarField {
        &self.phi_mass
    }

    /// The two pseudo-species as the [`crate::energy::GasMixture`] the
    /// driver's [`crate::energy::GasState::use_mixture`] takes: solved
    /// DUST_B, inert DUST_U (§152.4, Y_B = c~).
    pub fn mixture(&self) -> Result<crate::energy::GasMixture> {
        crate::energy::GasMixture::new(
            &[("DUST_B", self.setup.gas.w_b())],
            ("DUST_U", self.setup.gas.w_u()),
        )
    }

    /// The progress variable c~ (Y_B = c~).
    pub fn c(&self) -> &GpuScalarField {
        self.progress.field()
    }

    /// The transported flame-age alpha field, when the case runs the local
    /// clock (SPEC-LIT §160.5); `None` with the global clock.
    pub fn age(&self) -> Option<&GpuScalarField> {
        self.progress.age_field()
    }

    /// The ignition bracket W of (155.4) as `before_flow` last formed it;
    /// zeros without an ignition.
    pub fn w_ign(&self) -> &DevBuf<Scalar> {
        &self.w_ign
    }

    /// K of (154.11) as `before_flow` last formed it (0 where the zero rule
    /// wrote it, or under TFC).
    pub fn k_rate(&self) -> &DevBuf<Scalar> {
        &self.k_rate
    }

    /// The net rate `wdot_c = S_U - S_P' c~` (155.2) as `before_flow` last
    /// formed it, kg/(m3 s).
    pub fn wdot(&self) -> &DevBuf<Scalar> {
        &self.wdot
    }

    /// The face `rho_face` and the MASS flux `phi_mass = rho_f phi` of the
    /// flux `phi_volume` (m3/s) the flow solve holds - (155.3)'s convection
    /// runs on the mass-weighted assembly, so it reads kg/s.
    pub fn form_mass_flux(
        &mut self,
        gpu: &Gpu,
        m: &GpuMesh,
        gas: &crate::energy::GasState<'_>,
        phi_volume: &GpuSurfaceScalarField,
    ) -> Result<()> {
        fv::interpolate_linear(gpu, &self.fvk, &mut self.rho_face, gas.rho(), m)?;
        field_ops::copy_field(gpu, &self.fldk, &mut self.phi_mass.f, &self.rho_face.f, self.nf)?;
        field_ops::multiply_field(gpu, &self.fldk, &mut self.phi_mass.f, &phi_volume.f, self.nf)?;
        field_ops::copy_field(gpu, &self.fldk, &mut self.phi_mass.bf, &self.rho_face.bf, self.nb)?;
        field_ops::multiply_field(gpu, &self.fldk, &mut self.phi_mass.bf, &phi_volume.bf, self.nb)?;
        Ok(())
    }

    /// Starts a time step: the c~ equation's old level and its step-long
    /// `|grad c~|` (§155.2), then - as §157.4's step does - the closures'
    /// U_t held over the step is NOT re-formed here: `before_flow` forms
    /// everything, once per call, from the fields as they stand.
    pub fn open_step(&mut self, gpu: &Gpu) -> Result<()> {
        self.progress.begin_step(gpu)
    }

    /// Step 1 of the outer iteration: the `inletOutlet` faces of c~ switch
    /// on the flux the flow solve last made (§155.3's convection follows).
    /// The age field switches with them, the same call (SPEC-LIT §160.5).
    pub fn update_inlet_outlet(&mut self, gpu: &Gpu, phi: &GpuSurfaceScalarField) -> Result<()> {
        field_ops::update_inlet_outlet_scalar(gpu, &self.fldk, self.progress.field_mut(), phi)?;
        if let Some(alpha) = self.progress.age_field_mut() {
            field_ops::update_inlet_outlet_scalar(gpu, &self.fldk, alpha, phi)?;
        }
        Ok(())
    }

    /// Step 2: (133.20) - the density of the two pseudo-species at the c~
    /// of each time level, `Y_B = c~` the one solved member.
    pub fn update_density(
        &self,
        gpu: &Gpu,
        gas: &mut crate::energy::GasState<'_>,
        t: &GpuScalarField,
    ) -> Result<()> {
        gas.update_density_mixture(gpu, t, &[self.c()])
    }

    /// One outer iteration of the dust model, in §157.4's order: the
    /// mixture cp (133.18) onto the energy equation's field, the closures,
    /// the ignition bracket (155.4), Gamma = k_mol/(rho cp) + D_t, the face
    /// fluxes, one solve of (155.3), the net rate `wdot_c = S_U - S_P' c~`,
    /// the heat release (152.12) and the enthalpy diffusion (133.17)
    /// registered on the energy sources, and (133.24)'s S_B on the gas's
    /// molar change. `t_new` is the run's time; the closures' t_fd runs
    /// from [`DustSetup::t_fd_origin`]. `turb = None` is the laminar step.
    #[allow(clippy::too_many_arguments)]
    pub fn before_flow(
        &mut self,
        gpu: &Gpu,
        m: &GpuMesh,
        energy: &mut crate::energy::Energy<'_>,
        gas: &mut crate::energy::GasState<'_>,
        phi_volume: &GpuSurfaceScalarField,
        u: &GpuVectorField,
        turb: Option<DustTurbulence<'_>>,
        t_new: f64,
    ) -> Result<ProgressSolve> {
        let n = self.n;
        if n == 0 {
            return Ok(ProgressSolve {
                perf: Default::default(),
                out_of_bounds: 0,
            });
        }
        // The local clock's levels move once per step, before the first
        // outer solve (SPEC-LIT §160.5); nothing with the global clock.
        self.progress.begin_age_step(gpu, t_new)?;
        // 1. (133.18) on the cells and the boundary faces, onto the cp field.
        dev::mixture_cp(
            gpu,
            &self.sk,
            &mut self.cp_cells,
            &energy.field().f,
            &self.progress.field().f,
            &self.row_u,
            &self.row_b,
            n,
        )?;
        dev::mixture_cp(
            gpu,
            &self.sk,
            &mut self.cp_faces,
            &energy.field().bf,
            &self.progress.field().bf,
            &self.row_u,
            &self.row_b,
            self.nb,
        )?;
        energy.set_cp_field(gpu, &self.cp_cells, &self.cp_faces)?;
        // 2. The closures of §154 at this step's clock - a* of (160.4) with
        //    the local flame age on, the global clock without (§160.5).
        let (k_b, o_b, n_b) = match &turb {
            Some(t) => (t.k, t.omega, t.nu_t),
            None => (&self.zero, &self.zero, &self.zero),
        };
        if let Some(age) = self.progress.closure_age() {
            dev::fsc_aged(
                gpu,
                &self.fk,
                &self.setup.fsc,
                FscFields {
                    d_t: &mut self.d_t,
                    u_t: &mut self.u_t,
                    u_t_isp: &mut self.u_t_isp,
                    tau_l: &mut self.tau_l,
                    k_rate: &mut self.k_rate,
                },
                &mut self.n_zero,
                FscInputsDev { t: &energy.field().f, k: k_b, omega: o_b, nu_t: n_b },
                age,
                n,
            )?;
        } else {
            dev::fsc(
                gpu,
                &self.fk,
                &self.setup.fsc,
                FscFields {
                    d_t: &mut self.d_t,
                    u_t: &mut self.u_t,
                    u_t_isp: &mut self.u_t_isp,
                    tau_l: &mut self.tau_l,
                    k_rate: &mut self.k_rate,
                },
                &mut self.n_zero,
                FscInputsDev { t: &energy.field().f, k: k_b, omega: o_b, nu_t: n_b },
                development_time(t_new, self.setup.t_fd_origin),
                t_new,
                n,
            )?;
        }
        // 2b. Gate DST-VES's prescribed U_t (SPEC-LIT §157.6): the case's
        //     law at the CURRENT p0 replaces the closures' U_t - the TFC
        //     kernel already gives D_t = 0 and K = 0 on a laminar case.
        if let Some(law) = self.setup.prescribed_ut {
            field_ops::set_field(
                gpu,
                &self.fldk,
                &mut self.u_t,
                law.s_u(f64::from(gas.p0()), self.setup.p_init, self.setup.gamma_u)
                    as Scalar,
                n,
            )?;
        }
        // 3. The ignition bracket (155.4), or nothing without a source.
        if let (Some(src), Some(p)) = (&self.setup.ignition, self.p_local) {
            dev::ignition(gpu, &self.pk, &mut self.w_ign, &m.c, src, p, t_new, n)?;
        }
        // 4. Gamma = k_mol/(rho cp(T, c~)) + D_t, Le = 1: the SAME scalar
        //    c~ and T diffuse with. The boundary faces read the owner cell
        //    through the zeroGradient kinds `new` wrote.
        field_ops::copy_field(gpu, &self.fldk, &mut self.tmp, &gas.rho().f, n)?;
        field_ops::multiply_field(gpu, &self.fldk, &mut self.tmp, &self.cp_cells, n)?;
        field_ops::set_field(gpu, &self.fldk, &mut self.diff.f, self.setup.k_mol as Scalar, n)?;
        field_ops::divide_field(gpu, &self.fldk, &mut self.diff.f, &self.tmp, n)?;
        field_ops::add_field(gpu, &self.fldk, &mut self.diff.f, &self.d_t, n)?;
        field_ops::correct_boundary_conditions(gpu, &self.fldk, &mut self.diff, m)?;
        // 5. rho_f and the mass flux on the flow's volumetric flux.
        self.form_mass_flux(gpu, m, gas, phi_volume)?;
        // 6. One solve of (155.3), mass-weighted, rho_u(p) of (155.10).
        let has_ign = self.setup.ignition.is_some();
        let solve = self.progress.solve(
            gpu,
            &ProgressCoupling {
                rho: gas.rho(),
                rho_face: &self.rho_face,
                phi_mass: &self.phi_mass,
                u,
                diffusivity: &self.diff,
                u_t: &self.u_t,
                k_rate: &self.k_rate,
                w_ign: has_ign.then_some(&self.w_ign),
                rho_u: self.setup.propagation_density(f64::from(gas.p0())),
            },
        )?;
        // 7. wdot_c = S_U - S_P' c~ (155.2), kg/(m3 s).
        let (su, sp) = self.progress.source_terms();
        field_ops::copy_field(gpu, &self.fldk, &mut self.wdot, sp, n)?;
        field_ops::multiply_field(gpu, &self.fldk, &mut self.wdot, &self.progress.field().f, n)?;
        field_ops::scale_field(gpu, &self.fldk, &mut self.wdot, -1.0, n)?;
        field_ops::add_field(gpu, &self.fldk, &mut self.wdot, su, n)?;
        // 8. (152.12) registered as the temperature equation's heat release.
        dev::heat_release(
            gpu,
            &self.sk,
            &mut self.q,
            &energy.field().f,
            &self.wdot,
            &self.row_u,
            &self.row_b,
            n,
        )?;
        energy.sources_mut().register_heat_release(gpu, &self.q)?;
        // 9. (133.17) on the Gauss gradients of c~ and T, registered.
        fv::fvc_grad_scalar(gpu, &self.fvk, &mut self.grad_c, self.progress.field(), m)?;
        fv::fvc_grad_scalar(gpu, &self.fvk, &mut self.grad_t, energy.field(), m)?;
        dev::enthalpy_diffusion(
            gpu,
            &self.sk,
            &mut self.qhd,
            &energy.field().f,
            &gas.rho().f,
            &self.diff.f,
            &self.grad_c,
            &self.grad_t,
            &self.row_u,
            &self.row_b,
            n,
        )?;
        energy.sources_mut().register_enthalpy_diffusion(gpu, &self.qhd)?;
        // 10. (133.24)'s diffusion half on the faces: rho_f Gamma_f |S_f|
        //     snGrad(c~) summed over a cell's faces, plus the reaction half
        //     wdot_c; handed over as the one solved member's S_B.
        fv::interpolate_linear(gpu, &self.fvk, &mut self.gamma_face, &self.diff, m)?;
        field_ops::multiply_field(gpu, &self.fldk, &mut self.gamma_face.f, &self.rho_face.f, self.nf)?;
        field_ops::multiply_field(gpu, &self.fldk, &mut self.gamma_face.f, &m.mag_sf, self.nf)?;
        field_ops::multiply_field(gpu, &self.fldk, &mut self.gamma_face.bf, &self.rho_face.bf, self.nb)?;
        field_ops::multiply_field(gpu, &self.fldk, &mut self.gamma_face.bf, &m.b_mag_sf, self.nb)?;
        fv::sn_grad_flux(gpu, &self.fvk, &mut self.flux, self.progress.field(), &self.gamma_face.f, &self.gamma_face.bf, m)?;
        fv::fvc_div_surface(gpu, &self.fvk, &mut self.net, &self.flux, m)?;
        field_ops::add_field(gpu, &self.fldk, &mut self.net, &self.wdot, n)?;
        gas.set_molar_change(gpu, &[&self.net])?;
        Ok(solve)
    }

    /// The PREDICTED flame-Courant number of a fixed `dt` (155.9): the
    /// closures run ONCE on the fields as they stand at t_fd = 1e30
    /// (developed, so U_t <= U_t,ISP), `u_max` the largest |u| and dx_min
    /// the smallest `V^{1/3}` on the host in f64, then
    /// [`crate::dust::courant::check_fixed_step`] - which REFUSES, naming
    /// `maxFlameCourant`, a step above the bound. Reads the counters back,
    /// so it is not capturable; call it between steps.
    pub fn predicted_courant(
        &mut self,
        gpu: &Gpu,
        hm: &HostMesh,
        u: &GpuVectorField,
        turb: Option<DustTurbulence<'_>>,
        dt: f64,
    ) -> Result<f64> {
        let n = self.n;
        let (k_b, o_b, n_b) = match &turb {
            Some(t) => (t.k, t.omega, t.nu_t),
            None => (&self.zero, &self.zero, &self.zero),
        };
        dev::fsc(
            gpu,
            &self.fk,
            &self.setup.fsc,
            FscFields {
                d_t: &mut self.d_t,
                u_t: &mut self.u_t,
                u_t_isp: &mut self.u_t_isp,
                tau_l: &mut self.tau_l,
                k_rate: &mut self.k_rate,
            },
            &mut self.n_zero,
            FscInputsDev { t: &self.t_pred, k: k_b, omega: o_b, nu_t: n_b },
            1.0e30,
            1.0e30,
            n,
        )?;
        let isp_max = gpu
            .download(&self.u_t_isp)?
            .iter()
            .take(n)
            .fold(0.0f64, |a, &v| a.max(f64::from(v)));
        // Gate DST-VES's prescribed U_t (SPEC-LIT §157.6): the prediction
        // reads the case's law at BOTH ends of the pressure range, the
        // initial and the AICC end state, not the closures' U_t,ISP.
        let isp_max = match self.setup.prescribed_ut {
            Some(law) => {
                let p_e = self.setup.p_init
                    * self.setup.gas.aicc(self.setup.t_u)?.p_ratio;
                let s0 = law.s_u(self.setup.p_init, self.setup.p_init, self.setup.gamma_u);
                let s_e = law.s_u(p_e, self.setup.p_init, self.setup.gamma_u);
                f64::max(s0, s_e)
            }
            None => isp_max,
        };
        let u_max = gpu
            .download(&u.f)?
            .iter()
            .take(n)
            .fold(0.0f64, |a, v| a.max((f64::from(v.x) * f64::from(v.x)
                + f64::from(v.y) * f64::from(v.y)
                + f64::from(v.z) * f64::from(v.z))
                .sqrt()));
        let dx_min = hm
            .v
            .iter()
            .map(|v| f64::from(*v).cbrt())
            .fold(f64::INFINITY, f64::min);
        let predicted = crate::dust::courant::predicted(
            dt,
            dx_min,
            u_max,
            self.setup.sigma,
            isp_max,
        );
        crate::dust::courant::check_fixed_step(dt, predicted, self.setup.max_flame_courant)
    }

    /// The MEASURED flame-Courant number (155.9) of the step just taken,
    /// `dustFlameCourant` into the Co_f buffer, its max returned in f64.
    pub fn measured_courant(
        &mut self,
        gpu: &Gpu,
        m: &GpuMesh,
        u: &GpuVectorField,
        gas: &crate::energy::GasState<'_>,
        dt: f64,
    ) -> Result<f64> {
        let n = self.n;
        dev::flame_courant(
            gpu,
            &self.pk,
            &mut self.co,
            &u.f,
            &gas.rho().f,
            &self.u_t,
            &m.v,
            self.setup.propagation_density(f64::from(gas.p0())),
            dt,
            n,
        )?;
        Ok(gpu
            .download(&self.co)?
            .iter()
            .take(n)
            .fold(0.0f64, |a, &v| a.max(f64::from(v))))
    }

    /// The names [`Self::output`] writes, in order.
    pub fn field_names() -> [&'static str; 6] {
        ["c", "cbar", "Ut", "Cof", "HRR", "p0"]
    }

    /// A `calculated` copy of the c field's seed: every patch whose type
    /// names a value rule is rewritten to `calculated`, and the constraint
    /// patches (empty, symmetry, wedge) keep their type; the values are the
    /// caller's.
    fn calculated_seed(&self, name: &str, dims: &str) -> RawScalarField {
        let mut raw = self.raw_c.types_only();
        for spec in raw.boundary.values_mut() {
            if matches!(
                spec.type_name.as_str(),
                "fixedValue" | "zeroGradient" | "inletOutlet" | "fixedGradient" | "mixed" | "calculated"
            ) {
                spec.type_name = "calculated".to_string();
                spec.extra.clear();
            }
        }
        raw.boundary_patterns.clear();
        raw.name = name.to_string();
        raw.dimensions = dims.to_string();
        raw.internal.clear();
        for spec in raw.boundary.values_mut() {
            spec.value.clear();
        }
        raw
    }

    /// The step's six fields as the case format holds them: `c` with its
    /// OWN patch types, the other five as `calculated` copies of that
    /// seed's types - `cbar` (155.2) at rho_b(p0), `Ut`, `Cof` (155.9),
    /// `HRR` (152.12) and the uniform `p0`. Dimensions: c, cbar and Cof are
    /// dimensionless, Ut is m/s, HRR W/m3, p0 Pa.
    pub fn output(
        &mut self,
        gpu: &Gpu,
        hm: &HostMesh,
        gas: &crate::energy::GasState<'_>,
    ) -> Result<Vec<(String, RawScalarField)>> {
        let n = self.n;
        let mut c_raw = self.raw_c.types_only();
        crate::field_setup::harvest_scalar_field(gpu, &mut c_raw, self.c(), hm)?;
        let bvals_of = |b: &DevBuf<Scalar>| -> Result<Vec<Scalar>> {
            Ok(gpu.download(b)?.into_iter().take(self.nb).collect())
        };
        let owner = |cell: &[Scalar], f: usize| -> Scalar {
            cell[hm.b_face_cells[f] as usize]
        };
        let fill_boundary = |raw: &mut RawScalarField, per_face: &[Scalar]| {
            for p in &hm.patches {
                if let Some(spec) = raw.boundary.get_mut(&p.name) {
                    spec.value = per_face[p.start..p.start + p.size].to_vec();
                }
            }
        };
        // cbar (155.2) on the faces from the owners; cells on the device.
        let rho_b = self.setup.rho_b(f64::from(gas.p0()))?;
        dev::reynolds_mean(gpu, &self.pk, &mut self.cbar, &gas.rho().f, &self.progress.field().f, rho_b, n)?;
        let (rho_bf, c_bf) = (bvals_of(&gas.rho().bf)?, bvals_of(&self.c().bf)?);
        let cbar_b: Vec<Scalar> =
            (0..self.nb).map(|f| ((f64::from(rho_bf[f]) * f64::from(c_bf[f])) / rho_b) as Scalar).collect();
        let mut cbar_raw = self.calculated_seed("cbar", "[0 0 0 0 0 0 0]");
        cbar_raw.internal = gpu.download(&self.cbar)?.into_iter().take(n).collect();
        fill_boundary(&mut cbar_raw, &cbar_b);
        // Ut: the closures' own U_t, boundary faces from their owners.
        let ut_c: Vec<Scalar> = gpu.download(&self.u_t)?.into_iter().take(n).collect();
        let ut_b: Vec<Scalar> = (0..self.nb).map(|f| owner(&ut_c, f)).collect();
        let mut ut_raw = self.calculated_seed("Ut", "[0 1 -1 0 0 0 0]");
        ut_raw.internal = ut_c.clone();
        fill_boundary(&mut ut_raw, &ut_b);
        // Cof: the measured number, if it was taken; zeros otherwise.
        let co_c: Vec<Scalar> = gpu.download(&self.co)?.into_iter().take(n).collect();
        let co_b: Vec<Scalar> = (0..self.nb).map(|f| owner(&co_c, f)).collect();
        let mut co_raw = self.calculated_seed("Cof", "[0 0 0 0 0 0 0]");
        co_raw.internal = co_c.clone();
        fill_boundary(&mut co_raw, &co_b);
        // HRR: (152.12)'s qdot, boundary faces from their owners.
        let q_c: Vec<Scalar> = gpu.download(&self.q)?.into_iter().take(n).collect();
        let q_b: Vec<Scalar> = (0..self.nb).map(|f| owner(&q_c, f)).collect();
        let mut hrr_raw = self.calculated_seed("HRR", "[1 -1 -3 0 0 0 0]");
        hrr_raw.internal = q_c;
        fill_boundary(&mut hrr_raw, &q_b);
        // p0: the uniform thermodynamic pressure.
        let p0 = gas.p0();
        let mut p0_raw = self.calculated_seed("p0", "[1 -1 -2 0 0 0 0]");
        p0_raw.internal = vec![p0; n];
        let p0_b: Vec<Scalar> = vec![p0; self.nb];
        fill_boundary(&mut p0_raw, &p0_b);
        Ok(vec![
            ("c".to_string(), c_raw),
            ("cbar".to_string(), cbar_raw),
            ("Ut".to_string(), ut_raw),
            ("Cof".to_string(), co_raw),
            ("HRR".to_string(), hrr_raw),
            ("p0".to_string(), p0_raw),
        ])
    }

    /// `c`, `T`, `rho`, `HRR` and `Ut`, the CELL values at full precision,
    /// for a restart or a probe.
    pub fn snapshot(
        &self,
        gpu: &Gpu,
        t: &GpuScalarField,
        gas: &crate::energy::GasState<'_>,
    ) -> Result<Vec<(String, Vec<Scalar>)>> {
        Ok(vec![
            ("c".to_string(), gpu.download(&self.c().f)?),
            ("T".to_string(), gpu.download(&t.f)?),
            ("rho".to_string(), gpu.download(&gas.rho().f)?),
            ("HRR".to_string(), gpu.download(&self.q)?),
            ("Ut".to_string(), gpu.download(&self.u_t)?),
        ])
    }

    /// (152.12)'s heat release as `before_flow` last formed it, W/m3.
    pub fn heat_release(&self) -> &DevBuf<Scalar> {
        &self.q
    }
}

// ==========================================================================
//  The compressible coupling (SPEC-LIT §158.4-§158.6, DST-10 part 1)
// ==========================================================================

/// The Phase A dust model inside the COMPRESSIBLE solver (§158.6): the c~
/// equation (155.3) on the solver's own mass flux `mdot`, the mixture gas
/// law (152.8)-(152.9) about `p_op`, the local `rho_u(p)` of (155.10) as a
/// scaled `U_t'`, and the energy equation's mixture mode - `w = rho
/// cp(c~)`, the face flux `mdot cp(c~_f)` and S_dust in W/m3, with lambda =
/// `kappa + rho cp(c~) D_t` the diffusion the energy equation runs on
/// ((157.6) with the LOCAL `kappa` field). [`DustSource`] is untouched: the
/// low-Mach coupling stays as it was.
pub struct CompressibleDust<'m> {
    setup: DustSetup,
    hc: DustHConst,
    progress: ProgressEquation<'m>,
    turb: Option<Box<dyn CoupledTurbulence + 'm>>,
    nu_ref: f64,

    fk: crate::dust::device::FscKernels,
    pk: ProgressKernels,
    fvk: FvKernels,
    fldk: field_ops::FieldKernels,

    d_t: DevBuf<Scalar>,
    u_t_raw: DevBuf<Scalar>,
    u_t: DevBuf<Scalar>,
    u_t_isp: DevBuf<Scalar>,
    tau_l: DevBuf<Scalar>,
    k_rate: DevBuf<Scalar>,
    w_ign: DevBuf<Scalar>,
    zero: DevBuf<Scalar>,
    t_pred: DevBuf<Scalar>,
    n_zero: DevBuf<u64>,
    one_c: DevBuf<Scalar>,
    one_f: DevBuf<Scalar>,
    one_b: DevBuf<Scalar>,
    tmp: DevBuf<Scalar>,
    tmp_b: DevBuf<Scalar>,
    cp_cells: DevBuf<Scalar>,
    cp_faces: DevBuf<Scalar>,
    cp_bfaces: DevBuf<Scalar>,
    div_c: DevBuf<Scalar>,
    wdot: DevBuf<Scalar>,
    q: DevBuf<Scalar>,
    co: DevBuf<Scalar>,

    d_t_fld: GpuScalarField,
    gamma: GpuScalarField,
    lambda: GpuScalarField,
    mu_eff: Option<GpuScalarField>,
    c_face: GpuSurfaceScalarField,
    t_face: GpuSurfaceScalarField,
    rho_face: GpuSurfaceScalarField,
    gamma_face: GpuSurfaceScalarField,
    face_coeff: GpuSurfaceScalarField,
    sn_grad: GpuSurfaceScalarField,
    rho_face_t: GpuSurfaceScalarField,
    phi_volume: GpuSurfaceScalarField,

    p_local: Option<Vec3>,
    last: Option<ProgressSolve>,
    n: usize,
    nf: usize,
    nb: usize,
}

impl<'m> CompressibleDust<'m> {
    /// The coupling on one mesh: the hConst pair of §158.4 from the setup's
    /// gas at `T_u`, the c~ equation seeded with the case's `raw_c` then
    /// initialised (as `DustSource::new` does), a Gamma field and a D_t
    /// field with zeroGradient boundary faces, and every buffer
    /// `before_energy` runs through. Refused by name (§158.6): a prescribed
    /// `U_t` law with `alpha != 0` or `beta != 0` - the constant law only.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: &Gpu,
        hm: &HostMesh,
        m: &'m GpuMesh,
        setup: DustSetup,
        raw_c: &RawScalarField,
        ctrl: TurbulenceControls,
        turb: Option<Box<dyn CoupledTurbulence + 'm>>,
        nu_ref: f64,
    ) -> Result<Self> {
        if let Some(law) = setup.prescribed_ut {
            if law.alpha != 0.0 || law.beta != 0.0 {
                return Err(Error::Refused(format!(
                    "dust::source::CompressibleDust::new: the prescribed U_t law has \
                     alpha = {}, beta = {}; the compressible step (SPEC-LIT §158.6) \
                     runs the constant law only. Alternative: alpha = beta = 0, or \
                     the §154 closures",
                    law.alpha, law.beta
                )));
            }
        }
        let hc = DustHConst::from_gas(&setup.gas, setup.t_u)?;
        let (n, nf, nb) = (m.n_cells, m.n_internal_faces, m.n_boundary_faces);
        let cells = n.max(1);
        let mut progress = ProgressEquation::new(gpu, hm, m, ctrl)?;
        setup_scalar_field(gpu, progress.field_mut(), raw_c, hm)?;
        progress.initialise(gpu)?;
        // The local flame-age clock (SPEC-LIT §160.5): the case's call.
        if setup.flame_age == FlameAge::Local {
            progress.enable_age(gpu, setup.t_fd_origin, C_A)?;
        }
        let zg = |name: &str| -> Result<GpuScalarField> {
            let mut f = GpuScalarField::zeros(gpu, m, name)?;
            gpu.write(&mut f.bc_kind, &vec![BcKind::ZeroGradient as crate::Label; nb])?;
            Ok(f)
        };
        let p_local = setup.ignition.map(|src| src.local_point(hm));

        Ok(Self {
            fk: crate::dust::device::FscKernels::new(gpu)?,
            pk: ProgressKernels::new(gpu)?,
            fvk: FvKernels::new(gpu)?,
            fldk: field_ops::FieldKernels::new(gpu)?,
            d_t: gpu.zeros(cells)?,
            u_t_raw: gpu.zeros(cells)?,
            u_t: gpu.zeros(cells)?,
            u_t_isp: gpu.zeros(cells)?,
            tau_l: gpu.zeros(cells)?,
            k_rate: gpu.zeros(cells)?,
            w_ign: gpu.zeros(cells)?,
            zero: gpu.zeros(cells)?,
            t_pred: gpu.zeros(cells)?,
            n_zero: gpu.zeros(1)?,
            one_c: gpu.upload(&vec![1.0 as Scalar; cells])?,
            one_f: gpu.upload(&vec![1.0 as Scalar; nf.max(1)])?,
            one_b: gpu.upload(&vec![1.0 as Scalar; nb.max(1)])?,
            tmp: gpu.zeros(cells)?,
            tmp_b: gpu.zeros(nb.max(1))?,
            cp_cells: gpu.zeros(cells)?,
            cp_faces: gpu.zeros(nf.max(1))?,
            cp_bfaces: gpu.zeros(nb.max(1))?,
            div_c: gpu.zeros(cells)?,
            wdot: gpu.zeros(cells)?,
            q: gpu.zeros(cells)?,
            co: gpu.zeros(cells)?,
            d_t_fld: zg("dustDt")?,
            gamma: zg("dustMixGamma")?,
            lambda: zg("dustLambda")?,
            mu_eff: turb.is_some().then(|| GpuScalarField::zeros(gpu, m, "dustMuEff")).transpose()?,
            c_face: GpuSurfaceScalarField::zeros(gpu, m, "dustMixCf")?,
            t_face: GpuSurfaceScalarField::zeros(gpu, m, "dustMixTf")?,
            rho_face: GpuSurfaceScalarField::zeros(gpu, m, "dustMixRhoF")?,
            gamma_face: GpuSurfaceScalarField::zeros(gpu, m, "dustMixGamF")?,
            face_coeff: GpuSurfaceScalarField::zeros(gpu, m, "dustMixCoeff")?,
            sn_grad: GpuSurfaceScalarField::zeros(gpu, m, "dustMixSnGrad")?,
            rho_face_t: GpuSurfaceScalarField::zeros(gpu, m, "dustSstRhoF")?,
            phi_volume: GpuSurfaceScalarField::zeros(gpu, m, "dustSstPhiV")?,
            progress,
            setup,
            hc,
            turb,
            nu_ref,
            p_local,
            last: None,
            n,
            nf,
            nb,
        })
    }

    /// The constants the step runs with.
    pub fn setup(&self) -> &DustSetup {
        &self.setup
    }

    /// §158.4's hConst pair the mixture gas law and the energy equation run
    /// on.
    pub fn hconst(&self) -> &DustHConst {
        &self.hc
    }

    /// The progress variable c~ (Y_B = c~).
    pub fn c(&self) -> &GpuScalarField {
        self.progress.field()
    }

    /// `U_t'` - the propagation speed carried at the LOCAL pressure, as
    /// `before_energy` last scaled it (155.10).
    pub fn u_t_local(&self) -> &DevBuf<Scalar> {
        &self.u_t
    }

    /// The per-cell flame-Courant number (155.9) as
    /// [`Self::measured_courant`] last formed it; zeros before the first
    /// measurement.
    pub fn courant(&self) -> &DevBuf<Scalar> {
        &self.co
    }

    /// The net rate `wdot_c = S_U - S_P' c~` (155.2) as `before_energy` last
    /// formed it, kg/(m3 s).
    pub fn wdot(&self) -> &DevBuf<Scalar> {
        &self.wdot
    }

    /// §158.4's `q~ = Δh_c wdot_c` as `before_energy` last formed it, W/m3.
    pub fn heat_release(&self) -> &DevBuf<Scalar> {
        &self.q
    }

    /// The diffusion the energy equation's laplacian runs on,
    /// `lambda = kappa + rho cp(c~) D_t` ((157.6) with the local kappa), as
    /// `before_energy` last formed it - zeros before the first call.
    pub fn lambda(&self) -> &GpuScalarField {
        &self.lambda
    }

    /// The last [`ProgressEquation::solve`]'s report.
    pub fn last_solve(&self) -> Option<ProgressSolve> {
        self.last
    }

    /// The attached turbulence model, if any.
    pub fn turbulence(&self) -> Option<&(dyn CoupledTurbulence + 'm)> {
        self.turb.as_deref()
    }

    /// The flame-age field alpha of (160.4), when the local flame age is on
    /// (SPEC-LIT §160.5); `None` with the global clock.
    pub fn progress_age(&self) -> Option<&GpuScalarField> {
        self.progress.age_field()
    }

    /// The vent's opening (SPEC-LIT §158.10): the coupling of a closed run,
    /// carried onto the merged mesh. `hm`, `m` and `raw_c` are the MERGED
    /// (case) mesh and the case's own c~ field, as the no-vent path passes
    /// them, and `turb` is a model freshly built on that mesh (or `None` for
    /// a laminar case, on both sides alike). The new coupling is built with
    /// [`Self::new`] on this one's setup, c~ controls and `nu_ref`; c~
    /// (`f`, `f0`, `f00` and the shared boundary prefix), the ddt clock and
    /// the flame age's clock are carried by
    /// [`ProgressEquation::carry_from`]; each field of this coupling's
    /// turbulence model (k, omega and nut for the SST) is copied into the
    /// same-named field of the new model with
    /// [`crate::dust::vent::hand_over_scalar`], and `last` is copied. The new
    /// model's wall distance is its own: it was built on the merged mesh,
    /// where the membrane is no longer a wall. The SST's ddt clock is not
    /// carried: a dust run is Euler only (SPEC-LIT §158.6), and Euler's
    /// coefficients read only `dt` (`DdtScheme::coeffs`).
    ///
    /// Every other buffer is per-step scratch, written before it is read in
    /// a step (`open_step`, `update_mu_eff`, `before_energy` and
    /// `correct_turbulence` of the loop), and is not carried - except the
    /// five that a driver reads BETWEEN steps for its written fields and
    /// reports: `U_t'`, the heat release `q`, the net rate `wdot`, the
    /// flame-Courant buffer and `lambda`, which are carried so that a state
    /// written at the opening step is the closed run's last one. Refused by
    /// name (SPEC-LIT §158.10): different cell counts, a merged boundary
    /// longer than this coupling's (the merged arrays must be a prefix),
    /// `turb` present on one side only, and a field of this coupling's model
    /// that the new one lacks.
    pub fn hand_over<'n>(
        &self,
        gpu: &Gpu,
        hm: &HostMesh,
        m: &'n GpuMesh,
        raw_c: &RawScalarField,
        turb: Option<Box<dyn CoupledTurbulence + 'n>>,
    ) -> Result<CompressibleDust<'n>> {
        const W: &str = "dust::source::CompressibleDust::hand_over";
        if m.n_cells != self.n {
            return Err(Error::Refused(format!(
                "{W}: the merged mesh holds {} cells, this coupling {} (SPEC-LIT §158.10)",
                m.n_cells, self.n
            )));
        }
        if m.n_boundary_faces > self.nb {
            return Err(Error::Refused(format!(
                "{W}: the merged boundary holds {} faces, longer than this coupling's {}; \
                 the merged arrays must be a prefix (SPEC-LIT §158.10)",
                m.n_boundary_faces, self.nb
            )));
        }
        if self.turb.is_some() != turb.is_some() {
            return Err(Error::Refused(format!(
                "{W}: a turbulence model is attached on {} side only; the closed and the \
                 merged coupling carry one alike (SPEC-LIT §158.10)",
                if self.turb.is_some() { "the closed" } else { "the merged" }
            )));
        }
        if let (Some(old), Some(new)) = (&self.turb, &turb) {
            let have = new.output_fields();
            for (name, _) in old.output_fields() {
                if !have.iter().any(|(n, _)| *n == name) {
                    return Err(Error::Refused(format!(
                        "{W}: the new turbulence model carries no field {name}, which the \
                         closed one does (SPEC-LIT §158.10)"
                    )));
                }
            }
        }
        let mut new = CompressibleDust::new(
            gpu,
            hm,
            m,
            self.setup.clone(),
            raw_c,
            *self.progress.controls(),
            turb,
            self.nu_ref,
        )?;
        new.progress.carry_from(gpu, &self.progress, m.n_boundary_faces)?;
        if let (Some(old), Some(model)) = (&self.turb, &mut new.turb) {
            let mut to = model.output_fields_mut();
            for (name, from) in old.output_fields() {
                if let Some((_, field)) = to.iter_mut().find(|(n, _)| *n == name) {
                    crate::dust::vent::hand_over_scalar(gpu, from, field, m.n_boundary_faces)?;
                }
            }
        }
        // The between-step reads: U_t', q, wdot, the flame-Courant buffer.
        for (from, to) in [
            (&self.u_t, &mut new.u_t),
            (&self.q, &mut new.q),
            (&self.wdot, &mut new.wdot),
            (&self.co, &mut new.co),
        ] {
            gpu.write(to, &gpu.download(from)?)?;
        }
        crate::dust::vent::hand_over_scalar(gpu, &self.lambda, &mut new.lambda, m.n_boundary_faces)?;
        new.last = self.last;
        Ok(new)
    }

    /// Starts a time step: the c~ equation's old level and its step-long
    /// `|grad c~|` (§155.2).
    pub fn open_step(&mut self, gpu: &Gpu) -> Result<()> {
        self.progress.begin_step(gpu)
    }

    /// The MIXTURE gas law (152.8)-(152.9) on the cells and on the boundary
    /// faces, about the gauge datum `p_op`: `psi = 1/(r(c~) T)` and
    /// `rho = psi (p + p_op)`, with `r(c~) = r_u + c~ (r_b - r_u)` from
    /// (152.8) per kg. `p`'s and `T`'s boundary values must be evaluated.
    pub fn gas_state(
        &self,
        gpu: &Gpu,
        k: &MixKernels,
        s: &mut CompressibleState,
        p_op: f64,
    ) -> Result<()> {
        crate::compressible::thermo::gas_state_mix(
            gpu, k, &self.hc, &mut s.psi.f, &mut s.rho.f, &s.p.f, &s.t.f, &self.c().f, self.n,
            p_op,
        )?;
        crate::compressible::thermo::gas_state_mix(
            gpu, k, &self.hc, &mut s.psi.bf, &mut s.rho.bf, &s.p.bf, &s.t.bf, &self.c().bf, self.nb,
            p_op,
        )
    }

    /// `mu_eff = mu + rho nu_t` on the cells and the boundary faces (§158.6
    /// lifts CMP-15's non-SA refusal for the SST), into the owned field
    /// [`Self::mu_eff`] hands back. Nothing without a model: the loop then
    /// passes `s.mu`.
    pub fn update_mu_eff(&mut self, gpu: &Gpu, s: &CompressibleState) -> Result<()> {
        let Some(t) = &self.turb else { return Ok(()) };
        let Some(me) = &mut self.mu_eff else { return Ok(()) };
        let nut = t.nut();
        let (n, nb) = (self.n, self.nb);
        field_ops::copy_field(gpu, &self.fldk, &mut self.tmp, &s.rho.f, n)?;
        field_ops::multiply_field(gpu, &self.fldk, &mut self.tmp, &nut.f, n)?;
        field_ops::copy_field(gpu, &self.fldk, &mut me.f, &s.mu.f, n)?;
        field_ops::add_field(gpu, &self.fldk, &mut me.f, &self.tmp, n)?;
        field_ops::copy_field(gpu, &self.fldk, &mut self.tmp_b, &s.rho.bf, nb)?;
        field_ops::multiply_field(gpu, &self.fldk, &mut self.tmp_b, &nut.bf, nb)?;
        field_ops::copy_field(gpu, &self.fldk, &mut me.bf, &s.mu.bf, nb)?;
        field_ops::add_field(gpu, &self.fldk, &mut me.bf, &self.tmp_b, nb)
    }

    /// The `mu_eff` the momentum predictor runs on; `None` without a model.
    pub fn mu_eff(&self) -> Option<&GpuScalarField> {
        self.mu_eff.as_ref()
    }

    /// One outer iteration of the dust model in the compressible loop, in
    /// §158.6's order: the closures, `U_t'` at the local pressure, the
    /// ignition bracket, `cp(c~)` and Gamma, `rho_f`, one solve of (155.3)
    /// on the solver's own mass flux, `wdot_c` and `q~ = Δh_c wdot_c`, the
    /// energy equation's mixture state (`w`, the flux, S_dust), and
    /// `lambda`. `t_new` is the time at the END of the step.
    #[allow(clippy::too_many_arguments)]
    pub fn before_energy(
        &mut self,
        gpu: &Gpu,
        m: &GpuMesh,
        k: &MixKernels,
        s: &CompressibleState,
        energy: &mut RhoEnergy<'_>,
        p_op: f64,
        t_new: f64,
    ) -> Result<()> {
        let n = self.n;
        if n == 0 {
            return Ok(());
        }
        // The local clock's levels move once per step, before the first
        // outer solve (SPEC-LIT §160.5); nothing with the global clock.
        self.progress.begin_age_step(gpu, t_new)?;
        // (a) The closures of §154 at this step's clock, on the solver's T
        //     and the model's frozen k, omega, nu_t (zeros when laminar).
        let fields;
        let (k_b, o_b, n_b): (&DevBuf<Scalar>, &DevBuf<Scalar>, &DevBuf<Scalar>) = match &self.turb
        {
            Some(t) => {
                fields = t.output_fields();
                let pick = |name: &str| -> Result<&DevBuf<Scalar>> {
                    fields.iter().find(|(nm, _)| *nm == name).map(|(_, f)| &f.f).ok_or_else(|| {
                        Error::Refused(format!(
                            "dust::source::CompressibleDust::before_energy: the turbulence \
                             model carries no field {name}; the closures read k and omega of \
                             the rho-weighted kOmegaSST (SPEC-LIT §158.6)"
                        ))
                    })
                };
                (pick("k")?, pick("omega")?, &t.nut().f)
            }
            None => (&self.zero, &self.zero, &self.zero),
        };
        //     a* of (160.4) with the local flame age on, the global clock
        //     without (SPEC-LIT §160.5).
        if let Some(age) = self.progress.closure_age() {
            dev::fsc_aged(
                gpu,
                &self.fk,
                &self.setup.fsc,
                FscFields {
                    d_t: &mut self.d_t,
                    u_t: &mut self.u_t_raw,
                    u_t_isp: &mut self.u_t_isp,
                    tau_l: &mut self.tau_l,
                    k_rate: &mut self.k_rate,
                },
                &mut self.n_zero,
                FscInputsDev { t: &s.t.f, k: k_b, omega: o_b, nu_t: n_b },
                age,
                n,
            )?;
        } else {
            dev::fsc(
                gpu,
                &self.fk,
                &self.setup.fsc,
                FscFields {
                    d_t: &mut self.d_t,
                    u_t: &mut self.u_t_raw,
                    u_t_isp: &mut self.u_t_isp,
                    tau_l: &mut self.tau_l,
                    k_rate: &mut self.k_rate,
                },
                &mut self.n_zero,
                FscInputsDev { t: &s.t.f, k: k_b, omega: o_b, nu_t: n_b },
                development_time(t_new, self.setup.t_fd_origin),
                t_new,
                n,
            )?;
        }
        //     Gate DST-VES's prescribed constant U_t, as `DustSource` does.
        if let Some(law) = self.setup.prescribed_ut {
            field_ops::set_field(
                gpu,
                &self.fldk,
                &mut self.u_t_raw,
                law.s_u(self.setup.p_init, self.setup.p_init, self.setup.gamma_u) as Scalar,
                n,
            )?;
        }
        // (b) U_t' = U_t ((p + p_op)/p_init)^(1/gamma_u), the LOCAL rho_u(p)
        //     of (155.10) folded into the speed.
        crate::compressible::thermo::rho_u_ratio(
            gpu,
            k,
            &mut self.u_t,
            &self.u_t_raw,
            &s.p.f,
            n,
            p_op,
            self.setup.p_init,
            self.setup.gamma_u,
        )?;
        // (c) The ignition bracket (155.4), or nothing without a source.
        if let (Some(src), Some(p)) = (&self.setup.ignition, self.p_local) {
            dev::ignition(gpu, &self.pk, &mut self.w_ign, &m.c, src, p, t_new, n)?;
        }
        // (d) cp(c~) on the cells, then Gamma = kappa/(rho cp) + D_t - Le = 1
        //     with the LOCAL kappa field (157.6) - zeroGradient boundary.
        let (cp_u, dcp) = (self.hc.cp_u, self.hc.cp_b - self.hc.cp_u);
        let c_all = self.progress.field();
        crate::compressible::thermo::scale_cp(
            gpu, k, &mut self.cp_cells, &self.one_c, &c_all.f, n, cp_u, dcp,
        )?;
        field_ops::copy_field(gpu, &self.fldk, &mut self.tmp, &s.rho.f, n)?;
        field_ops::multiply_field(gpu, &self.fldk, &mut self.tmp, &self.cp_cells, n)?;
        field_ops::set_field(gpu, &self.fldk, &mut self.gamma.f, 1.0 as Scalar, n)?;
        field_ops::divide_field(gpu, &self.fldk, &mut self.gamma.f, &self.tmp, n)?;
        field_ops::multiply_field(gpu, &self.fldk, &mut self.gamma.f, &s.kappa.f, n)?;
        field_ops::add_field(gpu, &self.fldk, &mut self.gamma.f, &self.d_t, n)?;
        field_ops::correct_boundary_conditions(gpu, &self.fldk, &mut self.gamma, m)?;
        // (e) rho_f, linear.
        fv::interpolate_linear(gpu, &self.fvk, &mut self.rho_face, &s.rho, m)?;
        // (f) One solve of (155.3) on the solver's own mass flux, rho_u held
        //     at rho_u0: the pressure dependence rides U_t' of (b).
        let has_ign = self.setup.ignition.is_some();
        let solve = self.progress.solve(
            gpu,
            &ProgressCoupling {
                rho: &s.rho,
                rho_face: &self.rho_face,
                phi_mass: &s.mdot,
                u: &s.u,
                diffusivity: &self.gamma,
                u_t: &self.u_t,
                k_rate: &self.k_rate,
                w_ign: has_ign.then_some(&self.w_ign),
                rho_u: self.setup.rho_u0,
            },
        )?;
        self.last = Some(solve);
        // (g) wdot_c = S_U - S_P' c~ (155.2), and q~ = Δh_c wdot_c of
        //     (158.4), W/m3.
        let (su, sp) = self.progress.source_terms();
        field_ops::copy_field(gpu, &self.fldk, &mut self.wdot, sp, n)?;
        field_ops::multiply_field(gpu, &self.fldk, &mut self.wdot, &self.progress.field().f, n)?;
        field_ops::scale_field(gpu, &self.fldk, &mut self.wdot, -1.0, n)?;
        field_ops::add_field(gpu, &self.fldk, &mut self.wdot, su, n)?;
        field_ops::set_field(gpu, &self.fldk, &mut self.q, self.hc.delta_h_c() as Scalar, n)?;
        field_ops::multiply_field(gpu, &self.fldk, &mut self.q, &self.wdot, n)?;
        // (h) The energy equation's mixture state: rho cp(c~) at the three
        //     levels (c~'s OLD level on both old weights), the face flux
        //     mdot cp(c~_f), and S_dust = q~ + div(rho_f Gamma_f T_f
        //     (cp_b - cp_u) |Sf| snGrad c~) - the enthalpy-of-mixture
        //     transport the h_s = cp(c~) T datum moves to the source side.
        let Some(mx) = energy.mixture_mut() else {
            return Err(Error::Refused(
                "dust::source::CompressibleDust::before_energy: the energy equation \
                 runs no mixture state; RhoEnergy::enable_mixture must come first \
                 (SPEC-LIT §158.4)"
                    .into(),
            ));
        };
        crate::compressible::thermo::scale_cp(gpu, k, &mut mx.w, &s.rho.f, &self.c().f, n, cp_u, dcp)?;
        crate::compressible::thermo::scale_cp(gpu, k, &mut mx.w0, &s.rho.f0, &self.c().f0, n, cp_u, dcp)?;
        crate::compressible::thermo::scale_cp(gpu, k, &mut mx.w00, &s.rho.f00, &self.c().f0, n, cp_u, dcp)?;
        fv::interpolate_linear(gpu, &self.fvk, &mut self.c_face, self.progress.field(), m)?;
        fv::interpolate_linear(gpu, &self.fvk, &mut self.t_face, &s.t, m)?;
        crate::compressible::thermo::scale_cp(
            gpu, k, &mut self.cp_faces, &self.one_f, &self.c_face.f, self.nf, cp_u, dcp,
        )?;
        crate::compressible::thermo::scale_cp(
            gpu, k, &mut self.cp_bfaces, &self.one_b, &self.c_face.bf, self.nb, cp_u, dcp,
        )?;
        field_ops::copy_field(gpu, &self.fldk, &mut mx.flux.f, &s.mdot.f, self.nf)?;
        field_ops::multiply_field(gpu, &self.fldk, &mut mx.flux.f, &self.cp_faces, self.nf)?;
        field_ops::copy_field(gpu, &self.fldk, &mut mx.flux.bf, &s.mdot.bf, self.nb)?;
        field_ops::multiply_field(gpu, &self.fldk, &mut mx.flux.bf, &self.cp_bfaces, self.nb)?;
        fv::interpolate_linear(gpu, &self.fvk, &mut self.gamma_face, &self.gamma, m)?;
        field_ops::copy_field(gpu, &self.fldk, &mut self.face_coeff.f, &self.rho_face.f, self.nf)?;
        field_ops::multiply_field(gpu, &self.fldk, &mut self.face_coeff.f, &self.gamma_face.f, self.nf)?;
        field_ops::multiply_field(gpu, &self.fldk, &mut self.face_coeff.f, &self.t_face.f, self.nf)?;
        field_ops::multiply_field(gpu, &self.fldk, &mut self.face_coeff.f, &m.mag_sf, self.nf)?;
        field_ops::scale_field(gpu, &self.fldk, &mut self.face_coeff.f, dcp as Scalar, self.nf)?;
        field_ops::copy_field(gpu, &self.fldk, &mut self.face_coeff.bf, &self.rho_face.bf, self.nb)?;
        field_ops::multiply_field(gpu, &self.fldk, &mut self.face_coeff.bf, &self.gamma_face.bf, self.nb)?;
        field_ops::multiply_field(gpu, &self.fldk, &mut self.face_coeff.bf, &self.t_face.bf, self.nb)?;
        field_ops::multiply_field(gpu, &self.fldk, &mut self.face_coeff.bf, &m.b_mag_sf, self.nb)?;
        field_ops::scale_field(gpu, &self.fldk, &mut self.face_coeff.bf, dcp as Scalar, self.nb)?;
        fv::sn_grad_flux(gpu, &self.fvk, &mut self.sn_grad, self.progress.field(), &self.face_coeff.f, &self.face_coeff.bf, m)?;
        fv::fvc_div_surface(gpu, &self.fvk, &mut self.div_c, &self.sn_grad, m)?;
        field_ops::copy_field(gpu, &self.fldk, &mut mx.source, &self.q, n)?;
        field_ops::add_field(gpu, &self.fldk, &mut mx.source, &self.div_c, n)?;
        // (i) lambda = kappa + rho cp(c~) D_t (152.7 with §158.4's kappa),
        //     cells and boundary faces; D_t's boundary value from its owner,
        //     rho and c~ their boundary values.
        field_ops::multiply_field(gpu, &self.fldk, &mut self.tmp, &self.d_t, n)?;
        field_ops::copy_field(gpu, &self.fldk, &mut self.lambda.f, &s.kappa.f, n)?;
        field_ops::add_field(gpu, &self.fldk, &mut self.lambda.f, &self.tmp, n)?;
        field_ops::copy_field(gpu, &self.fldk, &mut self.d_t_fld.f, &self.d_t, n)?;
        field_ops::correct_boundary_conditions(gpu, &self.fldk, &mut self.d_t_fld, m)?;
        let c_all = self.progress.field();
        crate::compressible::thermo::scale_cp(
            gpu, k, &mut self.cp_bfaces, &s.rho.bf, &c_all.bf, self.nb, cp_u, dcp,
        )?;
        field_ops::multiply_field(gpu, &self.fldk, &mut self.cp_bfaces, &self.d_t_fld.bf, self.nb)?;
        field_ops::copy_field(gpu, &self.fldk, &mut self.lambda.bf, &s.kappa.bf, self.nb)?;
        field_ops::add_field(gpu, &self.fldk, &mut self.lambda.bf, &self.cp_bfaces, self.nb)
    }

    /// Step 7: the rho-weighted SST correction (§156) on the solver's own
    /// density and mass flux. Nothing without a model, and the model stays
    /// FROZEN while `t_new < 2 t_fd_origin` (§157.7 decision D6).
    pub fn correct_turbulence(
        &mut self,
        gpu: &Gpu,
        m: &GpuMesh,
        s: &CompressibleState,
        t_new: f64,
    ) -> Result<()> {
        if self.turb.is_none() || t_new < 2.0 * self.setup.t_fd_origin {
            return Ok(());
        }
        {
            let (fvk, fldk) = (&self.fvk, &self.fldk);
            compressible_sst_inputs(
                gpu,
                fvk,
                fldk,
                m,
                &s.rho,
                &s.mdot,
                &mut self.rho_face_t,
                &mut self.phi_volume,
            )?;
        }
        let nu_ref = self.nu_ref;
        let flow = FlowState::new(&s.u, &s.mdot, nu_ref as Scalar);
        let dens = SstDensity {
            rho: &s.rho.f,
            rho0: &s.rho.f0,
            rho00: &s.rho.f00,
            rho_face: &self.rho_face_t,
            phi_volume: &self.phi_volume,
        };
        self.turb.as_mut().expect("turbulence").correct_rho(gpu, &flow, &dens)?;
        Ok(())
    }
}

/// The `SstDensity` inputs the rho-weighted SST correction reads from the
/// compressible state (§158.6): `rho_face` the linear interpolation of rho,
/// `phi_volume = mdot / rho_face` on the internal and the boundary faces -
/// the MASS flux `mdot` is (155.3)'s and §86.3's contract, the correction
/// runs on the volume flux.
pub fn compressible_sst_inputs(
    gpu: &Gpu,
    fvk: &FvKernels,
    fldk: &field_ops::FieldKernels,
    m: &GpuMesh,
    rho: &GpuScalarField,
    mdot: &GpuSurfaceScalarField,
    rho_face: &mut GpuSurfaceScalarField,
    phi_volume: &mut GpuSurfaceScalarField,
) -> Result<()> {
    let (nif, nb) = (m.n_internal_faces, m.n_boundary_faces);
    fv::interpolate_linear(gpu, fvk, rho_face, rho, m)?;
    field_ops::copy_field(gpu, fldk, &mut phi_volume.f, &mdot.f, nif)?;
    field_ops::divide_field(gpu, fldk, &mut phi_volume.f, &rho_face.f, nif)?;
    field_ops::copy_field(gpu, fldk, &mut phi_volume.bf, &mdot.bf, nb)?;
    field_ops::divide_field(gpu, fldk, &mut phi_volume.bf, &rho_face.bf, nb)
}

impl<'m> CompressibleDust<'m> {
    /// The PREDICTED flame-Courant number of a fixed `dt` (155.9), as
    /// `DustSource`'s: the closures run ONCE on the fields as they stand at
    /// t_fd = 1e30, `u_max` the largest |u| and `dx_min` the smallest
    /// `V^(1/3)` on the host in f64 - at `p_init`, where the prediction is
    /// formed. Refuses, naming `maxFlameCourant`, a step above the bound.
    pub fn predicted_courant(
        &mut self,
        gpu: &Gpu,
        hm: &HostMesh,
        u: &GpuVectorField,
        dt: f64,
    ) -> Result<f64> {
        let n = self.n;
        dev::fsc(
            gpu,
            &self.fk,
            &self.setup.fsc,
            FscFields {
                d_t: &mut self.d_t,
                u_t: &mut self.u_t_raw,
                u_t_isp: &mut self.u_t_isp,
                tau_l: &mut self.tau_l,
                k_rate: &mut self.k_rate,
            },
            &mut self.n_zero,
            FscInputsDev { t: &self.t_pred, k: &self.zero, omega: &self.zero, nu_t: &self.zero },
            1.0e30,
            1.0e30,
            n,
        )?;
        let isp_max = gpu
            .download(&self.u_t_isp)?
            .iter()
            .take(n)
            .fold(0.0f64, |a, &v| a.max(f64::from(v)));
        // The prescribed law at BOTH ends of the pressure range, as
        // `DustSource` reads it; the prediction itself forms at p_init.
        let isp_max = match self.setup.prescribed_ut {
            Some(law) => {
                let p_e =
                    self.setup.p_init * self.setup.gas.aicc(self.setup.t_u)?.p_ratio;
                let s0 = law.s_u(self.setup.p_init, self.setup.p_init, self.setup.gamma_u);
                let s_e = law.s_u(p_e, self.setup.p_init, self.setup.gamma_u);
                f64::max(s0, s_e)
            }
            None => isp_max,
        };
        let u_max = gpu
            .download(&u.f)?
            .iter()
            .take(n)
            .fold(0.0f64, |a, v| {
                a.max((f64::from(v.x) * f64::from(v.x)
                    + f64::from(v.y) * f64::from(v.y)
                    + f64::from(v.z) * f64::from(v.z))
                    .sqrt())
            });
        let dx_min =
            hm.v.iter().map(|v| f64::from(*v).cbrt()).fold(f64::INFINITY, f64::min);
        let predicted =
            crate::dust::courant::predicted(dt, dx_min, u_max, self.setup.sigma, isp_max);
        crate::dust::courant::check_fixed_step(dt, predicted, self.setup.max_flame_courant)
    }

    /// The MEASURED flame-Courant number (155.9) of the step just taken, on
    /// `U_t'` and `rho_u0` - the pressure dependence rides the speed - its
    /// max returned in f64.
    pub fn measured_courant(
        &mut self,
        gpu: &Gpu,
        m: &GpuMesh,
        s: &CompressibleState,
        dt: f64,
    ) -> Result<f64> {
        let n = self.n;
        dev::flame_courant(
            gpu,
            &self.pk,
            &mut self.co,
            &s.u.f,
            &s.rho.f,
            &self.u_t,
            &m.v,
            self.setup.rho_u0,
            dt,
            n,
        )?;
        Ok(gpu
            .download(&self.co)?
            .iter()
            .take(n)
            .fold(0.0f64, |a, &v| a.max(f64::from(v))))
    }
}

#[cfg(test)]
#[path = "tests_source.rs"]
mod tests;
