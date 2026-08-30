// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! DES97, DDES and IDDES - the shielded length scale, on either background
//! (SPEC-LIT §57).
//!
//! **A detached-eddy hybrid is a RANS model and an LES model with a switch
//! between them, and the switch is the model.** That sentence used to be the
//! comment beside `simulationType DES;`'s refusal in
//! [`crate::models::registry`]; it is correct, it is the whole content of
//! this module, and it survives here rather than being deleted with the
//! refusal it justified.
//!
//! Written from:
//!   Spalart, Jou, Strelets & Allmaras, "Comments on the feasibility of LES
//!     for wings, and on a hybrid RANS/LES approach", in *Advances in
//!     DNS/LES*, Greyden Press (1997) 137-147 - DES97
//!   Shur, Spalart, Strelets & Travin, *Engineering Turbulence Modelling and
//!     Experiments 4* (1999) 669-678 - `C_DES = 0.65` on the SA background
//!   Strelets, *AIAA Paper* 2001-0879 (2001) - SST-DES
//!   Spalart, Deck, Shur, Squires, Strelets & Travin, *Theor. Comput. Fluid
//!     Dyn.* 20 (2006) 181-195 - DDES: `r_d`, `f_d`, and the grid-induced
//!     separation they fix
//!   **Herr, Radespiel & Probst, arXiv:2301.07223v2 (2023)**, *Computers &
//!     Fluids* 265 (2023) 106014 - open access, READ IN FULL. Appendix A is a
//!     complete restatement of IDDES and is where every equation here comes
//!     from.
//!   **Savino, Griffin, Lee, Vijayakumar, Wu & Sprague, arXiv:2603.08875
//!     (2026)** - open access, READ. §2 states SST-IDDES: `C_DES1 = 0.78`,
//!     `C_DES2 = 0.61`, `C_w = 0.15`, and the simplified filter width.
//!   Nikitin, Nicoud, Wasistho, Squires & Spalart, *Phys. Fluids* 12 (2000)
//!     1629-1632 - the log-layer mismatch `f_e` removes
//!   Spalart, *Annu. Rev. Fluid Mech.* 41 (2009) 181-202 - the review
//!   ofgpu `SPEC-LIT.md` §57
//! No GPL-licensed source was consulted. OpenFOAM's and SU2's DES
//! implementations were not opened, searched or quoted.
//!
//! **NOT read, and therefore not relied on:** Shur et al., *Int. J. Heat
//! Fluid Flow* 29 (2008) 1638-1649 (IDDES itself) and Gritskevich et al.,
//! *Flow Turbul. Combust.* 88 (2012) 431-449 (the SST recalibration) are both
//! paywalled. `C_dt1 = 20`, `c_t = 1.87` and `c_l = 5.0` on the SST
//! background come from the design note's reading of the latter and are NOT
//! independently verified here - SPEC-LIT §57.5 says so, and so does the run
//! banner.
//!
//! **NOT implemented, and named rather than silently absent:** the
//! low-Reynolds correction `Psi` that Shur et al. (2008) multiply into
//! `l_LES`. Neither open-access restatement read here carries it, and this
//! implementation follows what was read (SPEC-LIT §57.5).
//!
//! # Why the shielding is provable rather than measurable
//!
//! In an equilibrium log layer `r_d = 1 + 1/(kappa y+) >= 1`, and `tanh`
//! saturates to exactly `1.0` in IEEE-754 double once its argument passes
//! `19.0615`, which `(8 r_d)^3` does for every `r_d > 0.33391`. So `f_d` is
//! **exactly `0.0`** through an attached boundary layer and
//! `d - 0.0*max(0, d - C_DES Delta)` returns `d` **bitwise**. DDES therefore
//! reproduces its background model bit for bit where it is shielded, on any
//! mesh whatever - which is the grid-induced-separation fix, stated as an
//! identity instead of as a tolerance (SPEC-LIT §57.3).

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::les::{cell_extents, LesKernels};
use crate::mesh::GpuMesh;
use crate::{Label, Scalar, Vec3};

// ==========================================================================
//  The three enumerations the dictionary selects
// ==========================================================================

/// Which of the three hybrids - SPEC-LIT (57.5)/(57.6)/§57.4.
///
/// The discriminants are mirrored by `OFDES_*` in `cuda/des.cu`; they are a
/// LAUNCH parameter, constant across the grid, so the launch sequence is the
/// same whichever branch runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DesBranch {
    /// `dtil = min(d, C_DES Delta)` - a pure grid criterion, and the one that
    /// suffers grid-induced separation.
    Des97 = 0,
    /// `dtil = d - f_d max(0, d - C_DES Delta)` - shielded.
    Ddes = 1,
    /// `dtil = l_hyb`, the full blend of arXiv:2301.07223 (A.15).
    Iddes = 2,
}

impl DesBranch {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Des97 => "DES",
            Self::Ddes => "DDES",
            Self::Iddes => "IDDES",
        }
    }
}

/// Which background model carries the hybrid - SPEC-LIT §57.5.
///
/// It is not decoration: `C_DES`, `C_dt1`, `c_t`, `c_l` and the default
/// filter width are all per-background calibrations, and mixing them is a
/// §13.4 refusal rather than a preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridBackground {
    /// Spalart-Allmaras: `l_RANS = d_w`, one constant `C_DES = 0.65`.
    Sa,
    /// k-omega SST: `l_RANS = sqrt(k)/(beta* omega)`, `C_DES` blended by
    /// `F1`.
    Sst,
}

impl HybridBackground {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Sa => "SpalartAllmaras",
            Self::Sst => "kOmegaSST",
        }
    }
}

/// Which filter width - SPEC-LIT §57.4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridDelta {
    /// `Delta = h_max`. What DES97 and DDES take, and what Shur et al. (1999)
    /// calibrated `C_DES = 0.65` against.
    MaxEdge = 0,
    /// arXiv:2301.07223 (A.1):
    /// `min(max(C_w d_w, C_w h_max, h_wn), h_max)`. Needs `h_wn` (§57.6).
    IddesFull = 1,
    /// arXiv:2603.08875 (14): `min(C_w max(d_w, h_max), h_max)`. Drops
    /// `h_wn`; the SST background's own published width.
    IddesSimple = 2,
}

impl HybridDelta {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::MaxEdge => "maxDeltaxyz",
            Self::IddesFull => "IDDESDelta",
            Self::IddesSimple => "IDDESDeltaSimple",
        }
    }

    /// What the branch and background default to when the case names no
    /// width - each background's own published choice, never a substitution.
    #[must_use]
    pub fn default_for(branch: DesBranch, background: HybridBackground) -> Self {
        match (branch, background) {
            (DesBranch::Iddes, HybridBackground::Sa) => Self::IddesFull,
            (DesBranch::Iddes, HybridBackground::Sst) => Self::IddesSimple,
            _ => Self::MaxEdge,
        }
    }
}

// ==========================================================================
//  The calibration constants
// ==========================================================================

/// SPEC-LIT §57.5's table, and §58.1's dictionary.
///
/// **These are calibrations, not universals**, and which of them a given
/// background reads is why writing `CDES` under an SST hybrid or `CDES1`
/// under an SA one is refused by name rather than ignored.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DesCoeffs {
    /// SA background only: the single `C_DES`, `0.65` (Shur et al. 1999).
    pub cdes: Scalar,
    /// SST background only: `C_DES = C_DES1 F1 + C_DES2 (1 - F1)`.
    pub cdes1: Scalar,
    pub cdes2: Scalar,
    /// `f_d = 1 - tanh((C_dt1 r_d)^C_dt2)`. `8`/`3` on SA; `20`/`3` on SST
    /// (NOT independently verified - see the module doc).
    pub cdt1: Scalar,
    pub cdt2: Scalar,
    /// IDDES's `f_t = tanh((c_t^2 r_dt)^3)` and `f_l = tanh((c_l^2 r_dl)^10)`.
    pub ct: Scalar,
    pub cl: Scalar,
    /// `C_w = 0.15` in both published filter widths.
    pub cw: Scalar,
    pub kappa: Scalar,
}

impl DesCoeffs {
    /// The SA-background calibration - Shur et al. (1999) and
    /// arXiv:2301.07223.
    #[must_use]
    pub fn sa() -> Self {
        Self {
            cdes: 0.65,
            cdes1: 0.78,
            cdes2: 0.61,
            cdt1: 8.0,
            cdt2: 3.0,
            ct: 1.63,
            cl: 3.55,
            cw: 0.15,
            kappa: 0.41,
        }
    }

    /// The SST-background calibration - arXiv:2603.08875 §2 for `C_DES1`,
    /// `C_DES2` and `C_w`, and the design note's reading of Gritskevich et
    /// al. (2012) for `C_dt1`, `c_t` and `c_l`.
    #[must_use]
    pub fn sst() -> Self {
        Self {
            cdes: 0.65,
            cdes1: 0.78,
            cdes2: 0.61,
            cdt1: 20.0,
            cdt2: 3.0,
            ct: 1.87,
            cl: 5.0,
            cw: 0.15,
            kappa: 0.41,
        }
    }

    #[must_use]
    pub fn for_background(b: HybridBackground) -> Self {
        match b {
            HybridBackground::Sa => Self::sa(),
            HybridBackground::Sst => Self::sst(),
        }
    }

    /// The run banner's line. Printed because SPEC-LIT §57.5 requires the
    /// calibration in use to be visible: the same three names carry different
    /// numbers on the two backgrounds and a silent default is exactly how the
    /// wrong set gets used.
    #[must_use]
    pub fn describe(&self, b: HybridBackground) -> String {
        match b {
            HybridBackground::Sa => format!(
                "CDES {} Cdt1 {} Cdt2 {} ct {} cl {} Cw {} kappa {}",
                self.cdes, self.cdt1, self.cdt2, self.ct, self.cl, self.cw, self.kappa
            ),
            HybridBackground::Sst => format!(
                "CDES1 {} CDES2 {} Cdt1 {} Cdt2 {} ct {} cl {} Cw {} kappa {} \
                 (Cdt1/ct/cl are Gritskevich et al. 2012 via the design note \
                 and are NOT verified against a source read here)",
                self.cdes1,
                self.cdes2,
                self.cdt1,
                self.cdt2,
                self.ct,
                self.cl,
                self.cw,
                self.kappa
            ),
        }
    }

    pub fn check(&self) -> Result<()> {
        let bad = |what: &str, v: Scalar| {
            Err(Error::Config(format!(
                "DES: `{what}` = {v} is not usable; SPEC-LIT §57.5"
            )))
        };
        if !(self.cdes > 0.0) {
            return bad("CDES", self.cdes);
        }
        if !(self.cdes1 > 0.0) {
            return bad("CDES1", self.cdes1);
        }
        if !(self.cdes2 > 0.0) {
            return bad("CDES2", self.cdes2);
        }
        if !(self.cdt1 > 0.0) {
            return bad("Cdt1", self.cdt1);
        }
        if !(self.cw > 0.0) {
            return bad("Cw", self.cw);
        }
        if !(self.kappa > 0.0) {
            return bad("kappa", self.kappa);
        }
        Ok(())
    }
}

// ==========================================================================
//  The closed forms, on the host - SPEC-LIT §57.3, §57.4
//
//  Independently transcribed from `cuda/des.cu`'s device arithmetic, and
//  compared against it pointwise by the tests. Every threshold SPEC-LIT
//  §57.11 gates is expressed here as a function rather than as a literal, so
//  that a changed constant moves the gate with it.
// ==========================================================================

/// `r_d = (nu_t + nu)/(kappa^2 d^2 F)` - SPEC-LIT (57.7).
///
/// `F` is the Frobenius norm of the FULL velocity gradient: not `S`, not
/// `Omega`. In a pure shear all three coincide, which is why a log-layer
/// profile cannot tell them apart and the gate for this uses a strain state
/// where they differ.
#[must_use]
pub fn r_d(nut: Scalar, nu: Scalar, kappa: Scalar, d: Scalar, grad_frob: Scalar) -> Scalar {
    (nut + nu) / (kappa * kappa * d * d * grad_frob)
}

/// `f_d = 1 - tanh((C_dt1 r_d)^C_dt2)` - SPEC-LIT (57.8).
#[must_use]
pub fn f_d(rd: Scalar, cdt1: Scalar, cdt2: Scalar) -> Scalar {
    1.0 - ((cdt1 * rd).powf(cdt2)).tanh()
}

/// The `r_d` above which `f_d` is **exactly `0.0`** in IEEE-754 double -
/// SPEC-LIT (57.10).
///
/// `tanh(x)` rounds to `1.0` once `2 exp(-2x) <= 2^-54`, i.e. `x >= 19.0615`.
/// At `C_dt1 = 8`, `C_dt2 = 3` that is `r_d > 0.33391`, and the log layer's
/// own `r_d` is `1 + 1/(kappa y+) >= 1`.
#[must_use]
pub fn f_d_zero_threshold(cdt1: Scalar, cdt2: Scalar) -> Scalar {
    let x = tanh_saturation_argument();
    x.powf(1.0 / cdt2) / cdt1
}

/// The argument past which `tanh` returns exactly `1.0` in this crate's
/// `Scalar` - `-0.5 ln(eps/8)`, `19.0615475` in double precision.
///
/// Derived rather than tabulated. `1 - tanh(x) = 2 exp(-2x)` to leading
/// order, doubles just below `1` are spaced `eps/2 = 2^-53`, and a value
/// rounds to `1` once it is within HALF that spacing, `2^-54`. So the
/// condition is `2 exp(-2x) <= 2^-54`, i.e. `x >= -0.5 ln(eps/8)`.
///
/// The half-ulp is the part that is easy to get wrong: the first draft of
/// this function used `eps/4`, which is where `1 - tanh` equals a FULL ulp
/// and `tanh` is therefore still one ulp below `1`. A test that bisects for
/// the true switch point found it, and finds this value exactly - see
/// `tests::f_d_is_exactly_zero_above_a_threshold_this_test_locates`.
#[must_use]
pub fn tanh_saturation_argument() -> Scalar {
    let eps = Scalar::EPSILON;
    -0.5 * (eps / 8.0).ln()
}

/// `f_B = min(2 exp(-9 alpha^2), 1)`, `alpha = 0.25 - d_w/h_max` -
/// arXiv:2301.07223 (A.9).
#[must_use]
pub fn f_b(alpha: Scalar) -> Scalar {
    (2.0 * (-9.0 * alpha * alpha).exp()).min(1.0)
}

/// The `d_w/h_max` below which `f_B == 1` exactly, i.e. the RANS inner layer
/// of the WMLES branch: `0.25 + sqrt(ln 2/9) = 0.5275183`.
#[must_use]
pub fn f_b_unity_threshold() -> Scalar {
    0.25 + ((2.0 as Scalar).ln() / 9.0).sqrt()
}

/// `f_e1` - arXiv:2301.07223 (A.11), both branches.
#[must_use]
pub fn f_e1(alpha: Scalar) -> Scalar {
    let a2 = alpha * alpha;
    if alpha >= 0.0 {
        2.0 * (-11.09 * a2).exp()
    } else {
        2.0 * (-9.0 * a2).exp()
    }
}

/// `f_e2 = 1 - max(f_t, f_l)` - arXiv:2301.07223 (A.12)/(A.13).
#[must_use]
pub fn f_e2(rdt: Scalar, rdl: Scalar, ct: Scalar, cl: Scalar) -> Scalar {
    let ft = ((ct * ct * rdt).powi(3)).tanh();
    let fl = ((cl * cl * rdl).powi(10)).tanh();
    1.0 - ft.max(fl)
}

/// `f_e = f_e2 max(f_e1 - 1, 0)` - arXiv:2301.07223 (A.10).
#[must_use]
pub fn f_e(alpha: Scalar, rdt: Scalar, rdl: Scalar, ct: Scalar, cl: Scalar) -> Scalar {
    f_e2(rdt, rdl, ct, cl) * (f_e1(alpha) - 1.0).max(0.0)
}

/// `Delta_IDDES` - arXiv:2301.07223 (A.1), SPEC-LIT (57.17).
#[must_use]
pub fn delta_iddes_full(dw: Scalar, hmax: Scalar, hwn: Scalar, cw: Scalar) -> Scalar {
    (cw * dw).max(cw * hmax).max(hwn).min(hmax)
}

/// `Delta` - arXiv:2603.08875 (14), SPEC-LIT (57.18): the simplified width
/// that drops `h_wn`.
#[must_use]
pub fn delta_iddes_simple(dw: Scalar, hmax: Scalar, cw: Scalar) -> Scalar {
    (cw * dw.max(hmax)).min(hmax)
}

// ==========================================================================
//  Kernels
// ==========================================================================

/// Every entry point in `cuda/des.cu`, resolved once.
pub struct DesKernels {
    hmax: CudaFunction,
    wall_normal_step: CudaFunction,
    length_scale: CudaFunction,
    sst_rans_length: CudaFunction,
    sst_k_sink: CudaFunction,
    les_mode_mask: CudaFunction,
}

impl DesKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::DES)?;
        Ok(Self {
            hmax: k.func("desHmax")?,
            wall_normal_step: k.func("desWallNormalStep")?,
            length_scale: k.func("desLengthScale")?,
            sst_rans_length: k.func("desSstRansLength")?,
            sst_k_sink: k.func("desSstKSink")?,
            les_mode_mask: k.func("desLesModeMask")?,
        })
    }
}

// ==========================================================================
//  The length scale
// ==========================================================================

/// The hybrid length scale, and the four grid metrics it stands on.
///
/// Owns its buffers, allocates nothing in an outer iteration, and knows
/// nothing about either background model beyond the two buffers `update_sa`
/// and `update_sst` fill.
pub struct DesLengthScale {
    kern: DesKernels,
    coeffs: DesCoeffs,
    branch: DesBranch,
    delta_form: HybridDelta,
    background: HybridBackground,
    n_cells: usize,

    /// `[n_cells]` the grid metrics, all computed ONCE at setup.
    ///
    /// `hmax` is the componentwise max of the extents `lesCellExtents`
    /// gathers over the cell->face CSR - a gather with a deterministic
    /// maximum order and no atomic, which is what makes `h_max` reproducible
    /// (SPEC-LIT §57.2). `hwn` is (57.19). `dw` is the wall distance.
    dx: DevBuf<Vec3>,
    hmax: DevBuf<Scalar>,
    hwn: DevBuf<Scalar>,
    dw: DevBuf<Scalar>,

    /// `[n_cells]` `C_DES` - a constant on the SA background, `F1`-blended on
    /// the SST one - and the RANS length scale.
    cdes: DevBuf<Scalar>,
    l_rans: DevBuf<Scalar>,

    /// `[n_cells]` the output, and the three diagnostics SPEC-LIT §57.11's
    /// gates read: `f_d` (DDES) or `fdt~` (IDDES), `f_e`, and the filter
    /// width actually used.
    l_out: DevBuf<Scalar>,
    fd: DevBuf<Scalar>,
    fe: DevBuf<Scalar>,
    delta: DevBuf<Scalar>,
}

impl DesLengthScale {
    /// Build the grid metrics once.
    ///
    /// `y` and `grad_y` come from [`crate::walldistance::WallDistance`] - the
    /// same Poisson solve §6.6 already runs for SST, which is what hands
    /// IDDES its `h_wn` with no search and no new connectivity (§57.6).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: &Gpu,
        mesh: &GpuMesh,
        y: &DevBuf<Scalar>,
        grad_y: &DevBuf<Vec3>,
        branch: DesBranch,
        delta_form: HybridDelta,
        background: HybridBackground,
        coeffs: DesCoeffs,
    ) -> Result<Self> {
        coeffs.check()?;
        let n = mesh.n_cells;
        let nc = n.max(1);

        let les = LesKernels::new(gpu)?;
        let mut dx: DevBuf<Vec3> = gpu.zeros(nc)?;
        cell_extents(gpu, &les, &mut dx, mesh)?;

        let kern = DesKernels::new(gpu)?;
        let mut hmax: DevBuf<Scalar> = gpu.zeros(nc)?;
        let mut hwn: DevBuf<Scalar> = gpu.zeros(nc)?;

        if n > 0 {
            let nl = n as Label;
            let f = kern.hmax.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut hmax)
                    .arg(&dx)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
            let f = kern.wall_normal_step.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut hwn)
                    .arg(&dx)
                    .arg(grad_y)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
        }

        let fld = crate::field_ops::FieldKernels::new(gpu)?;
        let mut dw: DevBuf<Scalar> = gpu.zeros(nc)?;
        crate::field_ops::copy_field(gpu, &fld, &mut dw, y, n)?;

        // SA's `C_DES` is one number; SST's is rebuilt from `F1` every
        // update. Filling it here rather than launching a constant-fill
        // kernel every iteration keeps the hot loop to two launches.
        let cdes = gpu.upload(&vec![coeffs.cdes; nc])?;

        Ok(Self {
            kern,
            coeffs,
            branch,
            delta_form,
            background,
            n_cells: n,
            dx,
            hmax,
            hwn,
            dw,
            cdes,
            l_rans: gpu.zeros(nc)?,
            l_out: gpu.zeros(nc)?,
            fd: gpu.zeros(nc)?,
            fe: gpu.zeros(nc)?,
            delta: gpu.zeros(nc)?,
        })
    }

    #[must_use]
    pub fn branch(&self) -> DesBranch {
        self.branch
    }
    #[must_use]
    pub fn delta_form(&self) -> HybridDelta {
        self.delta_form
    }
    #[must_use]
    pub fn background(&self) -> HybridBackground {
        self.background
    }
    #[must_use]
    pub fn coeffs(&self) -> &DesCoeffs {
        &self.coeffs
    }

    /// The hybrid length: `dtil` on the SA background, `l_DES` on the SST one.
    #[must_use]
    pub fn length(&self) -> &DevBuf<Scalar> {
        &self.l_out
    }
    /// `f_d` under DDES, `fdt~` under IDDES, `0` under DES97.
    #[must_use]
    pub fn shielding(&self) -> &DevBuf<Scalar> {
        &self.fd
    }
    #[must_use]
    pub fn elevating(&self) -> &DevBuf<Scalar> {
        &self.fe
    }
    /// The filter width actually used - the one quantity a case cannot infer
    /// from its own dictionary once (57.17)'s three-way max has run.
    #[must_use]
    pub fn filter_width(&self) -> &DevBuf<Scalar> {
        &self.delta
    }
    #[must_use]
    pub fn h_max(&self) -> &DevBuf<Scalar> {
        &self.hmax
    }
    /// The wall-normal grid step of (57.19) - SPEC-LIT §57.6.
    #[must_use]
    pub fn h_wn(&self) -> &DevBuf<Scalar> {
        &self.hwn
    }
    #[must_use]
    pub fn cell_extents(&self) -> &DevBuf<Vec3> {
        &self.dx
    }
    #[must_use]
    pub fn wall_distance(&self) -> &DevBuf<Scalar> {
        &self.dw
    }
    #[must_use]
    pub fn rans_length(&self) -> &DevBuf<Scalar> {
        &self.l_rans
    }

    /// SPEC-LIT §57.1 on the Spalart-Allmaras background: `l_RANS = d_w`.
    ///
    /// `nut` is the PREVIOUS outer iteration's eddy viscosity - the lag
    /// SPEC-LIT §57.9 names as a fixed point in the outer iteration rather
    /// than an order dependence inside a kernel.
    pub fn update_sa(
        &mut self,
        gpu: &Gpu,
        nut: &DevBuf<Scalar>,
        grad_frob: &DevBuf<Scalar>,
        _y: &DevBuf<Scalar>,
        nu: Scalar,
        n: usize,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        self.launch_length_scale(gpu, nut, grad_frob, nu, n, true)
    }

    /// SPEC-LIT §57.1 on the k-omega SST background:
    /// `l_RANS = sqrt(k)/(beta* omega)` and `C_DES = C_DES1 F1 + C_DES2(1-F1)`,
    /// both per cell, then the same length-scale kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn update_sst(
        &mut self,
        gpu: &Gpu,
        k: &DevBuf<Scalar>,
        omega: &DevBuf<Scalar>,
        f1: &DevBuf<Scalar>,
        nut: &DevBuf<Scalar>,
        grad_frob: &DevBuf<Scalar>,
        nu: Scalar,
        beta_star: Scalar,
        n: usize,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let nl = n as Label;
        let f = self.kern.sst_rans_length.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut self.l_rans)
                .arg(&mut self.cdes)
                .arg(k)
                .arg(omega)
                .arg(f1)
                .arg(&beta_star)
                .arg(&self.coeffs.cdes1)
                .arg(&self.coeffs.cdes2)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        self.launch_length_scale(gpu, nut, grad_frob, nu, n, false)
    }

    fn launch_length_scale(
        &mut self,
        gpu: &Gpu,
        nut: &DevBuf<Scalar>,
        grad_frob: &DevBuf<Scalar>,
        nu: Scalar,
        n: usize,
        l_rans_is_dw: bool,
    ) -> Result<()> {
        let nl = n as Label;
        let branch = self.branch as Label;
        let delta_form = self.delta_form as Label;
        let c = self.coeffs;
        let f = self.kern.length_scale.clone();

        // On the SA background `l_RANS` IS the wall distance, so the same
        // buffer is handed in twice. Both are read-only, and `cuda/des.cu`
        // marks neither `__restrict__` for exactly that reason - which is
        // also what makes plain SA "the hybrid with the substitution not
        // made" rather than a second code path.
        let Self {
            l_out,
            fd,
            fe,
            delta,
            l_rans,
            dw,
            hmax,
            hwn,
            cdes,
            ..
        } = self;
        let lr: &DevBuf<Scalar> = if l_rans_is_dw { dw } else { l_rans };

        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(l_out)
                .arg(fd)
                .arg(fe)
                .arg(delta)
                .arg(lr)
                .arg(nut)
                .arg(grad_frob)
                .arg(&*dw)
                .arg(&*hmax)
                .arg(&*hwn)
                .arg(&*cdes)
                .arg(&nu)
                .arg(&c.kappa)
                .arg(&c.cdt1)
                .arg(&c.cdt2)
                .arg(&c.ct)
                .arg(&c.cl)
                .arg(&c.cw)
                .arg(&branch)
                .arg(&delta_form)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    /// SPEC-LIT (57.4): overwrite the `k` equation's `sp` with
    /// `beta* omega (l_RANS/l_DES)`.
    ///
    /// Called only when a hybrid is attached, and AFTER `sstKSources` has
    /// written `sp`. `cuda/sst.cu` is byte-for-byte unmodified; the ratio
    /// form is what makes `l_DES == l_RANS` reproduce `beta* omega` **bit for
    /// bit**, because multiplication by an exact `1.0` is exact.
    pub fn stamp_sst_k_sink(
        &self,
        gpu: &Gpu,
        sp: &mut DevBuf<Scalar>,
        omega: &DevBuf<Scalar>,
        beta_star: Scalar,
        n: usize,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let nl = n as Label;
        let f = self.kern.sst_k_sink.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(sp)
                .arg(omega)
                .arg(&self.l_rans)
                .arg(&self.l_out)
                .arg(&beta_star)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    /// SPEC-LIT §57.8's counter: `1` where the hybrid has put a
    /// boundary-layer cell into LES mode, and the destruction amplification
    /// `(d/dtil)^2` beside it.
    ///
    /// The count comes from the SAME device buffer the model uses, never from
    /// a host re-derivation that could agree with a wrong length scale.
    pub fn les_mode_mask(
        &self,
        gpu: &Gpu,
        mask: &mut DevBuf<Scalar>,
        amplification: &mut DevBuf<Scalar>,
        delta_bl: Scalar,
        n: usize,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let nl = n as Label;
        let f = self.kern.les_mode_mask.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(mask)
                .arg(amplification)
                .arg(&self.l_out)
                .arg(&self.dw)
                .arg(&delta_bl)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    #[must_use]
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }
}

#[cfg(test)]
mod tests;
